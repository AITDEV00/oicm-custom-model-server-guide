"""A single synthetic local-expert Triton trial.

Isolated invocation releases Ray worker/Triton JIT memory after each trial or
whitelisted timeout. No request, model, InferenceEngine or context ever exists.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import m2_moe_core as core


def tensor_digest(x) -> dict:
    import torch
    return {"sha256": hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
            "is_finite": bool(torch.isfinite(x).all().item())}


def minimum_runtime_guard():
    assert os.environ.get("M2_TUNING_CONFIRM") == "standalone-offline"
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") == 0, (
        "Each phase-1 trial subprocess is pinned to exactly one granted H200"
    )
    assert os.environ.get("M2_NO_MODEL_DOWNLOAD") == "1"


def verify_baseline_shape():
    import triton
    report = {"triton_version": triton.__version__,
              "vendor_checkpoint_fingerprint": core.benchmark_source_hash()}
    assert report["vendor_checkpoint_fingerprint"] == core.BENCHMARK_SHA256
    return report


def workload_inputs(seed: int, m: int):
    import torch
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    torch.set_default_device("cuda")
    torch.manual_seed(7500 + seed)
    # This exactly matches vendor benchmark tensor shape/dtypes and positive
    # block scales, while fixing source tensors between running configs.
    x = torch.randn(m, 6144, dtype=torch.bfloat16)
    w1 = (torch.randn((24, 3584, 6144), dtype=torch.float16) / 24).to(
        torch.float8_e4m3fn)
    w2 = (torch.randn((24, 6144, 1792), dtype=torch.float16) / 40).to(
        torch.float8_e4m3fn)
    gating = torch.randn(64, m, 24, dtype=torch.float32) / 3
    w1_scale = torch.rand((24, 28, 48), dtype=torch.float32) * 1e-2 + 1e-4
    w2_scale = torch.rand((24, 48, 14), dtype=torch.float32) * 1e-2 + 1e-4
    quant = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=w1_scale, w2_scale=w2_scale, block_shape=[128, 128])
    topk_weights, topk_ids, _ = fused_topk(x, gating[0], 8, renormalize=True)
    x_original = x.clone()
    return x, x_original, w1, w2, topk_weights, topk_ids, quant


def run_once(config, m):
    import torch
    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    x, x_copy, w1, w2, weights, ids, quant = workload_inputs(91, m)
    with override_config(core.tile(config)):
        result = fused_experts(x, w1, w2, weights, ids, inplace=True,
                               quant_config=quant, allow_deep_gemm=False)
    torch.cuda.synchronize()
    assert x_copy.data_ptr() == x.data_ptr()
    assert result.shape == x_copy.shape
    assert torch.isfinite(result).all().item()
    return result.detach().clone()


def compare(baseline, result):
    import torch
    assert torch.isfinite(result).all().item() and torch.isfinite(baseline).all().item()
    # Tile changes affect associativity and low bits. Use same inputs and seed.
    # A loose tolerance does NOT validate M2 model quality; it's just a kernel
    # equivalence diagnostic restricting obvious wrong-scale/layout failures.
    torch.testing.assert_close(result, baseline, atol=0.008, rtol=1e-3)
    delta = (result.float() - baseline.float()).abs()
    return {
        "all_close_atol_0p008_rtol_0p001": True,
        "max_abs_diff": delta.max().item(),
        "mean_abs_diff": delta.mean().item(),
        "checksum_finite": True,
    }


def execute(size, config, iterations, numeric_reference=True):
    from benchmarks.kernels.benchmark_moe import benchmark_config
    import torch
    torch.manual_seed(size)
    started = time.time()
    actual = run_once(config, size)
    timing = benchmark_config(
        core.tile(config), size, 24, 3584, 6144, 8,
        torch.bfloat16, True, False,
        num_iters=iterations, block_quant_shape=[128, 128], use_deep_gemm=False,
    )
    result = {
        "config": core.tile(config), "M": size, "time_us": timing,
        "finite": True, "tried": True,
    }
    if numeric_reference:
        assert torch.equal(run_once(config, size), actual)
        reference = run_once(core.DEFAULT, size)
        result["numeric_reference"] = compare(reference, actual)
        result["comparison"] = "same-input synthetic kernel A/B vs current default"
    result["elapsed_s"] = time.time() - started
    return result


def main():
    # Single trial when driven by the coordinator. Ray reporting stays on stdout
    # as structured JSON, even if Python stderr has progress logs.
    size = int(os.environ["M2_TRIAL_M"])
    config = json.loads(os.environ["M2_TRIAL_CONFIG"])
    iterations = int(os.environ["M2_TRIAL_ITERS"])
    reference = os.environ.get("M2_TRIAL_NUMERIC", "0") == "1"
    result = execute(size, config, iterations, reference)
    print("M2_TRIAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()