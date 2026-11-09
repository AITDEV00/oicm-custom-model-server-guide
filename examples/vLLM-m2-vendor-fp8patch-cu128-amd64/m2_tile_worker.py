#!/usr/bin/env python3
"""One bounded synthetic local-expert FP8 Triton trial on ONE allocated H200.

The unchanged vendor benchmark uses CUDA graphs on exact E/N/K/block shapes.
The numerical check compares identical FP8 weights/scales/routing/activations
against the current default tile. It is a kernel sanity check, NOT model-quality
proof. No model, PVC, downloader, service, or source-code patch is involved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import m2_moe_core as core


def runtime_guard():
    assert os.environ.get("M2_TUNING_CONFIRM") == "standalone-offline"
    assert os.environ.get("M2_TUNING_TARGET_UID") == core.TARGET_UID
    assert os.environ.get("M2_NO_MODEL_DOWNLOAD") == "1"
    assert os.environ.get("M2_NO_SERVICE_PATCH") == "1"
    assert os.environ.get("M2_TUNING_TRIAL_GPU_COUNT") == "1"
    assert core.benchmark_source_hash() == core.BENCHMARK_SHA256
    assert not Path("/pvc-home").exists(), "Model PVC is unnecessary and forbidden"


def workload_inputs(seed: int, m: int):
    import torch
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    # Same deterministic tensors for default and candidates, as in the vendor
    # benchmark: positive block scales, FP8 e4m3, BF16 input and top-8 routing.
    torch.manual_seed(seed + 7500)
    x = torch.randn(m, core.SHAPE["K"], dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn((24, 3584, 6144), dtype=torch.float16).to(torch.float8_e4m3fn)
    w2 = torch.randn((24, 6144, 1792), dtype=torch.float16).to(torch.float8_e4m3fn)
    gating = torch.randn(m, 24, dtype=torch.float32)
    w1_scale = torch.rand((24, 28, 48), dtype=torch.float32) * 1e-2
    w2_scale = torch.rand((24, 48, 14), dtype=torch.float32) * 1e-2
    quant = FusedMoEQuantConfig.make(quant_dtype=torch.float8_e4m3fn,
        w1_scale=w1_scale, w2_scale=w2_scale, block_shape=[128, 128])
    topk_weights, topk_ids, _ = fused_topk(x, gating, 8, renormalize=True)
    return x, x.clone(), w1, w2, topk_weights, topk_ids, quant


def dense_once(config, inputs):
    import torch
    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    x, x_pointer, w1, w2, weights, ids, quant = inputs
    with override_config(core.tile(config)):
        result = fused_experts(x, w1, w2, weights, ids, inplace=True,
                               quant_config=quant, allow_deep_gemm=False)
    torch.cuda.synchronize()
    assert result.shape == x_pointer.shape and torch.isfinite(result).all().item()
    # Vendor fused_experts_impl may return a distinct output buffer despite the
    # in-place named dispatch argument. Do not assert an incorrect alias identity.
    return result.detach().clone()


def numerical_comparison(reference, result):
    import torch
    diff = (reference.float() - result.float()).abs()
    return {"max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
            "max_reference_abs": reference.float().abs().max().item(),
            "finite": bool(torch.isfinite(result).all().item()),
            "atol": 0.008, "rtol": 0.001, "seed": 91,
            "note": "synthetic kernel sanity only, not full-model quality proof",
            "passed": bool(torch.allclose(result, reference, atol=0.008, rtol=1e-3))}


def execute(m, config, iterations, numeric, seed):
    import torch
    from benchmarks.kernels.benchmark_moe import benchmark_config
    runtime_guard()
    torch.set_default_device("cuda")  # Vendor BenchmarkWorker does this too.
    torch.manual_seed(seed)
    assert torch.cuda.device_count() == 1
    gpu = torch.cuda.get_device_name(0)
    assert "H200" in gpu and "H200" in core.ARTIFACT_NAME
    start = time.perf_counter()
    inputs = workload_inputs(91, m)
    actual = dense_once(config, inputs)
    metrics = None
    if numeric:
        reference = dense_once(core.DEFAULT, inputs)
        metrics = numerical_comparison(reference, actual)
    # Restore exactly the vendor's benchmark seed after numerical checks.
    torch.manual_seed(seed)
    dense_us = float(benchmark_config(core.tile(config), m, 24, 3584, 6144,
        8, torch.bfloat16, True, False, num_iters=iterations,
        block_quant_shape=[128, 128], use_deep_gemm=False))
    assert dense_us > 0
    return {"schema": "m2.moe.trial.v1", "M": m, "config": core.tile(config),
            "time_us": dense_us, "numeric_reference": metrics,
            "numeric_pass": None if metrics is None else metrics["passed"],
            "iterations": iterations, "seed": seed, "gpu": gpu,
            "triton_version": __import__("triton").__version__,
            "elapsed_s": time.perf_counter() - start,
            "benchmark_sha256": core.BENCHMARK_SHA256, "source_shape": core.SHAPE,
            "scope": "synthetic local-expert w13+w2 only, not end-to-end speed"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--config", type=json.loads, required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--numeric", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print("M2_TRIAL_RESULT=" + json.dumps(
        execute(args.m, args.config, args.iters, args.numeric, args.seed), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
