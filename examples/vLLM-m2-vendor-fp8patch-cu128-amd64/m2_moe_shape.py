#!/usr/bin/env python3
"""Prepare and validate a *shape-only* M2 FusedMoE-profile descriptor.

This is not an M2 serving config and has no association with model code.
M2 is still XllmForCausalLM. Qwen3MoeForCausalLM is selected solely to reach
this vendor benchmark's existing dimension-aware `elif` parser branch.
GPU tuning uses the vendor synthetic-weight function and no model blobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TARGET_ID = "7ec325bf-0033-4a97-a688-e8ac30334349"
PINNED_IMAGE_DIGEST = (
    "sha256:d3ced94ef036e8469bbbb85b8973a31c9b48bf441b565db3b3197cef36309ba9"
)
FP8_FILENAME = (
    "E=24,N=1792,device_name=NVIDIA_H200,"
    "dtype=fp8_w8a8,block_shape=[128,128].json"
)
BLOCK = [128, 128]
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}
CONFIG_KEYS = (
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages",
)
DEFAULT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def canonical(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(canonical(value))
        f.flush()
    temp.replace(path)


def load_original(path: Path) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    groups = source["quantization_config"]["config_groups"]
    block_groups = [g["weights"] for g in groups.values()
                    if g["weights"].get("block_structure")]
    assert len(block_groups) == 1
    weights = block_groups[0]
    assert (weights["block_structure"], weights["strategy"],
            weights["num_bits"], weights["dynamic"]) == ([128, 128], "block", 8, False)
    assert source["num_experts"] == 192
    assert source["num_experts_per_tok"] == 8
    assert source["moe_intermediate_size"] == 1792
    assert source["hidden_size"] == 6144
    return source


def descriptor(source: dict) -> dict:
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "dtype": "bfloat16",
        "num_experts": source["num_experts"],
        "num_experts_per_tok": source["num_experts_per_tok"],
        "moe_intermediate_size": source["moe_intermediate_size"],
        "hidden_size": source["hidden_size"],
        "vocab_size": 32000,
        "quantization_config": {
            "weight_block_size": [128, 128],
            "input_activation": {"dynamic": True, "group_size": 128},
        },
    }


def geometry(config: dict) -> dict:
    # Stock tuner: experts/TP when EP, shard_width=2*width (no TP divide).
    assert config["num_experts"] % 8 == 0
    local = config["num_experts"] // 8
    width = config["moe_intermediate_size"]
    hidden = config["hidden_size"]
    block = config["quantization_config"]["weight_block_size"]
    assert (local, width, hidden, block) == (24, 1792, 6144, [128, 128])
    return {
        "local_expert_count": local,
        "weight_shapes": {"w13": [local, width * 2, hidden],
                          "w2": [local, hidden, width]},
        "topk": config["num_experts_per_tok"],
        "dtype": "fp8_w8a8",
        "activation_dtype_before_dynamic_quant": "bfloat16",
        "block_shape": block,
        "tp_size_argument": 8,
        "enable_expert_parallel": True,
        "benchmark_physical_gpu_count": 1,
        "filename": FP8_FILENAME,
    }


def prepare(source_path: Path, destination: Path, required_sha: str | None) -> dict:
    original_bytes = source_path.read_bytes()
    digest = hashlib.sha256(original_bytes).hexdigest()
    if required_sha:
        assert digest == required_sha, "Original model config fingerprint changed"
    source = load_original(source_path)
    synthetic = descriptor(source)
    atomic_json(destination / "config.json", synthetic)
    evidence = {
        "schema": "m2.moe.tuning.input.v1",
        "purpose": "offline local-expert synthetic FP8 Triton tile tuning only",
        "target_workload": TARGET_ID,
        "pinned_production_runtime": PINNED_IMAGE_DIGEST,
        "original_config_sha256": digest,
        "original_config_unchanged": True,
        "not_serving_config": True,
        "vendor_source": "LLM360/vllm v0.12.0-ifm ce42a74a8",
        "geometry": geometry(synthetic),
        "qv_warning": {
            "production_torch_dtype": "bfloat16",
            "actual_dynamic_activation_group_size": 128,
        },
    }
    atomic_json(destination / "provenance.json", evidence)
    return evidence


def check_parser() -> dict:
    from vllm.transformers_utils.config import get_config
    c = get_config(model=".", trust_remote_code=False)
    actual = (
        c.architectures, c.num_experts, c.num_experts_per_tok,
        c.moe_intermediate_size, c.hidden_size,
        c.quantization_config["weight_block_size"],
    )
    assert actual == (
        ["Qwen3MoeForCausalLM"], 192, 8, 1792, 6144, [128, 128]
    ), actual
    assert str(c.dtype) in ("torch.bfloat16", "bfloat16"), str(c.dtype)
    return {key: getattr(c, key) for key in
            ("num_experts", "num_experts_per_tok",
             "moe_intermediate_size", "hidden_size")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--source-sha256")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    dest = args.destination.resolve()
    assert dest not in (Path("/"), Path("/pvc-home"), Path("/license"))
    assert args.destination.name in ("m2-tuner-meta", "meta", "tuner-meta")
    assert not any(part in ("model", "base_model", "checkpoint")
                   for part in dest.parts), "Synthetic meta cannot shadow model/checkpoint"
    if args.verify:
        summary = {"vendor_hf_and_geometry": check_parser(),
                   "geometry": geometry(json.loads((dest / "config.json").read_text()))}
    else:
        assert args.source is not None, "--source required for prepare"
        summary = prepare(args.source, dest, args.source_sha256)
    print(canonical(summary), end="")


if __name__ == "__main__":
    main()
