#!/usr/bin/env python3
"""Constants and CPU-testable operations for the isolated M2 FP8 tile tuner.

Everything here is independent of checkpoint storage and vLLM engine setup.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

TARGET_DEPLOYMENT = "j-7ec325bf-0033-4a97-a688-e8ac30334349"
TARGET_UID = "cbe870c3-263f-4460-a6ae-05b9d4fb9b76"
IMAGE_DIGEST = (
    "sha256:d3ced94ef036e8469bbbb85b8973a31c9b48bf441b565db3b3197cef36309ba9"
)
BENCHMARK_SHA256 = "01c06be4a33d0cff4790410249a01d38ebc5cc81a733de2e9f7fc2af1f70f76c"
ARTIFACT_NAME = (
    "E=24,N=1792,device_name=NVIDIA_H200,"
    "dtype=fp8_w8a8,block_shape=[128,128].json"
)
SHAPE = {"local_experts": 24, "N": 1792, "K": 6144, "topk": 8,
         "block": [128, 128]}
DEFAULT = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
           "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 3}
KEYS = tuple(DEFAULT)
VALID = {
    "BLOCK_SIZE_M": (16, 32, 64, 128, 256),
    "BLOCK_SIZE_N": (128, 256),
    "BLOCK_SIZE_K": (128,),
    "GROUP_SIZE_M": (1, 16, 32, 64),
    "num_warps": (4, 8),
    "num_stages": (2, 3, 4, 5),
}
SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def load_original(path: Path) -> tuple[dict, str]:
    text = path.read_bytes()
    source = json.loads(text)
    groups = source["quantization_config"]["config_groups"]
    weights = [g["weights"] for g in groups.values()
               if g["weights"].get("block_structure")]
    assert len(weights) == 1
    assert (weights[0]["block_structure"], weights[0]["strategy"],
            weights[0]["num_bits"], weights[0]["dynamic"]) == (
                list(SHAPE["block"]), "block", 8, False)
    for group in groups.values():
        activations = group["input_activations"]
        assert activations["dynamic"] and activations["group_size"] == 128
    assert (source["num_experts"], source["num_experts_per_tok"],
            source["moe_intermediate_size"], source["hidden_size"]) == (
                192, 8, SHAPE["N"], SHAPE["K"])
    return source, hashlib.sha256(text).hexdigest()


def descriptor(source: dict) -> dict:
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe", "dtype": "bfloat16", "vocab_size": 32000,
        "num_experts": source["num_experts"],
        "num_experts_per_tok": source["num_experts_per_tok"],
        "moe_intermediate_size": source["moe_intermediate_size"],
        "hidden_size": source["hidden_size"],
        "quantization_config": {
            "weight_block_size": SHAPE["block"],
            "input_activation": {"dynamic": True, "group_size": 128},
        },
    }


def prepare(source_path: Path, work: Path, pinned_digest: str | None) -> dict:
    source, digest = load_original(source_path)
    assert not pinned_digest or digest == pinned_digest
    work = work.resolve()
    assert work.name in ("meta", "m2-tuner-meta", "tuner-meta")
    assert work != Path("/pvc-home") and not str(work).startswith("/pvc-home")
    synthetic = descriptor(source)
    write_json(work / "config.json", synthetic)
    expected = {
        "schema": "m2.moe.tuning.input.v1",
        "purpose": "shape-only local-expert offline FP8 Triton tile tuner",
        "target": TARGET_UID, "runtime_image": IMAGE_DIGEST,
        "vendor_reference": "LLM360/vllm v0.12.0-ifm ce42a74a8",
        "original_config_sha256": digest, "original_source_untouched": True,
        "tuner_needs_weights": False, "not_serving_config": True,
        "geometry": synthetic_geometry(synthetic),
    }
    write_json(work / "provenance.json", expected)
    return expected


def synthetic_geometry(synthetic: dict) -> dict:
    # Same as stock tuner's TP8+EP branch, without importing torch/vLLM.
    experts = synthetic["num_experts"] // 8
    width = synthetic["moe_intermediate_size"]
    hidden = synthetic["hidden_size"]
    block = synthetic["quantization_config"]["weight_block_size"]
    assert (experts, width, hidden, block) == (24, 1792, 6144, [128, 128])
    return {"E": experts, "N": width, "K": hidden, "topk": synthetic["num_experts_per_tok"],
            "block": block, "filename": ARTIFACT_NAME,
            "synthetic_weight_shapes": {"w13": [experts, 2 * width, hidden],
                                        "w2": [experts, hidden, width]}}


def synthetic_parser_check(meta_path: Path):
    from vllm.transformers_utils.config import get_config
    config = get_config(str(meta_path), trust_remote_code=False)
    # Flatten model_config vs nested quantization config carefully for this
    # particular synthetic Qwen shape. This is intentionally NOT source auto-
    # detection of M2/Xllm, whose dimensions are not compatible with the old
    # tuner's generic shape field names.
    synthetic = config.to_dict()
    synthetic["num_experts"] = config.num_experts
    synthetic["num_experts_per_tok"] = config.num_experts_per_tok
    synthetic["moe_intermediate_size"] = config.moe_intermediate_size
    synthetic["hidden_size"] = config.hidden_size
    synthetic["quantization_config"] = {"weight_block_size": [128, 128]}
    return {"hf_parser_inputs": synthetic_geometry(synthetic),
            "engine_architecture_unchanged": "XllmForCausalLM"}


def parse_native_artifact(contents: dict, sizes: list[int] | None = None):
    """Only triton_version and integer-M are accepted by the actual loader."""
    assert set(contents) == {"triton_version", *[str(m) for m in sizes]}
    for m in sizes:
        assert type(contents[str(m)]) is dict
        assert contents[str(m)] == tile(contents[str(m)])
    assert isinstance(contents["triton_version"], str)
    return {"schema": "m2.moe.native-loader.precheck.v1", "sizes": sizes,
            "artifact_name": ARTIFACT_NAME}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, sort_keys=True, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


def validate_sizes(sizes: list[int]) -> list[int]:
    assert sizes and len(sizes) == len(set(sizes))
    assert all(type(m) is int and 1 <= m <= 2048 for m in sizes)
    return sizes


def tile(config: dict) -> dict[str, int]:
    assert type(config) is dict and set(config) == set(KEYS)
    # Effective Triton parameters for vendor 128-block FP8. Source clamps K.
    result = dict(config)
    result["BLOCK_SIZE_K"] = 128
    for key, allowed in VALID.items():
        assert type(result[key]) is int and result[key] in allowed, (key, config)
    return result


def tile_signature(config: dict) -> str:
    return json.dumps(tile(config), separators=(",", ":"), sort_keys=True)


def effective_candidates(raw: list[dict]) -> list[dict]:
    """Drop BLOCK_K=256 and other duplicate execution specialisations.

    Under production's [128,128] both BLOCK_K choices execute K=128; N must
    be128/256 from this old Triton source. Other keys may stay unchanged.
    """
    records = [DEFAULT, *raw]
    out, seen = [], set()
    canonical = tile_signature(DEFAULT)
    for record in raw:
        for key, allowed in VALID.items():
            if key != "BLOCK_SIZE_K":
                assert record[key] in allowed
        assert record["BLOCK_SIZE_K"] in (128, 256)
        signature = tile_signature(record)
        if signature not in seen:
            seen.add(signature)
            out.append(tile(record))
    assert len(out) == 320
    return out


def candidate_budget(m_counts: list[int]) -> dict:
    return {"M_sizes": validate_sizes(m_counts), "effective_candidates_per_M": 320,
            "native_vendor_count_per_M": 640,
            "candidate_numbers_include_initial_default": True,
            "source_restricted_K_effective_128": True}


def scratch_dir(path: Path) -> Path:
    path = path.resolve()
    assert path.name in ("m2-tuned-moe", "m2-tuner-work")
    assert not str(path).startswith(("/pvc-home", "/license", "/opt/vllm"))
    return path


def plan(sizes: list[int], seed: int, iters: int, benchmark_sha256: str) -> dict:
    assert seed >= 0 and iters >= 1
    return {"schema": "m2.moe.tuning.progress.v1", **SHAPE,
            "sizes": validate_sizes(sizes), "seed": seed, "tune_iters": iters,
            "benchmark_sha256": benchmark_sha256}


def benchmark_source_hash() -> str:
    path = Path("/opt/vllm/benchmarks/kernels/benchmark_moe.py")
    if not path.exists():
        raise RuntimeError("Exact pinned vendor's benchmark_moe.py is required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifact(path: Path, folder: Path):
    folder = scratch_dir(folder)
    path = path.resolve()
    assert path.name == ARTIFACT_NAME and path.parent == folder
    contents = json.loads(path.read_text(encoding="utf-8"))
    sizes = sorted(int(key) for key in contents if key != "triton_version")
    assert validate_sizes(sizes) == sizes
    # Calls the REAL same-image vLLM loader; no custom wrapper as deployment.
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_moe_configs, try_get_optimal_moe_config,
    )
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(folder)
    get_moe_configs.cache_clear()
    loaded = get_moe_configs(24, 1792, "fp8_w8a8", 128, 128)
    assert loaded is not None
    for m in sizes:
        actual = try_get_optimal_moe_config(
            (24, 3584, 6144), (24, 6144, 1792), 8, "fp8_w8a8", m, [128, 128])
        assert actual == contents[str(m)], (m, actual)
    return {"validated_vendor_loader": True, "filename": ARTIFACT_NAME,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "loaded_contents": contents,
            "shape": SHAPE, "sizes": sorted(loaded), "production_mutation_count": 0}


def plan_for_artifact(contents: dict) -> Any:
    return {k: contents["shape"][k] for k in SHAPE}
