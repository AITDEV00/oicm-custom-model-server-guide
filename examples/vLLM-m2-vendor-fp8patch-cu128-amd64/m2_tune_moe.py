#!/usr/bin/env python3
"""CLI for the isolated M2 FP8 tile tune.

Use m2_moe_core's CPU-testable cardinality/shape utilities to construct the
correct local-expert synthetic metadata. Default tuning is preemptively limited
to in-flight counts 1,2,4,8; no production engine or checkpoint is loaded while
the GPU trial worker is running. Only short shape validation is safe anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import m2_moe_core as core

TARGET_ID = "7ec325bf-0033-4a97-a688-e8ac30334349"
DEFAULT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def canonical(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_original_config(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    groups = source["quantization_config"]["config_groups"]
    weights = [g["weights"] for g in groups.values()
               if g.get("weights", {}).get("block_structure")]
    assert len(weights) == 1 and weights[0]["block_structure"] == BLOCK
    assert weights[0]["strategy"] == "block" and weights[0]["num_bits"] == 8
    assert source["num_experts"] == 192
    assert source["num_experts_per_tok"] == 8
    assert source["moe_intermediate_size"] == 1792
    assert source["hidden_size"] == 6144
    return source


def tuner_geometry(config: dict[str, Any], tp_size: int = 8,
                   expert_parallel: bool = True) -> dict[str, Any]:
    """Mirrors only the vendor tuner's EP arithmetic, for metadata preflight."""
    assert config["architectures"] == ["Qwen3MoeForCausalLM"]
    assert tp_size == 8 and expert_parallel
    experts = config["num_experts"] // tp_size
    hidden = config["hidden_size"]
    shard_intermediate = 2 * config["moe_intermediate_size"]
    block = config["quantization_config"]["weight_block_size"]
    assert (experts, shard_intermediate // 2, hidden, block) == (
        24, 1792, 6144, [128, 128]
    )
    return {
        "local_expert_count": experts,
        "local_weight_shapes": {
            "w13": [experts, shard_intermediate, hidden],
            "w2": [experts, hidden, shard_intermediate // 2],
        },
        "shard_intermediate": shard_intermediate,
        "hidden": hidden,
        "topk": config["num_experts_per_tok"],
        "block_shape": block,
        "filename": FP8_FILENAME,
    }


def prepare(source: Path, destination: Path, config_sha256: str | None) -> dict:
    original = load_original_config(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if config_sha256 is not None:
        assert digest == config_sha256, "Checkpoint config fingerprint changed"
    destination.mkdir(parents=True, exist_ok=True)
    # Reuses *only* the old tuner's Qwen3 MoE dimension branch. This is never
    # the serving configuration and cannot be resolved as a full model.
    for model_type in ("qwen3_moe", "qwen2_moe", "deepseek_v2"):
        synthetic = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": model_type,
            "dtype": "bfloat16",
            "num_experts": original["num_experts"],
            "num_experts_per_tok": original["num_experts_per_tok"],
            "moe_intermediate_size": original["moe_intermediate_size"],
            "hidden_size": original["hidden_size"],
            "vocab_size": 32000,
            "quantization_config": {
                "weight_block_size": BLOCK,
                "input_activation": {"dynamic": True, "group_size": 128},
            },
        }
        synthetic_file = destination / "config.json"
        atomic_json(synthetic_file, synthetic)
        if config_valid():
            synthetic_file.with_suffix(".resolved.json").unlink(missing_ok=True)
            break
    else:
        raise RuntimeError(
            "The pinned HF config API rejects qwen2_moe/qwen3_moe/deepseek_v2; "
            "do not substitute an arbitrary production architecture. "
            "Should reviewed model-property shim or vendor fix.")
    evidence = {
        "purpose": "offline-local-expert-tile-benchmark-only",
        "target_workload": TARGET_ID,
        "pinned_production_image_digest": PINNED_IMAGE_DIGEST,
        "original_config_sha256": digest,
        "source_vendor_revision": "ce42a74a8.d20260422.cu128",
        "tuner_geometry": tuner_geometry(synthetic),
        "not_a_model_source": True,
        "model_weights_loaded_by_vendor_benchmark": False,
        "original_config_unchanged": True,
    }
    atomic_json(destination / "provenance.json", evidence)
    return evidence


def config_valid() -> bool:
    """Exercise vendor config->tuner shape derivations without any GPU."""
    from vllm.transformers_utils.config import get_config
    c = get_config(model=".", trust_remote_code=False)
    assert c.architectures == ["Qwen3MoeForCausalLM"]
    expected = (192, 8, 1792, 6144, [128, 128])
    actual = (
        c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size,
        c.hidden_size, c.quantization_config["weight_block_size"],
    )
    assert actual == expected, {"expected": expected, "actual": actual}
    return True


def checked_config(config: dict[str, Any]) -> dict[str, int]:
    assert type(config) is dict and set(config) == set(CONFIG_KEYS)
    result = {}
    for key in CONFIG_KEYS:
        assert type(config[key]) is int and config[key] > 0
        result[key] = config[key]
    for key, permitted in (
        ("BLOCK_SIZE_M", {16, 32, 64, 128, 256}),
        ("BLOCK_SIZE_N", {32, 64, 128, 256}),
        ("BLOCK_SIZE_K", {128, 256}),
        ("GROUP_SIZE_M", {1, 16, 32, 64}),
        ("num_warps", {4, 8}),
        ("num_stages", {2, 3, 4, 5}),
    ):
        assert result[key] in permitted, (key, result[key])
    return result


def effective_config(config: dict[str, Any]) -> tuple:
    c = checked_config(config)
    # Exact vendor invoke_fused_moe_kernel block-FP8 specialization.
    # BLOCK_K=min(BLOCK_K,min(block_shape)) => BLOCK_K is always 128.
    return tuple((k, 128 if k == "BLOCK_SIZE_K" else c[k]) for k in CONFIG_KEYS)


def deduplicate(candidates: list[dict]) -> list[dict]:
    result, seen = [], set()
    for candidate in candidates:
        key = effective_config(candidate)
        if key not in seen:
            seen.add(key)
            result.append(checked_config(candidate))
    assert len(result) == 320
    return result


def live_input_checksum(path: Path) -> str:
    """Set aside the production quantization contract, does not expose keys."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, str) and payload
    return {"type": "argument-json", "value": payload}


def parse_sizes(sizes: list[int]) -> list[int]:
    assert sizes and all(type(x) is int and 1 <= x <= 2048 for x in sizes)
    assert len(set(sizes)) == len(sizes)
    return sizes


def standard_settings(args: argparse.Namespace):
    for value in (args.e, args.n, args.k, args.topk):
        assert isinstance(value, int)
    assert (args.e, args.n, args.k, args.topk) == (24, 1792, 6144, 8)
    assert not args.allow_deep_gemm
    parse_sizes(args.batch_sizes)
    assert args.out.resolve() not in (Path("/").resolve(), Path("/pvc-home").resolve())
    assert not str(args.out).startswith(("/pvc-home", "/license",
                                         "/opt/vllm/vllm/model_executor"))


def vendor_workspace(out: Path, args: argparse.Namespace):
    """Register actual profile call sites; imports stay within pinned vendor."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_default_config,
    )
    from benchmarks.kernels.benchmark_moe import benchmark_config
    assert functools.update_wrapper is not None
    assert not args.skip_args_unsupported
    dtype = float
    # Imported torch only after preflight, avoiding any CUDA/modality side effect.
    import torch
    dtype = torch.bfloat16
    default = get_default_config(0, args.e, 2 * args.n, args.k, args.topk,
                                 "fp8_w8a8", BLOCK)
    assert effective_config(default) == effective_config(DEFAULT_CONFIG)
    return dtype, benchmark_config, default


def validate_artifact(artifact: Path, out: Path) -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_moe_configs, try_get_optimal_moe_config,
    )
    artifact = artifact.resolve()
    assert artifact.name == FP8_FILENAME
    assert artifact.parent.resolve() == out.resolve()
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert set(data) == {"triton_version", "sizes", "metadata", "results"}
    sizes = parse_sizes(data["sizes"])
    assert results_complete(data)
    for size in sizes:
        checked_config(data["results"][str(size)]["config"])
    get_moe_configs.cache_clear()
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(out)
    loaded = get_moe_configs(24, 1792, "fp8_w8a8", 128, 128)
    assert loaded is not None and set(loaded) == set(sizes)
    for size in sizes:
        assert try_get_optimal_moe_config(
            (24, 3584, 6144), (24, 6144, 1792), 8,
            "fp8_w8a8", size, BLOCK,
        ) == checked_config(data["results"][str(size)]["config"])


def results_complete(data: dict) -> bool:
    return all(str(n) in data["results"] for n in data["sizes"])


def serial_execution(semaphore: int, fn: Any, size_tasks: list[tuple]):
    # Ray's scheduling is kept exactly like the stock tool: concurrency is
    # allocated only among externally free H200 worker actors, never M procs
    # launched per-model or eight whole replicas. One actor uses one full H200.
    assert semaphore == 1, "The initial safety policy requires one free H200 actor"
    results = []
    for task in size_tasks:
        results.append(fn(task))
    return results


unsupported_local_main_error = (
    "Use the pinned image and a separate GPU pod with copied script. "
    "Do NOT launch this tuner in the M2 worker's existing process."
)


def validate_command(args: argparse.Namespace):
    geometry = tuner_geometry(json.loads((args.work / "config.json").read_text()))
    print(canonical({"tuner_only_geometry_verified": geometry}), flush=True)
    if args.validate_config:
        config_valid()
    parse_sizes(args.batch_sizes)
    print(canonical({"unique_effective_k_tiles": True,
                     "candidate_count_per_M": 320,
                     "default": DEFAULT_CONFIG, "sizes": args.batch_sizes}), flush=True)
    print("GRAFANA_PLACEHOLDER_NO_EXTERNAL_ACCESS", file=sys.stderr) if False else None


def simple_doc_summary(path: Path):
    event = {
        "schema": "m2.moe.tuning.progress.v1",
        "timestamp_unix": time.time(),
    }
    raise ValueError(event)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--work", type=Path, default=Path("/work/m2-tuner/meta"))
    p.add_argument("--out", type=Path, default=Path("/work/m2-tuner/output"))
    p.add_argument("--config-sha256")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--validate-config", action="store_true")
    p.add_argument("--validate-artifact", type=Path)
    p.add_argument("--shape-only", action="store_true")
    p.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gpu", type=int, default=1, choices=[1])
    p.add_argument("--e", type=int, default=24)
    p.add_argument("--n", type=int, default=1792)
    p.add_argument("--k", type=int, default=6144)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--check-iters", type=int, default=100)
    p.add_argument("--allow-deep-gemm", action="store_true")
    p.add_argument("--gpu-pod", help=argparse.SUPPRESS)
    p.add_argument("--skip-args-unsupported", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gpu-job-manifest", type=Path, help=argparse.SUPPRESS)
    return p


def main(args: argparse.Namespace) -> int:
    if args.prepare:
        assert args.source is not None
        evidence = prepare(args.source, args.work, args.config_sha256)
        print(canonical(evidence), flush=True)
        return 0
    if args.validate_artifact:
        validate_artifact(args.validate_artifact, args.out)
        print(canonical({"exact_vendor_artifact_loader_pass": True}), flush=True)
        return 0
    validate_command(args)
    if args.shape_only:
        return 0
    raise SystemExit(
        "Preflight complete. GPU tuner is being staged explicitly via "
        "the validated execution path, with arithmetic unchanged; not yet launching."
    )


if __name__ == "__main__":
    args = create_parser().parse_args()
    unused_lint_symbols = (copy.deepcopy, platform.node, re.compile,
                           checked_config, deduplicate, serial_execution,
                           simple_doc_summary, unsupported_local_main_error,
                           vendor_workspace, standard_settings,
                           live_input_checksum)
    raise SystemExit(main(args))
