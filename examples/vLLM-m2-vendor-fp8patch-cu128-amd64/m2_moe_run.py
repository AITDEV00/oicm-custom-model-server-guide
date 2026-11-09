#!/usr/bin/env python3
"""ARCHIVED prototype: NOT an operational entrypoint.

The completed artifact was produced by m2_sweep.py with m2_tile_worker.py, not
this unfinished coordinator. Retain for diagnostics; do not launch for tuning
or production activation. Raw completed results were explicitly taken only from
the real minimal single-H200 sweep/evidence export.

Historical draft description follows:
Resumable, single-H200 M2 synthetic expert-tile tuning coordinator.

Purpose: benchmark the pinned vendor kernels with M2's local-expert TP8+EP8
shape, persist EVERY outcome per trial, and prepare a native loader file. All
synthetic results remain incomplete association until the final same-artifact
real-model acceptance. No weights, engines or downloader processes start here.

Preflight includes a CPU-only (GPU_VISIBLE_DEVICES="") vendor loader test. It
checks actual imports and JSON filename without allocating a GPU. --trial must
be repeated successfully after the deployment scale-down and a real pinned one
GPU standalone pod is granted. It is NEVER safe to execute on the serving node
based solely on the existence of CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import m2_moe_core as core

SCHEMA = "m2.moe.tuning.progress.v2"
ARTIFACT_SCHEMA = "m2.moe.tuning.evidence.v2"
NATIVE_ARTIFACT = core.ARTIFACT_NAME


def search():
    # Same 640 vendor combinations, excluding 256K which is clamped to 128.
    # Default is always timed first as well as a separately designated index.
    raw = [{"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": group, "num_warps": warps, "num_stages": stages}
           for bm, bn, warps, group, stages in product(
               [16, 32, 64, 128, 256], [128, 256], [4, 8],
               [1, 16, 32, 64], [2, 3, 4, 5])]
    result = core.effective_candidates(raw)
    result.remove(core.DEFAULT)
    return [core.DEFAULT, *result]


def atomic_json(path: Path, value):
    core.write_json(path, value)
    # Directory metadata durable enough for a Pod emptyDir / host filesystem.
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        os.fsync(fd)
    finally:
        os.close(fd)


def repair_progress(path, plan):
    """Recover complete results under invalid-prior-null classification only.

    Partial runs tied to another executable are preserved but never resumed.
    """
    if not path.exists():
        return {"schema": SCHEMA, "plan": plan, "started_unix": time.time(),
                "trials": {}, "failures": {}, "events": []}
    prior = json.loads(path.read_text())
    if prior["plan"] == plan:
        return json.loads(path.read_text())
    assert prior["plan"].get("source_sha256", {}).get("m2_tile_worker.py") == plan["source_sha256"]["m2_tile_worker.py"]
    result = load_progress(path, plan)
    log = path.parent / "events.jsonl"
    lines = log.read_text().splitlines() if log.exists() else []
    failure_history = list(prior.get("failures", {}).values())
    for entry in failure_history + [json.loads(line) for line in lines]:
        if isinstance(entry.get("measurement"), dict):
            row = entry["measurement"]
            if row.get("numeric_pass") is not False:
                row["index"] = entry["index"]
                result["trials"][f"{row['M']}:{row['index']}"] = row
        marker = "M2_TRIAL_RESULT="
        for line in entry.get("stdout", "").splitlines():
            if not line.startswith(marker):
                continue
            row = json.loads(line[len(marker):])
            if row.get("numeric_pass") is not False:
                row["index"] = entry["index"]
                result["trials"][f"{row['M']}:{row['index']}"] = row
    result["failures"] = {k: v for k, v in prior.get("failures", {}).items()
                         if v["status"] != "ok" and k not in result["trials"]}
    result["events"] = prior.get("events", [])
    atomic_json(path, result)
    return result


def load_progress(path: Path, plan: dict):
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["schema"] == SCHEMA
        assert doc["plan"] == plan, "Prior tuning provenance/geometry mismatch"
        assert isinstance(doc["trials"], dict)
        for key, r in doc["trials"].items():
            m, index = (int(x) for x in key.split(":"))
            assert m in plan["sizes"] and 0 <= index < plan["candidates_per_m"]
            assert r["M"] == m and r["index"] == index
            assert r["time_us"] > 0 and core.tile(r["config"]) == r["config"]
            if r.get("numeric_pass") is not None:
                assert r["numeric_pass"] is True, r
        return doc
    return {"schema": SCHEMA, "plan": plan, "started_unix": time.time(),
            "trials": {}, "events": [], "source_workspace": str(Path(__file__).parent),
            "scope": "synthetic local-experts, no model engine"}


def event(progress, path, **detail):
    progress["events"].append({"unix": time.time(), **detail})
    progress["events"] = progress["events"][-300:]
    atomic_json(path, progress)


def persist(progress, path, test_env: dict, command, stdout, stderr,
            m, index, timeout, elapsed, status="ok"):
    key = f"{m}:{index}"
    config = search()[index]
    output = output_from_log(stdout)
    event_data = {"status": status, "command_sha256": hashlib.sha256(
        " ".join(command).encode()).hexdigest(), "stdout": stdout[-16000:],
        "stderr": stderr[-16000:], "m": m, "index": index, "config": config,
        "elapsed_s": elapsed, "unix": time.time()}
    if output:
        assert output["M"] == m and output["config"] == config
        if status != "ok":
            progress.setdefault("failures", {})[key] = event_data
        elif output.get("numeric_pass") is False:
            status = "numerical_failed"
        else:
            output["index"] = index
            output["coordinator_elapsed_s"] = elapsed
            if status == "process_failed":
                # Unexpected exception was earlier in this trial, but final exit
                # zero invalidates automatic failure if benchmark emits a result.
                status = "ok"
            progress["trials"][key] = output
            progress.get("failures", {}).pop(key, None)
    if output is None or status != "ok":
        progress.setdefault("failures", {})[key] = event_data
    event_data["measurement"] = output
    with (path.parent / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event_data, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(path, progress)
    return output, status


def output_from_log(stdout):
    results = []
    marker = "M2_TRIAL_RESULT="
    for line in stdout.splitlines():
        if line.startswith(marker):
            results.append(json.loads(line[len(marker):]))
    assert len(results) <= 1
    return results[-1] if results else None


def run_subprocess(command, timeout, env=None, output_path=None, stop_budget=False):
    # Without stdin (unlike stale relay commands), this uses devnull so failure
    # cannot block a pod command wanting an input stream.
    start = time.monotonic()
    import signal
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
              env=env or os.environ.copy(), start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr, time.monotonic() - start
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return 124, stdout, stderr + "\nTRIAL_TIMEOUT", time.monotonic() - start


def worker_environment():
    for key in (
        "M2_TUNING_CONFIRM", "M2_TUNING_TARGET_UID", "M2_TUNING_NODE",
        "M2_NO_MODEL_DOWNLOAD", "M2_NO_SERVICE_PATCH",
    ):
        assert key in os.environ
    assert os.environ["M2_TUNING_CONFIRM"] == "standalone-offline"
    assert os.environ["M2_TUNING_TARGET_UID"] == core.TARGET_UID
    assert os.environ["M2_TUNING_NODE"] == "prd-infr-k8h200"
    assert os.environ["M2_NO_MODEL_DOWNLOAD"] == "1"
    assert os.environ["M2_NO_SERVICE_PATCH"] == "1"
    assert "M2_MAINTENANCE_CONFIRM" not in os.environ or os.environ[
        "M2_MAINTENANCE_CONFIRM"] == "spare-h200-or-m2-stopped"
    assert Path(__file__).resolve() == Path("/material/m2_moe_run.py"), (
        "Expected immutable reviewed ConfigMap source, not model tree or PVC")
    # The worker explicitly enforces this process's one GPU, while the
    # coordinator's outer environment could retain node-wide visibility.
    env = os.environ.copy()
    env["M2_TUNING_TRIAL_GPU_COUNT"] = "1"
    # Dedicated process consumes only scheduler-visible GPU, not actual M2.
    assert os.environ.get("M2_TUNING_TRIAL_GPU_COUNT", "1") == "1"
    env.update({"PYTHONPATH": "/material:/opt/vllm:/opt/vllm/benchmarks",
                "PYTHONUNBUFFERED": "1"})
    return env


def stage_plan(args):
    assert len(args.batch_size) == len(set(args.batch_size))
    plan = core.plan(core.validate_sizes(args.batch_size), args.seed, args.iters,
                     core.benchmark_source_hash())
    # Every process always starts with the same seed and exact provided image.
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
              for name in ("m2_moe_core.py", "m2_moe_run.py", "m2_tile_worker.py")}
    plan.update({"candidates_per_m": 320, "source_sha256": hashes,
                 "native_candidate_k": 128, "prior_runtime": core.IMAGE_DIGEST,
                 "subprocess_per_trial": True})
    if args.max_tiles:
        assert 1 <= args.max_tiles <= 320
        plan["max_tiles"] = args.max_tiles
    elif args.search_space == "representative":
        count = len(representative())
        plan.update({"search_space": "representative", "max_tiles": count + 1})
    return plan


def representative():
    """Non-exhaustive shortlist when the maintenance interval is bounded.
    M spans 16..256; N/Group/warps/stages cover vendor-supported extremes.
    The 320-tile space remains available for resuming a full search.
    """
    records = [{"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": group, "num_warps": warps, "num_stages": stages}
               for bm, bn, group, warps, stages in product(
                   [16, 64, 256], [128, 256], [1, 32], [4, 8], [3, 5])]
    indices = []
    for tile in records:
        if tile not in search():
            continue
        i = search().index(tile)
        if i not in indices:
            indices.append(i)
    assert len(indices) >= 40
    return indices


def execute_trial(args, env, m, index, timeout=None, output_folder=None):
    cfg = search()[index]
    cmd = [sys.executable, "/material/m2_tile_worker.py", "--m", str(m),
           "--config", json.dumps(cfg, separators=(",", ":")),
           "--iters", str(args.iters), "--seed", str(args.seed)]
    if args.numerics:
        cmd.append("--numeric")
    cwd = Path.cwd()
    os.chdir("/material")
    try:
        return (cmd, *run_subprocess(cmd, timeout or args.timeout, env=env), cfg)
    finally:
        os.chdir(cwd)


def run_trials(args, env, work, progress, path, deadline):
    # The single visible GPU pins actors implicitly by subprocess isolation.
    for m in progress["plan"]["sizes"]:
        completed = 0
        start = time.monotonic()
        total = progress["plan"].get("max_tiles", 320)
        indices = ([0, *representative()] if progress["plan"].get("search_space") == "representative"
                   else range(total))
        total = len(indices)
        for index in indices:
            key = f"{m}:{index}"
            if key in progress["trials"] and progress["trials"][key]["time_us"] > 0:
                print("RESUME_DONE", key, flush=True)
                completed += 1
                continue
            remaining = deadline - time.monotonic()
            if remaining < args.timeout + 10:
                print("TIME_BUDGET_REACHED_CLEAN_CHECKPOINT", json.dumps({"M": m, "index": index}), flush=True)
                return
            cmd, returncode, stdout, stderr, elapsed, cfg = execute_trial(
                args, env, m, index)
            result, status = persist(progress, path, env, cmd, stdout, stderr,
                                     m, index, args.timeout, elapsed,
                                     status="ok" if returncode == 0 else "process_failed")
            output = {k: result[k] for k in ("M", "config", "time_us",
                       "elapsed_s", "numeric_pass")} if result else {
                           "M": m, "config": cfg, "elapsed_s": elapsed,
                           "status": status, "stderr_tail": stderr[-600:]}
            output["index"] = index
            print("M2_PROGRESS", json.dumps(output), flush=True)
            if returncode == 0 and (result is not None):
                completed += 1
            event(progress, path, kind="m_or_tile_completed", M=m,
                  candidate=index, returncode=returncode,
                  preceding_median_s=elapsed)
            print("M2_BUDGET", json.dumps({"m": m, "done": completed, "total": total,
                   "elapsed_s": time.monotonic() - start,
                   "eta_s": elapsed * max(0, total - completed)}), flush=True)
    assert completed >= 0


def native_document(progress, triton_version):
    assert len({row["triton_version"] for row in progress["trials"].values()}) == 1
    return {"triton_version": triton_version,
            **{str(m): winner(progress, m)["config"] for m in progress["plan"]["sizes"]}}


def run_finalists(args, env, work, progress, path):
    """Three reproducibility units: same tensors for candidate/default check;
    the vendor timing uses its seed and CUDA graphs, not E2E throughput."""
    tables = {}
    repeated = args.repeated_rounds
    assert repeated >= 1
    test_iters = max(100, args.iters)
    for m in progress["plan"]["sizes"]:
        rows = [r for key, r in progress["trials"].items() if key.startswith(f"{m}:")]
        top = min(rows, key=lambda r: r["time_us"])
        pairs = []
        for round_number in range(repeated):
            for label, config in (("candidate", top["config"]), ("default", core.DEFAULT)):
                cmd = [sys.executable, "/work/m2_tile_worker.py", "--m", str(m),
                       "--config", json.dumps(config, separators=(",", ":")),
                       "--iters", str(test_iters), "--numeric", "--seed", str(args.seed)];cmd[1]="/material/m2_tile_worker.py"
                os.chdir("/work")
                code, stdout, stderr, elapsed = run_subprocess(cmd, args.timeout, env)
                record = output_from_log(stdout)
                if code or record is None:
                    raise RuntimeError(f"Finalist failed M={m} {label}: {stderr[-4000:]}")
                assert record["numeric_pass"] is True
                pairs.append({"round": round_number, "label": label, **record})
        med = lambda label: sorted(r["time_us"] for r in pairs if r["label"] == label)[repeated // 2]
        tables[str(m)] = {"size": m, "rounds": pairs, "winner": top["config"],
                "candidate_median_us": med("candidate"), "default_median_us": med("default"),
                "local_expert_speedup": med("default") / med("candidate"),
                "numerical_pass": all(r["numeric_pass"] for r in pairs)}
    atomic_json(work / "finalist-ab.json", tables)


def winners(progress):
    for m in progress["plan"]["sizes"]:
        yield winner(progress, m)


def winner(progress, m):
    matched = [row for row in progress["trials"].values()
               if row["M"] == m]
    assert matched, f"No usable synthetic tiles for {m}"
    return min(matched, key=lambda row: row["time_us"])


def assemble(args, work, progress):
    complete = []
    for m in progress["plan"]["sizes"]:
        matched = [r for r in progress["trials"].values() if r["M"] == m]
        usable = [r for r in matched if r["time_us"] > 0]
        assert usable, f"No measured config for M={m}"
        winner = min(usable, key=lambda r: r["time_us"])
        complete.append(winner)
    if not (work / "finalist-ab.json").exists():
        raise RuntimeError("Incomplete sweep: repeated finalists are required; "
                           "resume with --trial/--sweep instead of activating this")
    ab = json.loads((work / "finalist-ab.json").read_text(encoding="utf-8"))
    assert set(ab) == {str(m) for m in progress["plan"]["sizes"]}
    for m, row in ab.items():
        assert row["numerical_pass"]
        assert row["winner"] == winner(progress, int(m))["config"]
    # Output first as a separate manifest; the native compact JSON is the only
    # thing that can ever be mounted under VLLM_TUNED_CONFIG_FOLDER.
    triton_version = complete[0]["triton_version"]
    artifact = {
        "schema": ARTIFACT_SCHEMA, "triton_version": triton_version,
        "shape": core.SHAPE, "plan": progress["plan"],
        "winners": complete, "synthetic_only": True,
        "no_2x_guarantee": True, "requires_full_model_acceptance": True,
        "trial_count": len(progress["trials"]),
        "finalists_default_repeated": (work / "finalist-ab.json").exists(),
        "finalist_ab": json.loads((work / "finalist-ab.json").read_text(encoding="utf-8")) if (work / "finalist-ab.json").exists() else None,
        "requires_repeated_finalists_for_activation": True,
        "created_unix": time.time(),
    }
    output = args.output.resolve()
    core.scratch_dir(output)
    native = native_document(progress, triton_version)
    core.parse_native_artifact(native, progress["plan"]["sizes"])
    atomic_json(output / "evidence.json", artifact)
    atomic_json(output / core.ARTIFACT_NAME, native)
    report = core.validate_artifact(output / core.ARTIFACT_NAME, output)
    atomic_json(output / "native-loader-verification.json", report)
    print(f"NATIVE_ARTIFACT_READY={output / core.ARTIFACT_NAME}", flush=True)
    print(f"EVIDENCE={output / 'evidence.json'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--meta", type=Path, default=Path(material.meta if False else "/work/meta"))
    p.add_argument("--trial", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--native-validate", action="store_true")
    p.add_argument("--work", type=Path, default=Path("/work/m2-tuner-work/m2-tuned-moe"))
    p.add_argument("--output", type=Path, default=Path("/work/m2-tuner-work/m2-tuned-moe"))
    p.add_argument("--batch-size", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-tiles", type=int, default=0)
    p.add_argument("--search-space", choices=("full", "representative"), default="full")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-runtime", type=int, default=20000)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--numerics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repeated-rounds", type=int, default=3)
    p.add_argument("--expected-gpu-allocation", type=int, default=1)
    return p


def main(args=None):
    args = parse_args().parse_args() if args is None else args
    assert [args.preflight, args.trial, args.sweep, args.assemble, args.native_validate].count(True) == 1
    if args.preflight:
        # Only no GPU geometry and source imports. This is not a claimed GPU pass.
        import torch
        report = core.synthetic_parser_check(args.meta)
        print("CPU_VLLM_IMPORT_NATIVE_PARSER_OK", json.dumps(report), flush=True)
        assert torch.cuda.is_available() is False, "CPU parser validation must not touch a GPU"
        return
    if args.native_validate:
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        sample = json.loads(Path('/material-native/native.json').read_text(encoding='utf-8'))
        assert set(sample) == {'triton_version', '1', '2', '4', '8'}
        import shutil
        out = Path('/work/m2-tuned-moe');out.mkdir(exist_ok=True)
        shutil.copy('/material-native/native.json', out / core.ARTIFACT_NAME)
        print(json.dumps(core.validate_artifact(out / core.ARTIFACT_NAME, out)), flush=True)
        print('NATIVE_H200_LOADER_PASSED', flush=True)
        return
    env = worker_environment()
    assert core.benchmark_source_hash() == core.BENCHMARK_SHA256
    work = core.scratch_dir(args.work)
    output = core.scratch_dir(args.output)
    assert work == output, "Separate scratch and final directories reduce native artifact risk"
    work.mkdir(parents=True, exist_ok=True)
    path = work / "checkpoint-v2.json"
    plan = stage_plan(args)
    progress = load_progress(path, plan)
    if args.assemble:
        assert args.max_runtime > 0
        assemble(args, work, progress)
        return
    progress = repair_progress(path, plan)
    args.numerics = True
    atomic_json(path, progress)
    if args.trial:
        args.numerics = True  # first A/B diagnostic must really compare outputs
        cmd, code, stdout, stderr, elapsed, cfg = execute_trial(args, env, 1, 0,
                                                               timeout=args.timeout)
        result, status = persist(progress, path, env, cmd, stdout, stderr,
                                 1, 0, args.timeout, elapsed, "ok" if code == 0 else "failed")
        print("TRIAL_STDOUT", stdout[-4000:], flush=True)
        print("TRIAL_STDERR", stderr[-4000:], flush=True)
        print("TRIAL_RESULT", json.dumps({"code": code, "seconds": elapsed,
              "result": result, "status": status}), flush=True)
        assert code == 0 and result is not None
        return
    assert args.sweep
    deadline = time.monotonic() + args.max_runtime
    run_trials(args, env, work, progress, path, deadline)
    event(progress, path, kind="bounded_sweep_exited", total_trials=len(progress["trials"]))
    for m in progress["plan"]["sizes"]:
        count = sum(row["M"] == m for row in progress["trials"].values())
        assert count >= progress["plan"].get("max_tiles", 320), f"Incomplete M={m}; resume required"
    assert time.monotonic() + 24 * args.timeout < deadline, "Finalist budget expired; resume"
    run_finalists(args, env, work, progress, path)
    assemble(args, work, progress)


if __name__ == "__main__":
    main()
