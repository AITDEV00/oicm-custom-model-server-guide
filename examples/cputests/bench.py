"""
Diagnostic benchmark core.

Three independent probes, all CPU-only:
  * cpu_info()        -> core count + topology + model/flags/cache
  * cpu_benchmark()   -> sysbench events/sec (if available) + a pure-python baseline
  * vad_benchmark()   -> Silero VAD per-chunk latency / RTF via onnxruntime (no torch)

execute_run() ties them together and is a *generator* that yields progress
events, so the web layer can stream them and/or write them to a log.
"""
from __future__ import annotations

import glob
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# CPU INFO
# --------------------------------------------------------------------------- #

def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    """Run a command, return stdout text or None if it isn't available/fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout.strip() or out.stderr.strip() or None
    except Exception as exc:  # noqa: BLE001
        return f"<error running {' '.join(cmd)}: {exc}>"


def _proc_cpuinfo_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def cpu_info() -> Dict[str, Any]:
    """Collect core count and topology. Everything degrades gracefully."""
    # Cores actually usable by this process (respects cgroup/cpuset pinning).
    try:
        affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = None

    info: Dict[str, Any] = {
        "logical_cpus_os": os.cpu_count(),
        "logical_cpus_available": affinity,
        "model_name": _proc_cpuinfo_model() or platform.processor() or "unknown",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cgroup_quota": _cgroup_cpu_quota(),
    }

    lscpu = _run(["lscpu"])
    if lscpu:
        info["lscpu"] = lscpu
        # pull a few structured fields out of lscpu for convenience
        parsed: Dict[str, str] = {}
        for line in lscpu.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                parsed[k.strip()] = v.strip()
        for key in (
            "Architecture", "CPU(s)", "Thread(s) per core", "Core(s) per socket",
            "Socket(s)", "NUMA node(s)", "Model name", "CPU max MHz", "L1d cache",
            "L2 cache", "L3 cache", "Flags",
        ):
            if key in parsed:
                info.setdefault("lscpu_fields", {})[key] = parsed[key]

    topo = _run(["lstopo-no-graphics", "--no-io"])
    if topo:
        info["topology"] = topo

    return info


def _cgroup_cpu_quota() -> Optional[float]:
    """Effective CPU limit imposed by a container runtime, in cores, or None."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as fh:
            quota, period = fh.read().split()
            if quota != "max":
                return round(int(quota) / int(period), 3)
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8") as fh:
            period = int(fh.read())
        if quota > 0:
            return round(quota / period, 3)
    except (OSError, ValueError):
        pass
    return None


# --------------------------------------------------------------------------- #
# CPU BENCHMARK
# --------------------------------------------------------------------------- #

def _python_cpu_baseline(target_seconds: float = 2.0) -> Dict[str, Any]:
    """
    Deterministic, dependency-free CPU micro-benchmark.

    Counts how many integer sqrt/prime checks complete in ~target_seconds.
    Useful as a relative score even when sysbench is unavailable.
    """
    deadline = time.perf_counter() + target_seconds
    n = 2
    ops = 0
    while time.perf_counter() < deadline:
        # batch work between clock reads to keep timing overhead low
        for _ in range(2000):
            is_prime = n > 1
            limit = int(math.isqrt(n))
            for d in range(2, limit + 1):
                if n % d == 0:
                    is_prime = False
                    break
            ops += 1
            n += 1
    elapsed = target_seconds  # we ran ~this long; ops is the score
    return {
        "ops": ops,
        "approx_seconds": round(elapsed, 3),
        "ops_per_sec": round(ops / elapsed, 1),
        "last_n": n,
    }


def _parse_sysbench(text: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("events per second"):
            try:
                res["events_per_sec"] = float(s.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
        elif s.startswith("total number of events"):
            try:
                res["total_events"] = int(s.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
        elif s.startswith("total time"):
            res["total_time"] = s.split(":", 1)[1].strip()
    return res


def cpu_benchmark(threads: int, max_prime: int = 20000) -> Dict[str, Any]:
    """sysbench cpu if present, plus the python baseline (always)."""
    result: Dict[str, Any] = {
        "threads_requested": threads,
        "max_prime": max_prime,
    }

    if shutil.which("sysbench"):
        out = _run(
            [
                "sysbench", "cpu",
                f"--cpu-max-prime={max_prime}",
                f"--threads={threads}",
                "--time=0", "--events=10000",
                "run",
            ],
            timeout=120,
        )
        if out:
            result["sysbench"] = _parse_sysbench(out)
            result["sysbench_raw"] = out
    else:
        result["sysbench"] = None  # not installed in this image/host

    result["python_baseline"] = _python_cpu_baseline()
    return result


# --------------------------------------------------------------------------- #
# SILERO VAD BENCHMARK  (onnxruntime, CPU, no torch)
# --------------------------------------------------------------------------- #

_VAD_SESSION_CACHE: Dict[int, Any] = {}


def _find_silero_onnx() -> str:
    """
    Locate the Silero .onnx model.

    Priority:
      1. $SILERO_ONNX_PATH if it points to an existing file
      2. the model downloaded into the image at build time
      3. the silero-vad pip package, if it happens to be installed (local dev)
    """
    env_path = os.environ.get("SILERO_ONNX_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    default_path = "/srv/models/silero_vad.onnx"
    if os.path.exists(default_path):
        return default_path

    # local-dev fallback: only if the package is present (not in the image)
    try:
        import silero_vad  # noqa: WPS433

        base = os.path.dirname(silero_vad.__file__)
        candidates = sorted(glob.glob(os.path.join(base, "**", "*.onnx"), recursive=True))
        for c in candidates:
            if c.endswith("silero_vad.onnx"):
                return c
        if candidates:
            return candidates[0]
    except Exception:  # noqa: BLE001
        pass

    raise FileNotFoundError(
        "Silero ONNX model not found. Set SILERO_ONNX_PATH or place it at "
        f"{default_path}."
    )


def _vad_session(intra_threads: int):
    import onnxruntime as ort  # lazy import

    key = intra_threads
    if key in _VAD_SESSION_CACHE:
        return _VAD_SESSION_CACHE[key]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        _find_silero_onnx(), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    _VAD_SESSION_CACHE[key] = sess
    return sess


def vad_benchmark(
    num_chunks: int = 2000,
    threads: int = 1,
    sample_rate: int = 16000,
    chunk_samples: int = 512,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Stream `num_chunks` of audio through Silero v5 one chunk at a time
    (the realistic streaming-VAD access pattern) and report latency / RTF.

    Synthetic noise is used: the model runs the same graph regardless of the
    input, so compute cost is identical to real speech.
    """
    sess = _vad_session(threads)
    rng = np.random.default_rng(seed)
    chunks = rng.standard_normal((num_chunks, chunk_samples)).astype("float32")
    sr = np.array(sample_rate, dtype="int64")

    def fresh_state() -> np.ndarray:
        return np.zeros((2, 1, 128), dtype="float32")

    # warmup
    state = fresh_state()
    for c in chunks[: min(50, num_chunks)]:
        out = sess.run(None, {"input": c[None, :], "state": state, "sr": sr})
        state = out[1]

    # timed run, capturing per-chunk latency for percentiles
    state = fresh_state()
    per_chunk_ms: List[float] = []
    t0 = time.perf_counter()
    for c in chunks:
        c0 = time.perf_counter()
        out = sess.run(None, {"input": c[None, :], "state": state, "sr": sr})
        per_chunk_ms.append((time.perf_counter() - c0) * 1e3)
        state = out[1]
    total = time.perf_counter() - t0

    audio_seconds = num_chunks * chunk_samples / sample_rate
    per_chunk_ms.sort()

    def pct(p: float) -> float:
        if not per_chunk_ms:
            return 0.0
        idx = min(len(per_chunk_ms) - 1, int(round(p / 100 * (len(per_chunk_ms) - 1))))
        return round(per_chunk_ms[idx], 4)

    return {
        "model": os.path.basename(_find_silero_onnx()),
        "threads": threads,
        "sample_rate": sample_rate,
        "chunk_samples": chunk_samples,
        "chunk_ms": round(chunk_samples / sample_rate * 1e3, 2),
        "num_chunks": num_chunks,
        "audio_seconds": round(audio_seconds, 2),
        "total_seconds": round(total, 4),
        "mean_chunk_ms": round(statistics.fmean(per_chunk_ms), 4),
        "p50_chunk_ms": pct(50),
        "p95_chunk_ms": pct(95),
        "p99_chunk_ms": pct(99),
        "max_chunk_ms": round(per_chunk_ms[-1], 4),
        "rtf": round(total / audio_seconds, 5),
        "x_realtime": round(audio_seconds / total, 1),
    }


# --------------------------------------------------------------------------- #
# ORCHESTRATION  (generator: yields progress events)
# --------------------------------------------------------------------------- #

def _aggregate(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {}
    return {
        "mean": round(statistics.fmean(samples), 4),
        "min": round(min(samples), 4),
        "max": round(max(samples), 4),
        "stdev": round(statistics.pstdev(samples), 4) if len(samples) > 1 else 0.0,
    }


def execute_run(params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Run the benchmark `repetitions` times and yield progress events.

    Event shape: {"event": <str>, "ts": <float>, ...}
    The final event is {"event": "result", "result": {...}} containing the
    full aggregated record.
    """
    reps = int(params.get("repetitions", 1))
    skip_cpu = bool(params.get("skip_cpu", False))
    skip_vad = bool(params.get("skip_vad", False))
    cpu_threads = int(params.get("cpu_threads") or os.cpu_count() or 1)
    cpu_max_prime = int(params.get("cpu_max_prime", 20000))
    vad_chunks = int(params.get("vad_chunks", 2000))
    vad_threads = int(params.get("vad_threads", 1))

    def ev(event: str, **kw: Any) -> Dict[str, Any]:
        return {"event": event, "ts": round(time.time(), 3), **kw}

    yield ev("start", message="collecting CPU info",
             params={"repetitions": reps, "cpu_threads": cpu_threads,
                     "cpu_max_prime": cpu_max_prime, "vad_chunks": vad_chunks,
                     "vad_threads": vad_threads, "skip_cpu": skip_cpu,
                     "skip_vad": skip_vad})

    info = cpu_info()
    yield ev("cpu_info", message="CPU info collected",
             logical_cpus=info.get("logical_cpus_available") or info.get("logical_cpus_os"),
             model=info.get("model_name"))

    cpu_reps: List[Dict[str, Any]] = []
    vad_reps: List[Dict[str, Any]] = []

    for i in range(1, reps + 1):
        yield ev("rep_start", message=f"repetition {i}/{reps}", rep=i, total=reps)

        if not skip_cpu:
            yield ev("cpu_begin", message=f"[{i}/{reps}] CPU benchmark running", rep=i)
            cpu = cpu_benchmark(threads=cpu_threads, max_prime=cpu_max_prime)
            cpu_reps.append(cpu)
            sb = (cpu.get("sysbench") or {}).get("events_per_sec")
            base = cpu["python_baseline"]["ops_per_sec"]
            yield ev("cpu_done", message=f"[{i}/{reps}] CPU done", rep=i,
                     sysbench_events_per_sec=sb, python_ops_per_sec=base)

        if not skip_vad:
            yield ev("vad_begin",
                     message=f"[{i}/{reps}] Silero VAD ({vad_chunks} chunks, "
                             f"{vad_threads} thread/s)", rep=i)
            try:
                vad = vad_benchmark(num_chunks=vad_chunks, threads=vad_threads)
                vad_reps.append(vad)
                yield ev("vad_done", message=f"[{i}/{reps}] VAD done", rep=i,
                         mean_chunk_ms=vad["mean_chunk_ms"],
                         p95_chunk_ms=vad["p95_chunk_ms"],
                         x_realtime=vad["x_realtime"])
            except Exception as exc:  # noqa: BLE001
                yield ev("vad_error", message=f"VAD failed: {exc}", rep=i)

        yield ev("rep_done", message=f"repetition {i}/{reps} complete", rep=i)

    # ---- aggregate ----
    summary: Dict[str, Any] = {}
    if cpu_reps:
        sb_vals = [c["sysbench"]["events_per_sec"] for c in cpu_reps
                   if c.get("sysbench") and "events_per_sec" in c["sysbench"]]
        py_vals = [c["python_baseline"]["ops_per_sec"] for c in cpu_reps]
        summary["cpu"] = {
            "sysbench_events_per_sec": _aggregate(sb_vals) if sb_vals else None,
            "python_ops_per_sec": _aggregate(py_vals),
        }
    if vad_reps:
        summary["vad"] = {
            "mean_chunk_ms": _aggregate([v["mean_chunk_ms"] for v in vad_reps]),
            "p95_chunk_ms": _aggregate([v["p95_chunk_ms"] for v in vad_reps]),
            "x_realtime": _aggregate([v["x_realtime"] for v in vad_reps]),
        }

    result = {
        "cpu_info": info,
        "summary": summary,
        "cpu_runs": cpu_reps,
        "vad_runs": vad_reps,
    }
    yield ev("result", message="run complete", result=result)
