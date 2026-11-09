"""Capture one bounded ready-model decode using the worker’s normal profile API.

Run inside the M2 main container only after VLLM_TORCH_PROFILER_DIR is set at a
controlled new worker startup. No Deployment modifications are made by this
script. Do not start profile during weight loading/torch.compile/CUDA graph
capture. /start_profile may parallelize async CPU profiling overhead; do not
infer a measured speedup from this profiling request alone.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

PROMPT = "Write three concise bullet points explaining how expert parallelism improves mixture-of-experts inference."


def post(url: str, payload: dict | None, timeout=300):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        # vLLM profile control endpoints return successful empty bodies.
        return response.status, json.loads(body) if body else None


def warmup(url: str) -> list[dict]:
    """Single first warm request validates output splits, never profile output."""
    payload = {
        "model": "M2", "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 8, "temperature": 0,
    }
    status, data = post(url + "/chat/completions", payload)
    assert status == 200
    assert data["model"] == "M2"
    return [{"http": status, "finish_reason": data["choices"][0]["finish_reason"]}]


def snapshot(directory: pathlib.Path) -> list[pathlib.Path]:
    if not directory.exists():
        raise RuntimeError(f"Missing profiler dir: {directory}")
    return list(directory.iterdir())


def assert_stopped_profiles(directory: pathlib.Path, before, attempts=180):
    """Wait for all rank traces and summaries to remain stable for 5 seconds.
    tensorboard_trace_handler writes `.json.tmp` files, then renames them
    to `.json` once their export is complete. Count only finite completed
    names, not `.tmp` contents, and never output GB-scale raw traces here.
    """
    previous = None
    stable_count = 0
    for _ in range(attempts):
        all_current = [(p, p.stat().st_size) for p in snapshot(directory)
                       if p not in before and p.is_file()]
        complete = [p for p, _ in all_current if p.name.endswith(".json")]
        current = sorted(all_current, key=lambda row: row[0].name)
        if len(complete) == 8 and len(current) == 16:
            if current == previous:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= 5:
                return complete
        else:
            stable_count = 0
        previous = current
        time.sleep(1)
    raise RuntimeError(
        f"Profiler files not settled: {[str(p) for p in snapshot(directory)]}. "
        "Expected eight complete trace jsons and eight rank summaries. "
        "Do not restart profiling without explicit completion check."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--trace-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1, choices=(1, 8))
    args = parser.parse_args()
    assert args.trace_dir != pathlib.Path("/tmp"), "Dedicated trace dir required"
    assert not str(args.trace_dir).startswith("/pvc-home/"), "Do not write PVC"
    if not args.trace_dir.is_dir():
        raise RuntimeError("Create a dedicated writable trace dir before capture")
    actual = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if pathlib.Path(actual or "").resolve() != args.trace_dir.resolve():
        raise RuntimeError(
            "Refuse: profiler must be enabled from the new engine startup env. "
            f"Expected VLLM_TORCH_PROFILER_DIR={args.trace_dir}, got {actual!r}. "
            "Nothing staged by this audit can enable the existing engine."
        )
    if any(args.trace_dir.iterdir()):
        raise RuntimeError("Trace dir must be fresh to avoid stale exports")

    # The engine must be warm and profiler routes must have been registered
    # at that engine startup. An absent flag is a hard stop, not a permission
    # to attach arbitrary code to a live worker.
    with urllib.request.urlopen(args.base_url + "/metrics", timeout=30) as response:
        metric_names = [line for line in response.read().decode().splitlines()
                        if line.startswith("vllm:generation_tokens_total{")]
    assert metric_names, "Expected token counter not found on ready server"
    # Finish an unprofiled warm-up first. Start's synchronous vendor RPC arms
    # the workers, and the main traced request's step drives collection. Do not
    # put any extra generation between start and the target request.
    warm = warmup(args.base_url + "/v1")
    try:
        status, data = post(args.base_url + "/start_profile", None, timeout=60)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Profiler unavailable HTTP {exc.code}") from exc
    assert status == 200, (status, data)
    before = snapshot(args.trace_dir)
    print("PROFILE_STARTED", status, "warmup_unprofiled", warm, flush=True)
    captured = False
    payload = {
        "model": "M2", "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 32, "temperature": 0,
    }
    try:
        begin = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            responses = list(pool.map(lambda _: post(
                args.base_url + "/v1/chat/completions", payload), range(args.concurrency)))
        elapsed = time.perf_counter() - begin
        assert all(status == 200 and response["model"] == "M2" for status, response in responses)
        stop_status, stop_body = post(args.base_url + "/stop_profile", None, timeout=120)
        captured = True
        traces = assert_stopped_profiles(args.trace_dir, before)
    finally:
        if not captured:
            # Cleanup at most once; finally never starts a profile.
            try:
                post(args.base_url + "/stop_profile", None, timeout=120)
            except Exception:
                pass
            raise
    result = {
        "model": "M2", "prompt": PROMPT, "mode": "vendor TorchProfilerWrapper",
        "profile_start": start_status, "profile_stop": stop_status,
        "concurrency": args.concurrency, "completion_cap": 32,
        "profiled_request_elapsed_s": elapsed,
        "profiled_request_elapsed_is_diagnostic_not_acceptance_benchmark": True,
        "completion_usages": [x[1]["usage"] for x in responses],
        "traces": [{"name": p.name, "size": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in traces],
        "trace_dir": str(args.trace_dir), "warmups": warm,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
