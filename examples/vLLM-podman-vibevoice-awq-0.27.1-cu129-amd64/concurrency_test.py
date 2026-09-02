#!/usr/bin/env python3
"""Concurrency test for the local VibeVoice ASR server.

Fires N simultaneous transcription requests (all using the same ~46 s clip,
matching real-world load where many clients send audio at once) and reports:
  * per-request latency and throughput (tokens/s, audio-seconds/s)
  * wall-clock speedup vs serial
  * success/failure counts
  * server health after the burst (engine-crash regression check)

Usage:
    python3 concurrency_test.py --concurrency 4 --rounds 2
"""
import argparse
import base64
import concurrent.futures as cf
import json
import re
import subprocess
import time

import requests

BASE = "http://localhost:8321"
AUDIO = "/home/jyao/ADEO/vibevoice.cpp/audio/2p_argument_1_5x.wav"  # 45.7 s
SYSTEM = ("You are a helpful assistant that transcribes audio input into "
          "text output in JSON format.")
SEG_RE = re.compile(
    r'\{"Start":[\d.]+,"End":[\d.]+,"Speaker":\d+,"Content":"(?:[^"\\]|\\.)*"\}')


def audio_len(path: str) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())


def one_request(b64: str, audio_len: float, max_tokens: int):
    """One transcription request. Returns (ok, elapsed_s, segs, finish, err)."""
    payload = {
        "model": "vibevoice",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text",
                 "text": f"This is a {audio_len:.1f} seconds audio, please "
                         f"transcribe it with these keys: Start time, "
                         f"End time, Speaker ID, Content"},
                {"type": "input_audio",
                 "input_audio": {"data": b64, "format": "wav"}},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "repetition_penalty": 1.05,
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{BASE}/v1/chat/completions", json=payload,
                          timeout=600)
        el = time.perf_counter() - t0
        r.raise_for_status()
        d = r.json()
        content = d["choices"][0]["message"]["content"] or ""
        segs = [json.loads(m.group(0)) for m in SEG_RE.finditer(content)]
        kept = [s for s in segs if s["End"] <= audio_len * 1.05]
        return True, el, len(kept), d["choices"][0]["finish_reason"], None
    except Exception as e:
        return False, time.perf_counter() - t0, 0, None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="simultaneous in-flight requests")
    ap.add_argument("--rounds", type=int, default=2,
                    help="how many bursts to run")
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()

    a = audio_len(AUDIO)
    b64 = base64.b64encode(open(AUDIO, "rb").read()).decode()

    # warmup (compile kernels, prime caches) so the burst measures steady state
    print("[warmup] 1 request ...")
    ok, el, segs, fin, err = one_request(b64, a, args.max_tokens)
    print(f"[warmup] ok={ok} {el:.2f}s segs={segs} finish={fin} {err or ''}")

    lat, seg_counts = [], []
    for rd in range(1, args.rounds + 1):
        print(f"\n[burst {rd}/{args.rounds}] {args.concurrency} concurrent requests")
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(one_request, b64, a, args.max_tokens)
                       for _ in range(args.concurrency)]
            results = [f.result() for f in futures]
        wall = time.perf_counter() - t0

        oks = [r for r in results if r[0]]
        fails = [r for r in results if not r[0]]
        for i, (ok, el, segs, fin, err) in enumerate(results):
            lat.append(el)
            seg_counts.append(segs)
            tag = "OK " if ok else "ERR"
            extra = f"segs={segs} finish={fin}" if ok else f"error={err}"
            print(f"  req{i + 1}: {tag} {el:6.2f}s  {extra}")
        audio_s = sum(a for (ok, _, _, _, _) in results if ok)
        print(f"  wall: {wall:.2f}s | success {len(oks)}/{len(results)} | "
              f"audio processed: {audio_s:.0f}s in {wall:.1f}s "
              f"-> aggregate {audio_s / wall:.1f}x real-time")

    # engine-crash regression check after the burst
    h = requests.get(f"{BASE}/health", timeout=5)
    print(f"\n[post-burst] health: HTTP {h.status_code}")
    ok, el, segs, fin, err = one_request(b64, a, args.max_tokens)
    print(f"[post-burst] single request: ok={ok} {el:.2f}s segs={segs} "
          f"finish={fin} {err or ''}")
    if lat:
        lat_sorted = sorted(lat)
        n = len(lat_sorted)
        print(f"\n=== latency over {n} concurrent-mix requests: "
              f"min {lat_sorted[0]:.2f}s | median {lat_sorted[n // 2]:.2f}s | "
              f"max {lat_sorted[-1]:.2f}s ===")


if __name__ == "__main__":
    main()
