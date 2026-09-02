#!/usr/bin/env python3
"""Real-Time Factor (RTF) benchmark for the VibeVoice ASR vLLM server.

RTF = processing_time / audio_duration.  RTF < 1 means faster than real time.

Sends each audio file as a base64 data-URL through /v1/chat/completions
(the VibeVoice multimodal path) and measures wall-clock server time, which
includes audio preprocessing (24 kHz resample + acoustic/semantic tokenizers)
plus decode. Reports per-file RTF and aggregate stats.

Usage:
    python3 rtf_benchmark.py --base-url http://localhost:8321 \
        --audio-dir /home/jyao/ADEO/vibevoice.cpp/audio \
        --runs 3 [--warmup 1]
"""

import argparse
import base64
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import requests

MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".m4a": "audio/mp4", ".ogg": "audio/ogg"}

SEG_RE = re.compile(
    r'\{"Start":[\d.]+,"End":[\d.]+,"Speaker":\d+,'
    r'"Content":"(?:[^"\\]|\\.)*"\}')


def audio_duration(path: Path) -> float:
    """Duration via ffprobe (fallback: wave header)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        import wave
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()


def transcribe(base_url: str, path: Path, model: str, timeout: float,
               max_tokens: int):
    """POST one chat-completions ASR request. Returns (elapsed, raw_content)."""
    b64 = base64.b64encode(path.read_bytes()).decode()
    mime = MIME.get(path.suffix.lower(), "application/octet-stream")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url",
                 "audio_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }],
        # VibeVoice emits a JSON array of {Start,End,Speaker,Content} segments.
        # NOTE: the model does NOT stop after the audio ends and will keep
        # generating (hallucinated segments with growing timestamps). Size
        # max_tokens to the audio and post-filter segments whose End exceeds
        # the real duration (see main()).
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"] or ""
    return elapsed, content


def in_duration_segments(content: str, audio_len: float, margin: float = 1.15):
    """Keep only complete segments whose End <= audio_len * margin."""
    keep = []
    for m in SEG_RE.finditer(content):
        try:
            end = float(re.search(r'"End":([\d.]+)', m.group(0)).group(1))
        except AttributeError:
            continue
        if end <= audio_len * margin:
            keep.append(json.loads(m.group(0)))
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8321")
    ap.add_argument("--model", default="vibevoice")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup runs per file (not counted)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--max-tokens", type=int, default=1600,
                    help="decode cap per request; sized so the post-audio "
                         "hallucination loop is cut off early")
    args = ap.parse_args()

    files = sorted([p for p in Path(args.audio_dir).iterdir()
                    if p.suffix.lower() in MIME])
    if not files:
        raise SystemExit(f"no audio files found in {args.audio_dir}")

    print(f"{'file':<28} {'dur_s':>7} {'proc_s':>8} {'RTF':>7} {'segs':>6}")
    print("-" * 64)
    rtf_all = []
    for f in files:
        dur = audio_duration(f)
        for _ in range(args.warmup):
            try:
                transcribe(args.base_url, f, args.model, args.timeout,
                           args.max_tokens)
            except Exception as e:
                print(f"  [warmup failed: {e}]")
                break
        times, nsegs = [], 0
        for run in range(args.runs):
            try:
                elapsed, content = transcribe(args.base_url, f, args.model,
                                              args.timeout, args.max_tokens)
            except Exception as e:
                print(f"{f.name:<28} FAILED: {e}")
                break
            times.append(elapsed)
            nsegs = len(in_duration_segments(content, dur))
        if not times:
            continue
        best = min(times)
        rtf = best / dur
        rtf_all.append(rtf)
        print(f"{f.name:<28} {dur:>7.2f} {best:>8.2f} {rtf:>7.3f} {nsegs:>6}")
    print("-" * 64)
    if rtf_all:
        print(f"files: {len(rtf_all)}  median RTF: "
              f"{statistics.median(rtf_all):.3f}  min: {min(rtf_all):.3f}  "
              f"max: {max(rtf_all):.3f}  (RTF < 1 = faster than real-time)")


if __name__ == "__main__":
    main()
