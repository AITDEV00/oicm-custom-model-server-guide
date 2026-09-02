#!/usr/bin/env python3
"""Transcribe long audio files via the VibeVoice vLLM endpoint, save JSON.

For each audio file:
  * POST base64 audio to /v1/chat/completions (audio_url part)
  * stream=false; generous max_tokens sized for long audio
  * salvage all complete {Start,End,Speaker,Content} segments from the content
  * keep segments whose End <= duration * margin (cuts post-audio hallucination)
  * write <name>.transcript.json next to the audio, plus a small metadata block

Usage:
  python3 transcribe_files.py --base-url http://localhost:8321 \
      --audio-dir /home/jyao/ADEO/vibevoice.cpp/video/audio_1_5x
"""

import argparse
import base64
import json
import re
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
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip())


def transcribe(base_url: str, path: Path, model: str, max_tokens: int,
               timeout: float) -> tuple[float, str, str]:
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
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    content = choice["message"]["content"] or ""
    return elapsed, content, choice["finish_reason"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8321")
    ap.add_argument("--model", default="vibevoice")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=24000,
                    help="long audio: ~14 tokens/s of speech at 7.5 Hz frames")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--margin", type=float, default=1.05,
                    help="keep segments with End <= duration * margin")
    ap.add_argument("--files", nargs="*", help="subset of filenames")
    args = ap.parse_args()

    d = Path(args.audio_dir)
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in MIME)
    if args.files:
        files = [p for p in files if p.name in args.files]

    for path in files:
        out_path = path.with_suffix(".transcript.json")
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        dur = audio_duration(path)
        print(f"[run ] {path.name} ({dur:.1f}s audio) ...", flush=True)
        try:
            elapsed, content, finish = transcribe(
                args.base_url, path, args.model, args.max_tokens, args.timeout)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        segs = [json.loads(m.group(0)) for m in SEG_RE.finditer(content)]
        kept = [s for s in segs if s["End"] <= dur * args.margin]
        result = {
            "file": path.name,
            "audio_duration_s": round(dur, 3),
            "processing_time_s": round(elapsed, 3),
            "rtf": round(elapsed / dur, 4),
            "finish_reason": finish,
            "raw_segments_total": len(segs),
            "segments_in_duration": len(kept),
            "last_end_s": kept[-1]["End"] if kept else None,
            "transcript": kept,
        }
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"  done: {elapsed:.1f}s -> RTF {elapsed/dur:.3f}, "
              f"{len(kept)}/{len(segs)} segments (last End "
              f"{kept[-1]['End']:.1f}s)" if kept else
              f"  done: {elapsed:.1f}s, NO segments in duration")


if __name__ == "__main__":
    main()
