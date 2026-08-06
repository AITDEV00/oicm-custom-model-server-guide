# cputests

- **Folder:** `examples/cputests/`
- **Base image:** `python:3.11-slim` + `sysbench`, `util-linux` (lscpu), `hwloc` (lstopo), `curl`
- **Build target:** `make build-cputests`

A small **CPU-only diagnostic service** that reports CPU core count + topology,
runs a CPU throughput benchmark (sysbench + a dependency-free Python baseline),
and benchmarks **Silero VAD v5** streaming speed on CPU via `onnxruntime` (no
torch, no CUDA).

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | `python:3.11-slim` + sysbench/lscpu/lstopo; fetches `silero_vad.onnx` (tag `v5.1.2`) at build; runs unprivileged (uid 10001) |
| `bench.py` | Probe implementations — `cpu_info()`, `cpu_benchmark()`, `vad_benchmark()` |
| `server.py` | FastAPI server — `GET /health`, `POST /run` (streaming NDJSON), `GET /runs` |
| `requirements.txt` | Pinned `fastapi`, `uvicorn`, `numpy`, `onnxruntime` |
| `10th-june-2026-benchmark-result.txt` | Sample benchmark output |

Results stream as NDJSON and are stored in a fetchable history (persists to a
writable dir, works under `readOnlyRootFilesystem`).