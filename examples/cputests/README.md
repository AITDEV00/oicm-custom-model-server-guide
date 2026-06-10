# CPU + Silero-VAD Diagnostic

A small CPU-only diagnostic service. It reports CPU core count + topology,
runs a CPU throughput benchmark (sysbench + a dependency-free Python baseline),
and benchmarks **Silero VAD v5** streaming speed on CPU via `onnxruntime`
(no torch, no GPU). Results stream live and are kept in a history you can
fetch back.

## Build & run

```bash
docker compose up --build
# or plain docker:
docker build -t cpu-vad-diag .
docker run --rm -p 8000:8000 -v diag-data:/data cpu-vad-diag
```

Benchmark a specific core allocation (mirrors a k8s cpuset / `--cpus` limit):

```bash
docker run --rm -p 8000:8000 --cpuset-cpus="0-3" cpu-vad-diag
```

## Endpoints

| Method | Path                  | Purpose                                            |
|--------|-----------------------|----------------------------------------------------|
| GET    | `/`                   | endpoint index                                     |
| GET    | `/health-check`       | liveness + capability probe (sysbench/lscpu/VAD)   |
| POST   | `/run`                | run benchmark; **streams NDJSON** progress, logged |
| GET    | `/runs`               | all stored run outputs (history); `?full=true`     |
| GET    | `/runs/{run_id}`      | single run's full output                           |
| GET    | `/runs/{run_id}/log`  | single run's raw progress log                      |

### Run a benchmark (streaming)

```bash
curl -N -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"repetitions":3,"cpu_max_prime":20000,"vad_chunks":2000,"vad_threads":1}'
```

Each line is one JSON event (`accepted`, `start`, `cpu_info`, `rep_start`,
`cpu_begin`/`cpu_done`, `vad_begin`/`vad_done`, `rep_done`, and finally
`result` with the aggregated record). The same stream is written to
`/data/runs/{run_id}.log`, so progress is never lost even if the client
disconnects.

### Request parameters

| field           | default | meaning                                        |
|-----------------|---------|------------------------------------------------|
| `repetitions`   | 1       | how many times to repeat (mean/min/max/stdev)  |
| `cpu_threads`   | all     | sysbench worker threads                         |
| `cpu_max_prime` | 20000   | sysbench cpu workload size                      |
| `vad_chunks`    | 2000    | 512-sample (~32ms) chunks streamed through VAD  |
| `vad_threads`   | 1       | onnxruntime intra-op threads (per-stream)       |
| `skip_cpu`      | false   | skip the CPU benchmark                           |
| `skip_vad`      | false   | skip the VAD benchmark                            |

### Fetch history

```bash
curl http://localhost:8000/runs            # slim summaries
curl http://localhost:8000/runs?full=true  # everything
curl http://localhost:8000/runs/<run_id>   # one run
```

## Notes on the VAD number

- The benchmark streams audio one 512-sample chunk at a time — the realistic
  streaming-VAD access pattern — and reports mean / p50 / p95 / p99 / max
  per-chunk latency plus RTF and ×realtime.
- Synthetic noise is used: Silero runs the same graph regardless of input, so
  compute cost equals that of real speech. Use real audio only to validate
  detection quality, not speed.
- Silero per-stream inference doesn't parallelise well, so `vad_threads=1` is
  the meaningful per-session number; raise it to observe (limited) scaling.
- `OMP_NUM_THREADS` is set to 1 in the image for predictable per-stream timing.
