# Diagnostics & tooling

Utility containers that help validate the OICM runtime environment.

## [diagnostics](diagnostics/index.md)

A lightweight **OICM runtime environment diagnostic** image that mirrors the
real serving image's security surface (uid 10000, gid 0, `/home/runner`, port
8080) **without running vLLM**, so probe results reflect what the real image
faces. It runs 16 probe sections (identity, filesystem, mounts, GPU/CUDA 12.9,
cgroups, network, env redaction, pod spec, EXTRA_ARGS verdict) and never aborts
early — every probe reports pass/fail.

## [cputests](cputests.md)

A small CPU-only diagnostic service that reports CPU core count + topology,
runs a CPU throughput benchmark (sysbench + Python baseline), and benchmarks
Silero VAD v5 streaming speed on CPU via onnxruntime (no torch, no CUDA).