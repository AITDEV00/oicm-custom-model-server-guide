# diagnostics

- **Folder:** `examples/diagnostics/`
- **Base image:** `python:3.12-slim` + `curl` + `ca-certificates`
- **Build target:** `make build-diagnostics`

A lightweight **OICM runtime environment diagnostic** image that mirrors the
real serving image's security surface (uid 10000, gid 0, `/home/runner`,
numeric non-root USER, group-writable dirs, port 8080) **without running
vLLM**, so probe results reflect what the real image faces.

## Startup.sh — 16 probe sections (never aborts early)

1. Identity & security context (`id`, UID/GID, capabilities, SA token mount)
2. Root filesystem mode (ro/rw)
3–5. Write-probes: standard paths, likely model/volume mounts (`/pvc-home`,
   `/data-volume`, ...), vLLM cache dirs
6. Mounts
7. Disk space
8. Model presence check
9. **GPU/CUDA 12.9 support** — drives `VERDICT`: OK / MARGINAL / NOT SUPPORTED
   based on driver CUDA version vs R575
10. cgroup resource limits
11. Network / air-gap check
12. Environment vars (secrets redacted)
13. Tooling present
14. How the container was launched
15. Pod spec fetched from k8s API (best-effort, RBAC-aware)
16. EXTRA_ARGS normalization verdict

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | `python:3.12-slim`, runner uid 10000/gid 0, `NVIDIA_VISIBLE_DEVICES=all` |
| `startup.sh` | The 16-section probe driver |
| `arg_normalizer.py` | Shared single-source-of-truth normalizer (imported by `serve_report.py`) |
| `serve_report.py` | Stdlib HTTP server — `GET /health`, `GET /`, `POST /test-args`, `GET /test-runs` |
| `tests/run_arg_tests.py` | Test driver that POSTs each case in `tests/cases.txt` to `/test-args` |
| `tests/cases.txt` | Test cases grouped by `# CATEGORY:` (quoting, shapes, separators, booleans) |