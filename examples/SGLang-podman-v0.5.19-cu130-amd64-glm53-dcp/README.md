# SGLang v0.5.19 + GLM-5.x DSA/DCP (PR #31821) — derived image example

Immutable backport of **only** PR #31821 (GLM-5.x DSA DCP) onto the SGLang
**v0.5.19** release, layered on top of the official cu130 image. No Kimi-K3
fixes, no GLM-5.3-Flash #36507, no other main commits.

## Provenance (frozen — never rebuilt from a live PR ref)

| Item | SHA |
|------|-----|
| v0.5.19 base | `59f20bffdd` (tag commit `0bcd822377`) |
| PR #31821 head at backport time | `58e13e93e3` |
| Backport commit | `f1eb59f54f` |

Patch: `glm53-dcp-v0.5.19.patch` (= `git diff --binary v0.5.19..HEAD` from
`/home/jyao/ADEO/mlops/sglang`, branch `backport/v0.5.19-glm53-dcp`).
The Docker build applies this local file — it never downloads the PR.

## Manual resolutions made during the backport

PR #31821 head sits on a much newer main, so four spots needed adapting:

1. **`server_args.py` → `arg_groups/parallel_hook.py`**: v0.5.19 validates DCP
   in `handle_dcp_validation()` in `parallel_hook.py`, not in
   `server_args.py::_handle_dcp_validation`. The experimental DCP+EAGLE
   warning was moved there (`get_platform().is_cuda` /
   `getattr(cfg, "speculative_algorithm", None)` adapted to the `cfg` view).
2. **`dsa_backend.py`**: `use_symmetric_memory` import applied manually (context
   line drift).
3. **`kv_cache_configurator.py`**: DSA pool keeps the kernel page size 64 —
   `page_size=get_schedule().page_size` instead of `self.pool_page_size`.
4. **`forward_mla.py`**: `lse = None` initialization in `forward_absorb_core`
   (hunk context drifted by ~88 lines).

## Build

```bash
make build-sglang-podman-v0.5.19-glm53-dcp
# or directly:
podman build --platform=linux/amd64 \
  -t oicm/sglang:v0.5.19-cu130-glm53-dcp-f1eb59f54f-amd64 \
  -f SGLang-podman-v0.5.19-cu130-amd64-glm53-dcp/Dockerfile \
  SGLang-podman-v0.5.19-cu130-amd64-glm53-dcp
```

Tag includes the backport commit so an image tag always pins exact code.

## Verify before pushing

```bash
IMG=oicm/sglang:v0.5.19-cu130-glm53-dcp-f1eb59f54f-amd64

# 1. Editable source tree is the one loaded
podman run --rm "$IMG" python3 -c \
  "import sglang, sglang.srt; print(sglang.__version__, sglang.srt.__file__)"
# expect: ... /sgl-workspace/sglang/python/sglang/srt/__init__.py

# 2. DCP args exist
podman run --rm "$IMG" python3 -m sglang.launch_server --help 2>&1 \
  | grep -E 'dcp-size|dcp-comm-backend'

# 3. DSA/DCP implementation present
podman run --rm "$IMG" grep -rl 'dcp' /sgl-workspace/sglang/python/sglang/srt/layers/dcp/
```

## Smoke test (B300 / GLM-5.3)

Follow PR #31821's validated progression — start small, do NOT begin with
DCP8 + EAGLE:

```bash
python3 -m sglang.launch_server \
  --model-path RadixArk/GLM-5.3-NVFP4 \
  --tp 4 --dcp-size 2 \
  --quantization modelopt_fp4 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.80
```

Ladder: TP4+DCP2 → TP4+DCP4 → TP4+DCP2+EAGLE 5/1/6 (experimental) → 8-GPU
topology.
