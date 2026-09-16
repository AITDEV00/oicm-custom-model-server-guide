# SGLang v0.5.19 + Kimi-K3 kv-free fix (PR #38982) — derived image example

Immutable backport of **only** PR #38982 onto the SGLang **v0.5.19** release,
layered on top of the official cu130 image. No GLM-5.x DSA/DCP (#31821), no
GLM-5.3-Flash #36507, no other main commits.

## What it fixes

`UnifiedRadixCache.cache_finished_req` frees a request aborted mid-prefill as
two ranges that touch at `effective_len`. Handed to the allocator separately,
the second starts mid-page and `_page_disjoint` asserts on the shared boundary
page — every scheduler rank dies with the request. Observed on **Kimi-K3**
(TP8, DSPARK, page_size 64) three times in production in two days, each right
after an abort of a 120–200K-character prompt.

The fix: coalesce touching `(kv_indices, start_pos)` segments of one kv row
into single segments (a slice plus its touching continuation is exactly the
slice they were cut from) before handing them to `free_kv_row_segments`.

## Provenance (frozen — never rebuilt from a live PR ref)

| Item | SHA |
|------|-----|
| v0.5.19 base | `59f20bffdd` (tag commit `0bcd822377`) |
| PR #38982 head at backport time | `9bf4ab4928` |
| Backport commit | `0e71fd8af0` |

Patch: `kimi-k3-v0.5.19.patch` (= `git diff --binary v0.5.19..HEAD` from
`/home/jyao/ADEO/mlops/sglang`, branch `backport/v0.5.19-kimi-k3-free-segments`).
The Docker build applies this local file — it never downloads the PR.

## Manual resolution made during the backport

Hunk 2 of the PR diff did not apply to v0.5.19: the SWA bookkeeping in
`free_kv_row_segments` differs slightly from PR-branch `main`
(`swa_dead` is `list[torch.Tensor]` here, not `list[tuple[...]]`). Applied
manually: docstring extension + iterate over `_coalesce_touching_segments(segments)`.
Hunk 1 (`_coalesce_touching_segments` helper) applied cleanly, and the PR's
unit test `test/registered/unit/mem_cache/test_free_kv_row_segments_touching.py`
is included.

## Build

```bash
# add a Makefile target or run directly:
podman build --platform=linux/amd64 \
  -t oicm/sglang:0.5.19-cu130-amd64-kimi-k3 \
  -f SGLang-podman-v0.5.19-cu130-amd64-kimi-k3/Dockerfile \
  SGLang-podman-v0.5.19-cu130-amd64-kimi-k3
```

## Verify before pushing

```bash
IMG=oicm/sglang:0.5.19-cu130-amd64-kimi-k3

# 1. Editable source tree is the one loaded
podman run --rm -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache "$IMG" python3 -c \
  "import sglang; print(sglang.__version__)"
# expect: 0.5.19

# 2. The fix is present in the patched tree
podman run --rm --entrypoint grep "$IMG" -c "_coalesce_touching_segments" \
  /sgl-workspace/sglang/python/sglang/srt/mem_cache/common.py
# expect: >= 2

# 3. Run the PR's unit test inside the image
podman run --rm -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache --entrypoint python3 \
  "$IMG" -m pytest test/registered/unit/mem_cache/test_free_kv_row_segments_touching.py -q
```
