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
| v0.5.19 base | tag commit `0bcd822377` (annotated tag object `59f20bffdd`) |
| PR #38982 head at backport time | `9bf4ab4928` |
| Backport tip | `f664ce5fb9` (chain below) |

### Backport chain (v0.5.19 → tip)

1. `0e71fd8af0` — PR #38982: coalesce touching kv-row segments before free
   (fixes the abort-mid-prefill scheduler crash on Kimi-K3 TP8/DSPARK page-64)
2. `a1bc7427e3` / `4a57e0e164` — adapt the PR's unit test to v0.5.19 allocator
   semantics (no `_page_disjoint` here: the raw shape LEAKS 52 slots instead
   of asserting; coalescing prevents the leak)
3. `f3a7f5b08a` — **PR #34820** (merged 2026-09-09, AFTER v0.5.19): store
   mamba prefix-cache checkpoints at the configured SSM state dtype (bf16
   KDA state → fp32 track buffer, no double-round)
4. `a6d0978f6f` + `bb60318c20` — **PR #36770** (OPEN upstream): graceful
   Mamba cache exhaustion — skip caching a chunk instead of killing the
   scheduler when HiCache DMA pins all candidate slots. Consciously carried
   production hardening; drop when merged.
5. `b31c387eb8` — **PR #38157** (OPEN upstream): read host MemAvailable once
   per TP group before splitting the HiCache budget — prevents false
   "Not enough host memory available" at multi-hundred-GB L2 sizes.
6. `cc63b2ffe1` / `ee8651a1c7` / `f664ce5fb9` — v0.5.19 adaptation commits
   for #34820 (missing ForwardMetadata fields, `to_device` helper, Mamba2
   on-grid/off-grid track split — each caught by the build-time tests).

### Already in v0.5.19 — verified via merge-base ancestry, do NOT re-pick

`#33112` DCP+HiCache L2 (`1a3bea7`) · `#33639` Mamba branching (`3c533ac`) ·
`#34808` mamba checkpoint depth under DCP (`c20acee`) · `#35084` DCP prefill
sync removal (`f44a130`) · `#35412` decode checkpoint grid (`eac91ac3`;
`mamba_track_grid` is page×LCM → 512 for TP8/DCP8/page-64, correct) ·
`#36317` HiCache auxiliary load-back ownership (`5263568`).

Patch: `kimi-k3-v0.5.19.patch` (= `git diff --binary v0.5.19..HEAD` from
`/home/jyao/ADEO/mlops/sglang`, branch `backport/v0.5.19-kimi-k3-free-segments`).
The Docker build applies this local file — it never downloads the PR.

## Manual resolutions made during the backports

- **#38982 hunk 2**: v0.5.19's SWA bookkeeping in `free_kv_row_segments`
  differs slightly from PR-branch main (`swa_dead` is `list[torch.Tensor]`
  here, not `list[tuple[...]]`). Applied manually.
- **#34820**: four conflicts resolved (assert reformat + track assertions in
  `chunk_delta_h.py`; signature extensions in `kda.py`/`kda_prefill.py`;
  metadata plumbing in `hybrid_linear_attn_backend.py`), plus three
  v0.5.19-shape commits the build-time tests forced out (missing
  `ForwardMetadata` fields, `to_device` helper, Mamba2 on-grid/off-grid
  producer split).
- **Tests at build time**: PR #38982 unit test + #34820 SSM-dtype tests +
  mamba2 track-index tests run during `podman build` (15/15 pass).

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
