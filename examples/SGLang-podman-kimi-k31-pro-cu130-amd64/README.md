# SGLang Kimi-K3.1-Pro (DFLASH) — OICM serving image

OICM-ready SGLang runtime for **`moonshotai/Kimi-K3.1-Pro`** (2.8T params, 104B
active, 1M context) with **DFLASH** speculative decoding.

Derived from the official `lmsysorg/sglang:dev-kimi-k31-pro` preview image plus
only the OICM harness (entrypoint + arg normalizer). No kernels are rebuilt.

---

## Why not a stock SGLang release

**SGLang v0.5.23 cannot serve this model at all.** This is a hard capability gap,
not a version-preference:

```
v0.5.23  models/kimi_k3.py
  EntryClass = [KimiK3ForConditionalGeneration, KimiK3LinearForCausalLM]
                                                    ↑ no K31

  'kimi_k31' in srt/configs/model_config.py  ->  0 hits
```

The model's own `config.json` declares:

```json
"architectures": ["KimiK31ForConditionalGeneration"],
"model_type":    "kimi_k31"
```

Neither resolves on v0.5.23, so SGLang fails at **model-class resolution** before
a single kernel runs. v0.5.23 was cut **2026-11-02**; K3.1 support landed in
`main` after that and lives on the **`kimi-k3.1-day-0`** branch (2 commits ahead
of main, 636 behind — a thin layer, not a fork).

| Ref | `KimiK31ForConditionalGeneration` | `kimi_k31` model_type | k31 kernel plugins | `k31.Dockerfile` |
|---|---|---|---|---|
| v0.5.23 | ❌ | ❌ | 0 | ❌ |
| main | ✅ | ✅ | 0 | ❌ |
| **kimi-k3.1-day-0** | ✅ | ✅ | **4** | ✅ |

## Why the base image is required (not a pip install)

Upstream's `docker/k31.Dockerfile` header is explicit: without the compiled
parts *"the server still starts and answers correctly, only **slower (silent
fallback)**"*. Startup success is therefore **not** evidence of a correct
deployment. Three mandatory components:

| Component | Requirement | Why |
|---|---|---|
| **FlashMLA** | **10** ops (stock has **8**) | +`sparse_decode_num_sm_parts` +`flashmla_token_pack_prepare` — native 96-head sparse decode and prefill token packing. PyPI `sgl-kernel` has neither (and currently tops out at 0.3.21, below even the 0.4.10 the Dockerfile mentions). |
| **FlashInfer DynBlock** | `DynBlockKernelMaxNumExperts = 1024` | K3.1 has **896 experts**; the stock header caps lower, so TRT-LLM MoE routing would fall back. |
| **Rust TreeCore** | native extension loads | Radix tree; needs `.git` (setuptools-scm) and Rust **1.92**. |

Plus three opt-in extras: **Prims-TS** (alternate DSA backends), **DeepGEMM**
(MegaMoE decode `block_m`), **Dynamo 1.6.0** (patched for K3.1 +
session-affinity).

## What this image verifies at build time

Three gates, all **hard failures** — a broken image never reaches the registry:

1. **`/opt/k31-image/verify_image.py`** (upstream's own script) — re-run to catch
   a mis-tagged or silently-repointed base.
2. **Model class importable** — imports `KimiK31ForConditionalGeneration` and
   asserts `kimi_k31` is registered in `model_config`.
3. **`k31_flags_check.py`** — asserts all 10 DFLASH flags exist in the real
   `launch_server` argparser, that `cutedsl_mla` is offered, and that
   `--enable-linear-replayssm-spec` is actually accepted.

Re-run any of them post-build:

```bash
podman run --rm -e HOME=/tmp --entrypoint python3 "$IMG" /opt/k31-image/verify_image.py
```

## Provenance

| Item | Value |
|---|---|
| Base (tag) | `lmsysorg/sglang:dev-kimi-k31-pro` |
| Base (pinned) | `@sha256:67fc77d1cbe67de27ca54d874c82757a4844743fea899ea7d0df547e30c2c391` (multi-arch index) |
| amd64 image manifest | `sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a` |
| Base config digest | `sha256:814578811527bf5d280d432e2a1e87e8f1e495accfe2b72cfa4ad5c32c0cead0` |
| Base size | 40.5 GB on disk (14.93 GiB compressed) |
| Platform | `linux/amd64` (also published for `arm64`) |
| K3.1 source commit | `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae` |
| sglang version in image | `0.0.0.dev1+g41208373c.d20261104` |
| Base last pushed | 2026-11-04T15:44:45Z |
| Verified locally | 2026-11-05 — `image verification passed` |

> ⚠️ **Pinned by digest on purpose.** `dev-kimi-k31-pro` is a *preview* tag on a
> *day-0* branch; upstream will keep moving it. The digest makes builds
> reproducible and prevents a silent base swap.

### Benchmark caveat (important)

The cookbook's published numbers were measured on a **different** image: built
from commit `9ab3c28176` *plus one unmerged change*, and the docs state the
public `dev-kimi-k31-pro` tag *"has not been checked against these commands"*.
This image is the officially recommended one, but treat its **exact performance
as unverified** until measured in your environment.

## Build

```bash
# from the examples/ directory
make build-sglang-podman-kimi-k31-pro
```

Or directly:

```bash
podman build --platform=linux/amd64 \
  -t oicm/sglang:kimi-k31-pro-cu130-amd64 \
  -f SGLang-podman-kimi-k31-pro-cu130-amd64/Dockerfile \
  SGLang-podman-kimi-k31-pro-cu130-amd64
```

The build is fast — the base layers are reused, only the harness is added.

## Prerequisites: the model AND the drafter

DFLASH needs **both** on the mounted volume. K3.1 has **no NextN/MTP head**, so
DFLASH is the only speculative algorithm available.

| | Repo | Size | Prefix |
|---|---|---|---|
| Target | `moonshotai/Kimi-K3.1-Pro` | 1,454.13 GiB | `models/moonshotai/Kimi-K3.1-Pro/` |
| **Drafter** | `RadixArk/Kimi-K3.1-Dflash2-Preview` | **5.19 GiB** | `models/moonshotai/Kimi-K3.1-Pro/Kimi-K3.1-Dflash2-Preview/` |

The drafter is **nested inside the target's prefix**. That is deliberate: OICM
mounts exactly one data volume per deployment, so a sibling prefix would never
reach the pod. In-bucket precedent: `models/Kimi-K3/Kimi-K3-DSpark/`.

### Nested layout is safe with model discovery

`startup.sh` crawls for `config.json` / `params.json` / `model_index.json` and
sorts by **shallowest depth first**, with mtime only as a tiebreaker. The target
sits at depth 1, the drafter at depth 2, so the target always wins — *even when
the drafter is newer* (verified by simulation). The drafter path is then passed
**explicitly** via `EXTRA_ARGS`, never discovered.

For a speculative-decoding deployment you may also set
`MODEL_DOWNLOAD_FOLDER` explicitly, which bypasses discovery entirely and makes
the layout deterministic.

## OICM configuration

**Image:** `registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:kimi-k31-pro-cu130-amd64`

**Model Server Arguments (`EXTRA_ARGS`)** — the cookbook's
B300 / Unified / Low-Latency (TP8) / DFLASH / HiCache-off cell, pointed at the
local drafter path:

```
--trust-remote-code
--tp-size=8
--mem-fraction-static=0.85
--max-running-requests=64
--weight-loader-prefetch-checkpoints
--model-loader-extra-config={"enable_multithread_load": true}
--reasoning-parser=kimi_k3
--tool-call-parser=kimi_k3
--speculative-algorithm=DFLASH
--speculative-draft-model-path=/pvc-home/app/download/base_model/Kimi-K3.1-Dflash2-Preview
--speculative-dflash-block-size=8
--speculative-draft-kv-cache-dtype=fp8_e4m3
--speculative-draft-attention-backend=cutedsl_mla
--enable-linear-replayssm-spec
```

> ⚠️ Use the **local path**, not the HF repo id. The cookbook writes
> `RadixArk/Kimi-K3.1-Dflash2-Preview`, which will fail in-cluster:
> `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are baked into these images.

`--port`, `--host`, `--model-path`, `--enable-metrics` are supplied by
`startup.sh` automatically — do **not** pass them.

### Flag notes

| Flag | Note |
|---|---|
| `--speculative-dflash-block-size=8` | Verify window 8 (K=7 drafted). Use **6** only for the high-throughput *decode* role. |
| `--enable-linear-replayssm-spec` | **Mandatory.** Without it DFLASH keeps one intermediate KDA state per draft token and the KV pool collapses (5.13M → 0.77M tokens measured on decode). |
| `--speculative-draft-attention-backend=cutedsl_mla` | The draft KV ring (default on) needs an MLA draft backend. |
| `--speculative-draft-kv-cache-dtype=fp8_e4m3` | Halves the draft pool. Draft shares the target's slot index space and costs ~10240 B/token in bf16. |
| `--speculative-use-block-verification` | Optional; **add it** to reproduce the drafter's published acceptance lengths (AL 4.26 GSM8K / 3.23 AIME26 at width 5). Used in the standalone recipe. |
| ~~`--kv-cache-dtype`~~ | **Do not set.** The model forces FP8 KV with page size 64 itself on this cell. |

## Verification before pushing

```bash
IMG=oicm/sglang:kimi-k31-pro-cu130-amd64

# 1. Compiled K3.1 components intact
podman run --rm -e HOME=/tmp --entrypoint python3 "$IMG" /opt/k31-image/verify_image.py
# expect: image verification passed

# 2. Model class + config hook
podman run --rm -e HOME=/tmp --entrypoint python3 "$IMG" -c \
  "from sglang.srt.models.kimi_k3 import KimiK31ForConditionalGeneration as K; print(K.__name__)"

# 3. DFLASH flags (kept in the image precisely so this is re-runnable)
podman run --rm -e HOME=/tmp --entrypoint python3 "$IMG" /app/k31_flags_check.py
# expect: all K3.1 DFLASH flags present

# 4. Entrypoint present + non-root
podman inspect "$IMG" --format '{{.Config.Entrypoint}} | {{.Config.User}}'
# expect: [/app/startup.sh] | 10000
```

Or just:

```bash
make verify-sglang-podman-kimi-k31-pro
```

## Push

### Supported cold-transfer default for v0.5.23

For a cold, authorized large-image transfer, use the [canonical Al Ain HF →
MinIO → Harbor runbook](../../../devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).
This repository-relative source link intentionally leaves the server guide's
MkDocs `docs/` tree; use the [canonical runbook on
GitHub](https://github.com/AITDEV00/devops-custom-models/blob/main/docs/oicm/runbooks/alain-image-hf-relay-runbook.md)
for the site-safe cross-repo reference. The tested SGLang v0.5.23 archive is
39.79 GB; it was imported and published as `sglang:0.5.23-cu130-amd64` on
2026-11-11. Image transport does not deploy a model or verify this K3.1
preview's serving path.

### OCI-directory MinIO relay — valid internal low-delta alternative

The **existing internal OCI-directory MinIO relay below remains a valid
low-delta alternative**, not the default v0.5.23 cold-transfer route. It avoids
the public HF leg and reduces network transfer when this image harness changes,
but its MinIO/node reuse and registry identity still require validation:

```bash
make deliver-sglang-podman-kimi-k31-pro
```

### Why not `podman push`

A ~40 GB push over this box's uplink runs at roughly **0.14 MB/s per TCP
connection**, and `podman push` opens one connection per layer — and re-uploads
layers **uncompressed** even when Harbor already has them as gzip (0% dedupe).
That is days. `deliver-image.sh` relays through MinIO instead, so each leg runs
on the network that suits it:

| Leg | Rate |
|---|---|
| dev box → MinIO (`mc mirror`, 64 threads) | ~8 MB/s |
| MinIO → node (in-cluster curl) | ~117 MB/s |
| node → Harbor (containerd push) | ~27 s |

### Re-delivery is fast — measured, not assumed

**The build side.** `startup.sh` / `arg_normalizer.py` are the **last single
COPY**, and the instruction uses `COPY --chmod` so there is no follow-up `RUN`.
Measured: after editing `startup.sh`, a rebuild takes **6 s** and produces only
**2 changed layers out of 80** — the other 78 are byte-identical.

**The wire side.** `deliver-image.sh` works on an OCI **directory**, not a
monolithic tar:

1. `podman save --format=oci-dir` — one file per blob, so a re-save after a
   harness edit rewrites only the changed blob, not all 16 GB.
2. A ledger (`_delivery.json`) in S3 records delivered blob digests. It is
   written **last**, so an aborted run never claims blobs it did not send.
3. `mc mirror` (no `--overwrite`) uses size equality as a **fast preflight
   only**, not a complete correctness test. Trusted reuse must validate the
   content-addressed OCI index and every expected blob digest/content, with the
   independently known source image identity. An equal-size wrong object is not
   equal bytes, and a stale size-only cache/ledger is not proof of publication.
4. The node fetches only missing blobs, then tars the oci-dir **locally** (disk
   speed, no network) and `ctr images import`s it, and removes its scratch.
   Validate the imported config/digest and read back Harbor's manifest (config,
   manifest digest and ordered layers); do not close out from process status.

| Run | Mirror (MinIO) | Node fetch | Total |
|---|---|---|---|
| First (cold) | 15.36 GiB in **991 s** (14.5 MiB/s) | 79 blobs | ~35 min |
| Repeat (no change) | **0 s** | 2 blobs (index.json + oci-layout) | **443 s** |

The repeat run is dominated by the node's local `tar` (~16 GB) + `ctr import` +
`ctr push` — not by the network. `FRESH=1` forces the upload-everything path.

### Delivery and pipeline pitfalls

The original script bugs were found only by running the delivery end to end.
Their historical fixes and publication caveats are retained here.

1. **`kubectl cp` needs `tar` inside the container.** The nettest relay image
   ships none, so step 6 died with `exec: "tar": executable file not found`.
   Replaced with `kubectl exec -i -- bash -c 'cat > file' < local`.
2. **The container also has no `python3`.** The completeness check silently
   became a no-op path. Fixed by `chroot /host /usr/bin/python3`, which is the
   only form that works: running `/host/usr/bin/python3` directly fails on
   `libexpat.so.1`, and **inside the chroot `/host/tmp/...` must be rewritten to
   `/tmp/...`** (`${D#/host}`). Same for `tar`/`du` — the container has neither,
   but `/host/usr/bin/{tar,du}` run fine directly.
3. **`grep -q` under `set -o pipefail` can cause a false failure via SIGPIPE.**
   The preflight producer is sensitive to its reader closing early. `grep -q`
   exits at the first match; if the producer then receives SIGPIPE, `pipefail`
   reports failure despite a match. The observed ~4% rate is workload dependent,
   not universal timing. Use a non-early-exit `grep -c`, or retain the full
   output/status separately. Note this instead of changing unrelated commands.
4. **A critical import/push piped to `tail` can have its failure masked** when
   `pipefail` is not enabled, and `$?` after a later pipeline does not recover
   the `ctr` status. Capture that original command status explicitly; never
   declare Harbor publication from pipeline completion, percent output, or a
   still-running process. Read back the actual registry manifest/digests.

A fourth, non-fatal issue: `mc mirror --overwrite` compares **mtimes**, and every
`podman save` writes fresh ones — so a no-op re-delivery re-uploaded all
15.36 GiB (991 s). `--overwrite` is now applied only when `FRESH=1`.

### Re-running after an interrupted delivery

The ledger makes this cheap, **conditional on integrity and identity checks**.
If a run dies at any point, re-run the same command: matching preflight entries
may avoid transfer, while every trusted reuse requires OCI content-address and
source/manifest verification. Do not assume an old ledger or retained node tar
proves successful publication. On failure retain diagnostic evidence, inspect
actual Harbor readback, and publish again only with known authenticated input.

⚠️ The node's oci-dir (`/host/tmp/${NAME}.oci`, ~16 GB) is a **hostPath**, and it
is deliberately **kept** after a delivery: it is the node-side blob cache that
can let a subsequent run skip most objects. A **stale entry can be wrong**, and
equal size or a content-addressed filename alone does not prove its contents.
Trusted reuse needs content/digest validation against the expected OCI image
and final registry config/manifest/layer readback. The transient `.tar` (~16 GB)
**is** removed at the end of step 7. If you ever want the space back, delete the
directory on `adeo-gpu-02` — the next delivery simply re-fetches from MinIO.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Derived image: pinned base + 3 build gates + OICM harness (harness last). |
| `startup.sh` | OICM entrypoint: model discovery, `/tmp` caches, CUDA forward-compat probe, `EXTRA_ARGS` normalization, exec on 8080. |
| `arg_normalizer.py` | stdlib-only `EXTRA_ARGS` normalizer (shared with the other SGLang examples). |
| `k31_flags_check.py` | build-gate 3: asserts the DFLASH flag set exists. Kept at `/app/k31_flags_check.py` so it can be re-run against any future base. |
| `deliver-image.sh` | MinIO relay delivery with per-blob delta. |
| `img-staging-pod.yaml` | Privileged staging pod on `adeo-gpu-02` for the relay. |

`startup.sh` and `arg_normalizer.py` are byte-identical to
`SGLang-podman-v0.5.22-cu130-amd64/` — the harness is version-agnostic.

## Known limitations

- **Not the measured image.** See the benchmark caveat above.
- **TP8 = 8 GPUs.** The recipe assumes one 8-GPU B300 node. A 4-GPU box cannot
  hold the 2.8T model in native MXFP4.
- **This cookbook cell is unverified upstream.** B300 `Unified` `Low-Latency` is
  marked *in progress*; the only **Verified** DFLASH cell is B300 `Decode`
  `Balanced (DCP8 · EP2)` (1P1D, AIME26 ×8 = 1.000).
- **DFLASH + DP attention** is rejected upstream on non-NPU devices, and
  `pp_size` must be 1.
