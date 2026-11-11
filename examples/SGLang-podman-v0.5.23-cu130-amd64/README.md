# SGLang 0.5.23 CUDA 13.0 OICM image

Build:

```bash
make build-sglang-podman-0.5.23-cu130
```

For an Al Ain cold transfer, use the verified **durable Docker archive →
authorized public HF transport repo → Al Ain bastion `hfs3` → MinIO →
containerd → Harbor** route. [HF_RELAY.md](HF_RELAY.md) records the exact
completed artifact, run and checks. The canonical full procedure is in the
[canonical full procedure](../../../devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).

The supported first-publication wrapper is
`make deliver-sglang-podman-0.5.23-cu130`: its Python script validates the
unchanged source ID/Docker archive and uses the protected write token for
`upload-public-hf.py`. It requires an explicit public-disclosure approval and
refuses an exact already-present archive. Use a single manually managed
hfs3 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` session for the following
bastion handoff. No MinIO secret is written to its YAML. Install the existing
tested `huggingface-hub` package where the local Python needs it; no virtualenv
activation changes code behavior. After the uploader succeeds, compare public
LFS size/SHA and record the resolved main commit, then pin that exact commit
in `hfs3-alain.yaml` before starting bastion ingest.

MinIO creates no Harbor publication by itself. For only the authorized
incomplete remaining node-Harbor leg use `complete_harbor.py --retry-push`
once, not the retired OCI helpers. Its registry Basic auth comes from the
existing protected non-public runtime store.
It uses Bash `-c` with protected `kubectl -i` stdin, so userdata never appears
in the controller's `kubectl argv`; ctr's ultimate Basic `--user` **will** remain
visible to appropriately privileged same-host process inspection. This is a
limitation of the installed 1.7.28 `ctr` interface and is not a claimed fix.
A standard Podman auth store with delegated support can be used where its
native tooling supports it; there is no allowed inline manual secret fallback.

Verified complete Harbor tag:

```text
registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.23-cu130-amd64
```

No model deployment was changed. No GPU/model inference test or DeepSeek crash
fix is implied by registry delivery.

[Complete Harbor](complete_harbor.py) is the reviewed archive-based finalizer
for a future incomplete delivery; `--retry-push` doesn't re-transfer anything
when the local raw-node archive and target descriptor are already right. All
credentials come from the existing protected runtime store, not source/history.
The original successful execution operated `ctr` via a protected local 0600
credentials file through the separate single-writer staging-session stdin.
It was not a `make` invocation. No --auth-file is supported by ctr--see the
notes before relying on an alternative method.

[Verify Al Ain](verify-alain.py) is the separate read-only registry verifier;
supply the trusted CA via `--tls-ca-bundle` when available and keep the scoped
TLS exception only for the tested registry path when required. The optional
`harbor-auth.sh preflight` is read-only; its direct-push mode is not the default.
`sleep` callers and helper wrappers, not process output alone, decide completion.
Unsupported historical recovery code and its exposed credential defaults were
removed. Only a nonsecret retirement note remains in
`../retired-v0.5.23-recovery-drafts/`; no old OCI helper is part of this route.
