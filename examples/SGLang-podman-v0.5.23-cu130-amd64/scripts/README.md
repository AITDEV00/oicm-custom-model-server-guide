# SGLang 0.5.23 transport helpers

Use `make -f ../Makefile <target>` from the example directory (or run from
`examples/` with its `-C` equivalent) as the supported entry point. If invoking
`python3` or `bash` directly, run it from this example directory using the table
below—not from `scripts/`, because configuration and evidence are siblings.

| Helper | Responsibility | Default behavior |
|--------|---------------|------------------|
| `scripts/publish_docker_archive.py` | Source-ID/archive guards, first HF publication | Validates the source image and archive; refuses the completed archive. A new artifact requires explicit public-transit approval, after which export/upload can occur. |
| `scripts/upload-public-hf.py` | Uses stored HF write auth; verifies committed LFS bytes/hash then writes public metadata | **Never invoke this as a documentation check.** Execution invokes external write operations. |
| `scripts/complete_harbor.py` | Checks MinIO state/source archive/containerd target; completes an authorized incomplete registry leg | Default inspection has no node/registry write. `--verify-only` skips MinIO/Kubernetes; `--retry-push` needs explicit authorization. |
| `scripts/verify-alain.py` | Registry manifest/config/layer-descriptor readback | Read-only; optional `--check-blobs` loads and hashes the full image layers. |
| `scripts/harbor-auth.sh` | Uses supported Podman delegated auth stores | `preflight` is a read-only registry lookup; `push-only` is gated, slow, and non-default. |

The relocated helpers retain all evidence and archive paths. Specifically,
`image-archive/` is a local-only sibling of `scripts/`, never a helper source
directory. The existing archive's SHA-256, Docker config, and registry manifest
identity are unchanged by this file organization. The two non-runtime image
files, `hfs3-alain.yaml` and `img-staging-pod.yaml`, are declarative inputs;
these helper paths do not alter the Dockerfile's build contract.

Do not copy credentials into any helper or log. Read more in
[HF_RELAY.md](../HF_RELAY.md) and the [canonical runbook](../../../../devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).
