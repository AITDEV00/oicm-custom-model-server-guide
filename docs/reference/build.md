# Build & push

All image build targets live in `examples/Makefile`. Each builds
`--platform=linux/amd64` and tags `oicm/<server>:<ver>-cu129(-amd64)` →
`localhost/oicm/...`.

## Targets

| Target | Image |
|--------|-------|
| `build-vllm` | vLLM docker, v0.19.0 |
| `build-vllm-podman-0.22.0` … `0.25.1` | vLLM podman versions |
| `build-vllm-podman-0.27.1` | vLLM podman, `vllm[audio]` + librosa |
| `build-vllm-podman-minimaxm3` | MiniMax-M3 |
| `build-vllm-podman-nightly` | Nightly |
| `build-sglang` | SGLang docker, v0.5.9 |
| `build-sglang-podman-0.5.14/0.5.15/0.5.16/0.5.17/0.5.18` | SGLang podman versions |
| `build-sglang-podman-0.5.19/0.5.21/0.5.22/0.5.23-cu130` | SGLang podman CUDA 13.0 versions |
| `build-sglang-podman-glm-5.3-flash` | SGLang podman, `glm-5.3-flash` |
| `build-cputests` | cputests diagnostic |
| `build-diagnostics` | diagnostics image |

For v0.5.23, `deliver-sglang-podman-0.5.23-cu130` is the guarded durable
archive/HF first-publication entry point; `continue-sglang-podman-0.5.23-cu130`
is the guarded continuation and `verify-sglang-podman-0.5.23-cu130` is read-only.
`push-sglang-podman-0.5.23-cu130` remains an explicitly slow-direct alternative.
They are deliberately absent from the build-only table; see the workflow below.

## Push workflow

Local `commands.txt` scratch files are not authoritative or tracked. The
canonical procedures publish to:

- **Al Ain** registry — `registry.adeoaiengine.ecouncil.ae/...`
- **Abu Dhabi** Harbor — `harbor.ai.ecouncil.ae`

### Supported Al Ain HF relay (v0.5.23)

For an authorized large-image cold transfer, use the durable Docker archive →
public HF → hfs3/MinIO → Al Ain containerd → Harbor route in the [canonical
runbook](../../../devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).
The SGLang v0.5.23 delivery was verified on 2026-11-11; image transport does
not create or change a deployment.

The [v0.5.23 example](../../examples/SGLang-podman-v0.5.23-cu130-amd64/HF_RELAY.md)
contains its dedicated `scripts/` upload/import helpers. Complete with actual
registry config/manifest/digest readback, not a successful process observation.

**Source-link caveat:** these repository-relative links work in the source
tree to directories outside MkDocs' `docs/` root. Do not move them into
top-level MkDocs navigation: use the [canonical runbook on
GitHub](https://github.com/AITDEV00/devops-custom-models/blob/main/docs/oicm/runbooks/alain-image-hf-relay-runbook.md)
for the cross-site link instead.

Some commented-out `podman save` / `rsync` transfer steps reference the remote
`adeo@10.34.104.99` host for air-gapped transfer. This **optional Kubernetes
API access** is separate from the HF/MinIO relay's working
`oiadmin@10.34.104.99` bastion and `~/hfs3-runs` configurations.

!!! warning "Credentials"
    `commands.txt` contains registry login credentials. It must remain
    uncommitted / gitignored. Previously exposed registry literals require
    rotation; use only existing protected authentication stores for new work.