# SGLang 0.5.23 HF relay: completed artifact evidence

This page records the delivered SGLang 0.5.23 CUDA 13.0 Linux/AMD64 **runtime
image archive**, not model weights, an SGLang distribution signature, or
GPU/inference validation. The generic procedure is in the [Al Ain image/HF
relay runbook](../../../devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).
**Do not start another upload, import, push, or rollout for this completed
artifact.**

## Artifact and immutable public source

| Identity | Exact value |
|---|---|
| Harbor target | `registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.23-cu130-amd64` |
| Source config | `sha256:ab8fc368a60473e329bde89157d84b20ce3bef8fddd5b6bf4999df65c4392cf5` |
| Raw archive | 39,787,218,432 bytes; SHA-256 `570648111409bdece6ec93192643808a30787311059c90f451e487cfe145c20a` |
| Docker contents | One exact target entry, config plus 75 uncompressed layer members; aggregate 39,786,978,626 bytes |
| Public source | [`maxJasper/sglang-0.5.23-cu130-amd64`](https://huggingface.co/maxJasper/sglang-0.5.23-cu130-amd64) |
| Pinned HF revision | `25bb4121e30d40d2257ed7b9007763f7ca0c3b0e` |
| Archive URL | [Archive at the pinned revision](https://huggingface.co/maxJasper/sglang-0.5.23-cu130-amd64/resolve/25bb4121e30d40d2257ed7b9007763f7ca0c3b0e/sglang-0.5.23-cu130-amd64.tar) |

The durable worktree archive is `image-archive/sglang-0.5.23-cu130-amd64.tar`,
created with `podman save --format=docker-archive`, not temporary storage.
For a **new**, authorized delivery, `publish_docker_archive.py` performs this
source-ID/tag guard and verified export before calling the one-HF uploader:

```bash
umask 077
podman image inspect --format '{{.Id}}' localhost/oicm/sglang:0.5.23-cu130-amd64
INITIATE_PUBLIC_HF_RELAY=I_HAVE_APPROVED_PUBLIC_ARTIFACT_TRANSIT \
  make -C .. deliver-sglang-podman-0.5.23-cu130
# The wrapper tags SOURCE to TARGET, exports the absent durable archive with
# --format=docker-archive, verifies the known raw SHA/tag/config/layers, then
# calls the uploader. Do not pre-save SOURCE: that archive lacks the final tag.
# Only after independent public LFS and resolved-main SHA metadata comparison:
# edit hfs3-alain.yaml to pin that exact commit before starting bastion work.
```

Re-running the wrapper against this existing archive intentionally exits
nonzero: its durable exact archive/source guard prevents an accidental fresh
upload. That is not a test of completed HF or Harbor publication.
The initial temporary tar disappeared after an upload did not commit it; that
alone was not proof the remote commit was absent or that later bytes were a
verified resume. The eventual upload committed the raw archive and matching
LFS SHA-256. Initial metadata retained only the original resolved main SHA
until independent readback refreshed its status. Now [hfs3-alain.yaml](hfs3-alain.yaml)
pins the exact revision, not moving `main`. Public HEAD and `/resolve/main/manifest.sha256`
readback matched the source; the bastion needs no HF token for this public
repository. MinIO's private TLS exception does not disable HF TLS.

## Required operator authentication

Use `hf auth login` with a write-capable `maxJasper` token. The upload script
explicitly reads HF CLI's stored token and verifies its write permission before
starting. An old read-only `HF_TOKEN` inherited from the environment must not be
used. Do not paste credentials into chat, source files, or public metadata.

Run only one instance of `upload-public-hf.py`. It uses HF's resumable large
folder workflow with one worker so partial progress is recorded locally and
large-file uploads can be retried. It verifies that the committed HF LFS object
has the expected size, records that exact SHA-256 in `manifest.sha256`, then
publishes metadata matching that verified digest. Do not begin bastion replay
before this verification succeeds.

The bastion uses the existing authenticated `oiadmin@10.34.104.99` session;
no remote passwords are stored in source code. The runbook shows a one-pass
Python process-substitution reader for an authorized protected hfs3 config.

## MinIO completion evidence

After HF publication, hfs3 ran **on the Al Ain bastion**. S3
credentials come from existing protected stores through the supported
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` variables. No `HFS3_*`
access-key aliases exist. Do not prompt for or store remote passwords here.
The actual journal is `/home/oiadmin/hfs3-runs/sglang-0.5.23.db`, with its tar
row last observed `completed`, `verified='hash'`.

[hfs3-alain.yaml](hfs3-alain.yaml) contains no credentials and pins the exact
public HF revision:

The independently fetched MinIO `_READY.json` is 320 bytes:

```json
{"status":"READY","repo":"maxJasper/sglang-0.5.23-cu130-amd64","revision":"25bb4121e30d40d2257ed7b9007763f7ca0c3b0e","bucket":"oi-deployment-files","prefix":"runtime/sglang-0.5.23-cu130-amd64-hf","files":5,"bytes":39787221225,"verified":"hash","completed_at":"2026-11-11T07:03:00Z"}
```

Its aggregate includes transport metadata, potentially `_READY.json`, not just
the archive. It is not one archive plus only one marker, and must not be forced
to equal the archive size. The tar alone is 39,787,218,432 bytes with ETag
`16114660a316db9969f58a1da9bf04fd-2372` (2,372 nominal 16 MiB parts); this
multipart ETag is not content SHA-256. Old MinIO aliases and one-file marker
assertions are stale; validate the current store, exact marker, journal, and
archive hash independently.

The public repo requires no HF token on the bastion. MinIO `insecure: true`
concerns only its S3 endpoint, not public HF TLS trust.

## Node import and registry completion

The existing Running `pause-push/img-staging` pod on `adeo-gpu-02` downloaded
to host `/tmp/sglang-0.5.23-cu130-amd64.tar` (`/host/tmp/...` in the pod),
with matching raw SHA-256. A presigned `mc share download --expire 12h`
`Share:` value is a secret: provide it through protected stdin, not arguments
or `tee`. The executed node import returned `done` and unpacked the embedded
target without retag. This records that import shape; do not rerun it here:

```bash
kubectl exec -n pause-push img-staging --quiet -- \
  /host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io \
  images import /host/tmp/sglang-0.5.23-cu130-amd64.tar
```

Native `ctr images list "name==$TARGET"` returns the exact target and
manifest descriptor. A 2026-11-11 live Distribution read returned Docker schema 2
(`application/vnd.docker.distribution.manifest.v2+json`) in both the HTTP
Content-Type and manifest `mediaType`, with the same proven raw digest. Offering
OCI in Accept does not prove the registry chooses an OCI response:

- Manifest `sha256:01e0d5637028f59e2836bfc8c385c393a9d722d4b971acbcfa66875569eee960`.
- Config `sha256:ab8fc368a60473e329bde89157d84b20ce3bef8fddd5b6bf4999df65c4392cf5`.
- Independent OCI Distribution read: HTTP 200, matching raw manifest digest,
  source config, and 75 ordered layer descriptors.

**The archive is imported and its Harbor push verified. No model deployment
was changed, restarted, rolled out, or runtime-certified.**

## Reviewed continuation and Basic-auth limitation

[complete_harbor.py](complete_harbor.py) is the supported extensible route,
using `--retry-push` and protected `REGISTRY_CREDS_FILE`. It is **reviewed
archive-based continuation code, not literal executed commands** for this
delivery. Compilation, focused offline tests and identity review passed; a
successful registry read returns before any write and permits this exact
provenance as already complete. Its MinIO aggregate includes all 5 files and
39,787,221,225 bytes, matching the fetched ready marker, not a one-file alias.
It checks raw source order and diff IDs against config using host `chroot`
Python on a dedicated stdin, separate from protected URL/Basic-data streams.
It documents the actual ctr `--user` limitation rather than claiming unsupported
authfile or hidden FIFO secret arguments; protected stdin only for containing
`bash -c` handoff. `verify-alain.py` is read-only; retired OCI scripts not part
of the supported delivery route. A full new-helper production redelivery was
NOT performed; no need to repeat already verified transport for testing prose.

For standard credentials, prefer local Podman `--authfile AUTH.json` with a
delegated `credsStore` through normal tooling; retain inline Basic auth JSON
in the protected file. Do not decode delegated helpers. Node-local
`podman --authfile /host/tmp/auth.json` remains unverified and is not the main
route. The original supervised single-writer push used a protected 0600
`user:password` file via the scoped privileged staging pod's stdin:

```bash
tmux new -A -s sglang0523-harbor-push
TARGET=registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.23-cu130-amd64
REGISTRY_CREDS_FILE="$HOME/.config/containers/registry-basic"
umask 077
test "$(stat -c '%a' "$REGISTRY_CREDS_FILE")" = 600
kubectl exec -i -n pause-push img-staging --quiet -- bash -c '
  set -euo pipefail
  IFS= read -r REG_CREDS
  T="$1"
  CTR=(/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io)
  test "$("${CTR[@]}" images list -q "name==$T")" = "$T"
  exec "${CTR[@]}" images push --skip-verify -u "$REG_CREDS" "$T"
' -- "$TARGET" < "$REGISTRY_CREDS_FILE"
```

This records the legacy Basic path, not another command to execute for the
completed artifact. The original `bash -s -- "$TARGET" < input` shape passed
shell script on stdin before `IFS= read -r REG_CREDS` consumed that same
stream. Do not combine a script heredoc with a protected-file redirection:
only one stdin source remains. The corrected `bash -c` shape above places only
nonsecret shell text in argv. Neither that, stdin, nor a FIFO hides the
ultimate `ctr -u "$REG_CREDS"` child argument; ctr 1.7.28 has no authfile flag.
This constraint remains unless native environment credential augmentation is
provided. Prefer a secrets-aware registry credential mechanism if this
limitation matters for the next new artifact; do not use ad hoc shell variables
as a claim that the child ctr argv is protected.

The pod has Bash, `sha256sum`, and `curl`, but no unqualified Python or
`pgrep`. Invoking `/host/usr/bin/python3` from it lacks the host shared-library
environment; do not use it without `chroot /host`. Network timings from other
transfers are not artifact evidence. Inspect the actual writer and manifest
before any narrow retry. Do not repeat source, HF, or MinIO legs once exact
registry identity is established.
Current executing read-only verifier: `verify-alain.py --insecure-registry`
locates existing protected exact-registry Basic store and reads the actual HTTP
200 config/manifest; optional config/blob SHA checks run only when explicitly
requested with `--check-blobs`.
In the supported code MinIO store alias has strict JSON checks for this exact
ready object. The live independent 0.5.23 tag reader never starts new writers
and does not request `--check-blobs` unless asked with explicit budget/space.
