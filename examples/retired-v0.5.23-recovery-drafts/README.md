# Retired v0.5.23 recovery attempts — retained conclusion only

The ten former OCI-directory recovery drafts were renamed to plain text and
then removed from the repository because they were unsupported and some still
contained exposed registry credential defaults. This directory intentionally
contains **no runnable or archival recovery code**, so stale secrets and unsafe
commands cannot be committed accidentally. Exposed values need owner rotation.

Use instead:

- The canonical Al Ain HF relay runbook in
  `devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md`.
- The supported, guarded publication/continuation/read-only verification
  helpers in `../SGLang-podman-v0.5.23-cu130-amd64/`.
- `HF_RELAY.md` and `delivery-evidence.json` in that same example directory
  for the already completed archive/HF/MinIO/containerd/Harbor evidence.

The original artifact was imported natively from its Docker archive and pushed
to Harbor; no OCI-directory helper needs to repeat that delivery. No model
deployment was changed by the transport, and registry delivery does not prove
inference correctness or fix any separate model incident.

Historical files were renamed to `.txt` and made non-executable at the
operator's request. They remain reference material only. Any credential-like
text in a discarded draft must not be treated as current authentication or
used for a new operation.
