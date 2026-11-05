#!/usr/bin/env python3
"""Build-time gate: the K3.1 DFLASH launch flags must exist in this image.

Kimi-K3.1-Pro's serving recipe hard-depends on every flag below. They are not
optional niceties:

  --speculative-algorithm DFLASH
      K3.1 has NO NextN / MTP head, so DFLASH (a separate draft checkpoint) is
      the only speculative algorithm available for this model.

  --speculative-draft-model-path
      Required unconditionally by _handle_dflash(); omitting it is a hard
      ValueError at startup, not a fallback.

  --speculative-dflash-block-size
      The verify window. The cookbook uses 8 (K=7 drafted) for unified
      low-latency, 6 (K=5) for the high-throughput decode. Also propagates to
      --speculative-num-draft-tokens, which _handle_dflash() forces them to
      agree on.

  --speculative-draft-kv-cache-dtype fp8_e4m3
      Halves the draft KV pool. The draft shares the target's slot index space
      and costs ~10240 B/token in bf16, so this is what buys back real KV
      capacity.

  --speculative-draft-attention-backend cutedsl_mla
      The draft KV ring (default on) needs an MLA draft backend.

  --speculative-use-block-verification
      The drafter's published acceptance lengths (AL 4.26 GSM8K / 3.23 AIME26
      at width 5) were measured WITH block verification. Without it, DFLASH
      degrades to the sampled path and those numbers do not apply.

  --enable-linear-replayssm-spec
      MANDATORY for DFLASH on this model. Without ReplaySSM, DFLASH keeps one
      intermediate KDA state per draft token and the KV pool collapses
      (5.13M -> 0.77M tokens measured on decode).

  --mamba-full-memory-ratio, --weight-loader-prefetch-checkpoints,
  --model-loader-extra-config
      Carried by the low-latency recipe (KDA state pool sizing and load speed).

This runs against `sglang.launch_server --help`, i.e. the argparser the pod will
actually use. A missing flag is a guaranteed startup failure on a GPU node, so
we fail the BUILD and keep the image out of the registry.

A script file rather than a Dockerfile heredoc: podman's builder misparses
heredoc bodies passed to RUN (the same reason the kimi-k3 example ships
import_check.py as a file).

Needs no GPU. Safe to re-run any time:
    python3 /app/k31_flags_check.py
"""

import subprocess
import sys

REQUIRED = [
    ("--speculative-algorithm", "DFLASH selector (K3.1 has no NextN/MTP head)"),
    ("--speculative-draft-model-path", "drafter directory (hard error if absent)"),
    ("--speculative-dflash-block-size", "verify window: 8 low-latency / 6 high-throughput"),
    ("--speculative-draft-kv-cache-dtype", "fp8_e4m3 for the draft KV pool"),
    ("--speculative-draft-attention-backend", "cutedsl_mla (MLA draft backend)"),
    ("--speculative-use-block-verification", "block verify (how AL numbers were measured)"),
    ("--enable-linear-replayssm-spec", "MANDATORY: without it the KV pool collapses"),
    ("--mamba-full-memory-ratio", "KDA state pool sizing"),
    ("--weight-loader-prefetch-checkpoints", "low-latency recipe"),
    ("--model-loader-extra-config", "low-latency recipe"),
]


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "sglang.launch_server", "--help"],
        capture_output=True,
        text=True,
    )
    # `--help` exits 0 on success; some builds emit the usage on stderr.
    help_text = (proc.stdout or "") + (proc.stderr or "")
    if not help_text.strip():
        print("FATAL: `sglang.launch_server --help` produced no output",
              file=sys.stderr)
        return 2

    missing = []
    for flag, _why in REQUIRED:
        present = flag in help_text
        print(f"  {'OK     ' if present else 'MISSING'}  {flag}")
        if not present:
            missing.append(flag)

    if missing:
        print("\nargparser is missing K3.1 DFLASH flags:", file=sys.stderr)
        for flag in missing:
            why = dict(REQUIRED)[flag]
            print(f"  {flag}: {why}", file=sys.stderr)
        return 1

    # The draft attention backend is validated as a VALUE, not a flag name.
    if "cutedsl_mla" not in help_text:
        print("FATAL: cutedsl_mla is not among the draft attention backends; "
              "DFLASH's draft KV ring needs an MLA backend", file=sys.stderr)
        return 1
    print("  OK      cutedsl_mla draft attention backend available")

    # --enable-linear-replayssm-spec lives outside the spec namespace, so assert
    # it is a real, resolvable flag rather than a help-text coincidence.
    probe = subprocess.run(
        [sys.executable, "-m", "sglang.launch_server",
         "--enable-linear-replayssm-spec", "--help"],
        capture_output=True, text=True,
    )
    if "unrecognized arguments" in (probe.stderr or "") or probe.returncode != 0:
        print("FATAL: --enable-linear-replayssm-spec is not accepted by the "
              "argparser: " + (probe.stderr or "").strip()[-500:],
              file=sys.stderr)
        return 1
    print("  OK      --enable-linear-replayssm-spec is accepted by the argparser")

    print("all K3.1 DFLASH flags present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
