#!/usr/bin/env python3
"""Apply the single reviewed sparse-to-fused Xllm mapping atomically.

Used as fallback when a vendor base has the exact source anchor but no patch(1).
The anchor is unique in the pinned image and matching is required; nothing is
silently ignored. No model or credential files are touched.
"""
from pathlib import Path
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    additions = [
        line[1:] + "\n" for line in args.patch.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    needle = (
        "        expert_params_mapping = self.get_expert_mapping()\n"
        "        for name, loaded_weight in weights:\n"
    )
    assert source.count(needle) == 1, "unique audited source context not found"
    patched = source.replace(needle, needle + "".join(additions), 1)
    compile(patched, str(args.source), "exec")
    tmp = args.source.with_suffix(".py.tmp")
    tmp.write_text(patched, encoding="utf-8")
    os.chmod(tmp, args.source.stat().st_mode)
    tmp.replace(args.source)
    print("Applied reviewed sparse-to-fused derived mapping: syntax accepted")


if __name__ == "__main__":
    main()
