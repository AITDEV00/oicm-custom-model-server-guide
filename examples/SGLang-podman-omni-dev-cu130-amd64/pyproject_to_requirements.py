#!/usr/bin/env python3
"""Emit a PEP 508 requirements file from a pyproject.toml.

`sglang-omni`'s pyproject carries its runtime deps in [project.dependencies]
and its pinned-baseline overrides in [tool.uv.override-dependencies]. `uv pip
install -r` only understands requirements files (and .toml only as a PEP 751
lockfile), so the TOML has to be flattened first.

Environment markers are preserved verbatim so platform-specific entries
(e.g. "sglang==0.5.22; sys_platform != 'darwin'") still resolve correctly.
"""
import sys
import tomllib

src, out = sys.argv[1], sys.argv[2]

with open(src, "rb") as fh:
    data = tomllib.load(fh)

reqs = list(data.get("project", {}).get("dependencies", []))

# Optional extras named on the command line are appended too, so a model extra
# can be installed from the same generated file.
for extra in sys.argv[3:]:
    extras = data.get("project", {}).get("optional-dependencies", {})
    if extra not in extras:
        raise SystemExit(f"error: no such optional-dependency group: {extra}")
    reqs.extend(extras[extra])

with open(out, "w") as fh:
    fh.write("\n".join(reqs) + "\n")

print(f"wrote {len(reqs)} requirements to {out}", file=sys.stderr)
