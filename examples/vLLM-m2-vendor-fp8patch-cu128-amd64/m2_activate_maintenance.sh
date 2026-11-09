#!/usr/bin/env bash
# Forward to the guarded Python implementation. No cluster mutation by default.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/m2_activate_maintenance.py" "$@"
