#!/usr/bin/env bash
set -euo pipefail

DQECG_ROOT="${DQECG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DQECG_ROOT

exec bash "$DQECG_ROOT/server/setup_runtime.sh" "$@"
