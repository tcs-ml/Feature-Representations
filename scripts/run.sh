#!/usr/bin/env sh
# Rerun the experiments, or run any JSON config.
#
# Usage:
#   scripts/run.sh                               # rerun every experiment, then the analysis (hours)
#   scripts/run.sh configs/analysis.json         # only the analysis of the saved outputs -> outputs/analysis/
#   scripts/run.sh configs/experiment/smoke.json # small training check
#
# The environment comes from `uv sync` or `scripts/setup.sh` (.venv at the
# repository root); set PYTHON to use another interpreter with the project
# installed.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${1:-$ROOT/configs/rerun.json}"

if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found; run 'uv sync' or 'scripts/setup.sh' first" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "error: config not found: $CONFIG" >&2
    exit 1
fi

exec "$PYTHON" -c 'import sys; from runner import run_config; run_config(sys.argv[1])' "$CONFIG"
