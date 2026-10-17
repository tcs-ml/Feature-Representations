#!/usr/bin/env sh
# Create the project virtualenv at .venv WITHOUT uv (cluster-friendly).
#
# Usage:
#   scripts/setup.sh [python]        # e.g. scripts/setup.sh python3.13
#
# Environment:
#   PYTHON           interpreter to use (same as the positional argument)
#   VENV_DIR         where to create the venv (default .venv; activate it
#                    before importing the Python experiment entry point)
#   TORCH_INDEX_URL  optional torch wheel index for a specific CUDA build,
#                    e.g. https://download.pytorch.org/whl/cu126
#
# The project needs Python >= 3.13 (pyproject.toml). On clusters, load the
# interpreter first (e.g. `module load python/3.13`) or pass its path.
# On Linux x86_64 the default PyPI torch wheel already bundles CUDA support;
# set TORCH_INDEX_URL only if that wheel mismatches the node's driver.
set -eu
cd "$(dirname "$0")/.."

version_ok() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)' 2>/dev/null
}

PYTHON="${1:-${PYTHON:-}}"
if [ -n "$PYTHON" ]; then
    if ! version_ok "$PYTHON"; then
        echo "error: $PYTHON is not a Python >= 3.13 interpreter" >&2
        exit 1
    fi
else
    for cand in python3.16 python3.15 python3.14 python3.13 python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && version_ok "$cand"; then
            PYTHON="$cand"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "error: no Python >= 3.13 found on PATH." >&2
        echo "Load one (e.g. 'module load python/3.13') or pass its path:" >&2
        echo "  scripts/setup.sh /path/to/python3.13" >&2
        exit 1
    fi
fi
echo "using $PYTHON ($("$PYTHON" -V 2>&1))"

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"

"$VENV_PY" -m pip install --upgrade pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
    echo "installing torch from $TORCH_INDEX_URL"
    "$VENV_PY" -m pip install torch --index-url "$TORCH_INDEX_URL"
fi

# project + dependencies (editable), then the dev tools (pyproject dev group)
"$VENV_PY" -m pip install -e .
"$VENV_PY" -m pip install "pytest>=9.1.1" "ruff>=0.9.0"

echo
"$VENV_PY" - <<'EOF'
import torch
from runner import run_config

print(f"ok: {run_config.__module__}.run_config importable; torch {torch.__version__}; "
      f"cuda available: {torch.cuda.is_available()}")
EOF

echo
echo "environment ready at $VENV_DIR - next steps:"
echo "  $VENV_PY -m pytest -q                                  # test suite"
echo "  Activate the environment and open Python or a notebook."
echo "  from runner import run_config"
echo '  run_config("configs/analysis.json")                      # analysis of the saved outputs only'
echo '  run_config("configs/rerun.json")                         # rerun every experiment'
