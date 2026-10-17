"""The installed source tree exposes its modules directly from any directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_flat_source_imports_outside_the_repository(tmp_path):
    source = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, json; "
            "from runner import run_config; "
            "modules = ['runner', 'config', 'analysis.paper', 'concepts.matrix', "
            "'data.atoms', 'evaluation.confidence', 'experiments.greedy', "
            "'models.sae', 'solvers.greedy', 'theory.reference', "
            "'training.reconstruction', 'utils.io']; "
            "assert callable(run_config); "
            "print(json.dumps({name: importlib.util.find_spec(name).origin for name in modules}))",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    locations = json.loads(result.stdout)
    for name, location in locations.items():
        assert Path(location).resolve() == source / f"{name.replace('.', '/')}.py"
