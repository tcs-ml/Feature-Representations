"""The SAE sweep covers exactly the paper's 19 instances on the seed grid the analysis expects."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import load_config
from experiments.common import enumerate_concepts

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
EXPERIMENT_DIR = CONFIG_ROOT / "experiment"


@pytest.fixture(scope="module")
def cfgs():
    return {
        "sae": load_config(EXPERIMENT_DIR / "sae.json"),
        "smoke": load_config(EXPERIMENT_DIR / "smoke.json"),
    }


def test_paper_instances(cfgs):
    # 8 powerset sizes, parity r = 1..6, coordinates r = 2..6: 19 instances
    families: dict[str, list[int]] = {}
    for block in cfgs["sae"]["concept"]["families"]:
        sizes = block.get("c_values", block.get("d_values"))
        families.setdefault(block["family"], []).extend(sizes)
    assert sorted(families["powerset"]) == [2, 3, 4, 6, 8, 10, 12, 16]
    assert sorted(families["parity"]) == [1, 2, 3, 4, 5, 6]
    assert sorted(families["transposed_powerset"]) == [2, 3, 4, 5, 6]
    concepts = enumerate_concepts(cfgs["sae"]["concept"])
    assert len(concepts) == 19
    assert len({c.name for c in concepts}) == 19
    assert max(c.metadata().num_inputs for c in concepts) == 64
    assert max(c.metadata().num_tasks for c in concepts) == 65536
    # every instance instantiates the entropy lower-bound construction
    for c in concepts:
        assert "dual_shatter_mass" in c.metadata().extra, c.name


def test_sweep_dimensions(cfgs):
    model = cfgs["sae"]["model"]
    assert model["type"] == "topk_sae"
    assert model["dictionary_widths"] == [4, 8, 16, 32]
    assert model["active_ks"] == [1, 2, 4]
    assert model["value_bits_grid"] == [1, 2, 4]
    assert cfgs["sae"]["experiment"]["seeds"] == [5, 6, 7, 8, 9]
    analysis = load_config(CONFIG_ROOT / "analysis.json")
    assert analysis["epsilon_grid"] == [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
    assert Path(analysis["sweep"]).name == "sae.json"


def test_seeds_are_disjoint_from_the_development_sweep(cfgs):
    # the analysis rules were fixed on a development sweep with seeds 0 to 4;
    # the reported study must not reuse any of its data or initializations
    assert not set(cfgs["sae"]["experiment"]["seeds"]) & set(range(5))


def test_no_device_keys(cfgs):
    # the compute device is auto-detected at runtime; configs must not pin it
    for name, cfg in cfgs.items():
        assert "device" not in cfg, name


def test_experiment_name_pins_resume_identity(cfgs):
    # run IDs embed experiment.name; renaming it orphans the completed result set
    assert cfgs["sae"]["experiment"]["name"] == "sae"


def test_smoke_config_is_a_tiny_sae_stage(cfgs):
    smoke = cfgs["smoke"]
    assert smoke["model"]["type"] == "topk_sae"
    assert smoke["experiment"]["seeds"] == [0]
    concepts = enumerate_concepts(smoke["concept"])
    assert len(concepts) <= 3
    assert max(c.metadata().num_inputs for c in concepts) <= 8
