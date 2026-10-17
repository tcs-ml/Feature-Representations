"""Plain JSON configurations, validated without merging or changing hash inputs.

The `observations`, `training`, and `probe` blocks of an SAE stage and the
SAE variant grid are copied verbatim into every run identity (see
experiments.sae.sae_run_config), as are the `family`, `samples`, `model`,
and `training` blocks of a threshold pilot. Validation never inserts
defaults into them, so an unchanged JSON file resumes the same runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Invalid or unsupported experiment configuration."""


FAMILY_LIMITS = {"powerset": 20, "parity": 12, "transposed_powerset": 14}


def _keys(value: Any, allowed: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be an object")
    if unknown := value.keys() - allowed:
        raise ConfigError(f"unsupported keys in {where}: {sorted(unknown)}")


def _require(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"missing required config key: {dotted}")
        node = node[part]
    return node


def _number(value: Any, where: str, minimum: float = 0, *, positive: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < minimum
        or (positive and value == minimum)
    ):
        relation = ">" if positive else ">="
        raise ConfigError(f"{where} must be finite and {relation} {minimum}")


def _integer(value: Any, where: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}")


def _grid(values: Any, where: str, minimum: int = 1) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{where} must be a non-empty list")
    for value in values:
        _integer(value, where, minimum)
    if len(set(values)) != len(values):
        raise ConfigError(f"{where} contains duplicates")


def _epsilons(values: Any, where: str) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{where} must be a non-empty list")
    for value in values:
        _number(value, where)
        if value >= 0.5:
            raise ConfigError(f"{where} entries must lie in [0, 0.5)")
    if len(set(values)) != len(values):
        raise ConfigError(f"{where} contains duplicates")


def _path(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty path string")


def validate_concept_config(cfg: dict) -> None:
    concept = _require(cfg, "concept")
    _keys(concept, {"families", "family", "c_values", "d_values"}, "concept")
    if "families" in concept and len(concept) > 1:
        raise ConfigError("use either concept.families or one concept.family block")
    blocks = concept.get("families", [concept])
    if not isinstance(blocks, list) or not blocks:
        raise ConfigError("concept.families must be a non-empty list")
    for block in blocks:
        _keys(block, {"family", "c_values", "d_values"}, "concept family")
        family = _require(block, "family")
        if not isinstance(family, str) or family not in FAMILY_LIMITS:
            raise ConfigError(f"unsupported concept family {family!r}")
        size_key = "d_values" if family == "parity" else "c_values"
        _keys(block, {"family", size_key}, f"concept.{family}")
        _grid(_require(block, size_key), f"concept.{size_key}")
        if max(block[size_key]) > FAMILY_LIMITS[family]:
            raise ConfigError(f"{family} supports {size_key} <= {FAMILY_LIMITS[family]}")


def _validate_observations(cfg: dict) -> None:
    obs = _require(cfg, "observations")
    _keys(
        obs,
        {
            "signal_dim",
            "observed_dim",
            "samples_per_atom",
            "signal_noise_std",
            "mixing_condition_number",
        },
        "observations",
    )
    for key in ("signal_dim", "observed_dim"):
        _integer(_require(obs, key), f"observations.{key}")
    if obs["observed_dim"] < obs["signal_dim"]:
        raise ConfigError("observations.observed_dim must be >= signal_dim")
    splits = _require(obs, "samples_per_atom")
    _keys(splits, {"train", "validation", "test"}, "observations.samples_per_atom")
    for split in ("train", "validation", "test"):
        _integer(_require(splits, split), f"observations.samples_per_atom.{split}")
    _number(_require(obs, "signal_noise_std"), "observations.signal_noise_std")
    _number(_require(obs, "mixing_condition_number"), "observations.mixing_condition_number", 1)


def _validate_training(cfg: dict) -> None:
    training = _require(cfg, "training")
    _keys(
        training,
        {"epochs", "batch_size", "lr", "eval_every", "patience", "resample_dead"},
        "training",
    )
    for key in ("epochs", "batch_size", "lr"):
        _require(training, key)
    for key, value in training.items():
        if key == "lr":
            _number(value, "training.lr", positive=True)
        elif key == "resample_dead":
            if type(value) is not bool:
                raise ConfigError("training.resample_dead must be a boolean")
        else:
            _integer(value, f"training.{key}")


def _validate_sae(cfg: dict) -> None:
    _grid(_require(cfg, "experiment.seeds"), "experiment.seeds", 0)
    _validate_observations(cfg)
    _validate_training(cfg)
    # `probe` is a frozen component of the historical run identities. Only
    # the lookup decoder is evaluated; "linear" is accepted for identity
    # compatibility with the completed grid and is otherwise inert.
    probe = _require(cfg, "probe")
    _keys(probe, {"types"}, "probe")
    types = _require(probe, "types")
    if (
        not isinstance(types, list)
        or not all(isinstance(t, str) for t in types)
        or "lookup" not in types
        or not set(types) <= {"lookup", "linear"}
        or len(set(types)) != len(types)
    ):
        raise ConfigError("probe.types must contain lookup and optionally linear")
    analysis = _require(cfg, "analysis")
    _keys(analysis, {"confidence_level"}, "analysis")
    confidence = _require(analysis, "confidence_level")
    _number(confidence, "analysis.confidence_level", positive=True)
    if confidence >= 1:
        raise ConfigError("analysis.confidence_level must be < 1")
    model = _require(cfg, "model")
    _keys(model, {"type", "dictionary_widths", "active_ks", "value_bits_grid"}, "model")
    if model.get("type") != "topk_sae":
        raise ConfigError("SAE experiments require model.type=topk_sae")
    for key in ("dictionary_widths", "active_ks", "value_bits_grid"):
        _grid(_require(model, key), f"model.{key}")
    if max(model["active_ks"]) > min(model["dictionary_widths"]):
        raise ConfigError("every active_k must be <= every dictionary_width")


def _validate_greedy(exp: dict) -> None:
    pairs = _require(exp, "grid_sizes")
    if not isinstance(pairs, list) or not pairs:
        raise ConfigError("experiment.grid_sizes must be a non-empty list of [dimension, size]")
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError("each grid must be [dimension, size]")
        _integer(pair[0], "grid dimension")
        _integer(pair[1], "grid size", 2)
        if sum(eps >= 1 / pair[1] for eps in exp["epsilon_grid"]) < 2:
            raise ConfigError("each grid needs at least two epsilons >= 1 / grid size")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ConfigError("experiment.grid_sizes contains duplicates")


def _validate_threshold_pilot(cfg: dict) -> None:
    _keys(
        cfg, {"kind", "outputs", "experiment", "family", "samples", "model", "training"}, "config"
    )
    _path(_require(cfg, "outputs"), "outputs")
    exp = _require(cfg, "experiment")
    _keys(exp, {"name", "initialization_seeds", "data_seed"}, "experiment")
    _path(_require(exp, "name"), "experiment.name")
    _grid(_require(exp, "initialization_seeds"), "experiment.initialization_seeds", 0)
    _integer(_require(exp, "data_seed"), "experiment.data_seed", 0)
    family = _require(cfg, "family")
    _keys(family, {"grid_size", "dimensions"}, "family")
    _integer(_require(family, "grid_size"), "family.grid_size", 2)
    _grid(_require(family, "dimensions"), "family.dimensions")
    samples = _require(cfg, "samples")
    _keys(samples, {"train", "reconstruction_validation", "test"}, "samples")
    for split in ("train", "reconstruction_validation", "test"):
        _integer(_require(samples, split), f"samples.{split}")
    model = _require(cfg, "model")
    _keys(
        model, {"type", "width_per_dimension", "active_per_dimension", "value_bits_grid"}, "model"
    )
    if model.get("type") != "topk_sae":
        raise ConfigError("threshold pilots require model.type=topk_sae")
    _integer(_require(model, "width_per_dimension"), "model.width_per_dimension")
    _integer(_require(model, "active_per_dimension"), "model.active_per_dimension")
    if model["active_per_dimension"] > model["width_per_dimension"]:
        raise ConfigError("model.active_per_dimension must be <= model.width_per_dimension")
    _grid(_require(model, "value_bits_grid"), "model.value_bits_grid")
    _validate_training(cfg)


def validate_config(cfg: dict) -> None:
    kind = _require(cfg, "kind")
    if kind == "threshold_pilot":
        _validate_threshold_pilot(cfg)
        return
    if kind == "suite":
        _keys(cfg, {"kind", "configs"}, "suite")
        paths = _require(cfg, "configs")
        if not isinstance(paths, list) or not paths:
            raise ConfigError("suite.configs must be a non-empty list")
        for path in paths:
            _path(path, "suite.configs")
        return
    if kind == "analysis":
        from analysis.data import validate_analysis_config

        validate_analysis_config(cfg)
        return
    if not isinstance(kind, str) or kind not in {"greedy", "sae"}:
        raise ConfigError(f"unsupported config kind {kind!r}")
    specific = {
        "greedy": set(),
        "sae": {"concept", "observations", "model", "training", "probe", "analysis"},
    }
    _keys(cfg, {"kind", "outputs", "experiment"} | specific[kind], "config")
    _path(_require(cfg, "outputs"), "outputs")
    exp = _require(cfg, "experiment")
    exp_keys = {"greedy": {"name", "grid_sizes", "epsilon_grid"}, "sae": {"name", "seeds"}}[kind]
    _keys(exp, exp_keys, "experiment")
    _path(_require(exp, "name"), "experiment.name")
    if kind == "greedy":
        _epsilons(_require(exp, "epsilon_grid"), "experiment.epsilon_grid")
        _validate_greedy(exp)
        return
    validate_concept_config(cfg)
    _validate_sae(cfg)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ConfigError(f"non-finite JSON number: {value}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate JSON; preserve scalar types, key presence and list order."""
    path = Path(path)
    if path.suffix != ".json":
        raise ConfigError(f"experiment configs must be .json files: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            cfg = json.load(
                stream, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
            )
        if not isinstance(cfg, dict):
            raise ConfigError("configuration must be a JSON object")
        validate_config(cfg)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return cfg
