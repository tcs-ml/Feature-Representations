"""Noiseless coordinate observations on the grid [N]^r for threshold tasks.

Every coordinate is drawn independently and uniformly from {1, ..., N}, and
the observed vector holds the normalized coordinates y_j = (x_j - 1) / (N - 1).
Task labels are deterministic functions of the observation, so a decoded
coordinate determines every threshold prediction. Each split of each
dimension comes from its own deterministic stream; nothing is shared across
dimensions, and nothing depends on a model seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.seeds import numpy_rng

SPLITS = ("train", "reconstruction_validation", "test")


@dataclass(frozen=True)
class ThresholdProductConfig:
    grid_size: int  # N
    dimension: int  # r
    samples: dict[str, int]  # split -> number of observations
    seed: int  # root data seed; streams are keyed by dimension and split

    def __post_init__(self) -> None:
        if self.grid_size < 2 or self.dimension < 1:
            raise ValueError("need grid_size >= 2 and dimension >= 1")
        missing = [split for split in SPLITS if split not in self.samples]
        if missing:
            raise ValueError(f"samples missing splits: {missing}")
        if any(int(self.samples[split]) < 1 for split in SPLITS):
            raise ValueError("every split needs at least one observation")


@dataclass
class ThresholdSplit:
    x: np.ndarray  # [n, r] int64 grid coordinates in 1..N
    features: np.ndarray  # [n, r] float32 normalized coordinates in [0, 1]


@dataclass
class ThresholdDataset:
    config: ThresholdProductConfig
    splits: dict[str, ThresholdSplit]

    def split(self, name: str) -> ThresholdSplit:
        return self.splits[name]


def normalize_coordinates(x: np.ndarray, grid_size: int) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) - 1.0) / (grid_size - 1)).astype(np.float32)


def generate_threshold_dataset(cfg: ThresholdProductConfig) -> ThresholdDataset:
    """Generate every split from its own stream keyed by (seed, dimension, split)."""
    splits: dict[str, ThresholdSplit] = {}
    for split in SPLITS:
        rng = numpy_rng(cfg.seed, "threshold_product", cfg.dimension, split)
        x = rng.integers(
            1, cfg.grid_size + 1, size=(int(cfg.samples[split]), cfg.dimension), dtype=np.int64
        )
        splits[split] = ThresholdSplit(x=x, features=normalize_coordinates(x, cfg.grid_size))
    return ThresholdDataset(config=cfg, splits=splits)
