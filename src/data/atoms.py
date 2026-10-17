"""Synthetic raw observations from finite input atoms.

Each atom j yields observations x = A mu_j + eta: a unit-norm prototype mu_j
mixed by a fixed conditioned transform A, plus Gaussian signal noise eta in
observation space. Splits are generated independently (never by duplicating
vectors), and atom identities are drawn iid from the concept's input
distribution so task errors estimated on a split are unbiased for Pr_{x~D}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.seeds import numpy_rng

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class AtomObservationConfig:
    num_atoms: int
    signal_dim: int
    observed_dim: int
    samples_per_atom: dict[str, int]  # split -> average samples per atom
    signal_noise_std: float
    mixing_condition_number: float
    seed: int

    def __post_init__(self) -> None:
        if self.signal_dim < 1 or self.observed_dim < self.signal_dim:
            raise ValueError("need observed_dim >= signal_dim >= 1")
        if self.mixing_condition_number < 1.0:
            raise ValueError("mixing_condition_number must be >= 1")
        missing = [s for s in SPLITS if s not in self.samples_per_atom]
        if missing:
            raise ValueError(f"samples_per_atom missing splits: {missing}")


@dataclass
class AtomSplit:
    x: np.ndarray  # [n, observed_dim] float32
    atom_ids: np.ndarray  # [n] int64


@dataclass
class AtomDataset:
    config: AtomObservationConfig
    prototypes: np.ndarray  # [num_atoms, signal_dim]
    mixing: np.ndarray  # [observed_dim, signal_dim]
    atom_probs: np.ndarray  # [num_atoms]
    splits: dict[str, AtomSplit]

    @property
    def input_dim(self) -> int:
        return self.config.observed_dim

    def split(self, name: str) -> AtomSplit:
        return self.splits[name]


def _make_prototypes(cfg: AtomObservationConfig, rng: np.random.Generator) -> np.ndarray:
    """Unit-norm prototypes; orthogonalized when the signal space allows it."""
    raw = rng.standard_normal((cfg.num_atoms, cfg.signal_dim))
    if cfg.num_atoms <= cfg.signal_dim:
        q, _ = np.linalg.qr(raw.T)  # columns orthonormal
        return q.T[: cfg.num_atoms]
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def _make_mixing(cfg: AtomObservationConfig, rng: np.random.Generator) -> np.ndarray:
    """observed_dim x signal_dim transform with the requested condition number."""
    raw = rng.standard_normal((cfg.observed_dim, cfg.signal_dim))
    u, _, vt = np.linalg.svd(raw, full_matrices=False)
    singular = np.geomspace(1.0, 1.0 / cfg.mixing_condition_number, num=cfg.signal_dim)
    return (u * singular) @ vt


def generate_atom_dataset(
    cfg: AtomObservationConfig, atom_probs: np.ndarray | None = None
) -> AtomDataset:
    """Generate all splits deterministically from cfg.seed.

    `atom_probs` is the distribution over atoms shared by every split
    (uniform when omitted).
    """
    structure_rng = numpy_rng(cfg.seed, "structure")
    prototypes = _make_prototypes(cfg, structure_rng)
    mixing = _make_mixing(cfg, structure_rng)
    if atom_probs is None:
        atom_probs = np.full(cfg.num_atoms, 1.0 / cfg.num_atoms)
    atom_probs = np.asarray(atom_probs, dtype=float)
    atom_probs = atom_probs / atom_probs.sum()

    clean = (prototypes @ mixing.T).astype(np.float64)  # [num_atoms, observed_dim]

    splits: dict[str, AtomSplit] = {}
    for split in SPLITS:
        rng = numpy_rng(cfg.seed, "split", split)
        n = cfg.num_atoms * int(cfg.samples_per_atom[split])
        atom_ids = rng.choice(cfg.num_atoms, size=n, p=atom_probs)
        x = clean[atom_ids] + cfg.signal_noise_std * rng.standard_normal((n, cfg.observed_dim))
        splits[split] = AtomSplit(x=x.astype(np.float32), atom_ids=atom_ids.astype(np.int64))

    return AtomDataset(
        config=cfg,
        prototypes=prototypes,
        mixing=mixing,
        atom_probs=atom_probs,
        splits=splits,
    )
