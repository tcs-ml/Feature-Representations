"""Synthetic observation generator checks."""

from __future__ import annotations

import numpy as np
import pytest

from data.atoms import AtomObservationConfig, generate_atom_dataset


def make_config(**overrides) -> AtomObservationConfig:
    base = dict(
        num_atoms=4,
        signal_dim=8,
        observed_dim=16,
        samples_per_atom={"train": 200, "validation": 50, "test": 200},
        signal_noise_std=0.1,
        mixing_condition_number=3.0,
        seed=0,
    )
    base.update(overrides)
    return AtomObservationConfig(**base)


def test_shapes_and_dtypes():
    ds = generate_atom_dataset(make_config())
    assert ds.input_dim == 16
    for split, per_atom in (("train", 200), ("validation", 50), ("test", 200)):
        part = ds.split(split)
        assert part.x.shape == (4 * per_atom, 16)
        assert part.x.dtype == np.float32
        assert part.atom_ids.shape == (4 * per_atom,)
        assert part.atom_ids.max() < 4


def test_deterministic_and_seed_sensitive():
    a = generate_atom_dataset(make_config())
    b = generate_atom_dataset(make_config())
    c = generate_atom_dataset(make_config(seed=1))
    np.testing.assert_array_equal(a.split("train").x, b.split("train").x)
    assert not np.array_equal(a.split("train").x, c.split("train").x)


def test_splits_are_independently_generated():
    ds = generate_atom_dataset(
        make_config(samples_per_atom={"train": 50, "validation": 50, "test": 50})
    )
    assert not np.array_equal(ds.split("train").x, ds.split("test").x)


def test_clean_setting_recovers_atoms_from_prototypes():
    ds = generate_atom_dataset(make_config())
    clean = ds.prototypes @ ds.mixing.T
    test = ds.split("test")
    nearest = ((test.x[:, None, :] - clean[None]) ** 2).sum(axis=2).argmin(axis=1)
    assert (nearest == test.atom_ids).mean() >= 0.99


def test_atom_distribution_respected():
    probs = np.array([0.7, 0.1, 0.1, 0.1])
    ds = generate_atom_dataset(make_config(seed=3), atom_probs=probs)
    freq = np.bincount(ds.split("train").atom_ids, minlength=4) / ds.split("train").x.shape[0]
    np.testing.assert_allclose(freq, probs, atol=0.05)


def test_config_validation():
    with pytest.raises(ValueError):
        make_config(observed_dim=4, signal_dim=8)
    with pytest.raises(ValueError):
        make_config(mixing_condition_number=0.5)
    with pytest.raises(ValueError):
        make_config(samples_per_atom={"train": 10})


def test_mixing_condition_number():
    ds = generate_atom_dataset(make_config(mixing_condition_number=10.0))
    s = np.linalg.svd(ds.mixing, compute_uv=False)
    assert s.max() / s.min() == pytest.approx(10.0, rel=1e-6)
