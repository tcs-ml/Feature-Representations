"""Lookup decoder: majority fitting, unseen codes, and key layouts."""

from __future__ import annotations

import numpy as np

from concepts import make_powerset
from models.probes import lookup_task_errors


def test_lookup_matches_direct_majority_calculation():
    # 3 atoms, code 0 <- atoms {0,1}, code 1 <- atom {2}
    y = np.array([[1, 0, 0], [1, 1, 0]], dtype=bool)
    train_keys = np.array([0, 0, 0, 0, 1, 1])
    train_atoms = np.array([0, 0, 1, 1, 2, 2])
    test_keys = np.array([0, 0, 1, 1])
    test_atoms = np.array([0, 1, 2, 2])
    res = lookup_task_errors(train_keys, train_atoms, test_keys, test_atoms, y)
    # task 0: code0 pos mass 2/6 of 4/6 -> tie -> predict 1; atom1 (label 0)
    #   predicted 1 -> 0.25; code1 pos 0 -> predict 0, atom 2 ok
    # task 1: code0 majority 1 (both atoms labeled 1) -> no error; code1 label 0 -> ok
    np.testing.assert_allclose(res.per_task_errors, [0.25, 0.0])
    assert res.decoder.shape == (2, 2)
    assert res.test_code_ids.tolist() == [0, 0, 1, 1]


def test_lookup_unseen_code_falls_back_to_global_majority():
    y = np.array([[1, 1, 0]], dtype=bool)
    res = lookup_task_errors(
        train_keys=np.array([0, 0, 1]),
        train_atoms=np.array([0, 1, 2]),
        test_keys=np.array([2, 2]),  # never seen in training
        test_atoms=np.array([0, 2]),
        label_matrix=y,
    )
    # global majority = 1 (mass 2/3 positive): errs on atom 2 only
    np.testing.assert_allclose(res.per_task_errors, [0.5])


def test_lookup_accepts_2d_keys():
    y = np.array([[1, 0]], dtype=bool)
    train_keys = np.array([[0, 3], [0, 3], [1, 2]])
    train_atoms = np.array([0, 0, 1])
    res = lookup_task_errors(train_keys, train_atoms, train_keys, train_atoms, y)
    np.testing.assert_allclose(res.per_task_errors, [0.0])
    assert res.decoder.shape == (2, 1)


def test_lookup_zero_error_with_identity_codes():
    cls = make_powerset(4)
    y = cls.label_matrix()
    atoms = np.repeat(np.arange(4), 25)
    res = lookup_task_errors(atoms, atoms, atoms, atoms, y)
    assert res.per_task_errors.max() == 0.0
