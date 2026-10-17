"""Training-majority lookup decoder for discrete codes.

This is the theorem-aligned evaluation of a finite code: each (code, task)
pair predicts the training-majority label, the optimal deterministic decoder
for the observed code distribution. All task-label computations factor
through atoms: per-example labels are Y[:, atom_ids]. Tasks are processed in
chunks so that families with tens of thousands of functions never need a
dense codes-by-tasks matrix of floating-point masses at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LookupResult:
    per_task_errors: np.ndarray  # [m]
    decoder: np.ndarray  # [n_codes, m] fitted majority table
    test_code_ids: np.ndarray  # contiguous ids into decoder rows


def _canonical_code_ids(
    train_keys: np.ndarray, test_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    both = np.concatenate([train_keys, test_keys], axis=0)
    if both.ndim == 1:
        _, inverse = np.unique(both, return_inverse=True)
    else:
        _, inverse = np.unique(both, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return inverse[: len(train_keys)], inverse[len(train_keys) :]


def lookup_task_errors(
    train_keys: np.ndarray,
    train_atoms: np.ndarray,
    test_keys: np.ndarray,
    test_atoms: np.ndarray,
    label_matrix: np.ndarray,
    task_chunk: int = 4096,
) -> LookupResult:
    """Optimal train-majority lookup decoder, evaluated on the test split.

    Keys may be 1-D code ids or 2-D integer index rows of quantized sparse
    codes. Codes unseen in training fall back to the global train-majority
    label per task. Ties predict 1.
    """
    y = np.asarray(label_matrix, dtype=bool)
    m, n_atoms = y.shape
    train_ids, test_ids = _canonical_code_ids(np.asarray(train_keys), np.asarray(test_keys))
    n_codes = int(max(train_ids.max(), test_ids.max())) + 1

    cont_tr = np.zeros((n_codes, n_atoms))
    np.add.at(cont_tr, (train_ids, np.asarray(train_atoms)), 1.0 / len(train_ids))
    mass_tr = cont_tr.sum(axis=1, keepdims=True)
    unseen = mass_tr[:, 0] == 0.0
    global_mass = cont_tr.sum(axis=0)  # training mass per atom

    cont_te = np.zeros((n_codes, n_atoms))
    np.add.at(cont_te, (test_ids, np.asarray(test_atoms)), 1.0 / len(test_ids))
    mass_te = cont_te.sum(axis=1, keepdims=True)

    decoder = np.empty((n_codes, m), dtype=bool)
    errors = np.empty(m)
    for start in range(0, m, task_chunk):
        stop = min(start + task_chunk, m)
        labels = y[start:stop].T.astype(float)  # [atoms, chunk]
        pos_tr = cont_tr @ labels  # [codes, chunk]
        chunk = 2.0 * pos_tr >= mass_tr  # majority, ties -> 1
        chunk[unseen] = 2.0 * (global_mass @ labels) >= 1.0
        pos_te = cont_te @ labels
        errors[start:stop] = np.where(chunk, mass_te - pos_te, pos_te).sum(axis=0)
        decoder[:, start:stop] = chunk
    return LookupResult(per_task_errors=errors, decoder=decoder, test_code_ids=test_ids)
