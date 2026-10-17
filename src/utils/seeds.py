"""Deterministic seed derivation.

Data-generation seeds and model-initialization seeds must never be shared.
Derive both from one root seed with distinct labels:

    data_rng   = numpy_rng(root, "data", concept_name, nuisance_dim)
    model_seed = child_seed(root, "model", concept_name, nuisance_dim, variant)

Child seeds are stable across processes and platforms (sha256, not `hash()`).
"""

from __future__ import annotations

import hashlib

import numpy as np


def child_seed(root: int, *labels: object) -> int:
    """Stable 63-bit child seed derived from a root seed and string labels."""
    payload = repr((int(root), tuple(str(label) for label in labels))).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def numpy_rng(root: int, *labels: object) -> np.random.Generator:
    """Numpy generator seeded from a labeled child seed."""
    return np.random.default_rng(child_seed(root, *labels))
