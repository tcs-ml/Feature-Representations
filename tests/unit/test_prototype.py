"""Known-prototype decoder: nearest clean image, exact task errors, and certification counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.prototype import (
    decode_nearest_prototype,
    prototype_certification,
    prototype_rows,
)
from concepts import make_powerset
from data.atoms import AtomObservationConfig, generate_atom_dataset


def test_nearest_prototype_is_exact_without_noise_and_degrades_with_noise():
    concept = make_powerset(4)
    for noise, exact in ((0.0, True), (3.0, False)):
        cfg = AtomObservationConfig(
            num_atoms=4,
            signal_dim=4,
            observed_dim=6,
            samples_per_atom={"train": 10, "validation": 30, "test": 30},
            signal_noise_std=noise,
            mixing_condition_number=2.0,
            seed=3,
        )
        dataset = generate_atom_dataset(cfg, atom_probs=concept.input_probabilities())
        clean = dataset.prototypes @ dataset.mixing.T
        decoded = decode_nearest_prototype(dataset.split("test").x, clean)
        assert (decoded == dataset.split("test").atom_ids).all() == exact
        rows = prototype_rows(dataset, concept.label_matrix(), 0.95)
        assert [row["split"] for row in rows] == ["validation", "test"]
        test = rows[1]
        assert test["n"] == 120
        assert (test["worst_task_error"] == 0.0) == exact
        assert test["worst_task_error_ucb"] > test["worst_task_error"]
        # a decoding error on an atom flips every task that separates the two atoms
        assert test["worst_task_error"] <= test["atom_error"]


def test_certification_counts_seeds_per_split():
    table = pd.DataFrame(
        [
            {
                "concept_family": "parity",
                "condition": "parity-2",
                "zero_error_bits": 2,
                "seed": s,
                "split": split,
                "worst_task_error": e,
                "worst_task_error_ucb": u,
                "atom_error": e,
            }
            for s, split, e, u in [
                (0, "validation", 0.0, 0.01),
                (0, "test", 0.0, 0.004),
                (1, "validation", 0.02, 0.03),
                (1, "test", 0.02, 0.03),
            ]
        ]
    )
    out = prototype_certification(table, [0.005, 0.05]).set_index("epsilon")
    assert out.loc[0.005, "seeds_test_certified"] == 1
    assert out.loc[0.005, "seeds_validation_certified"] == 0
    assert out.loc[0.05, "seeds_test_certified"] == 2
    assert out.loc[0.05, "n_seeds"] == 2
    assert np.isclose(out.loc[0.05, "median_test_worst_task_error"], 0.01)
