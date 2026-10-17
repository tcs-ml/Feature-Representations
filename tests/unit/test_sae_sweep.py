"""SAE sweep plumbing: variant grids, fixed key layout, the representation bit budget."""

from __future__ import annotations

import numpy as np
import pytest

from concepts import make_powerset
from data.atoms import AtomObservationConfig, generate_atom_dataset
from evaluation.confidence import binomial_upper_bound
from experiments.common import evaluate_code_representation
from experiments.sae import enumerate_variants
from models.sae import bit_budget, quantize_sparse_codes


def test_enumerate_variants_sweeps_the_full_grid_and_skips_k_above_width():
    cfg = {
        "model": {
            "type": "topk_sae",
            "dictionary_widths": [4, 8],
            "active_ks": [1, 2, 8],
            "value_bits_grid": [1, 4],
        }
    }
    variants = enumerate_variants(cfg)
    assert len(variants) == 10  # 2 precisions x (2 ks for width 4 + 3 ks for width 8)
    assert len({v["label"] for v in variants}) == 10  # labels stay unique per configuration
    assert {v["value_bits"] for v in variants} == {1, 4}
    assert all(v["tied"] is False and v["type"] == "topk_sae" for v in variants)


def test_quantize_k_slots_gives_split_independent_layout():
    # train batch reaches support 2; a smaller batch only support 1 - with a
    # fixed slot count both must produce identical keys for identical codes
    z_two = np.array([[0.0, 0.5, 0.0, 1.0], [0.0, 0.5, 0.0, 0.0]])
    z_one = np.array([[0.0, 0.5, 0.0, 0.0]])
    keys_two = quantize_sparse_codes(z_two, value_bits=2, vmax=1.0, k_slots=2)
    keys_one = quantize_sparse_codes(z_one, value_bits=2, vmax=1.0, k_slots=2)
    assert keys_two.shape[1] == keys_one.shape[1] == 4
    np.testing.assert_array_equal(keys_two[1], keys_one[0])


def test_quantize_k_slots_rejects_overflow():
    z = np.array([[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="exceeds k_slots"):
        quantize_sparse_codes(z, value_bits=2, vmax=1.0, k_slots=2)


def test_bit_budget_counts_every_code_of_the_encoding_rule():
    # at most k of W features, q-bit nonzero values; identities + values
    assert bit_budget(4, 1, 1) == int(np.ceil(np.log2(1 + 4 * 1)))
    assert bit_budget(8, 2, 2) == int(np.ceil(np.log2(1 + 8 * 3 + 28 * 9)))


def test_evaluate_code_representation_reports_validation_and_test_fields():
    concept = make_powerset(2)
    cfg = AtomObservationConfig(
        num_atoms=2,
        signal_dim=4,
        observed_dim=8,
        samples_per_atom={"train": 50, "validation": 30, "test": 40},
        signal_noise_std=0.05,
        mixing_condition_number=2.0,
        seed=0,
    )
    ds = generate_atom_dataset(cfg, atom_probs=concept.input_probabilities())
    keys = {s: ds.split(s).atom_ids.copy() for s in ("train", "validation", "test")}
    fields = evaluate_code_representation(keys, ds, concept, confidence=0.95)
    # a perfect atom-identity code preserves every task on every split; the
    # binomial bound of a zero-error sample stays positive
    assert fields["max_task_error"] == 0.0
    assert fields["max_task_error_ucb"] == pytest.approx(binomial_upper_bound(0, 80, 0.95))
    assert fields["val_max_task_error"] == 0.0
    assert fields["val_max_task_error_ucb"] == pytest.approx(binomial_upper_bound(0, 60, 0.95))
    assert set(fields) == {
        "max_task_error",
        "max_task_error_ucb",
        "val_max_task_error",
        "val_max_task_error_ucb",
    }
    with pytest.raises(ValueError, match="discrete code"):
        evaluate_code_representation({s: keys[s] for s in ("train", "test")}, ds, concept, 0.95)
