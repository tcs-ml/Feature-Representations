"""Top-k SAE model, capacity formula, and code quantization."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest
import torch

from models.sae import TopKSAE, bit_budget, ceil_log2, quantize_sparse_codes


def test_topk_sae_sparsity_and_reconstruction_path():
    torch.manual_seed(0)
    sae = TopKSAE(in_dim=12, dictionary_width=32, active_k=4)
    x = torch.randn(9, 12)
    z, recon = sae(x)
    assert z.shape == (9, 32)
    assert ((z != 0).sum(dim=-1) <= 4).all()
    assert (z >= 0).all()
    assert recon.shape == x.shape
    sae.normalize_decoder_()
    np.testing.assert_allclose(sae.w_dec.norm(dim=1).detach().numpy(), 1.0, rtol=1e-5)
    with pytest.raises(ValueError):
        TopKSAE(in_dim=6, dictionary_width=4, active_k=5)


def test_sparse_codes_batches_match_single_pass():
    torch.manual_seed(1)
    sae = TopKSAE(in_dim=5, dictionary_width=8, active_k=2)
    x = np.random.default_rng(0).standard_normal((37, 5)).astype(np.float32)
    whole = sae.sparse_codes(x)
    batched = sae.sparse_codes(x, batch_size=10)
    np.testing.assert_array_equal(whole, batched)
    with torch.no_grad():
        np.testing.assert_array_equal(whole, sae.encode(torch.from_numpy(x)).numpy())


def test_ceil_log2_exact_integer_math():
    assert ceil_log2(1) == 0
    assert ceil_log2(2) == 1
    assert ceil_log2(3) == 2
    assert ceil_log2(2**300) == 300
    assert ceil_log2(2**300 + 1) == 301
    with pytest.raises(ValueError):
        ceil_log2(0)


def brute_force_at_most_k_count(width: int, kmax: int, q: int) -> int:
    """Distinct messages of the encoding rule: support of size <= kmax, one of
    2^q - 1 NONZERO levels per active feature (level 0 = inactive, and a code
    listing a level-0 feature is the same message as one omitting it)."""
    count = 0
    for k in range(kmax + 1):
        for _support in combinations(range(width), k):
            count += len(list(product(range(1, 2**q), repeat=k)))
    return count


def test_capacity_formula_matches_brute_force_counts():
    for width, k, q in [(4, 2, 1), (5, 2, 2), (6, 3, 1), (8, 3, 4)]:
        assert bit_budget(width, k, q) == ceil_log2(brute_force_at_most_k_count(width, k, q))
    with pytest.raises(ValueError):
        bit_budget(4, 5, 1)


def test_quantize_sparse_codes():
    z = np.array(
        [
            [0.0, 0.5, 0.0, 1.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.001, 0.0],  # quantizes to zero -> drops from support
        ]
    )
    keys = quantize_sparse_codes(z, value_bits=2, vmax=1.0, k_slots=2)
    assert keys.shape == (3, 4)
    assert keys[0].tolist() == [1, 3, 2, 3]  # support {1, 3} at levels 2 and 3 of 3
    assert keys[1].tolist() == [0, -1, 1, -1]
    assert keys[2].tolist() == [-1, -1, -1, -1]
    # rows with identical quantized codes get identical keys
    np.testing.assert_array_equal(keys, quantize_sparse_codes(z.copy(), 2, 1.0, k_slots=2))
    with pytest.raises(ValueError):
        quantize_sparse_codes(z, 2, 0.0, k_slots=2)
