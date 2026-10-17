"""Top-k sparse autoencoder, the representation bit budget, and code quantization."""

from __future__ import annotations

from math import comb

import numpy as np
import torch
from torch import nn


def ceil_log2(n: int) -> int:
    """Exact ceil(log2(n)) for arbitrary-precision positive ints."""
    if n < 1:
        raise ValueError(f"need n >= 1, got {n}")
    return (n - 1).bit_length()


def bit_budget(dictionary_width: int, active_k: int, value_bits: int) -> int:
    """The paper's representation bit budget B_bits(W, k, q).

    ceil(log2(sum_{i<=k} C(W, i) (2^q - 1)^i)): the length of a fixed-length
    index of every code the encoding rule allows, at most k of W features
    active, each carrying one of 2^q - 1 nonzero levels (level 0 means
    inactive). Codes the trained encoder never emits are counted.
    """
    if not 0 <= active_k <= dictionary_width:
        raise ValueError(f"need 0 <= k <= W, got k={active_k}, W={dictionary_width}")
    levels = 2**value_bits - 1
    code_count = sum(comb(dictionary_width, i) * levels**i for i in range(active_k + 1))
    return ceil_log2(code_count)


class TopKSAE(nn.Module):
    """z = TopK(ReLU(W_enc (x - b_dec) + b_enc)); xhat = W_dec z + b_dec.

    Decoder rows are unit-normalized by the trainer after each step. The
    module has no mode-dependent layers, so encodings are identical in train
    and eval mode.
    """

    def __init__(self, in_dim: int, dictionary_width: int, active_k: int) -> None:
        super().__init__()
        if not 1 <= active_k <= dictionary_width:
            raise ValueError(f"need 1 <= k <= W, got k={active_k}, W={dictionary_width}")
        self.in_dim = in_dim
        self.dictionary_width = dictionary_width
        self.active_k = active_k
        w = torch.randn(in_dim, dictionary_width) / (in_dim**0.5)
        self.w_enc = nn.Parameter(w)
        self.b_enc = nn.Parameter(torch.zeros(dictionary_width))
        self.w_dec = nn.Parameter(w.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(in_dim))

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.w_dec.div_(self.w_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.w_enc + self.b_enc
        acts = torch.relu(pre)
        top_vals, top_idx = acts.topk(self.active_k, dim=-1)
        z = torch.zeros_like(acts)
        z.scatter_(-1, top_idx, top_vals)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, z @ self.w_dec + self.b_dec

    @torch.no_grad()
    def sparse_codes(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Sparse codes of a whole split as a [N, W] float32 array."""
        device = self.w_enc.device
        parts = []
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32))
            parts.append(self.encode(batch.to(device)).cpu().numpy())
        return np.concatenate(parts)


def quantize_sparse_codes(z: np.ndarray, value_bits: int, vmax: float, k_slots: int) -> np.ndarray:
    """Post-training quantization of sparse activations into lookup-decoder keys.

    Nonzero values are quantized to 2^q - 1 positive levels on (0, vmax]
    (level 0 means inactive, so features quantized to zero drop out of the
    support). `k_slots` fixes the key layout to the encoding rule's maximum
    support size so codes from different splits share one layout. Returns
    [N, 2*k_slots] int64 rows: sorted support indices, then levels, -1 padded.
    """
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    levels = 2**value_bits - 1
    level_idx = np.rint(np.clip(z, 0.0, vmax) / vmax * levels).astype(np.int64)
    active = level_idx > 0
    k_max = int(active.sum(axis=1).max()) if len(active) else 0
    if k_max > k_slots:
        raise ValueError(f"observed support {k_max} exceeds k_slots={k_slots}")
    slots = max(k_slots, 1)
    keys = np.full((z.shape[0], 2 * slots), -1, dtype=np.int64)
    for i in range(z.shape[0]):
        support = np.flatnonzero(active[i])
        keys[i, : support.size] = support
        keys[i, slots : slots + support.size] = level_idx[i, support]
    return keys


def quantized_activations(z: np.ndarray, value_bits: int, vmax: float) -> np.ndarray:
    """Dequantized sparse activations, rounded exactly as quantize_sparse_codes rounds the keys.

    Each entry becomes its level on (0, vmax]; entries that quantize to level
    0 become exactly zero, so a reconstruction from this array sees only what
    the discrete code carries.
    """
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    levels = 2**value_bits - 1
    level_idx = np.rint(np.clip(np.asarray(z, dtype=np.float64), 0.0, vmax) / vmax * levels)
    return (level_idx / levels * vmax).astype(np.float32)
