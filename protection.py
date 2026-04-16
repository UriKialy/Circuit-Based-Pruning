# protection.py — protection percentage providers and mask builders
#
# Each provider returns a dict: (layer_idx, matrix_name) -> protection_pct
# Each mask builder returns a dict: (layer_idx, matrix_name) -> boolean tensor
#
# Designed to be pluggable. Mix and match protection providers with
# score providers to get different experiment variants.

import torch
import numpy as np
from config import LINEAR_LAYERS, PROTECTION_MULTIPLIERS, LAYER_PROTECTION_SPREAD


# ═══════════════════════════════════════════════════════════════
#  Protection percentage providers
# ═══════════════════════════════════════════════════════════════

def uniform_protection(n_layers, base_pct):
    """All matrices get the same protection percentage."""
    pct_map = {}
    for layer in range(n_layers):
        for m in LINEAR_LAYERS:
            pct_map[(layer, m)] = base_pct
    return pct_map


def matrix_multiplier_protection(n_layers, base_pct,
                                  multipliers=None):
    """
    Per-matrix-type protection: base_pct * multiplier for matrix type.
    No per-layer variation.
    """
    if multipliers is None:
        multipliers = PROTECTION_MULTIPLIERS
    pct_map = {}
    for layer in range(n_layers):
        for m in LINEAR_LAYERS:
            pct_map[(layer, m)] = base_pct * multipliers.get(m, 1.0)
    return pct_map


def relp_layered_protection(layer_importance, n_layers, base_pct,
                             matrix_multipliers=None,
                             layer_spread=None):
    """
    RelP-driven per-layer + matrix-type protection.
    protection(layer, matrix) = base_pct * layer_mult(layer) * matrix_mult(matrix)
    """
    if matrix_multipliers is None:
        matrix_multipliers = PROTECTION_MULTIPLIERS
    if layer_spread is None:
        layer_spread = LAYER_PROTECTION_SPREAD
    lo, hi = layer_spread

    scores = np.array([layer_importance.get(i, 0.0) for i in range(n_layers)])
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        layer_mults = np.ones(n_layers)
    else:
        norm = (scores - s_min) / (s_max - s_min)   # in [0,1]
        layer_mults = lo + norm * (hi - lo)          # map to [lo, hi]

    pct_map = {}
    for layer in range(n_layers):
        for m in LINEAR_LAYERS:
            pct_map[(layer, m)] = (
                base_pct
                * layer_mults[layer]
                * matrix_multipliers.get(m, 1.0)
            )
    return pct_map


# ═══════════════════════════════════════════════════════════════
#  Protection mask builder
# ═══════════════════════════════════════════════════════════════

def build_protection_mask(weight_tensor, score_tensor, protect_pct):
    """
    Mark top `protect_pct` fraction of weights by score as protected.

    Args:
        weight_tensor: (d_out, d_in) — just for shape
        score_tensor:  (d_out, d_in) — per-weight importance score
        protect_pct:   fraction in [0, 1] to protect

    Returns:
        mask: (d_out, d_in) bool — True = protected (won't be pruned)
    """
    if protect_pct <= 0:
        return torch.zeros_like(weight_tensor, dtype=torch.bool)
    if protect_pct >= 1:
        return torch.ones_like(weight_tensor, dtype=torch.bool)

    flat_scores = score_tensor.flatten()
    k = int(flat_scores.numel() * protect_pct)
    if k == 0:
        return torch.zeros_like(weight_tensor, dtype=torch.bool)

    threshold = torch.topk(flat_scores, k, largest=True).values[-1]
    return score_tensor >= threshold


def clamp_protection_vs_sparsity(protect_pct, target_sparsity, safety_margin=0.02):
    """
    Ensure protection + sparsity is feasible.
    Target sparsity must come from unprotected pool.
    If protect_pct + target_sparsity > 1 - safety_margin, reduce protection.

    Returns adjusted protect_pct.
    """
    max_protect = 1.0 - target_sparsity - safety_margin
    return min(protect_pct, max(0.0, max_protect))