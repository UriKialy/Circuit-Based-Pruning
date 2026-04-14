# sparsity.py — convert importance scores → per-layer/matrix sparsity budgets
#
# From the paper: inverse softmax-temperature scheme.
# Layers with higher importance → lower sparsity.

import torch
import numpy as np
from config import N_LAYERS, LINEAR_LAYERS


def allocate_layer_sparsity(layer_importance, model, target_sparsity,
                             temperature=5.0, min_sparsity=0.0, max_sparsity=0.95):
    """
    Allocate per-layer sparsity ratios using softmax-temperature scheme.

    Args:
        layer_importance: dict layer_idx → scalar importance
        model:            HF model (to count params per layer)
        target_sparsity:  global target (e.g. 0.5)
        temperature:      T > 0. Low = extreme non-uniform. High = uniform.
        min_sparsity:     floor per layer
        max_sparsity:     ceiling per layer

    Returns:
        sparsity_map: dict (layer_idx, matrix_name) → sparsity ratio
    """
    n_layers = len(model.model.layers)
    layers_list = list(range(n_layers))

    # Count params per layer
    param_counts = {}
    for layer_idx in layers_list:
        layer = model.model.layers[layer_idx]
        count = 0
        for name, param in layer.named_parameters():
            if "weight" in name:
                count += param.numel()
        param_counts[layer_idx] = count
    total_params = sum(param_counts.values())

    # Importance scores → softmax weights (inverse: higher importance → lower weight)
    scores = np.array([layer_importance.get(i, 0.0) for i in layers_list])
    # Normalize to prevent overflow
    scores = scores / (scores.max() + 1e-8)

    # Inverse softmax: high importance → low sparsity
    logits = -scores / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    # Scale to hit target sparsity (parameter-weighted)
    raw_sparsity = {}
    for i, layer_idx in enumerate(layers_list):
        raw_sparsity[layer_idx] = target_sparsity * weights[i] * total_params / param_counts[layer_idx]

    # Clamp + redistribute to hit target exactly
    sparsity_per_layer = _redistribute(
        raw_sparsity, param_counts, total_params,
        target_sparsity, min_sparsity, max_sparsity,
    )

    # Expand to per-matrix (all matrices in layer share same ratio)
    sparsity_map = {}
    for layer_idx, ratio in sparsity_per_layer.items():
        for matrix_name in LINEAR_LAYERS:
            sparsity_map[(layer_idx, matrix_name)] = ratio

    return sparsity_map


def _redistribute(raw, param_counts, total_params, target, floor, ceil,
                  max_iters=50):
    """Iteratively clamp and redistribute to hit global target."""
    ratios = dict(raw)
    for _ in range(max_iters):
        # Clamp
        clamped_mass = 0.0
        free_params = 0
        for idx, r in ratios.items():
            if r < floor:
                ratios[idx] = floor
                clamped_mass += (floor - r) * param_counts[idx]
            elif r > ceil:
                ratios[idx] = ceil
                clamped_mass += (ceil - r) * param_counts[idx]
            else:
                free_params += param_counts[idx]

        if abs(clamped_mass) < 1e-8 or free_params == 0:
            break

        # Redistribute clamped excess to free layers
        for idx in ratios:
            if floor < ratios[idx] < ceil:
                ratios[idx] += clamped_mass * param_counts[idx] / (free_params * total_params)

    # Verify
    achieved = sum(ratios[i] * param_counts[i] for i in ratios) / total_params
    if abs(achieved - target) > 0.01:
        print(f"  Warning: achieved sparsity {achieved:.4f} vs target {target:.4f}")

    return ratios


def uniform_sparsity(model, target_sparsity):
    """Uniform allocation — same ratio for every matrix."""
    n_layers = len(model.model.layers)
    sparsity_map = {}
    for layer_idx in range(n_layers):
        for matrix_name in LINEAR_LAYERS:
            sparsity_map[(layer_idx, matrix_name)] = target_sparsity
    return sparsity_map


def print_allocation(sparsity_map, n_layers):
    """Pretty-print the allocation table."""
    print(f"\n{'Layer':<6} {'q_proj':>8} {'k_proj':>8} {'v_proj':>8} "
          f"{'o_proj':>8} {'gate':>8} {'up':>8} {'down':>8}")
    print("-" * 70)
    for layer in range(n_layers):
        vals = []
        for m in LINEAR_LAYERS:
            r = sparsity_map.get((layer, m), 0.0)
            vals.append(f"{r:>7.1%}")
        print(f"{layer:<6} {' '.join(vals)}")
