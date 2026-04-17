#!/usr/bin/env python3
"""
run_full_paper.py — Full paper experiment suite on LLaMA-1 7B
==============================================================
Replicates the circuit-guided pruning experiments from the 3B notebook.
Uses cached RelP scores. No TransformerLens needed at runtime.

Usage:
    python run_full_paper.py                          # full suite
    python run_full_paper.py --only temp_sweep        # just temperature sweep
    python run_full_paper.py --only full_sweep        # just sparsity sweep
    python run_full_paper.py --only ablation_ushape   # just U-shape ablation
    python run_full_paper.py --only ablation_protect  # just protection ablation
    python run_full_paper.py --only ablation_shuffle  # just shuffle ablation
    python run_full_paper.py --only crossover         # just fine-grained crossover
    python run_full_paper.py --use_ro                 # add Regional Optimization
"""

import argparse
import gc
import json
import os
import random

import numpy as np
import torch

from config import MODEL_NAME, N_LAYERS, LINEAR_LAYERS
from utils import load_model, load_tokenizer, check_sparsity, free_memory, \
    load_scores, save_scores, print_gpu_memory
from evaluation import eval_perplexity_wikitext2
from sparsity import allocate_layer_sparsity, uniform_sparsity, print_allocation
from pruning import prune_model
from attribution_nodes import relp_to_layer_importance, relp_to_matrix_importance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None,
                   choices=["temp_sweep", "full_sweep", "ablation_ushape",
                            "ablation_protect", "ablation_shuffle", "crossover"])
    p.add_argument("--relp_scores", type=str,
                   default="./scores/relp_nodes_pile10k_s64.pkl")
    p.add_argument("--nsamples", type=int, default=128,
                   help="Calibration samples for Wanda pruning")
    p.add_argument("--use_ro", action="store_true",
                   help="Enable Regional Optimization after pruning")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str, default="./results")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def run_one(layer_importance, target_sparsity, temperature, args,
            protect_layers=None, max_sparsity=0.95, verbose=False):
    """Load model, allocate, prune, eval, free. Returns PPL."""
    model = load_model()
    tokenizer = load_tokenizer()
    device = torch.device(args.device)

    sp_map = allocate_layer_sparsity(
        layer_importance, model, target_sparsity,
        temperature=temperature,
        min_sparsity=0.0,
        max_sparsity=max_sparsity,
    )

    # Apply layer protection if requested
    if protect_layers:
        for layer_idx in protect_layers:
            for m in LINEAR_LAYERS:
                sp_map[(layer_idx, m)] = min(sp_map.get((layer_idx, m), 0), 0.1)
        # Redistribute to hit target
        _redistribute_after_protect(sp_map, model, target_sparsity)

    if verbose:
        print_allocation(sp_map, N_LAYERS)

    ro_fn = None
    if args.use_ro:
        from regional_optimizer import regional_optimize
        ro_fn = regional_optimize

    prune_model(
        model, tokenizer, sp_map,
        scoring_method="wanda",
        nsamples=args.nsamples,
        device=device,
        use_ro=args.use_ro,
        ro_fn=ro_fn,
        verbose=False,
    )

    actual_sp = check_sparsity(model)
    ppl = eval_perplexity_wikitext2(model, tokenizer, device)

    del model
    free_memory()
    return ppl, actual_sp


def _redistribute_after_protect(sp_map, model, target):
    """After capping protected layers, bump others to hit global target."""
    layers = model.model.layers
    total_params = 0
    total_pruned = 0
    layer_params = {}

    for i, layer in enumerate(layers):
        count = sum(p.numel() for n, p in layer.named_parameters() if "weight" in n)
        layer_params[i] = count
        total_params += count
        # Average sparsity for this layer
        sp_vals = [sp_map.get((i, m), 0) for m in LINEAR_LAYERS]
        total_pruned += count * np.mean(sp_vals)

    achieved = total_pruned / total_params
    if abs(achieved - target) < 0.005:
        return

    # Scale uncapped layers
    capped = {i for i, m in sp_map if sp_map[(i, m)] <= 0.1
              and any(sp_map.get((i, mm), 1.0) <= 0.1 for mm in LINEAR_LAYERS)}
    uncapped_params = sum(layer_params[i] for i in range(len(layers)) if i not in capped)
    if uncapped_params == 0:
        return
    deficit = (target - achieved) * total_params
    for i in range(len(layers)):
        if i not in capped:
            bump = deficit * layer_params[i] / uncapped_params / layer_params[i]
            for m in LINEAR_LAYERS:
                sp_map[(i, m)] = min(0.95, sp_map.get((i, m), 0) + bump)


def make_u_shaped_importance(n_layers):
    """U-shape: high at edges, low in middle. No circuit knowledge."""
    imp = {}
    center = (n_layers - 1) / 2.0
    for layer in range(n_layers):
        score = abs(layer - center) / center
        imp[layer] = score
    return imp


def make_shuffled_importance(layer_importance, seed):
    """Randomly permute layer scores."""
    layers = sorted(layer_importance.keys())
    scores = [layer_importance[l] for l in layers]
    random.seed(seed)
    random.shuffle(scores)
    return {l: scores[i] for i, l in enumerate(layers)}


# ═══════════════════════════════════════════════════════════════
#  Experiment 1: Temperature sweep at 30% and 50%
# ═══════════════════════════════════════════════════════════════

def temp_sweep(layer_importance, args):
    print("\n" + "=" * 60)
    print("  TEMPERATURE SWEEP — layer-only, no protection")
    print("=" * 60)

    results = {}
    for target_pct in [50, 70]:
        target = target_pct / 100.0
        results[target_pct] = {}
        for temp in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
            ppl, sp = run_one(layer_importance, target, temp, args)
            results[target_pct][temp] = ppl
            print(f"  {target_pct}% T={temp:<5} → PPL = {ppl:.2f}")

    # Find best T per sparsity
    for pct in results:
        best_t = min(results[pct], key=results[pct].get)
        print(f"\n  Best T at {pct}%: T={best_t} → PPL={results[pct][best_t]:.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Experiment 2: Full sparsity sweep at best T
# ═══════════════════════════════════════════════════════════════

def full_sweep(layer_importance, best_temp, args):
    print("\n" + "=" * 60)
    print(f"  FULL SWEEP — layer-only, T={best_temp}, no protection")
    print("=" * 60)

    results = {}
    for pct in [50, 70]:
        target = pct / 100.0
        if target == 0:
            model = load_model()
            tokenizer = load_tokenizer()
            ppl = eval_perplexity_wikitext2(model, tokenizer, torch.device(args.device))
            del model
            free_memory()
        else:
            ppl, _ = run_one(layer_importance, target, best_temp, args, verbose=(pct == 50))
        results[pct] = ppl
        print(f"  {pct}% → PPL = {ppl:.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Ablation A: U-shaped heuristic vs circuit-guided
# ═══════════════════════════════════════════════════════════════

def ablation_ushape(layer_importance, best_temp, args):
    print("\n" + "=" * 60)
    print("  ABLATION: U-shaped heuristic vs circuit-guided")
    print("=" * 60)

    u_importance = make_u_shaped_importance(N_LAYERS)

    print(f"  {'Sparsity':<10} {'Uniform':>10} {'U-shaped':>10} {'Circuit':>10}")
    print(f"  {'-'*42}")

    results = {"uniform": {}, "u_shaped": {}, "circuit": {}}
    for pct in [30, 50, 70]:
        target = pct / 100.0

        # Uniform
        ppl_u, _ = run_one(layer_importance, target, 1e6, args)  # huge T → uniform
        results["uniform"][pct] = ppl_u

        # U-shaped
        ppl_ush, _ = run_one(u_importance, target, best_temp, args)
        results["u_shaped"][pct] = ppl_ush

        # Circuit
        ppl_c, _ = run_one(layer_importance, target, best_temp, args)
        results["circuit"][pct] = ppl_c

        best = min(ppl_u, ppl_ush, ppl_c)
        winner = "U" if ppl_ush == best else ("C" if ppl_c == best else "Unif")
        print(f"  {pct}%{'':<6} {ppl_u:>10.2f} {ppl_ush:>10.2f} {ppl_c:>10.2f}  → {winner}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Ablation B: Protected vs unprotected (first/last layer cap)
# ═══════════════════════════════════════════════════════════════

def ablation_protect(layer_importance, best_temp, args):
    print("\n" + "=" * 60)
    print("  ABLATION: Protected (cap L0/L31 at 10%) vs unprotected")
    print("=" * 60)

    print(f"  {'Sparsity':<10} {'Protected':>12} {'Unprotected':>13} {'Uniform':>10}")
    print(f"  {'-'*48}")

    results = {"protected": {}, "unprotected": {}, "uniform": {}}
    for pct in [50, 70]:
        target = pct / 100.0

        ppl_u, _ = run_one(layer_importance, target, 1e6, args)
        results["uniform"][pct] = ppl_u

        ppl_p, _ = run_one(layer_importance, target, best_temp, args,
                           protect_layers=[0, N_LAYERS - 1])
        results["protected"][pct] = ppl_p

        ppl_np, _ = run_one(layer_importance, target, best_temp, args)
        results["unprotected"][pct] = ppl_np

        print(f"  {pct}%{'':<6} {ppl_p:>12.2f} {ppl_np:>13.2f} {ppl_u:>10.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Ablation C: Shuffled scores (5 random permutations)
# ═══════════════════════════════════════════════════════════════

def ablation_shuffle(layer_importance, best_temp, args, n_shuffles=5):
    print("\n" + "=" * 60)
    print(f"  ABLATION: Shuffled scores ({n_shuffles} random permutations)")
    print("=" * 60)

    results_by_sparsity = {30: [], 50: [], 70: []}

    for seed in range(n_shuffles):
        shuf_imp = make_shuffled_importance(layer_importance, seed * 42)
        for pct in [50, 70]:
            target = pct / 100.0
            ppl, _ = run_one(shuf_imp, target, best_temp, args)
            results_by_sparsity[pct].append(ppl)
            print(f"  Shuffle {seed}, {pct}% → PPL = {ppl:.2f}")

    # Circuit scores for comparison
    circuit_results = {}
    for pct in [50, 70]:
        ppl, _ = run_one(layer_importance, pct / 100.0, best_temp, args)
        circuit_results[pct] = ppl

    print(f"\n  {'Sparsity':<10} {'Shuffled (mean±std)':>22} {'Circuit':>10}")
    print(f"  {'-'*44}")
    for pct in [50, 70]:
        vals = results_by_sparsity[pct]
        mean, std = np.mean(vals), np.std(vals)
        c = circuit_results[pct]
        print(f"  {pct}%{'':<6} {mean:>10.2f} ± {std:<8.2f} {c:>10.2f}")

    return {"shuffled": results_by_sparsity, "circuit": circuit_results}


# ═══════════════════════════════════════════════════════════════
#  Experiment: Fine-grained crossover sweep
# ═══════════════════════════════════════════════════════════════

def crossover(layer_importance, best_temp, args):
    print("\n" + "=" * 60)
    print(f"  CROSSOVER SWEEP — uniform vs layer-only T={best_temp}")
    print("=" * 60)

    points = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    results = {"uniform": {}, "circuit": {}}

    print(f"  {'Sparsity':<10} {'Uniform':>10} {'Circuit':>10} {'Delta':>10} {'Winner':>10}")
    print(f"  {'-'*52}")

    for pct in points:
        target = pct / 100.0

        ppl_u, _ = run_one(layer_importance, target, 1e6, args)
        ppl_c, _ = run_one(layer_importance, target, best_temp, args)

        results["uniform"][pct] = ppl_u
        results["circuit"][pct] = ppl_c

        delta = ppl_u - ppl_c
        winner = "Circuit" if delta > 0.05 else ("Uniform" if delta < -0.05 else "Tie")
        print(f"  {pct}%{'':<6} {ppl_u:>10.2f} {ppl_c:>10.2f} {delta:>+10.2f} {winner:>10}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Load cached RelP scores
    print(f"Loading RelP scores from {args.relp_scores}...")
    relp_data = load_scores(args.relp_scores)

    if "layer_importance" in relp_data:
        layer_importance = relp_data["layer_importance"]
    elif "sub_scores" in relp_data:
        layer_importance = relp_to_layer_importance(relp_data["sub_scores"], N_LAYERS)
    else:
        raise ValueError("RelP scores file doesn't contain expected keys")

    print(f"Layer importance scores for {len(layer_importance)} layers loaded.")
    for l in [0, 1, N_LAYERS // 2, N_LAYERS - 2, N_LAYERS - 1]:
        print(f"  Layer {l:>2}: {layer_importance.get(l, 0):.4f}")

    all_results = {}

    # ── Temperature sweep ──
    if args.only is None or args.only == "temp_sweep":
        res = temp_sweep(layer_importance, args)
        all_results["temp_sweep"] = res
        # Find global best T
        best_temp = 5.0  # default
        if 50 in res:
            best_temp = min(res[50], key=res[50].get)
        print(f"\n  Using best T = {best_temp} for remaining experiments")
    else:
        best_temp = 5.0

    # ── Full sweep ──
    if args.only is None or args.only == "full_sweep":
        res = full_sweep(layer_importance, best_temp, args)
        all_results["full_sweep"] = res

    # ── Ablation: U-shaped ──
    if args.only is None or args.only == "ablation_ushape":
        res = ablation_ushape(layer_importance, best_temp, args)
        all_results["ablation_ushape"] = res

    # ── Ablation: Protection ──
    if args.only is None or args.only == "ablation_protect":
        res = ablation_protect(layer_importance, best_temp, args)
        all_results["ablation_protect"] = res

    # ── Ablation: Shuffle ──
    if args.only is None or args.only == "ablation_shuffle":
        res = ablation_shuffle(layer_importance, best_temp, args)
        all_results["ablation_shuffle"] = res

    # ── Crossover ──
    if args.only is None or args.only == "crossover":
        res = crossover(layer_importance, best_temp, args)
        all_results["crossover"] = res

    # Save everything
    suffix = "_ro" if args.use_ro else ""
    out_path = os.path.join(args.results_dir, f"full_paper_7b{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")
    print("Done!")