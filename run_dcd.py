#!/usr/bin/env python3
"""
run_dcd.py — DCD-guided pruning experiment
===========================================

Plugs DCD (your new paper) into the Circuit-Based-Pruning infrastructure
as a third scoring method alongside EAP-IG and RelP, targeting a win
over Wanda++ at 30 / 50 / 70 % sparsity on WikiText-2 PPL.

Two strategies implemented
──────────────────────────
  A) DCD as a weight-scoring criterion  (replaces |W|·‖X‖ in Wanda)
       Score = dcd_weight_score(W, X_stats)
       → drop-in replacement for the EAP-IG path in pruning.py / run.py

  B) DCD as a layer-importance signal   (drives non-uniform sparsity budgets)
       layer_importance = dcd_layer_importance(model, dataloader, device)
       → used with sparsity.allocate_layer_sparsity, then prune_wanda_nonuniform

Both are tested at 30 / 50 / 70 % sparsity and results printed / saved.

Usage
─────
  # RECOMMENDED: two separate calls to avoid double-loading the model
  python run_dcd.py --step scores_only   # score + save, then model is freed
  python run_dcd.py --step prune_only    # load scores + model fresh, prune + eval

  # Single-call (only safe if you have enough VRAM for two 7B loads back-to-back)
  python run_dcd.py --strategy both --sparsity 0.3 0.5 0.7

  # Strategy A only (weight criterion), single sparsity
  python run_dcd.py --strategy A --sparsity 0.5

  # Strategy B only (layer budget)
  python run_dcd.py --strategy B --sparsity 0.5

Dependencies
────────────
  Must be run from (or with sys.path pointing to) the Circuit-Based-Pruning
  workspace where /workspace/wanda is also present (same setup as run.py).
"""

import argparse
import json
import os
import sys
import copy
import gc
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm


def hard_free():
    """Aggressively release all CUDA memory — call after del model."""
    gc.collect()
    gc.collect()          # two passes catches cyclic refs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ── repo path ──────────────────────────────────────────────────────────────
# Adjust if your workspace layout differs.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, "/workspace/wanda")

# ── imports from YOUR existing modules (no functions copied) ───────────────
from config import MODEL_NAME, N_LAYERS, LINEAR_LAYERS, NUM_CALIBRATION_SAMPLES
from utils import load_model, load_tokenizer, free_memory, save_scores, load_scores
from evaluation import eval_perplexity_wikitext2
from data import get_calibration_loader, load_texts
from sparsity import allocate_layer_sparsity, uniform_sparsity, print_allocation
from pruning import prune_wanda_nonuniform, prune_wandapp_eap
from attribution_weights import (
    prepare_block_inputs,
    forward_block_unpruned,
    _make_fwd_kwargs,
    _get_position_embeddings,
    _get_linear_layers,
)

# ── imports from Wanda (no functions copied) ────────────────────────────────
from lib.prune import find_layers, check_sparsity, prepare_calibration_input
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders


# ═══════════════════════════════════════════════════════════════════════════
#  ██████╗  ██████╗ ██████╗
#  ██╔══██╗██╔════╝ ██╔══██╗
#  ██║  ██║██║      ██║  ██║
#  ██║  ██║██║      ██║  ██║
#  ██████╔╝╚██████╗ ██████╔╝
#  ╚═════╝  ╚═════╝ ╚═════╝
#
#  INSERT DCD SCORING HERE — one function per strategy
#  All other code is wired up and ready; only these two stubs need filling.
# ═══════════════════════════════════════════════════════════════════════════

def dcd_weight_scores_block(block, clean_inps, attention_mask,
                             position_ids, position_embeddings=None,
                             nsamples_per_block=None):
    """
    [STRATEGY A]  Compute a per-weight DCD importance score for one decoder block.

    Replace the body of this function with the DCD formula from the paper.

    The function must return:
        dict  {matrix_name (str) -> score_tensor (d_out, d_in) on CPU}

    The scores are used EXACTLY like EAP-IG scores in prune_wandapp_eap:
        W_metric = (alpha * dcd_score + input_norm) * |W|

    Current stub: falls back to gradient norm (similar to RGS) so the
    pipeline runs end-to-end while you fill in the real formula.

    Args:
        block            — one LlamaDecoderLayer (unpruned)
        clean_inps       — list of (1, seqlen, d_model) tensors on CPU
        attention_mask   — attention mask (or None)
        position_ids     — position ids (or None)
        position_embeddings — precomputed RoPE tuple or None
        nsamples_per_block  — how many samples to use (None = all)

    ── DCD PAPER DETAILS TO INSERT ────────────────────────────────────────
    Replace the stub body with the DCD scoring formula.
    Typical structure for a weight-level criterion:

        for each calibration sample:
            run forward (and/or backward) through block
            accumulate score contributions per weight matrix

        return {name: accumulated_score / n_total for name in linears}

    ── STUB (gradient norm, safe to run as placeholder) ───────────────────
    """
    device = next(block.parameters()).device
    linears = _get_linear_layers(block)
    samples = clean_inps if nsamples_per_block is None else clean_inps[:nsamples_per_block]

    # Enable grads temporarily
    for p in block.parameters():
        p.requires_grad_(True)

    sq_grad = {name: torch.zeros_like(layer.weight, dtype=torch.float32)
               for name, layer in linears.items()}
    n_total = 0

    fwd_kw = _make_fwd_kwargs(attention_mask, position_ids, position_embeddings, device)

    for inp in tqdm(samples, desc="  DCD block samples", leave=False):
        inp_dev = inp.to(device)
        for name, layer in linears.items():
            if layer.weight.grad is not None:
                layer.weight.grad.zero_()

        out = block(inp_dev, **fwd_kw)
        if isinstance(out, tuple):
            out = out[0]

        # ── REPLACE from here with DCD forward/backward ──────────────────
        loss = torch.norm(out)      # <─ placeholder: swap for DCD metric
        loss.backward()
        # ── REPLACE to here ──────────────────────────────────────────────

        with torch.no_grad():
            for name, layer in linears.items():
                g = layer.weight.grad
                if g is not None:
                    sq_grad[name] += g.float() ** 2

        n_total += 1
        del out, loss, inp_dev

    scores = {}
    for name in sq_grad:
        scores[name] = torch.sqrt(sq_grad[name] / max(n_total, 1)).cpu()
        linears[name].weight.requires_grad_(False)

    for p in block.parameters():
        p.requires_grad_(False)

    torch.cuda.empty_cache()
    return scores


def dcd_layer_importance_score(block, clean_inps, attention_mask,
                                position_ids, position_embeddings=None):
    """
    [STRATEGY B]  Compute a scalar DCD importance for one decoder block.

    Replace the body of this function with the DCD layer-level formula.

    Must return:
        float — higher = more important (will get lower sparsity)

    Current stub: uses mean output activation norm as a proxy.

    ── DCD PAPER DETAILS TO INSERT ────────────────────────────────────────
    Replace the stub body.  Typical structure:

        run forward (and/or compare clean vs corrupt) through block
        return scalar importance derived from DCD criterion

    ── STUB ───────────────────────────────────────────────────────────────
    """
    device = next(block.parameters()).device
    fwd_kw = _make_fwd_kwargs(attention_mask, position_ids, position_embeddings, device)
    total = 0.0
    n = 0
    with torch.no_grad():
        for inp in clean_inps:
            out = block(inp.to(device), **fwd_kw)
            if isinstance(out, tuple):
                out = out[0]
            # ── REPLACE with DCD layer importance metric ─────────────────
            total += out.float().abs().mean().item()   # <─ placeholder
            # ── REPLACE to here ──────────────────────────────────────────
            n += 1
    return total / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Full-model scorers  (call the block-level stubs above across all layers)
# ═══════════════════════════════════════════════════════════════════════════

def dcd_weight_scores_all_blocks(model, dataloader, device, scores_path=None,
                                  nsamples_per_block=None):
    """
    Compute per-weight DCD scores for every decoder block.

    STREAMING MODE (scores_path provided):
      Scores each block, immediately torch.saves that block's tensors to
      scores_path/block_{i}.pt, then frees the tensors. Peak CPU RAM =
      one block worth of scores (~810 MB for LLaMA-7B) instead of 26 GB.
      Returns None; caller must load via load_scores_streamed().

    LEGACY MODE (scores_path=None):
      Returns full dict: (layer_idx, matrix_name) -> tensor on CPU.
      Only use for small models that fit entirely in RAM.
    """
    streaming = scores_path is not None
    if streaming:
        os.makedirs(scores_path, exist_ok=True)

    print("Preparing block-0 inputs for DCD weight scoring...")
    clean_inps, attention_mask, position_ids = prepare_block_inputs(
        model, dataloader, device)

    position_embeddings = _get_position_embeddings(
        model, clean_inps, position_ids, device)

    layers = model.model.layers
    all_scores = {} if not streaming else None
    n_saved = 0

    for layer_idx in range(len(layers)):
        print(f"\n== DCD weight scoring: block {layer_idx}/{len(layers)-1} ==")
        block = layers[layer_idx]

        scores = dcd_weight_scores_block(
            block, clean_inps, attention_mask, position_ids,
            position_embeddings=position_embeddings,
            nsamples_per_block=nsamples_per_block,
        )

        if streaming:
            # Write this block immediately and free the tensors
            block_path = os.path.join(scores_path, f"block_{layer_idx:02d}.pt")
            torch.save({name: tensor for name, tensor in scores.items()}, block_path)
            del scores
            gc.collect()
            n_saved += 1
            print(f"  saved → {block_path}")
        else:
            for name, tensor in scores.items():
                all_scores[(layer_idx, name)] = tensor

        # Forward clean inputs through unpruned block for next layer
        clean_inps = forward_block_unpruned(
            block, clean_inps, attention_mask, position_ids,
            position_embeddings=position_embeddings,
        )
        torch.cuda.empty_cache()

    if streaming:
        print(f"\nDCD weight scoring complete: {n_saved} block files in {scores_path}")
        return None
    else:
        print(f"\nDCD weight scoring complete: {len(all_scores)} matrices scored.")
        return all_scores


def load_scores_streamed(scores_path, n_layers):
    """Load block-by-block score files back into a flat (layer_idx, name)->tensor dict."""
    all_scores = {}
    for layer_idx in range(n_layers):
        block_path = os.path.join(scores_path, f"block_{layer_idx:02d}.pt")
        block = torch.load(block_path, map_location="cpu", weights_only=True)
        for name, tensor in block.items():
            all_scores[(layer_idx, name)] = tensor
    return all_scores


def dcd_layer_importance_all_blocks(model, dataloader, device):
    """
    Compute per-layer DCD importance for every decoder block.

    Mirrors node_scores_to_layer_importance flow in attribution_nodes.py.
    Returns dict: layer_idx -> scalar importance  (higher = more important)
    """
    print("Preparing block-0 inputs for DCD layer scoring...")
    clean_inps, attention_mask, position_ids = prepare_block_inputs(
        model, dataloader, device)

    position_embeddings = _get_position_embeddings(
        model, clean_inps, position_ids, device)

    layers = model.model.layers
    layer_importance = {}

    for layer_idx in range(len(layers)):
        print(f"\n== DCD layer scoring: block {layer_idx}/{len(layers)-1} ==")
        block = layers[layer_idx]

        imp = dcd_layer_importance_score(
            block, clean_inps, attention_mask, position_ids,
            position_embeddings=position_embeddings,
        )
        layer_importance[layer_idx] = imp
        print(f"  importance = {imp:.6f}")

        clean_inps = forward_block_unpruned(
            block, clean_inps, attention_mask, position_ids,
            position_embeddings=position_embeddings,
        )
        torch.cuda.empty_cache()

    print(f"\nDCD layer importance computed for {len(layer_importance)} layers.")
    return layer_importance


# ═══════════════════════════════════════════════════════════════════════════
#  Pruning with DCD weight scores  (Strategy A)
#  Reuses prune_wandapp_eap from pruning.py with DCD scores in place of EAP-IG
# ═══════════════════════════════════════════════════════════════════════════

def run_strategy_A(args, device, tokenizer, scores_dir, results_dir):
    """
    DCD weight scores as pruning criterion (Strategy A).

    DCD score per weight replaces EAP-IG in the Wanda++ formula:
        W_metric = (alpha * dcd_score + input_norm) * |W|

    Reuses prune_wandapp_eap from pruning.py — no code copied.
    """
    print("\n" + "=" * 70)
    print("  STRATEGY A: DCD weight scores as pruning criterion")
    print("=" * 70)

    # scores_path is a DIRECTORY — one block_{i}.pt file per layer (streaming)
    scores_path = os.path.join(scores_dir, f"dcd_weight_scores_{args.dataset}_s{args.n_samples}")

    # ── 1. Attribution ────────────────────────────────────────────────────
    if args.step != "prune_only":
        print("\n[1/3] Computing DCD weight scores (streaming per-block)...")
        model = load_model(args.model)
        dataloader = get_calibration_loader(
            args.dataset, args.n_samples, seed=0,
            seqlen=model.seqlen, tokenizer=tokenizer)

        # passes scores_path as directory → saves block_{i}.pt on the fly, never
        # accumulates the full 26 GB dict in RAM
        dcd_weight_scores_all_blocks(model, dataloader, device,
                                     scores_path=scores_path,
                                     nsamples_per_block=None)
        del model
        hard_free()
        print(f"  GPU memory freed. Block files in {scores_path}/")
    else:
        if not os.path.isdir(scores_path):
            print(f"ERROR: scores dir not found at {scores_path}")
            print("Remove --step prune_only to compute them first.")
            return {}

    if args.step == "scores_only":
        print("Scores saved. Stopping (--step scores_only).")
        return {}

    import glob
    n_layers = len(glob.glob(os.path.join(scores_path, "block_*.pt")))
    print(f"\nLoading {n_layers} block score files from {scores_path}/...")
    dcd_scores = load_scores_streamed(scores_path, n_layers)

    # ── 2. Prune + eval at each sparsity ──────────────────────────────────
    results = {}
    print("\n[2/3] Pruning with DCD weight criterion...")

    for target_sparsity in args.sparsity:
        print(f"\n── Sparsity: {target_sparsity:.0%} ──")
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            print(f"  GPU free before load: {free_gb:.1f} GB")
        model = load_model(args.model)

        sp_map = uniform_sparsity(model, target_sparsity)

        # prune_wandapp_eap from pruning.py handles all Wanda infrastructure.
        # We pass DCD scores in place of EAP-IG scores.
        prune_wandapp_eap(
            model, tokenizer,
            sparsity_map=sp_map,
            eap_ig_scores=dcd_scores,   # DCD scores used here
            alpha=args.alpha,
            dataset="pile10k",
            nsamples=args.n_cal_samples,
            seed=0,
            device=device,
            verbose=True,
        )

        actual_sp = check_sparsity(model)
        ppl = eval_perplexity_wikitext2(model, tokenizer, device)

        print(f"  Verified sparsity : {actual_sp:.4f}")
        print(f"  WikiText-2 PPL    : {ppl:.2f}")
        results[f"{target_sparsity:.0%}"] = {"actual_sparsity": actual_sp, "ppl": ppl}

        del model
        hard_free()

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Pruning with DCD layer importance  (Strategy B)
#  Reuses allocate_layer_sparsity + prune_wanda_nonuniform — no code copied
# ═══════════════════════════════════════════════════════════════════════════

def run_strategy_B(args, device, tokenizer, scores_dir, results_dir):
    """
    DCD layer importance → non-uniform sparsity budgets (Strategy B).

    Mirrors Experiment 2 (RelP) in run.py:
      1. DCD layer importance scores
      2. allocate_layer_sparsity (inverse softmax-temperature)
      3. prune_wanda_nonuniform  (Wanda locally within each layer)
    """
    print("\n" + "=" * 70)
    print("  STRATEGY B: DCD layer importance → non-uniform budgets")
    print("=" * 70)

    scores_path = os.path.join(scores_dir, f"dcd_layer_importance_{args.dataset}_s{args.n_samples}.pkl")

    # ── 1. Attribution ────────────────────────────────────────────────────
    if args.step != "prune_only":
        print("\n[1/3] Computing DCD layer importance scores...")
        model = load_model(args.model)
        dataloader = get_calibration_loader(
            args.dataset, args.n_samples, seed=0,
            seqlen=model.seqlen, tokenizer=tokenizer)

        layer_importance = dcd_layer_importance_all_blocks(model, dataloader, device)
        save_scores({"layer_importance": layer_importance}, scores_path)
        print(f"DCD layer importance saved → {scores_path}")
        del layer_importance, model
        hard_free()
        print(f"  GPU memory freed. Scores on disk at {scores_path}")
    else:
        print(f"\n[1/3] Loading cached DCD layer importance from {scores_path}...")
        if not os.path.exists(scores_path):
            print(f"ERROR: scores not found at {scores_path}")
            print("Remove --step prune_only to compute them first.")
            return {}

    if args.step == "scores_only":
        print("Scores saved. Stopping (--step scores_only).")
        return {}

    data = load_scores(scores_path)
    layer_importance = data["layer_importance"]

    # Pretty-print layer importance
    print("\nLayer importance (DCD):")
    for li, v in sorted(layer_importance.items()):
        bar = "█" * int(v / max(layer_importance.values()) * 20 + 0.5)
        print(f"  L{li:02d}  {v:8.4f}  {bar}")

    # ── 2. Prune + eval at each sparsity ──────────────────────────────────
    results = {}
    print("\n[2/3] Pruning with DCD non-uniform sparsity budgets...")

    for target_sparsity in args.sparsity:
        print(f"\n── Sparsity: {target_sparsity:.0%}, temperature: {args.temperature} ──")
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            print(f"  GPU free before load: {free_gb:.1f} GB")
        model = load_model(args.model)

        sp_map = allocate_layer_sparsity(
            layer_importance=layer_importance,
            model=model,
            target_sparsity=target_sparsity,
            temperature=args.temperature,
            min_sparsity=0.0,
            max_sparsity=0.95,
        )
        print_allocation(sp_map, len(model.model.layers))

        prune_wanda_nonuniform(
            model, tokenizer,
            sparsity_map=sp_map,
            nsamples=args.n_cal_samples,
            seed=0,
            device=device,
            verbose=True,
        )

        actual_sp = check_sparsity(model)
        ppl = eval_perplexity_wikitext2(model, tokenizer, device)

        print(f"  Verified sparsity : {actual_sp:.4f}")
        print(f"  WikiText-2 PPL    : {ppl:.2f}")
        results[f"{target_sparsity:.0%}"] = {"actual_sparsity": actual_sp, "ppl": ppl}

        del model
        hard_free()

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Combined run + comparison table
# ═══════════════════════════════════════════════════════════════════════════

# Wanda++ reference PPLs for comparison (from your README / MASTER_RESULTS)
WANDAPP_REFERENCE = {
    "0.3": None,       # fill in if you have it
    "0.5": 7.02,       # LLaMA-1 7B unstructured
    "0.7": 55.52,
}
WANDA_REFERENCE = {
    "0.3": None,
    "0.5": 7.26,
    "0.7": 76.17,
}

def print_comparison(results_A, results_B, sparsities):
    """Print a tidy comparison table against Wanda / Wanda++ baselines."""
    header = f"\n{'Method':<30} {'30%':>10} {'50%':>10} {'70%':>10}"
    print("\n" + "═" * 65)
    print("  PPL COMPARISON  (WikiText-2, lower is better)")
    print("═" * 65)
    print(header)
    print("-" * 65)

    def row(label, d):
        vals = []
        for sp in ["30%", "50%", "70%"]:
            if d and sp in d:
                vals.append(f"{d[sp]['ppl']:>10.2f}")
            else:
                vals.append(f"{'—':>10}")
        print(f"  {label:<28} {''.join(vals)}")

    def ref_row(label, ref):
        vals = []
        for sp in ["30%", "50%", "70%"]:
            key = sp.rstrip("%") + ".0" if "." in sp else sp.rstrip("%")
            # try both formats
            v = ref.get(sp.rstrip("%")) or ref.get(sp)
            vals.append(f"{v:>10.2f}" if v is not None else f"{'—':>10}")
        print(f"  {label:<28} {''.join(vals)}")

    ref_row("Wanda (baseline)",    WANDA_REFERENCE)
    ref_row("Wanda++ (target)",    WANDAPP_REFERENCE)
    print("-" * 65)
    row("DCD (strategy A, weight)",   results_A)
    row("DCD (strategy B, layer)",    results_B)

    # Delta vs Wanda++
    print("-" * 65)
    for tag, res in [("Δ vs Wanda++ [A]", results_A), ("Δ vs Wanda++ [B]", results_B)]:
        if not res:
            continue
        vals = []
        for sp in ["30%", "50%", "70%"]:
            key = sp.rstrip("%")
            wpp = WANDAPP_REFERENCE.get(key)
            our = res.get(sp, {}).get("ppl")
            if wpp is not None and our is not None:
                delta = our - wpp
                sign = "+" if delta > 0 else ""
                flag = " ✓" if delta < 0 else " ✗"
                vals.append(f"{sign}{delta:>8.2f}{flag}")
            else:
                vals.append(f"{'—':>10}")
        print(f"  {tag:<28} {''.join(vals)}")

    print("═" * 65)


def parse_args():
    p = argparse.ArgumentParser(description="DCD-guided pruning experiment")
    p.add_argument("--model", type=str, default=MODEL_NAME,
                   help="HuggingFace model id or local path")
    p.add_argument("--strategy", type=str, default="both",
                   choices=["A", "B", "both"],
                   help="A=weight criterion, B=layer budget, both=run both")
    p.add_argument("--step", type=str, default="full",
                   choices=["full", "scores_only", "prune_only"],
                   help="full=score+prune+eval, scores_only, prune_only")
    p.add_argument("--sparsity", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                   help="Target sparsity ratios to test")
    p.add_argument("--alpha", type=float, default=500.0,
                   help="Strategy A: DCD score scaling factor (Wanda++ formula)")
    p.add_argument("--temperature", type=float, default=5.0,
                   help="Strategy B: inverse-softmax temperature for budget allocation")
    p.add_argument("--n_samples", type=int, default=128,
                   help="Attribution samples for DCD scoring")
    p.add_argument("--n_cal_samples", type=int, default=128,
                   help="Calibration samples for Wanda pruning step")
    p.add_argument("--dataset", type=str, default="pile10k",
                   choices=["pile10k", "c4"],
                   help="Calibration dataset")
    p.add_argument("--scores_dir", type=str, default="./scores",
                   help="Directory to save / load DCD score files")
    p.add_argument("--results_dir", type=str, default="./results",
                   help="Directory to save result JSON")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  DCD-GUIDED PRUNING EXPERIMENT" + " " * 32 + "║")
    print(f"║  model      : {args.model:<48}║")
    print(f"║  sparsities : {str(args.sparsity):<48}║")
    print(f"║  strategy   : {args.strategy:<48}║")
    print(f"║  alpha (A)  : {args.alpha:<48}║")
    print(f"║  temperature: {args.temperature:<48}║")
    print("╚" + "═" * 63 + "╝")

    tokenizer = load_tokenizer(args.model)

    results_A, results_B = {}, {}

    if args.strategy in ("A", "both"):
        results_A = run_strategy_A(
            args, device, tokenizer, args.scores_dir, args.results_dir)

    if args.strategy in ("B", "both"):
        results_B = run_strategy_B(
            args, device, tokenizer, args.scores_dir, args.results_dir)

    # ── Save JSON ──────────────────────────────────────────────────────────
    combined = {
        "model": args.model,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "strategy_A_weight_criterion": results_A,
        "strategy_B_layer_budget":     results_B,
        "reference_wanda":    WANDA_REFERENCE,
        "reference_wandapp":  WANDAPP_REFERENCE,
    }
    out_path = os.path.join(args.results_dir, "dcd_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Comparison table ───────────────────────────────────────────────────
    if args.step != "scores_only":
        print_comparison(results_A, results_B, args.sparsity)

    print("\nDone!")


if __name__ == "__main__":
    main()
