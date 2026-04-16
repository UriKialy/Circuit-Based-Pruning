#!/usr/bin/env python3
"""
run.py — Main experiment runner for Circuit-Guided EAP-IG Pruning
=================================================================

Three experiments on LLaMA-1 7B:

  Experiment 1: EAP-IG weight-level scoring
    - Compute per-weight EAP-IG scores (block-by-block)
    - Use as pruning criterion (replacing Wanda's |W|*||X||)
    - Eval PPL at 30%, 50%, 70% sparsity

  Experiment 2: RelP node-level → layer budget + Wanda local
    - RelP attribution → per-layer importance
    - Non-uniform sparsity allocation (softmax-temperature)
    - Wanda within each layer
    - (This is your published method, replicated on LLaMA-1 7B)

  Experiment 3: Wanda++ with EAP-IG replacing RGS
    - Uniform sparsity (like Wanda++)
    - But swap RGS gradient for EAP-IG integrated gradient
    - Compare to Wanda++ paper numbers directly

Usage:
    python run.py --experiment 1 --sparsity 50
    python run.py --experiment 2 --sparsity 50 --temperature 5.0
    python run.py --experiment 3 --sparsity 50 --alpha 100
    python run.py --experiment all
    python run.py --step attribution_only   # just compute and save scores
    python run.py --step prune_only         # load saved scores and prune
"""

import argparse
import json
import sys
import os
import torch

# ── Our modules ──
from config import *
from utils import load_model, load_tokenizer, check_sparsity, free_memory, \
    save_scores, load_scores, print_gpu_memory
from evaluation import eval_perplexity_wikitext2
from corruption import shuffle_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default="1",
                       choices=["1", "2", "3", "4a", "4b", "all"])
    p.add_argument("--step", type=str, default="full",
                   choices=["full", "attribution_only", "prune_only"])
    p.add_argument("--sparsity", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--alpha", type=float, nargs="+", default=[100.0],
                   help="Scaling factor(s) for gradient term. Pass multiple to sweep.")
    p.add_argument("--ig_steps", type=int, default=10)
    p.add_argument("--ig_metric", type=str, default="l2",
                   choices=["ce", "l2", "both"])
    p.add_argument("--n_attr_samples", type=int, default=128)
    p.add_argument("--n_cal_samples", type=int, default=128)
    p.add_argument("--corruption", type=str, default="shuffle",
                   choices=["shuffle", "noise", "both"])
    p.add_argument("--dataset", type=str, default="pile10k",
                   choices=["pile10k", "c4"])
    p.add_argument("--scores_dir", type=str, default="./scores")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--protect_pct", type=float, nargs="+", default=[0.2],
                   help="Protection percentage(s) — sweep supported.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Experiment 1: EAP-IG weight-level scoring
# ═══════════════════════════════════════════════════════════════

def run_experiment_1(args):
    """
    EAP-IG at weight level as pruning criterion.
    Uses HF model block-by-block — no TransformerLens needed.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: EAP-IG weight-level scoring")
    print("=" * 70)
    device = torch.device(args.device)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    scores_path = os.path.join(args.scores_dir,
        f"eap_ig_weights_{args.dataset}_{args.ig_metric}_s{args.n_attr_samples}.pkl")

    # ── Attribution (or load cached) ──
    if args.step != "prune_only":
        from attribution_weights import eap_ig_all_blocks
        from data import get_calibration_loader

        print("\n[1/3] Computing EAP-IG weight scores...")
        model = load_model()
        tokenizer = load_tokenizer()
        print_gpu_memory()

        dataloader = get_calibration_loader(
            args.dataset, args.n_attr_samples, seed=0,
            seqlen=MAX_SEQ_LEN, tokenizer=tokenizer)

        corruption_fn = shuffle_tokens  # token-level only for now

        eap_scores = eap_ig_all_blocks(
            model, dataloader, corruption_fn, device,
            n_steps=args.ig_steps, metric=args.ig_metric,
        )

        save_scores(eap_scores, scores_path)
        del model
        free_memory()
    else:
        eap_scores = load_scores(scores_path)

    if args.step == "attribution_only":
        print("Attribution complete. Scores saved.")
        return

    # ── Pruning + Eval ──
    print("\n[2/3] Pruning and evaluating...")
    from pruning import prune_model
    from sparsity import uniform_sparsity

    tokenizer = load_tokenizer()
    results = {}

    for target in args.sparsity:
        print(f"\n── Sparsity: {target:.0%} ──")
        model = load_model()

        sp_map = uniform_sparsity(model, target)
        prune_model(
            model, tokenizer, sp_map,
            scoring_method="eap_ig",
            eap_ig_scores=eap_scores,
            nsamples=args.n_cal_samples, device=device,
        )

        actual_sp = check_sparsity(model)
        ppl = eval_perplexity_wikitext2(model, tokenizer, device)
        print(f"  Verified sparsity: {actual_sp:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        results[f"{target:.0%}"] = {"sparsity": actual_sp, "ppl": ppl}

        del model
        free_memory()

    # ── Save results ──
    out_path = os.path.join(args.results_dir, "exp1_eap_ig_weights.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[3/3] Results saved to {out_path}")
    _print_results(results, "Exp 1: EAP-IG Weight-Level")


# ═══════════════════════════════════════════════════════════════
#  Experiment 2: RelP node-level → non-uniform budget
# ═══════════════════════════════════════════════════════════════

def run_experiment_2(args):
    """
    RelP for layer importance → non-uniform sparsity + Wanda local.
    Requires TransformerLens + RelP fork.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: RelP node-level + non-uniform Wanda")
    print("=" * 70)
    device = torch.device(args.device)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    scores_path = os.path.join(args.scores_dir,
        f"relp_nodes_{args.dataset}_s{args.n_attr_samples}.pkl")

    # ── Attribution ──
    if args.step != "prune_only":
        from transformer_lens import HookedTransformer
        from attribution_nodes import run_relp_nodes, relp_to_layer_importance
        from data import load_texts

        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16,
        )
        tl_model = HookedTransformer.from_pretrained(
            "llama-7b-hf", 
            hf_model=hf_model,
            center_writing_weights=False,
            center_unembed=False, 
            fold_ln=False,
        use_attn_result=True,
        )
        tl_model.cfg.use_attn_result = False
        tl_model.cfg.use_hook_mlp_in = False
        tl_model.cfg.use_attn_in = False
        tl_model.set_tokenizer(AutoTokenizer.from_pretrained(MODEL_NAME))
        del hf_model

        texts = load_texts(args.dataset, args.n_attr_samples)
        sub_scores = run_relp_nodes(tl_model, texts,
                                     num_samples=args.n_attr_samples,
                                     max_seq_len=MAX_SEQ_LEN)

        layer_importance = relp_to_layer_importance(sub_scores, N_LAYERS)
        save_scores({"sub_scores": sub_scores, "layer_importance": layer_importance},
                    scores_path)

        del tl_model
        free_memory()
    else:
        loaded = load_scores(scores_path)
        layer_importance = loaded["layer_importance"]

    if args.step == "attribution_only":
        print("Attribution complete. Scores saved.")
        return

    # ── Pruning + Eval ──
    print("\n[2/3] Non-uniform pruning...")
    from pruning import prune_model
    from sparsity import allocate_layer_sparsity, print_allocation

    tokenizer = load_tokenizer()
    results = {}

    for target in args.sparsity:
        print(f"\n── Sparsity: {target:.0%}, T={args.temperature} ──")
        model = load_model()

        sp_map = allocate_layer_sparsity(
            layer_importance, model, target,
            temperature=args.temperature,
        )
        print_allocation(sp_map, N_LAYERS)

        prune_model(
            model, tokenizer, sp_map,
            scoring_method="wanda",
            nsamples=args.n_cal_samples, device=device,
        )

        actual_sp = check_sparsity(model)
        ppl = eval_perplexity_wikitext2(model, tokenizer, device)
        print(f"  Verified sparsity: {actual_sp:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        results[f"{target:.0%}"] = {"sparsity": actual_sp, "ppl": ppl}

        del model
        free_memory()

    out_path = os.path.join(args.results_dir, "exp2_relp_nonuniform.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[3/3] Results saved to {out_path}")
    _print_results(results, "Exp 2: RelP + Non-uniform Wanda")


# ═══════════════════════════════════════════════════════════════
#  Experiment 3: Wanda++ with EAP-IG replacing RGS
# ═══════════════════════════════════════════════════════════════

def run_experiment_3(args):
    """
    Wanda++ formula but with EAP-IG instead of regional gradient.
    Score = (α * IG + ||X||) * |W|
    Uniform sparsity — direct comparison to Wanda++ paper numbers.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Wanda++ with EAP-IG (replacing RGS)")
    print("=" * 70)
    device = torch.device(args.device)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    scores_path = os.path.join(args.scores_dir,
        f"eap_ig_weights_{args.dataset}_{args.ig_metric}_s{args.n_attr_samples}.pkl")

    # ── Attribution (reuse exp 1 scores if available) ──
    if args.step != "prune_only":
        if os.path.exists(scores_path):
            print(f"Reusing cached scores from {scores_path}")
            eap_scores = load_scores(scores_path)
        else:
            from attribution_weights import eap_ig_all_blocks
            from data import get_calibration_loader

            print("\n[1/3] Computing EAP-IG weight scores...")
            model = load_model()
            tokenizer = load_tokenizer()

            dataloader = get_calibration_loader(
                args.dataset, args.n_attr_samples, seed=0,
                seqlen=MAX_SEQ_LEN, tokenizer=tokenizer)

            eap_scores = eap_ig_all_blocks(
                model, dataloader, shuffle_tokens, device,
                n_steps=args.ig_steps, metric=args.ig_metric,
            )
            save_scores(eap_scores, scores_path)
            del model
            free_memory()
    else:
        eap_scores = load_scores(scores_path)

    if args.step == "attribution_only":
        return

    # ── Pruning + Eval (sweep alpha) ──
    print("\n[2/3] Wanda++ EAP-IG pruning...")
    from pruning import prune_model
    from sparsity import uniform_sparsity

    tokenizer = load_tokenizer()
    all_results = {}

    for alpha_val in args.alpha:
        results = {}
        for target in args.sparsity:
            print(f"\n── Sparsity: {target:.0%}, α={alpha_val} ──")
            model = load_model()

            sp_map = uniform_sparsity(model, target)
            prune_model(
                model, tokenizer, sp_map,
                scoring_method="wandapp_eap",
                eap_ig_scores=eap_scores,
                alpha=alpha_val,
                nsamples=args.n_cal_samples, device=device,
            )

            actual_sp = check_sparsity(model)
            ppl = eval_perplexity_wikitext2(model, tokenizer, device)
            print(f"  Verified sparsity: {actual_sp:.4f}")
            print(f"  Perplexity: {ppl:.2f}")
            results[f"{target:.0%}"] = {"sparsity": actual_sp, "ppl": ppl}

            del model
            free_memory()

        all_results[f"alpha={alpha_val}"] = results
        _print_results(results, f"Exp 3: Wanda++ EAP-IG (α={alpha_val})")

    out_path = os.path.join(args.results_dir, "exp3_wandapp_eap.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[3/3] Results saved to {out_path}")


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _print_results(results, title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  {'Sparsity':<12} {'PPL':>10}")
    print(f"  {'-'*24}")
    for key in sorted(results.keys()):
        print(f"  {key:<12} {results[key]['ppl']:>10.2f}")


# ═══════════════════════════════════════════════════════════════
#  Paper baselines for comparison
# ═══════════════════════════════════════════════════════════════

PAPER_BASELINES = {
    "dense": {"ppl": 5.68}, # 
    "wanda": {
        "0.5_unstructured": 7.26, # 
        "0.5_2:4": 11.53,         # 
        "0.5_4:8": 8.57,          # 
        "0.6_unstructured": 9.71, # 
        "0.7_unstructured": 76.17,# 
        "0.8_unstructured": 1942.53 # 
    },
    "wandapp": {
        "0.5_unstructured": 7.02, # 
        "0.5_2:4": 9.43,          # 
        "0.5_4:8": 7.88,          # 
        "0.6_unstructured": 9.50, # 
        "0.7_unstructured": 55.52,# 
        "0.8_unstructured": 1586.69 # 
    },
    "sparsegpt": {
        "0.5_unstructured": 7.22, # 
        "0.5_2:4": 11.00,         # 
        "0.5_4:8": 8.61,          # 
    },
    "gblm": {
        "0.5_unstructured": 7.15, # 
        "0.5_2:4": 11.33,         # 
        "0.5_4:8": 8.48,          # 
        "0.6_unstructured": 10.37,# 
        "0.7_unstructured": 54.60,# 
        "0.8_unstructured": 2550.10 # 
    }
}

# ═══════════════════════════════════════════════════════════════
#  Experiment 4a: EAP-IG weight protection + Wanda pruning
# ═══════════════════════════════════════════════════════════════

def run_experiment_4a(args):
    """
    Protect top X% per matrix (by EAP-IG score + matrix multipliers).
    Wanda prunes the rest.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4a: EAP-IG protection + Wanda")
    print("=" * 70)
    device = torch.device(args.device)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load cached EAP-IG scores (from exp 1)
    scores_path = os.path.join(args.scores_dir,
        f"eap_ig_weights_{args.dataset}_{args.ig_metric}_s{args.n_attr_samples}.pkl")
    if not os.path.exists(scores_path):
        print(f"ERROR: EAP-IG scores not found at {scores_path}")
        print("Run: python run.py --experiment 1 --step attribution_only first")
        return
    eap_scores = load_scores(scores_path)

    from pruning import prune_with_protection
    from protection import matrix_multiplier_protection
    from sparsity import uniform_sparsity

    tokenizer = load_tokenizer()
    all_results = {}

    for protect in args.protect_pct:
        results = {}
        for target in args.sparsity:
            print(f"\n── Sparsity: {target:.0%}, Protect base: {protect:.0%} ──")
            model = load_model()

            sp_map = uniform_sparsity(model, target)
            pct_map = matrix_multiplier_protection(
                n_layers=len(model.model.layers),
                base_pct=protect,
            )

            prune_with_protection(
                model, tokenizer, sp_map,
                protect_scores=eap_scores,
                protect_pct_map=pct_map,
                nsamples=args.n_cal_samples, device=device,
            )

            actual_sp = check_sparsity(model)
            ppl = eval_perplexity_wikitext2(model, tokenizer, device)
            print(f"  Verified sparsity: {actual_sp:.4f}")
            print(f"  Perplexity: {ppl:.2f}")
            results[f"{target:.0%}"] = {"sparsity": actual_sp, "ppl": ppl}

            del model
            free_memory()

        all_results[f"protect={protect}"] = results
        _print_results(results, f"Exp 4a: EAP-IG protect (X={protect:.0%})")

    out_path = os.path.join(args.results_dir, "exp4a_eap_protect.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ═══════════════════════════════════════════════════════════════
#  Experiment 4b: RelP-layer + RGS-weight protection + Wanda pruning
# ═══════════════════════════════════════════════════════════════

def run_experiment_4b(args):
    """
    Per-layer protection driven by RelP importance.
    Within each matrix, top X% by Wanda++ RGS score protected.
    Wanda prunes the rest.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4b: RelP-layer + RGS-weight protection + Wanda")
    print("=" * 70)
    device = torch.device(args.device)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load RelP layer importance
    relp_path = os.path.join(args.scores_dir,
        f"relp_nodes_{args.dataset}_s{args.n_attr_samples}.pkl")
    if not os.path.exists(relp_path):
        print(f"ERROR: RelP scores not found at {relp_path}")
        print("Run: python run.py --experiment 2 --step attribution_only first")
        return
    relp_data = load_scores(relp_path)
    layer_importance = relp_data["layer_importance"]

    # Load or compute RGS weight scores
    rgs_path = os.path.join(args.scores_dir,
        f"rgs_weights_{args.dataset}_s{args.n_attr_samples}.pkl")
    if os.path.exists(rgs_path):
        print(f"Loading cached RGS scores...")
        rgs_scores = load_scores(rgs_path)
    else:
        print("Computing RGS scores...")
        from attribution_weights import compute_rgs_scores
        from data import get_calibration_loader
        model = load_model()
        tokenizer = load_tokenizer()
        dataloader = get_calibration_loader(
            args.dataset, args.n_attr_samples, 0,
            MAX_SEQ_LEN, tokenizer)
        rgs_scores = compute_rgs_scores(model, dataloader, device)
        save_scores(rgs_scores, rgs_path)
        del model
        free_memory()

    from pruning import prune_with_protection
    from protection import relp_layered_protection
    from sparsity import uniform_sparsity

    tokenizer = load_tokenizer()
    all_results = {}

    for protect in args.protect_pct:
        results = {}
        for target in args.sparsity:
            print(f"\n── Sparsity: {target:.0%}, Protect base: {protect:.0%} ──")
            model = load_model()

            sp_map = uniform_sparsity(model, target)
            pct_map = relp_layered_protection(
                layer_importance=layer_importance,
                n_layers=len(model.model.layers),
                base_pct=protect,
            )

            prune_with_protection(
                model, tokenizer, sp_map,
                protect_scores=rgs_scores,
                protect_pct_map=pct_map,
                nsamples=args.n_cal_samples, device=device,
            )

            actual_sp = check_sparsity(model)
            ppl = eval_perplexity_wikitext2(model, tokenizer, device)
            print(f"  Verified sparsity: {actual_sp:.4f}")
            print(f"  Perplexity: {ppl:.2f}")
            results[f"{target:.0%}"] = {"sparsity": actual_sp, "ppl": ppl}

            del model
            free_memory()

        all_results[f"protect={protect}"] = results
        _print_results(results, f"Exp 4b: RelP+RGS protect (X={protect:.0%})")

    out_path = os.path.join(args.results_dir, "exp4b_relp_rgs_protect.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
if __name__ == "__main__":
    args = parse_args()
    if args.experiment == "1" or args.experiment == "all":
        run_experiment_1(args)
    if args.experiment == "2" or args.experiment == "all":
        run_experiment_2(args)
    if args.experiment == "3" or args.experiment == "all":
        run_experiment_3(args)
    if args.experiment == "4a" or args.experiment == "all":
        run_experiment_4a(args)
    if args.experiment == "4b" or args.experiment == "all":
        run_experiment_4b(args)
    print("\nDone!")
