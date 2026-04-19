#!/usr/bin/env python3
"""
eval_zeroshot.py — Zero-shot evaluation on pruned models
=========================================================

Uses lm_eval 0.4.x API. Adjustable pruning method.

Usage:
    # Vanilla Wanda at 50%
    python eval_zeroshot.py --method wanda --sparsity 0.5

    # EAP-IG + Wanda (exp 3) at 50%
    python eval_zeroshot.py --method eapig --sparsity 0.5 --alpha 100

    # RelP non-uniform + Wanda (exp 2) at 70%
    python eval_zeroshot.py --method relp --sparsity 0.7 --temperature 3.0

    # Dense (no pruning)
    python eval_zeroshot.py --method dense

    # Custom task list
    python eval_zeroshot.py --method wanda --sparsity 0.5 --tasks boolq,piqa,hellaswag
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, "/workspace/wanda")

from config import MODEL_NAME, N_LAYERS
from utils import load_model, load_tokenizer, free_memory, load_scores
from pruning import (run_wanda_uniform, prune_wanda_nonuniform,
                     prune_wandapp_eap, check_sparsity)
from sparsity import allocate_layer_sparsity, uniform_sparsity
from evaluation import eval_perplexity_wikitext2


DEFAULT_TASKS = [
    "boolq", "piqa", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "openbookqa",
    "rte", "mrpc",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, required=True,
                   choices=["dense", "wanda", "eapig", "relp"])
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=100.0,
                   help="EAP-IG scaling (for method=eapig)")
    p.add_argument("--temperature", type=float, default=3.0,
                   help="Softmax temperature (for method=relp)")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task list. Default: boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,rte,mrpc")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--eapig_scores", type=str,
                   default="./scores/eap_ig_weights_pile10k_l2_s128.pkl")
    p.add_argument("--relp_scores", type=str,
                   default="./scores/relp_nodes_pile10k_s128.pkl")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eval_ppl", action="store_true",
                   help="Also evaluate WikiText-2 PPL")
    return p.parse_args()


def prune_model_by_method(model, tokenizer, args):
    """Apply pruning based on selected method. Returns method description string."""
    device = torch.device(args.device)

    if args.method == "dense":
        return "dense (no pruning)"

    elif args.method == "wanda":
        run_wanda_uniform(model, tokenizer, args.sparsity, device=device)
        return f"wanda_uniform_{args.sparsity:.0%}"

    elif args.method == "eapig":
        if not os.path.exists(args.eapig_scores):
            raise FileNotFoundError(
                f"EAP-IG scores not found: {args.eapig_scores}\n"
                "Run: python run.py --experiment 1 --step attribution_only")
        eap_scores = load_scores(args.eapig_scores)
        sp_map = uniform_sparsity(model, args.sparsity)
        prune_wandapp_eap(
            model, tokenizer, sp_map,
            eap_ig_scores=eap_scores, alpha=args.alpha,
            device=device,
        )
        return f"eapig_alpha{args.alpha}_{args.sparsity:.0%}"

    elif args.method == "relp":
        if not os.path.exists(args.relp_scores):
            raise FileNotFoundError(
                f"RelP scores not found: {args.relp_scores}\n"
                "Run: python run.py --experiment 2 --step attribution_only")
        from attribution_nodes import relp_to_layer_importance
        relp_data = load_scores(args.relp_scores)
        if "layer_importance" in relp_data:
            layer_imp = relp_data["layer_importance"]
        else:
            layer_imp = relp_to_layer_importance(relp_data["sub_scores"], N_LAYERS)
        sp_map = allocate_layer_sparsity(
            layer_imp, model, args.sparsity,
            temperature=args.temperature,
        )
        prune_wanda_nonuniform(
            model, tokenizer, sp_map, device=device,
        )
        return f"relp_T{args.temperature}_{args.sparsity:.0%}"


def run_zero_shot(model, tokenizer, tasks, num_fewshot=0, batch_size="auto"):
    """Run zero-shot eval using lm_eval 0.4.x API."""
    from lm_eval.models.huggingface import HFLM; from lm_eval.evaluator import simple_evaluate

    lm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    results = simple_evaluate(
        model=lm_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    return results


def extract_results(raw_results, tasks):
    """Extract clean accuracy numbers from lm_eval output."""
    clean = {}
    for task in tasks:
        if task in raw_results["results"]:
            r = raw_results["results"][task]
            # lm_eval 0.4.x uses "acc,none" or "acc_norm,none" keys
            acc = r.get("acc,none", r.get("acc", None))
            acc_norm = r.get("acc_norm,none", r.get("acc_norm", None))
            clean[task] = {
                "acc": round(acc * 100, 2) if acc is not None else None,
                "acc_norm": round(acc_norm * 100, 2) if acc_norm is not None else None,
            }
    return clean


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    tasks = args.tasks.split(",") if args.tasks else DEFAULT_TASKS

    print(f"\n{'='*60}")
    print(f"  Zero-Shot Evaluation")
    print(f"  Method: {args.method}")
    if args.method != "dense":
        print(f"  Sparsity: {args.sparsity:.0%}")
    if args.method == "eapig":
        print(f"  Alpha: {args.alpha}")
    if args.method == "relp":
        print(f"  Temperature: {args.temperature}")
    print(f"  Tasks: {tasks}")
    print(f"{'='*60}\n")

    # Load and prune
    print("Loading model...")
    model = load_model()
    tokenizer = load_tokenizer()

    print("Pruning...")
    desc = prune_model_by_method(model, tokenizer, args)
    sp = check_sparsity(model) if args.method != "dense" else 0.0
    print(f"  Method: {desc}")
    print(f"  Actual sparsity: {sp:.4f}")

    # Optional PPL
    if args.eval_ppl:
        print("\nEvaluating PPL...")
        ppl = eval_perplexity_wikitext2(model, tokenizer)
        print(f"  WikiText-2 PPL: {ppl:.2f}")
    else:
        ppl = None

    # Zero-shot
    print("\nRunning zero-shot evaluation...")
    raw_results = run_zero_shot(model, tokenizer, tasks,
                                 num_fewshot=args.num_fewshot,
                                 batch_size=args.batch_size)
    clean = extract_results(raw_results, tasks)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Results: {desc}")
    print(f"{'='*60}")
    print(f"  {'Task':<20} {'Acc':>8} {'Acc_norm':>10}")
    print(f"  {'-'*40}")
    accs = []
    for task in tasks:
        if task in clean:
            a = clean[task]["acc"]
            an = clean[task]["acc_norm"]
            accs.append(a if a is not None else 0)
            a_str = f"{a:.1f}" if a is not None else "—"
            an_str = f"{an:.1f}" if an is not None else "—"
            print(f"  {task:<20} {a_str:>8} {an_str:>10}")
    avg = sum(accs) / len(accs) if accs else 0
    print(f"  {'-'*40}")
    print(f"  {'Average':<20} {avg:>8.1f}")
    if ppl:
        print(f"  {'WikiText-2 PPL':<20} {ppl:>8.2f}")

    # Save
    output = {
        "method": desc,
        "sparsity": sp,
        "ppl": ppl,
        "tasks": clean,
        "average_acc": avg,
        "config": {
            "method": args.method,
            "sparsity": args.sparsity,
            "alpha": args.alpha if args.method == "eapig" else None,
            "temperature": args.temperature if args.method == "relp" else None,
            "num_fewshot": args.num_fewshot,
        }
    }
    out_path = os.path.join(args.results_dir, f"zeroshot_{desc}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")