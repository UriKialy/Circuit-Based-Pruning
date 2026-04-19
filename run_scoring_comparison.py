#!/usr/bin/env python3
"""
run_scoring_comparison.py — Circuit Score Stability & Data Efficiency
=====================================================================

4 scoring methods compared:
  1. EAP-IG node-level (pile10k)
  2. EAP-IG node-level (C4)
  3. RelP node-level (pile10k)
  4. RelP node-level (C4)

Experiments:
  A. Cross-dataset agreement (cosine similarity, top-K overlap, heatmaps)
  B. Data efficiency (how many samples needed for stable circuit)
  C. PPL validation (does fewer samples hurt downstream pruning)

Usage:
    python run_scoring_comparison.py                        # full suite
    python run_scoring_comparison.py --only score           # just compute scores
    python run_scoring_comparison.py --only compare         # just compare (needs cached scores)
    python run_scoring_comparison.py --only data_efficiency # just data efficiency
    python run_scoring_comparison.py --only ppl_check       # just PPL validation
"""

import argparse
import gc
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import stats as scipy_stats

from config import (MODEL_NAME, N_LAYERS, N_HEADS, D_MODEL, D_HEAD, D_FF,
                    MAX_SEQ_LEN, EMBEDDING_NOISE_STD, LINEAR_LAYERS)
from utils import (load_model, load_tokenizer, free_memory, save_scores,
                   load_scores, print_gpu_memory, check_sparsity)
from evaluation import eval_perplexity_wikitext2
from data import load_texts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None,
                   choices=["score", "compare", "data_efficiency", "ppl_check"])
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--scores_dir", type=str, default="./scores")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--plots_dir", type=str, default="./plots")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  C4 loader (streaming, capped at 10k texts)
# ═══════════════════════════════════════════════════════════════

def load_c4_texts_streaming(num_samples, min_words=20):
    """Load C4 via streaming — no full download."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for example in ds:
        t = example["text"].strip()
        if len(t.split()) > min_words:
            texts.append(t)
        if len(texts) >= num_samples:
            break
    print(f"Loaded {len(texts)} C4 texts via streaming")
    return texts


# ═══════════════════════════════════════════════════════════════
#  EAP-IG node-level scoring (attn heads + MLP as atomic nodes)
# ═══════════════════════════════════════════════════════════════

def run_eap_ig_nodes_on_texts(tl_model, texts, max_words=10, batch_size=1):
    """
    EAP-IG node-level: each node = one attention head or one MLP block.
    Returns raw scores tensor + extracted component dict.
    """
    from attribution_nodes import (run_eap_ig_nodes, extract_node_scores,
                                    node_scores_to_layer_importance)
    raw = run_eap_ig_nodes(tl_model, texts, max_words=max_words,
                            batch_size=batch_size)
    components = extract_node_scores(raw, N_LAYERS, N_HEADS)
    layer_imp = node_scores_to_layer_importance(components, N_LAYERS)
    return raw, components, layer_imp


# ═══════════════════════════════════════════════════════════════
#  RelP node-level scoring
# ═══════════════════════════════════════════════════════════════

def run_relp_on_texts(tl_model, texts, num_samples, max_seq_len):
    """RelP node-level. Returns sub_scores dict + layer importance."""
    from attribution_nodes import run_relp_nodes, relp_to_layer_importance
    sub_scores = run_relp_nodes(tl_model, texts, num_samples=num_samples,
                                 max_seq_len=max_seq_len, add_noise=True)
    layer_imp = relp_to_layer_importance(sub_scores, N_LAYERS)
    return sub_scores, layer_imp


# ═══════════════════════════════════════════════════════════════
#  Score vector extraction (for comparison)
# ═══════════════════════════════════════════════════════════════

def eap_components_to_vector(components):
    """
    Convert EAP-IG component dict to a flat vector.
    Order: [a0.h0, a0.h1, ..., a0.hN, m0, a1.h0, ..., mL]
    Length: N_LAYERS * (N_HEADS + 1)
    """
    vec = []
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            key = f"a{layer}.h{head}"
            vec.append(abs(components[key]["raw_score"]))
        key = f"m{layer}"
        vec.append(abs(components[key]["raw_score"]))
    return np.array(vec)


def relp_subscores_to_vector(sub_scores):
    """
    Convert RelP sub_scores to a flat vector matching node granularity.
    Sum q+k+v+z hooks → attn head score per layer.
    Sum mlp hooks → mlp score per layer.
    Length: N_LAYERS * 2 (one attn, one mlp per layer)
    """
    vec = []
    attn_hooks = ['attn.hook_q', 'attn.hook_k', 'attn.hook_v', 'attn.hook_z']
    mlp_hooks = ['mlp.hook_pre', 'mlp.hook_pre_linear', 'mlp.hook_post', 'hook_mlp_out']
    for layer in range(N_LAYERS):
        attn_score = sum(sub_scores.get((layer, h), 0) for h in attn_hooks)
        vec.append(attn_score)
        mlp_score = sum(sub_scores.get((layer, h), 0) for h in mlp_hooks)
        vec.append(mlp_score)
    return np.array(vec)


def layer_importance_to_vector(layer_imp):
    """Convert layer importance dict to vector of length N_LAYERS."""
    return np.array([layer_imp.get(l, 0) for l in range(N_LAYERS)])


# ═══════════════════════════════════════════════════════════════
#  Similarity metrics
# ═══════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def spearman_corr(a, b):
    """Spearman rank correlation."""
    corr, pval = scipy_stats.spearmanr(a, b)
    return float(corr), float(pval)


def top_k_overlap(a, b, k_frac=0.2):
    """Fraction of top-k% indices that overlap between two score vectors."""
    k = max(1, int(len(a) * k_frac))
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    overlap = len(top_a & top_b) / k
    return float(overlap)


# ═══════════════════════════════════════════════════════════════
#  TransformerLens model loader
# ═══════════════════════════════════════════════════════════════

def load_tl_model():
    """Load TransformerLens model with proper settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16)
    tl_model = HookedTransformer.from_pretrained(
        "llama-7b-hf",
        hf_model=hf_model,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        use_attn_result=True,
    )
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.cfg.use_attn_in = True
    tl_model.set_tokenizer(AutoTokenizer.from_pretrained(MODEL_NAME))
    del hf_model
    free_memory()
    return tl_model


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Compute all 4 score sets
# ═══════════════════════════════════════════════════════════════

def compute_all_scores(args):
    print("\n" + "=" * 60)
    print("  PHASE 1: Computing all 4 score sets")
    print("=" * 60)

    os.makedirs(args.scores_dir, exist_ok=True)
    n = args.n_samples

    # ── EAP-IG node-level on pile10k ──
    path_eap_pile = os.path.join(args.scores_dir, f"eapig_nodes_pile10k_s{n}.pkl")
    if os.path.exists(path_eap_pile):
        print(f"  EAP-IG pile10k: cached at {path_eap_pile}")
    else:
        print(f"\n  Computing EAP-IG node-level on pile10k ({n} samples)...")
        tl_model = load_tl_model()
        texts = load_texts("pile10k", n)
        raw, components, layer_imp = run_eap_ig_nodes_on_texts(tl_model, texts)
        save_scores({
            "raw": raw.cpu(), "components": components, "layer_importance": layer_imp
        }, path_eap_pile)
        del tl_model
        free_memory()

    # ── EAP-IG node-level on C4 ──
    path_eap_c4 = os.path.join(args.scores_dir, f"eapig_nodes_c4_s{n}.pkl")
    if os.path.exists(path_eap_c4):
        print(f"  EAP-IG C4: cached at {path_eap_c4}")
    else:
        print(f"\n  Computing EAP-IG node-level on C4 ({n} samples)...")
        tl_model = load_tl_model()
        texts = load_c4_texts_streaming(n)
        raw, components, layer_imp = run_eap_ig_nodes_on_texts(tl_model, texts)
        save_scores({
            "raw": raw.cpu(), "components": components, "layer_importance": layer_imp
        }, path_eap_c4)
        del tl_model
        free_memory()

    # ── RelP node-level on pile10k ──
    path_relp_pile = os.path.join(args.scores_dir, f"relp_nodes_pile10k_s{n}.pkl")
    # Also accept the old s64 file
    path_relp_pile_old = os.path.join(args.scores_dir, "relp_nodes_pile10k_s64.pkl")
    if os.path.exists(path_relp_pile):
        print(f"  RelP pile10k: cached at {path_relp_pile}")
    elif os.path.exists(path_relp_pile_old):
        print(f"  RelP pile10k: using older cache at {path_relp_pile_old}")
        path_relp_pile = path_relp_pile_old
    else:
        print(f"\n  Computing RelP node-level on pile10k ({n} samples)...")
        tl_model = load_tl_model()
        tl_model.cfg.use_attn_result = False
        tl_model.cfg.use_hook_mlp_in = False
        tl_model.cfg.use_attn_in = False
        texts = load_texts("pile10k", n)
        sub_scores, layer_imp = run_relp_on_texts(
            tl_model, texts, num_samples=n, max_seq_len=MAX_SEQ_LEN)
        save_scores({
            "sub_scores": sub_scores, "layer_importance": layer_imp
        }, path_relp_pile)
        del tl_model
        free_memory()

    # ── RelP node-level on C4 ──
    path_relp_c4 = os.path.join(args.scores_dir, f"relp_nodes_c4_s{n}.pkl")
    if os.path.exists(path_relp_c4):
        print(f"  RelP C4: cached at {path_relp_c4}")
    else:
        print(f"\n  Computing RelP node-level on C4 ({n} samples)...")
        tl_model = load_tl_model()
        tl_model.cfg.use_attn_result = False
        tl_model.cfg.use_hook_mlp_in = False
        tl_model.cfg.use_attn_in = False
        texts = load_c4_texts_streaming(n)
        sub_scores, layer_imp = run_relp_on_texts(
            tl_model, texts, num_samples=n, max_seq_len=MAX_SEQ_LEN)
        save_scores({
            "sub_scores": sub_scores, "layer_importance": layer_imp
        }, path_relp_c4)
        del tl_model
        free_memory()

    return path_eap_pile, path_eap_c4, path_relp_pile, path_relp_c4


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Compare all 4 score sets
# ═══════════════════════════════════════════════════════════════

def compare_scores(args, paths=None):
    print("\n" + "=" * 60)
    print("  PHASE 2: Cross-dataset & cross-method comparison")
    print("=" * 60)

    n = args.n_samples
    if paths:
        p_ep, p_ec, p_rp, p_rc = paths
    else:
        p_ep = os.path.join(args.scores_dir, f"eapig_nodes_pile10k_s{n}.pkl")
        p_ec = os.path.join(args.scores_dir, f"eapig_nodes_c4_s{n}.pkl")
        p_rp = os.path.join(args.scores_dir, f"relp_nodes_pile10k_s{n}.pkl")
        # Try old cache
        p_rp_old = os.path.join(args.scores_dir, "relp_nodes_pile10k_s64.pkl")
        if not os.path.exists(p_rp) and os.path.exists(p_rp_old):
            p_rp = p_rp_old
        p_rc = os.path.join(args.scores_dir, f"relp_nodes_c4_s{n}.pkl")

    # Load all 4
    eap_pile = load_scores(p_ep)
    eap_c4 = load_scores(p_ec)
    relp_pile = load_scores(p_rp)
    relp_c4 = load_scores(p_rc)

    # ── Layer-level vectors ──
    vecs_layer = {
        "EAP-IG pile10k": layer_importance_to_vector(eap_pile["layer_importance"]),
        "EAP-IG C4":      layer_importance_to_vector(eap_c4["layer_importance"]),
        "RelP pile10k":    layer_importance_to_vector(relp_pile["layer_importance"]),
        "RelP C4":         layer_importance_to_vector(relp_c4["layer_importance"]),
    }

    # ── Node-level vectors (EAP-IG has per-head, RelP has per-layer attn+mlp) ──
    vecs_eap_nodes = {
        "EAP-IG pile10k": eap_components_to_vector(eap_pile["components"]),
        "EAP-IG C4":      eap_components_to_vector(eap_c4["components"]),
    }
    vecs_relp_nodes = {
        "RelP pile10k": relp_subscores_to_vector(relp_pile["sub_scores"]),
        "RelP C4":      relp_subscores_to_vector(relp_c4["sub_scores"]),
    }

    # ── Pairwise comparisons ──
    results = {"layer_level": {}, "node_level": {}, "top_nodes": {}}

    # Layer-level: all 6 pairs
    names = list(vecs_layer.keys())
    print(f"\n  Layer-level cosine similarity ({N_LAYERS} scores):")
    print(f"  {'':30} ", end="")
    for n2 in names:
        print(f"{n2:>18}", end="")
    print()

    for i, n1 in enumerate(names):
        print(f"  {n1:30}", end="")
        for j, n2 in enumerate(names):
            cs = cosine_sim(vecs_layer[n1], vecs_layer[n2])
            results["layer_level"][f"{n1} vs {n2}"] = {
                "cosine": cs,
                "spearman": spearman_corr(vecs_layer[n1], vecs_layer[n2])[0],
            }
            print(f"{'':>4}{cs:>10.4f}    ", end="")
        print()

    # EAP-IG node-level: pile vs C4
    cs = cosine_sim(vecs_eap_nodes["EAP-IG pile10k"], vecs_eap_nodes["EAP-IG C4"])
    sp, _ = spearman_corr(vecs_eap_nodes["EAP-IG pile10k"], vecs_eap_nodes["EAP-IG C4"])
    tk = top_k_overlap(vecs_eap_nodes["EAP-IG pile10k"], vecs_eap_nodes["EAP-IG C4"], 0.2)
    results["node_level"]["EAP-IG pile vs C4"] = {"cosine": cs, "spearman": sp, "top20_overlap": tk}
    print(f"\n  EAP-IG node-level (pile vs C4): cosine={cs:.4f}, spearman={sp:.4f}, top-20% overlap={tk:.2%}")

    # RelP node-level: pile vs C4
    cs = cosine_sim(vecs_relp_nodes["RelP pile10k"], vecs_relp_nodes["RelP C4"])
    sp, _ = spearman_corr(vecs_relp_nodes["RelP pile10k"], vecs_relp_nodes["RelP C4"])
    tk = top_k_overlap(vecs_relp_nodes["RelP pile10k"], vecs_relp_nodes["RelP C4"], 0.2)
    results["node_level"]["RelP pile vs C4"] = {"cosine": cs, "spearman": sp, "top20_overlap": tk}
    print(f"  RelP node-level (pile vs C4):   cosine={cs:.4f}, spearman={sp:.4f}, top-20% overlap={tk:.2%}")

    # ── Top-scored nodes per method ──
    for method_name, components in [("EAP-IG pile10k", eap_pile["components"]),
                                      ("EAP-IG C4", eap_c4["components"])]:
        sorted_nodes = sorted(components.items(),
                               key=lambda x: abs(x[1]["raw_score"]), reverse=True)
        top20 = sorted_nodes[:int(len(sorted_nodes) * 0.2)]
        results["top_nodes"][method_name] = [name for name, _ in top20]
        print(f"\n  {method_name} top-20% nodes ({len(top20)}):")
        for name, info in top20[:10]:
            print(f"    {name:12} layer={info['layer']:>2} score={info['raw_score']:>10.4f}")
        if len(top20) > 10:
            print(f"    ... and {len(top20)-10} more")

    # ── Plot ──
    _plot_layer_heatmap(vecs_layer, args)
    _plot_layer_profiles(vecs_layer, args)

    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 3: Data efficiency
# ═══════════════════════════════════════════════════════════════

def data_efficiency(args):
    print("\n" + "=" * 60)
    print("  PHASE 3: Data efficiency — convergence of circuit scores")
    print("=" * 60)

    os.makedirs(args.scores_dir, exist_ok=True)
    full_n = args.n_samples

    # Reference: full-sample scores
    ref_eap_path = os.path.join(args.scores_dir, f"eapig_nodes_pile10k_s{full_n}.pkl")
    ref_relp_path = os.path.join(args.scores_dir, f"relp_nodes_pile10k_s{full_n}.pkl")
    ref_relp_old = os.path.join(args.scores_dir, "relp_nodes_pile10k_s64.pkl")

    if not os.path.exists(ref_eap_path):
        print(f"  ERROR: Need {ref_eap_path}. Run --only score first.")
        return None

    ref_eap = load_scores(ref_eap_path)
    ref_eap_vec = eap_components_to_vector(ref_eap["components"])
    ref_eap_layer = layer_importance_to_vector(ref_eap["layer_importance"])

    if os.path.exists(ref_relp_path):
        ref_relp = load_scores(ref_relp_path)
    elif os.path.exists(ref_relp_old):
        ref_relp = load_scores(ref_relp_old)
    else:
        print(f"  WARNING: No RelP reference. Skipping RelP efficiency.")
        ref_relp = None

    if ref_relp:
        ref_relp_vec = relp_subscores_to_vector(ref_relp["sub_scores"])
        ref_relp_layer = layer_importance_to_vector(ref_relp["layer_importance"])

    # ── EAP-IG at various sample counts ──
    eap_sample_counts = [8, 16, 32, 64, 128]
    eap_results = {}

    texts_pile = load_texts("pile10k", max(eap_sample_counts))

    for n_s in eap_sample_counts:
        if n_s >= full_n:
            eap_results[n_s] = {"cosine_node": 1.0, "cosine_layer": 1.0,
                                "spearman_node": 1.0, "top20_overlap": 1.0}
            continue

        path = os.path.join(args.scores_dir, f"eapig_nodes_pile10k_s{n_s}.pkl")
        if os.path.exists(path):
            data = load_scores(path)
        else:
            print(f"\n  EAP-IG with N={n_s}...")
            tl_model = load_tl_model()
            raw, components, layer_imp = run_eap_ig_nodes_on_texts(
                tl_model, texts_pile[:n_s])
            data = {"raw": raw.cpu(), "components": components,
                    "layer_importance": layer_imp}
            save_scores(data, path)
            del tl_model
            free_memory()

        vec = eap_components_to_vector(data["components"])
        lvec = layer_importance_to_vector(data["layer_importance"])
        eap_results[n_s] = {
            "cosine_node": cosine_sim(vec, ref_eap_vec),
            "cosine_layer": cosine_sim(lvec, ref_eap_layer),
            "spearman_node": spearman_corr(vec, ref_eap_vec)[0],
            "top20_overlap": top_k_overlap(vec, ref_eap_vec, 0.2),
        }
        print(f"  EAP-IG N={n_s:>3}: cos_node={eap_results[n_s]['cosine_node']:.4f} "
              f"cos_layer={eap_results[n_s]['cosine_layer']:.4f} "
              f"top20={eap_results[n_s]['top20_overlap']:.2%}")

    # ── RelP at various sample counts ──
    relp_sample_counts = [8, 16, 32, 64]
    relp_results = {}

    if ref_relp:
        for n_s in relp_sample_counts:
            path = os.path.join(args.scores_dir, f"relp_nodes_pile10k_s{n_s}.pkl")
            if os.path.exists(path):
                data = load_scores(path)
            else:
                print(f"\n  RelP with N={n_s}...")
                tl_model = load_tl_model()
                tl_model.cfg.use_attn_result = False
                tl_model.cfg.use_hook_mlp_in = False
                tl_model.cfg.use_attn_in = False
                texts = load_texts("pile10k", n_s)
                sub, limp = run_relp_on_texts(tl_model, texts, n_s, MAX_SEQ_LEN)
                data = {"sub_scores": sub, "layer_importance": limp}
                save_scores(data, path)
                del tl_model
                free_memory()

            vec = relp_subscores_to_vector(data["sub_scores"])
            lvec = layer_importance_to_vector(data["layer_importance"])
            relp_results[n_s] = {
                "cosine_node": cosine_sim(vec, ref_relp_vec),
                "cosine_layer": cosine_sim(lvec, ref_relp_layer),
                "spearman_node": spearman_corr(vec, ref_relp_vec)[0],
                "top20_overlap": top_k_overlap(vec, ref_relp_vec, 0.2),
            }
            print(f"  RelP  N={n_s:>3}: cos_node={relp_results[n_s]['cosine_node']:.4f} "
                  f"cos_layer={relp_results[n_s]['cosine_layer']:.4f} "
                  f"top20={relp_results[n_s]['top20_overlap']:.2%}")

    _plot_data_efficiency(eap_results, relp_results, args)

    return {"eap_ig": eap_results, "relp": relp_results}


# ═══════════════════════════════════════════════════════════════
#  Phase 4: PPL validation — does fewer samples hurt pruning?
# ═══════════════════════════════════════════════════════════════

def ppl_check(args):
    """
    Run Exp 3 (EAP-IG weight + Wanda, alpha=500) at 50% sparsity
    using weight-level scores computed from different sample counts.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: PPL validation — data efficiency downstream")
    print("=" * 60)

    from attribution_weights import eap_ig_all_blocks
    from corruption import shuffle_tokens
    from pruning import prune_model
    from sparsity import uniform_sparsity
    from data import get_calibration_loader

    device = torch.device(args.device)
    tokenizer = load_tokenizer()

    sample_counts = [16, 32, 64, 128]
    results = {}

    for n_s in sample_counts:
        # Check for cached weight-level scores
        path = os.path.join(args.scores_dir, f"eap_ig_weights_pile10k_l2_s{n_s}.pkl")
        if os.path.exists(path):
            eap_scores = load_scores(path)
        else:
            print(f"\n  Computing EAP-IG weight scores with N={n_s}...")
            model = load_model()
            dataloader = get_calibration_loader(
                "pile10k", n_s, seed=0, seqlen=MAX_SEQ_LEN, tokenizer=tokenizer)
            eap_scores = eap_ig_all_blocks(
                model, dataloader, shuffle_tokens, device,
                n_steps=10, metric="l2")
            save_scores(eap_scores, path)
            del model
            free_memory()

        # Prune at 50% with these scores
        print(f"\n  Pruning at 50% with N={n_s} EAP-IG scores (α=500)...")
        model = load_model()
        sp_map = uniform_sparsity(model, 0.5)
        prune_model(
            model, tokenizer, sp_map,
            scoring_method="wandapp_eap",
            eap_ig_scores=eap_scores,
            alpha=500.0,
            nsamples=128, device=device, verbose=False,
        )
        ppl = eval_perplexity_wikitext2(model, tokenizer, device)
        results[n_s] = ppl
        print(f"  N={n_s:>3} → PPL = {ppl:.2f}")

        del model
        free_memory()

    # Baseline: Wanda at 50% (no EAP-IG)
    print(f"\n  Wanda baseline at 50%...")
    model = load_model()
    sp_map = uniform_sparsity(model, 0.5)
    prune_model(model, tokenizer, sp_map, scoring_method="wanda",
                nsamples=128, device=device, verbose=False)
    ppl_wanda = eval_perplexity_wikitext2(model, tokenizer, device)
    results["wanda_baseline"] = ppl_wanda
    print(f"  Wanda baseline → PPL = {ppl_wanda:.2f}")
    del model
    free_memory()

    print(f"\n  Summary:")
    print(f"  {'N samples':<12} {'PPL at 50%':>12}")
    print(f"  {'-'*26}")
    for k in sorted(k for k in results if isinstance(k, int)):
        print(f"  {k:<12} {results[k]:>12.2f}")
    print(f"  {'Wanda':<12} {results['wanda_baseline']:>12.2f}")

    _plot_ppl_efficiency(results, args)

    return results


# ═══════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════

def _plot_layer_heatmap(vecs_layer, args):
    """Cosine similarity matrix between all 4 methods at layer level."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.plots_dir, exist_ok=True)
    names = list(vecs_layer.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_sim(vecs_layer[names[i]], vecs_layer[names[j]])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold')
    plt.colorbar(im, label='Cosine Similarity')
    ax.set_title('Layer-Level Circuit Score Agreement\n(4 methods × 2 datasets)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.plots_dir, 'cosine_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_layer_profiles(vecs_layer, args):
    """Per-layer importance profile for all 4 methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for i, (name, vec) in enumerate(vecs_layer.items()):
        # Normalize to [0,1] for comparison
        v = vec / (vec.max() + 1e-12)
        ax.plot(range(N_LAYERS), v, label=name, color=colors[i],
                linewidth=2, alpha=0.8, marker='o', markersize=4)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Normalized Importance', fontsize=12)
    ax.set_title('Layer Importance Profile — 4 Methods', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, N_LAYERS, 2))
    plt.tight_layout()
    path = os.path.join(args.plots_dir, 'layer_profiles.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_data_efficiency(eap_results, relp_results, args):
    """Convergence curves for both methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cosine similarity convergence
    ax = axes[0]
    if eap_results:
        ns = sorted(eap_results.keys())
        cos_vals = [eap_results[n]["cosine_node"] for n in ns]
        ax.plot(ns, cos_vals, 'o-', label='EAP-IG node', color='#e74c3c', linewidth=2)
        cos_layer = [eap_results[n]["cosine_layer"] for n in ns]
        ax.plot(ns, cos_layer, 's--', label='EAP-IG layer', color='#c0392b', linewidth=2)
    if relp_results:
        ns = sorted(relp_results.keys())
        cos_vals = [relp_results[n]["cosine_node"] for n in ns]
        ax.plot(ns, cos_vals, 'o-', label='RelP node', color='#3498db', linewidth=2)
        cos_layer = [relp_results[n]["cosine_layer"] for n in ns]
        ax.plot(ns, cos_layer, 's--', label='RelP layer', color='#2980b9', linewidth=2)
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Cosine Similarity to Full Run', fontsize=12)
    ax.set_title('Score Convergence', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    # Top-20% overlap convergence
    ax = axes[1]
    if eap_results:
        ns = sorted(eap_results.keys())
        overlap = [eap_results[n]["top20_overlap"] for n in ns]
        ax.plot(ns, overlap, 'o-', label='EAP-IG', color='#e74c3c', linewidth=2)
    if relp_results:
        ns = sorted(relp_results.keys())
        overlap = [relp_results[n]["top20_overlap"] for n in ns]
        ax.plot(ns, overlap, 'o-', label='RelP', color='#3498db', linewidth=2)
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Top-20% Node Overlap', fontsize=12)
    ax.set_title('Top Node Stability', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    path = os.path.join(args.plots_dir, 'data_efficiency.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_ppl_efficiency(results, args):
    """PPL vs number of attribution samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    int_keys = sorted(k for k in results if isinstance(k, int))
    ns = int_keys
    ppls = [results[k] for k in ns]

    ax.plot(ns, ppls, 'o-', color='#e74c3c', linewidth=2, markersize=8,
            label='EAP-IG + Wanda (α=500)')
    if "wanda_baseline" in results:
        ax.axhline(y=results["wanda_baseline"], color='#7f8c8d', linestyle='--',
                    linewidth=2, label=f'Wanda baseline ({results["wanda_baseline"]:.2f})')
    ax.set_xlabel('Attribution Samples', fontsize=12)
    ax.set_ylabel('Perplexity at 50% Sparsity', fontsize=12)
    ax.set_title('Data Efficiency: PPL vs Attribution Sample Count', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.plots_dir, 'ppl_efficiency.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    all_results = {}

    if args.only is None or args.only == "score":
        paths = compute_all_scores(args)
    else:
        paths = None

    if args.only is None or args.only == "compare":
        res = compare_scores(args, paths)
        all_results["comparison"] = res

    if args.only is None or args.only == "data_efficiency":
        res = data_efficiency(args)
        all_results["data_efficiency"] = res

    if args.only is None or args.only == "ppl_check":
        res = ppl_check(args)
        all_results["ppl_check"] = res

    out_path = os.path.join(args.results_dir, "scoring_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")
    print("Done!")