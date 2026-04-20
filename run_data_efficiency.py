#!/usr/bin/env python3
"""
Data efficiency: how many attribution samples needed?
Tests EAP-IG weight-level (50%) and RelP node-level (70%).
"""

import sys, json, os, torch, gc
sys.path.insert(0, "/workspace/wanda")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from utils import load_model, load_tokenizer, load_scores, save_scores, free_memory
from evaluation import eval_perplexity_wikitext2
from pruning import prune_wanda_nonuniform, prune_wandapp_eap, check_sparsity
from sparsity import allocate_layer_sparsity, uniform_sparsity
from corruption import shuffle_tokens
from attribution_weights import eap_ig_all_blocks
from attribution_nodes import run_relp_nodes, relp_to_layer_importance
from data import get_calibration_loader, load_texts
from config import N_LAYERS, MAX_SEQ_LEN

os.makedirs("results", exist_ok=True)
os.makedirs("scores", exist_ok=True)
os.makedirs("logs", exist_ok=True)

results = {"eapig_weight": {}, "relp_node": {}}

# ═══════════════════════════════════════════════════════════════
#  EAP-IG weight-level at 50% — vary attribution samples
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  EAP-IG weight-level: data efficiency at 50% sparsity")
print("=" * 60)

for n_attr in [8, 16, 32, 64, 128]:
    cache = f"./scores/eap_ig_weights_pile10k_l2_s{n_attr}.pkl"
    
    if os.path.exists(cache):
        print(f"\n  N={n_attr}: loading cached scores...")
        eap_scores = load_scores(cache)
    else:
        print(f"\n  N={n_attr}: computing EAP-IG weight scores...")
        model = load_model()
        tokenizer = load_tokenizer()
        dataloader = get_calibration_loader("pile10k", n_attr, seed=0,
                                             seqlen=MAX_SEQ_LEN, tokenizer=tokenizer)
        eap_scores = eap_ig_all_blocks(model, dataloader, shuffle_tokens,
                                        torch.device("cuda:0"), n_steps=10, metric="l2")
        print(f"  Skipping save (25GB per file)")
        del model
        free_memory()

    # Prune at 50% with alpha=100
    print(f"  N={n_attr}: pruning at 50%...")
    model = load_model()
    tokenizer = load_tokenizer()
    sp_map = uniform_sparsity(model, 0.5)
    prune_wandapp_eap(model, tokenizer, sp_map, eap_ig_scores=eap_scores,
                       alpha=100.0, verbose=False)
    ppl = eval_perplexity_wikitext2(model, tokenizer)
    results["eapig_weight"][n_attr] = round(ppl, 2)
    print(f"  N={n_attr}: PPL = {ppl:.2f}")
    del model, eap_scores
    free_memory()

# ═══════════════════════════════════════════════════════════════
#  RelP node-level at 70% — vary attribution samples
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  RelP node-level: data efficiency at 70% sparsity (T=3)")
print("=" * 60)

for n_attr in [8, 16, 32, 64, 128]:
    cache = f"./scores/relp_nodes_pile10k_s{n_attr}.pkl"
    
    if os.path.exists(cache):
        print(f"\n  N={n_attr}: loading cached scores...")
        relp_data = load_scores(cache)
    else:
        print(f"\n  N={n_attr}: computing RelP scores...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens import HookedTransformer
        hf_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", dtype=torch.float16)
        tl_model = HookedTransformer.from_pretrained("llama-7b-hf", hf_model=hf_model,
                    center_writing_weights=False, center_unembed=False, fold_ln=False)
        tl_model.set_tokenizer(AutoTokenizer.from_pretrained("huggyllama/llama-7b"))
        del hf_model; free_memory()
        
        texts = load_texts("pile10k", n_attr)
        sub_scores = run_relp_nodes(tl_model, texts, num_samples=n_attr, max_seq_len=MAX_SEQ_LEN)
        layer_imp = relp_to_layer_importance(sub_scores, N_LAYERS)
        relp_data = {"sub_scores": sub_scores, "layer_importance": layer_imp}
        save_scores(relp_data, cache)
        del tl_model
        free_memory()

    # Prune at 70% with T=3
    layer_imp = relp_data["layer_importance"]
    print(f"  N={n_attr}: pruning at 70% T=3...")
    model = load_model()
    tokenizer = load_tokenizer()
    sp_map = allocate_layer_sparsity(layer_imp, model, 0.7, temperature=3.0)
    prune_wanda_nonuniform(model, tokenizer, sp_map, verbose=False)
    ppl = eval_perplexity_wikitext2(model, tokenizer)
    results["relp_node"][n_attr] = round(ppl, 2)
    print(f"  N={n_attr}: PPL = {ppl:.2f}")
    del model, relp_data
    free_memory()

# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  DATA EFFICIENCY SUMMARY")
print("=" * 60)
print(f"\n  EAP-IG weight-level at 50% (Wanda baseline: 7.25):")
print(f"  {'N':>6} {'PPL':>8}")
for n, ppl in sorted(results["eapig_weight"].items()):
    marker = " <-- best" if ppl == min(results["eapig_weight"].values()) else ""
    print(f"  {n:>6} {ppl:>8.2f}{marker}")

print(f"\n  RelP node-level at 70% T=3 (Wanda baseline: 76.17):")
print(f"  {'N':>6} {'PPL':>8}")
for n, ppl in sorted(results["relp_node"].items()):
    marker = " <-- best" if ppl == min(results["relp_node"].values()) else ""
    print(f"  {n:>6} {ppl:>8.2f}{marker}")

with open("results/data_efficiency.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/data_efficiency.json")
