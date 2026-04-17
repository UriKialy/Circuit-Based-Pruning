# pruning.py — pruning using ACTUAL Wanda functions
#
# Imports core functions from wanda-main/lib/ directly.
# Only adds: non-uniform sparsity support + protection + EAP-IG scoring options.

import sys
import os
import torch
import torch.nn as nn

# ── Import from actual Wanda codebase ──
sys.path.insert(0, "/workspace/wanda")
from lib.prune import find_layers, check_sparsity, prepare_calibration_input, prune_wanda
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders
from lib.eval import eval_ppl, eval_ppl_wikitext

from config import LINEAR_LAYERS


# ═══════════════════════════════════════════════════════════════
#  WandaArgs — mimics the argparse namespace Wanda expects
# ═══════════════════════════════════════════════════════════════

class WandaArgs:
    def __init__(self, sparsity_ratio=0.5, nsamples=128, seed=0, use_variant=False):
        self.sparsity_ratio = sparsity_ratio
        self.nsamples = nsamples
        self.seed = seed
        self.use_variant = use_variant
        self.sparsity_type = "unstructured"


# ═══════════════════════════════════════════════════════════════
#  Run vanilla Wanda (exact paper implementation)
# ═══════════════════════════════════════════════════════════════

def run_wanda_uniform(model, tokenizer, sparsity_ratio, nsamples=128, device=torch.device("cuda:0")):
    """
    Run EXACT Wanda from the paper. No modifications.
    Uses C4 calibration, standard scoring, uniform sparsity.
    """
    args = WandaArgs(sparsity_ratio=sparsity_ratio, nsamples=nsamples)
    model.seqlen = getattr(model, 'seqlen', 2048)
    prune_wanda(args, model, tokenizer, device)


# ═══════════════════════════════════════════════════════════════
#  Non-uniform Wanda (circuit-guided sparsity allocation)
# ═══════════════════════════════════════════════════════════════

def prune_wanda_nonuniform(model, tokenizer, sparsity_map,
                            nsamples=128, seed=0,
                            device=torch.device("cuda:0"),
                            use_ro=False, ro_fn=None,
                            verbose=True):
    """
    Wanda pruning with per-matrix sparsity ratios.
    Uses Wanda's actual prepare_calibration_input + WrappedGPT.

    Args:
        sparsity_map: dict (layer_idx, matrix_name) -> sparsity ratio
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.seqlen = getattr(model, 'seqlen', 2048)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=seed,
                                 seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev),
                attention_mask.to(dev), position_ids.to(dev))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                               attention_mask=attention_mask,
                               position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            # Get per-matrix sparsity from the map
            sp = sparsity_map.get((i, name), 0.0)
            if sp <= 0:
                continue

            W_metric = torch.abs(subset[name].weight.data) * \
                       torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :int(W_metric.shape[1] * sp)]
            W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0

        if verbose:
            sub_z = sum((subset[n].weight.data == 0).sum().item() for n in subset)
            sub_t = sum(subset[n].weight.data.numel() for n in subset)
            print(f"  layer {i} sparsity {sub_z/sub_t:.6f}")

        # Optional: Regional Optimization
        if use_ro and ro_fn is not None:
            inps_list = [inps[j].unsqueeze(0).cpu() for j in range(nsamples)]
            ro_fn(layer, inps_list, attention_mask, position_ids)

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                               attention_mask=attention_mask,
                               position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
#  EAP-IG combined scoring (Wanda + EAP-IG + optional protection)
# ═══════════════════════════════════════════════════════════════

def prune_wandapp_eap(model, tokenizer, sparsity_map,
                       eap_ig_scores, alpha=500.0,
                       nsamples=128, seed=0,
                       device=torch.device("cuda:0"),
                       verbose=True):
    """
    Wanda++ formula with EAP-IG replacing RGS:
    Score = (alpha * IG + ||X||) * |W|

    Uses Wanda's actual infrastructure for calibration + input stats.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.seqlen = getattr(model, 'seqlen', 2048)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=seed,
                                 seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev),
                attention_mask.to(dev), position_ids.to(dev))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                               attention_mask=attention_mask,
                               position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            sp = sparsity_map.get((i, name), 0.0)
            if sp <= 0:
                continue

            W = subset[name].weight.data
            input_norm = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Get EAP-IG score for this matrix
            ig = eap_ig_scores.get((i, name))
            if ig is not None:
                ig = ig.to(W.device)
                W_metric = (alpha * ig + input_norm) * W.abs()
            else:
                # Fallback to standard Wanda if no IG score
                W_metric = W.abs() * input_norm

            W_mask = (torch.zeros_like(W_metric) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :int(W_metric.shape[1] * sp)]
            W_mask.scatter_(1, indices, True)
            W[W_mask] = 0

        if verbose:
            sub_z = sum((subset[n].weight.data == 0).sum().item() for n in subset)
            sub_t = sum(subset[n].weight.data.numel() for n in subset)
            print(f"  layer {i} sparsity {sub_z/sub_t:.6f}")

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                               attention_mask=attention_mask,
                               position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
#  Protection-aware pruning
# ═══════════════════════════════════════════════════════════════

def prune_with_protection(model, tokenizer, sparsity_map,
                           protect_scores, protect_pct_map,
                           nsamples=128, seed=0,
                           device=torch.device("cuda:0"),
                           safety_margin=0.02, verbose=True):
    """
    Wanda pruning with per-weight protection.
    Uses Wanda's actual infrastructure.
    """
    from protection import build_protection_mask, clamp_protection_vs_sparsity

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.seqlen = getattr(model, 'seqlen', 2048)

    dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=seed,
                                 seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev),
                attention_mask.to(dev), position_ids.to(dev))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                               attention_mask=attention_mask,
                               position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            key = (i, name)
            target_sp = sparsity_map.get(key, 0.0)
            protect_pct = protect_pct_map.get(key, 0.0)
            if target_sp <= 0:
                continue

            protect_pct_eff = clamp_protection_vs_sparsity(
                protect_pct, target_sp, safety_margin)

            W = subset[name].weight
            W_metric = W.data.abs() * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Build protection mask
            p_scores = protect_scores.get(key)
            if p_scores is not None and protect_pct_eff > 0:
                protected = build_protection_mask(W.data, p_scores.to(W.device), protect_pct_eff)
                W_metric[protected] = float('inf')

            W_mask = (torch.zeros_like(W_metric) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :int(W_metric.shape[1] * target_sp)]
            W_mask.scatter_(1, indices, True)
            W.data[W_mask] = 0

        if verbose:
            sub_z = sum((subset[n].weight.data == 0).sum().item() for n in subset)
            sub_t = sum(subset[n].weight.data.numel() for n in subset)
            print(f"  layer {i} sparsity {sub_z/sub_t:.6f}")

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                               attention_mask=attention_mask,
                               position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()