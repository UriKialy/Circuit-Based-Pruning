# pruning.py — weight pruning with multiple scoring methods
#
# Three modes:
#   1. 'wanda'   — |W| * ||X||  (baseline, from papers)
#   2. 'eap_ig'  — EAP-IG integrated gradient per weight
#   3. 'wandapp_eap' — Wanda++ RGS replaced with EAP-IG
#
# All support non-uniform sparsity via sparsity_map.

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

from config import LINEAR_LAYERS


# ═══════════════════════════════════════════════════════════════
#  Helpers (taken from Wanda/Wanda++)
# ═══════════════════════════════════════════════════════════════

def find_layers(module, layers_type=[nn.Linear], name=""):
    """Recursively find nn.Linear layers."""
    if type(module) in layers_type:
        return {name: module}
    res = {}
    for n, child in module.named_children():
        full = name + "." + n if name else n
        res.update(find_layers(child, layers_type, full))
    return res


class WrappedGPT:
    """Wrapper to accumulate input statistics for a linear layer (from Wanda)."""
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.weight.shape[0]
        self.cols = layer.weight.shape[1]
        self.scaler_row = torch.zeros(self.cols, device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t().float()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2
        self.nsamples += batch

    def get_input_norm(self):
        return torch.sqrt(self.scaler_row / self.nsamples)


# ═══════════════════════════════════════════════════════════════
#  Core pruning: apply mask to one linear layer
# ═══════════════════════════════════════════════════════════════

def prune_layer_by_score(weight, score, sparsity_ratio):
    """
    Zero out the bottom-k weights by score.

    Args:
        weight:         nn.Parameter (d_out, d_in)
        score:          (d_out, d_in) importance tensor
        sparsity_ratio: fraction to remove

    Returns:
        mask: (d_out, d_in) boolean — True = pruned
    """
    if sparsity_ratio <= 0:
        return torch.zeros_like(weight, dtype=torch.bool)

    flat_score = score.flatten()
    k = int(flat_score.numel() * sparsity_ratio)
    if k == 0:
        return torch.zeros_like(weight, dtype=torch.bool)

    threshold = torch.topk(flat_score, k, largest=False).values[-1]
    mask = score <= threshold
    weight.data[mask] = 0
    return mask


# ═══════════════════════════════════════════════════════════════
#  Scoring method 1: Wanda — |W| * ||X||
# ═══════════════════════════════════════════════════════════════

def wanda_score(weight, input_norm):
    """Standard Wanda: |W_ij| * ||X_j||_2."""
    return weight.abs() * input_norm.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
#  Scoring method 2: EAP-IG weight score (pre-computed)
# ═══════════════════════════════════════════════════════════════

def eap_ig_score(weight, ig_tensor):
    """
    Use pre-computed EAP-IG integrated gradient as the score.
    ig_tensor is already |averaged gradient| from attribution_weights.py.
    """
    return ig_tensor


def eap_ig_combined_score(weight, input_norm, ig_tensor, alpha=1.0):
    """
    Combine Wanda and EAP-IG: |W| * (||X|| + α * IG).
    Analogous to Wanda++ RGS formula: (α*G + ||X||) * |W|.
    """
    return weight.abs() * (input_norm.unsqueeze(0) + alpha * ig_tensor)


# ═══════════════════════════════════════════════════════════════
#  Scoring method 3: Wanda++ with EAP-IG replacing RGS
# ═══════════════════════════════════════════════════════════════

def wandapp_eap_score(weight, input_norm, ig_tensor, alpha=100.0):
    """
    Wanda++ formula but with EAP-IG instead of regional gradient.
    Score = (α * IG_ij + ||X_j||) * |W_ij|

    This is the direct replacement: wherever Wanda++ uses
    sqrt(Σ(∂L_RGS/∂W)² / N), we use the EAP-IG integrated gradient.
    """
    return (alpha * ig_tensor + input_norm.unsqueeze(0)) * weight.abs()


# ═══════════════════════════════════════════════════════════════
#  Main pruning loop
# ═══════════════════════════════════════════════════════════════

def prune_model(model, tokenizer, sparsity_map, scoring_method="wanda",
                eap_ig_scores=None, alpha=100.0,
                nsamples=128, seed=0, device=torch.device("cuda:0"),
                use_ro=False, ro_fn=None, verbose=True):
    """
    Prune the model using the given scoring method and sparsity map.

    Args:
        model:           HF LlamaForCausalLM
        tokenizer:       tokenizer
        sparsity_map:    dict (layer_idx, matrix_name) → sparsity ratio
        scoring_method:  'wanda', 'eap_ig', 'eap_ig_combined', 'wandapp_eap'
        eap_ig_scores:   dict (layer_idx, matrix_name) → weight-level score tensor
                         (required for eap_ig / eap_ig_combined / wandapp_eap)
        alpha:           scaling factor for gradient term
        nsamples:        calibration samples
        seed:            random seed
        device:          compute device
        use_ro:          whether to run Regional Optimization after pruning
        ro_fn:           callable for RO (from regional_optimizer.py)
        verbose:         print progress
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Prepare calibration inputs (same as Wanda)
    from data import get_calibration_loader
    dataloader = get_calibration_loader(
        "pile10k", nsamples, seed, model.seqlen, tokenizer)

    if verbose:
        print(f"Pruning with scoring={scoring_method}, α={alpha}")

    inps, outs, attention_mask, position_ids = _prepare_inputs(
        model, dataloader, device)

    # Pre-compute position embeddings if needed
    position_embeddings = _get_rope(model, inps, device)

    layers = model.model.layers
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        subset = find_layers(layer)

        # Determine layer device
        layer_dev = device
        if hasattr(model, "hf_device_map"):
            key = f"model.layers.{layer_idx}"
            if key in model.hf_device_map:
                layer_dev = model.hf_device_map[key]
                inps = [x.to(layer_dev) for x in inps] if isinstance(inps, list) else inps.to(layer_dev)

        # Collect input statistics (Wanda-style)
        wrapped = {}
        for name in subset:
            wrapped[name] = WrappedGPT(subset[name])

        def make_hook(name):
            def hook_fn(_, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data)
            return hook_fn

        handles = [subset[n].register_forward_hook(make_hook(n)) for n in subset]

        # Forward all calibration samples through this layer
        fwd_kwargs = _make_layer_fwd_kwargs(
            attention_mask, position_ids, position_embeddings, layer_dev)

        for j in range(nsamples):
            inp_j = inps[j] if isinstance(inps, list) else inps[j:j+1]
            if inp_j.dim() == 2:
                inp_j = inp_j.unsqueeze(0)
            with torch.no_grad():
                out_j = layer(inp_j.to(layer_dev), **fwd_kwargs)[0]
                if isinstance(inps, list):
                    pass  # outs handled below
                else:
                    outs[j] = out_j

        for h in handles:
            h.remove()

        # ── Score and prune each matrix ──
        for name in subset:
            matrix_key = _subset_name_to_config_name(name)
            sparsity = sparsity_map.get((layer_idx, matrix_key), 0.0)

            if sparsity <= 0:
                continue

            W = subset[name].weight
            input_norm = wrapped[name].get_input_norm()

            if scoring_method == "wanda":
                score = wanda_score(W.data, input_norm)

            elif scoring_method == "eap_ig":
                ig = eap_ig_scores[(layer_idx, matrix_key)].to(W.device)
                score = eap_ig_score(W.data, ig)

            elif scoring_method == "eap_ig_combined":
                ig = eap_ig_scores[(layer_idx, matrix_key)].to(W.device)
                score = eap_ig_combined_score(W.data, input_norm, ig, alpha)

            elif scoring_method == "wandapp_eap":
                ig = eap_ig_scores[(layer_idx, matrix_key)].to(W.device)
                score = wandapp_eap_score(W.data, input_norm, ig, alpha)

            else:
                raise ValueError(f"Unknown scoring method: {scoring_method}")

            prune_layer_by_score(W, score, sparsity)

        if verbose:
            # Count sparsity for this layer
            total_z, total_p = 0, 0
            for n in subset:
                w = subset[n].weight.data
                total_z += (w == 0).sum().item()
                total_p += w.numel()
            print(f"  layer {layer_idx}: sparsity {total_z/total_p:.4f}")

        # ── Optional: Regional Optimization ──
        if use_ro and ro_fn is not None:
            # RO is available but not called by default
            ro_fn(layer, inps, attention_mask, position_ids, position_embeddings)

        # Forward through pruned layer for next layer's inputs
        for j in range(nsamples):
            inp_j = inps[j] if isinstance(inps, list) else inps[j:j+1]
            if inp_j.dim() == 2:
                inp_j = inp_j.unsqueeze(0)
            with torch.no_grad():
                out_j = layer(inp_j.to(layer_dev), **fwd_kwargs)[0]
                if isinstance(inps, list):
                    inps[j] = out_j.detach().cpu()
                else:
                    outs[j] = out_j

        if not isinstance(inps, list):
            inps, outs = outs, inps

        torch.cuda.empty_cache()

    model.config.use_cache = use_cache


# ═══════════════════════════════════════════════════════════════
#  Protection-aware pruning (exp 4a / 4b)
# ═══════════════════════════════════════════════════════════════

def prune_with_protection(model, tokenizer, sparsity_map,
                           protect_scores, protect_pct_map,
                           nsamples=128, seed=0,
                           device=torch.device("cuda:0"),
                           safety_margin=0.02, verbose=True):
    """
    Prune with per-weight protection.

    Flow per matrix:
      1. Use protect_scores to rank weights; top protect_pct% are PROTECTED.
      2. Compute Wanda score |W|*||X|| on calibration data for ALL weights.
      3. Prune bottom weights among UNPROTECTED to hit matrix sparsity target.
      4. If target > (1 - protect_pct), cap at (1 - protect_pct - margin).

    Args:
        protect_scores:   dict (layer, matrix) -> (d_out, d_in) tensor
                          (EAP-IG for 4a, RGS for 4b, etc.)
        protect_pct_map:  dict (layer, matrix) -> float in [0,1]
        sparsity_map:     dict (layer, matrix) -> target sparsity
        safety_margin:    keep this much unprotected+unpruned space
    """
    from protection import clamp_protection_vs_sparsity

    use_cache = model.config.use_cache
    model.config.use_cache = False

    from data import get_calibration_loader
    dataloader = get_calibration_loader(
        "pile10k", nsamples, seed, model.seqlen, tokenizer)

    if verbose:
        print(f"Pruning with protection: scores + Wanda on unprotected")

    inps, outs, attention_mask, position_ids = _prepare_inputs(
        model, dataloader, device)
    position_embeddings = _get_rope(model, inps, device)

    layers = model.model.layers
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        subset = find_layers(layer)

        layer_dev = device
        if hasattr(model, "hf_device_map"):
            key = f"model.layers.{layer_idx}"
            if key in model.hf_device_map:
                layer_dev = model.hf_device_map[key]

        # Collect input stats (Wanda)
        wrapped = {name: WrappedGPT(subset[name]) for name in subset}

        def make_hook(name):
            def hook_fn(_, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data)
            return hook_fn

        handles = [subset[n].register_forward_hook(make_hook(n)) for n in subset]

        fwd_kwargs = _make_layer_fwd_kwargs(
            attention_mask, position_ids, position_embeddings, layer_dev)

        for j in range(nsamples):
            inp_j = inps[j] if isinstance(inps, list) else inps[j:j+1]
            if inp_j.dim() == 2:
                inp_j = inp_j.unsqueeze(0)
            with torch.no_grad():
                out_j = layer(inp_j.to(layer_dev), **fwd_kwargs)[0]
                if not isinstance(inps, list):
                    outs[j] = out_j

        for h in handles:
            h.remove()

        # Prune each matrix with protection
        for name in subset:
            key = (layer_idx, name)
            target_sp = sparsity_map.get(key, 0.0)
            protect_pct = protect_pct_map.get(key, 0.0)

            if target_sp <= 0:
                continue

            # Clamp protection vs sparsity
            protect_pct_eff = clamp_protection_vs_sparsity(
                protect_pct, target_sp, safety_margin)

            W = subset[name].weight
            input_norm = wrapped[name].get_input_norm()

            # Wanda score for ALL weights
            wanda = W.data.abs() * input_norm.unsqueeze(0)

            # Build protection mask from protect_scores
            from protection import build_protection_mask
            p_scores = protect_scores.get(key)
            if p_scores is None or protect_pct_eff <= 0:
                protected_mask = torch.zeros_like(W.data, dtype=torch.bool)
            else:
                protected_mask = build_protection_mask(
                    W.data, p_scores.to(W.device), protect_pct_eff)

            # Set protected weights' Wanda score to +inf so they never get picked
            wanda_for_pruning = wanda.clone()
            wanda_for_pruning[protected_mask] = float('inf')

            # Prune bottom-k by Wanda among unprotected
            prune_layer_by_score(W, wanda_for_pruning, target_sp)

        if verbose:
            tot_z, tot_p = 0, 0
            for n in subset:
                w = subset[n].weight.data
                tot_z += (w == 0).sum().item()
                tot_p += w.numel()
            print(f"  layer {layer_idx}: sparsity {tot_z/tot_p:.4f}")

        # Forward through pruned layer
        for j in range(nsamples):
            inp_j = inps[j] if isinstance(inps, list) else inps[j:j+1]
            if inp_j.dim() == 2:
                inp_j = inp_j.unsqueeze(0)
            with torch.no_grad():
                out_j = layer(inp_j.to(layer_dev), **fwd_kwargs)[0]
                if isinstance(inps, list):
                    inps[j] = out_j.detach().cpu()
                else:
                    outs[j] = out_j

        if not isinstance(inps, list):
            inps, outs = outs, inps

        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════

def _prepare_inputs(model, dataloader, device):
    """Catch block-0 inputs via Catcher. Returns (inps, outs, attn_mask, pos_ids)."""
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size),
        dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    return inps, outs, cache["attention_mask"], cache["position_ids"]


def _make_layer_fwd_kwargs(attention_mask, position_ids, position_embeddings, device):
    """Build kwargs for layer.forward()."""
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.to(device)
    if position_embeddings is not None:
        kwargs["position_embeddings"] = (
            position_embeddings[0].to(device),
            position_embeddings[1].to(device),
        )
    return kwargs


def _get_rope(model, inps, device):
    """Pre-compute RoPE if available."""
    if not hasattr(model.model, "rotary_emb"):
        return None
    try:
        with torch.no_grad():
            pos_ids = torch.arange(model.seqlen, device=device).unsqueeze(0)
            if isinstance(inps, list):
                dummy = inps[0].to(device)
            else:
                dummy = inps[:1].to(device)
            return model.model.rotary_emb(dummy, pos_ids)
    except Exception:
        return None


def _subset_name_to_config_name(subset_name):
    """Map find_layers name like 'self_attn.q_proj' to config LINEAR_LAYERS format."""
    # find_layers returns names like 'self_attn.q_proj', 'mlp.gate_proj'
    return subset_name
