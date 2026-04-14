# attribution_weights.py — EAP-IG at the weight level
#
# SCORING ONLY — no pruning happens here.
# Computes per-weight importance scores on the UNPRUNED model,
# block by block. Scores are saved and used later by pruning.py.

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

from config import LINEAR_LAYERS


# ═══════════════════════════════════════════════════════════════
#  Core: EAP-IG weight scores for a single decoder block
# ═══════════════════════════════════════════════════════════════

def _get_linear_layers(block):
    """Return dict of name -> nn.Linear for all 7 matrices in a decoder block."""
    return {
        "self_attn.q_proj": block.self_attn.q_proj,
        "self_attn.k_proj": block.self_attn.k_proj,
        "self_attn.v_proj": block.self_attn.v_proj,
        "self_attn.o_proj": block.self_attn.o_proj,
        "mlp.gate_proj":    block.mlp.gate_proj,
        "mlp.up_proj":      block.mlp.up_proj,
        "mlp.down_proj":    block.mlp.down_proj,
    }


def _zero_weight_grads(linears):
    """Zero all weight gradients."""
    for layer in linears.values():
        if layer.weight.grad is not None:
            layer.weight.grad.zero_()


def eap_ig_block_scores(block, clean_inps, corrupt_inps,
                         attention_mask, position_ids,
                         n_steps=10, metric="l2",
                         position_embeddings=None):
    """
    Compute per-weight EAP-IG scores for one UNPRUNED decoder block.

    For each sample and each interpolation step alpha:
      1. inp_a = corrupt + alpha * (clean - corrupt)
      2. Forward through block
      3. Backward (CE against clean output, or L2 norm)
      4. Accumulate |weight.grad|

    Average across samples x steps -> integrated gradient per weight.

    Args:
        block:            one LlamaDecoderLayer (UNPRUNED)
        clean_inps:       list of tensors, each (1, seqlen, d_model) on CPU
        corrupt_inps:     list of tensors, same shape, on CPU
        attention_mask:   attention mask tensor
        position_ids:     position ids tensor
        n_steps:          number of interpolation steps
        metric:           'ce', 'l2', or 'both'
        position_embeddings: precomputed RoPE tuple or None

    Returns:
        dict: layer_name -> (d_out, d_in) tensor of per-weight scores
        If metric='both': (ce_scores, l2_scores)
    """
    device = next(block.parameters()).device
    linears = _get_linear_layers(block)
    loss_fn = nn.CrossEntropyLoss()

    def _run_ig(use_ce):
        ig_accum = {name: torch.zeros_like(layer.weight, dtype=torch.float32)
                    for name, layer in linears.items()}
        n_total = 0

        for s_idx in tqdm(range(len(clean_inps)), desc="  IG samples", leave=False):
            clean = clean_inps[s_idx].to(device)
            corrupt = corrupt_inps[s_idx].to(device)

            # CE target: what the unpruned block produces on clean input
            if use_ce:
                with torch.no_grad():
                    fwd_kw = _make_fwd_kwargs(
                        attention_mask, position_ids, position_embeddings, device)
                    clean_out = block(clean, **fwd_kw)
                    if isinstance(clean_out, tuple):
                        clean_out = clean_out[0]
                    target = clean_out.detach()

            for step in range(n_steps):
                alpha = (step + 0.5) / n_steps  # midpoint rule
                inp_alpha = corrupt + alpha * (clean - corrupt)
                inp_alpha = inp_alpha.detach().requires_grad_(True)

                _zero_weight_grads(linears)
                for layer in linears.values():
                    layer.weight.requires_grad_(True)

                fwd_kw = _make_fwd_kwargs(
                    attention_mask, position_ids, position_embeddings, device)
                out = block(inp_alpha, **fwd_kw)
                if isinstance(out, tuple):
                    out = out[0]

                if use_ce:
                    loss = loss_fn(
                        out.reshape(-1, out.size(-1)),
                        target.reshape(-1, target.size(-1)).argmax(dim=-1),
                    )
                else:
                    loss = torch.norm(out)

                loss.backward()

                with torch.no_grad():
                    for name, layer in linears.items():
                        if layer.weight.grad is not None:
                            ig_accum[name] += layer.weight.grad.float().abs()

                n_total += 1
                del out, loss, inp_alpha

            del clean, corrupt
            if s_idx % 16 == 0:
                torch.cuda.empty_cache()

        # Average and detach weights from grad tracking
        for name in ig_accum:
            ig_accum[name] /= max(n_total, 1)
            linears[name].weight.requires_grad_(False)

        return ig_accum

    if metric == "ce":
        return _run_ig(use_ce=True)
    elif metric == "l2":
        return _run_ig(use_ce=False)
    elif metric == "both":
        ce_scores = _run_ig(use_ce=True)
        l2_scores = _run_ig(use_ce=False)
        return ce_scores, l2_scores
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ═══════════════════════════════════════════════════════════════
#  Prepare block-0 inputs (clean and corrupt separately)
# ═══════════════════════════════════════════════════════════════

def prepare_block_inputs(model, dataloader, device):
    """
    Run forward pass with clean data, catch hidden states at block 0.
    Model is NOT modified.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    inps = []
    cache = {"attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach().cpu())
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    model.config.use_cache = use_cache
    return inps, cache["attention_mask"], cache["position_ids"]


def prepare_corrupt_block_inputs(model, clean_dataloader, corruption_fn, device):
    """
    Same as prepare_block_inputs but applies corruption_fn to tokens first.
    Model is NOT modified.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    inps = []
    cache = {"attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach().cpu())
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in clean_dataloader:
        try:
            corrupted_ids = corruption_fn(batch[0])
            model(corrupted_ids.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    model.config.use_cache = use_cache
    return inps


# ═══════════════════════════════════════════════════════════════
#  Forward through a single UNPRUNED block
# ═══════════════════════════════════════════════════════════════

def forward_block_unpruned(block, inps, attention_mask, position_ids,
                            position_embeddings=None, output_device="cpu"):
    """
    Forward all samples through one UNPRUNED block.
    Used to propagate clean/corrupt hidden states to the next block.
    No weights are modified.
    """
    device = next(block.parameters()).device
    outs = []
    fwd_kw = _make_fwd_kwargs(
        attention_mask, position_ids, position_embeddings, device)

    with torch.no_grad():
        for inp in inps:
            out = block(inp.to(device), **fwd_kw)
            if isinstance(out, tuple):
                out = out[0]
            outs.append(out.detach().to(output_device))
    return outs


# ═══════════════════════════════════════════════════════════════
#  Full model: score all blocks on the UNPRUNED model
# ═══════════════════════════════════════════════════════════════

def eap_ig_all_blocks(model, clean_dataloader, corruption_fn, device,
                       n_steps=10, metric="l2"):
    """
    Run EAP-IG weight-level scoring across ALL decoder blocks.
    Model is NEVER modified — this is pure scoring.

    Flow:
      1. Catch block-0 inputs for clean and corrupt tokens
      2. For each block (unpruned):
         a. Compute EAP-IG weight scores
         b. Forward clean + corrupt through UNPRUNED block -> next inputs
      3. Return all per-weight scores

    Args:
        model:           HF LlamaForCausalLM (untouched throughout)
        clean_dataloader: list of (input_ids, targets)
        corruption_fn:   callable(token_ids) -> corrupt_token_ids
        device:          torch.device
        n_steps:         IG interpolation steps
        metric:          'ce', 'l2', or 'both'

    Returns:
        all_scores: dict of (layer_idx, matrix_name) -> score tensor on CPU
    """
    print("Preparing clean block-0 inputs...")
    clean_inps, attention_mask, position_ids = prepare_block_inputs(
        model, clean_dataloader, device)

    print("Preparing corrupt block-0 inputs...")
    corrupt_inps = prepare_corrupt_block_inputs(
        model, clean_dataloader, corruption_fn, device)

    print(f"Got {len(clean_inps)} clean, {len(corrupt_inps)} corrupt samples")
    assert len(clean_inps) == len(corrupt_inps), "Clean/corrupt sample count mismatch"

    # Pre-compute RoPE if needed
    position_embeddings = _get_position_embeddings(
        model, clean_inps, position_ids, device)

    layers = model.model.layers
    all_scores = {}

    for layer_idx in range(len(layers)):
        print(f"\n== Block {layer_idx}/{len(layers)-1} ==")
        block = layers[layer_idx]

        # Score this unpruned block
        scores = eap_ig_block_scores(
            block, clean_inps, corrupt_inps,
            attention_mask, position_ids,
            n_steps=n_steps, metric=metric,
            position_embeddings=position_embeddings,
        )

        # Store on CPU
        if isinstance(scores, tuple):
            ce_s, l2_s = scores
            for name, tensor in ce_s.items():
                all_scores[(layer_idx, name, "ce")] = tensor.cpu()
            for name, tensor in l2_s.items():
                all_scores[(layer_idx, name, "l2")] = tensor.cpu()
        else:
            for name, tensor in scores.items():
                all_scores[(layer_idx, name)] = tensor.cpu()

        # Forward BOTH through UNPRUNED block for next block's inputs
        print(f"  Forwarding through unpruned block {layer_idx}...")
        clean_inps = forward_block_unpruned(
            block, clean_inps, attention_mask, position_ids,
            position_embeddings=position_embeddings)
        corrupt_inps = forward_block_unpruned(
            block, corrupt_inps, attention_mask, position_ids,
            position_embeddings=position_embeddings)

        torch.cuda.empty_cache()

    print(f"\nScored {len(all_scores)} weight matrices total.")
    return all_scores


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _make_fwd_kwargs(attention_mask, position_ids, position_embeddings, device):
    """Build kwargs dict for block.forward()."""
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


def _get_position_embeddings(model, sample_inps, position_ids, device):
    """Pre-compute RoPE if the model has rotary_emb."""
    if not hasattr(model.model, "rotary_emb"):
        return None
    try:
        with torch.no_grad():
            dummy = sample_inps[0].to(device)
            pos_ids = torch.arange(dummy.size(1), device=device).unsqueeze(0)
            pos_emb = model.model.rotary_emb(dummy, pos_ids)
            return (pos_emb[0].cpu(), pos_emb[1].cpu())
    except Exception:
        return None
