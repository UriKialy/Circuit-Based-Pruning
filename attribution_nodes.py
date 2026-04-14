# attribution_nodes.py — node-level circuit discovery
#
# Two methods:
#   1. EAP-IG: uses the EAP library + TransformerLens (existing, from notebook)
#   2. RelP:   uses RelP-modified TransformerLens (existing, from notebook)
#
# Both produce per-layer importance scores for sparsity allocation.

import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import N_LAYERS, N_HEADS, D_MODEL, D_HEAD, D_FF


# ═══════════════════════════════════════════════════════════════
#  EAP-IG node-level (requires TransformerLens + EAP library)
# ═══════════════════════════════════════════════════════════════

class TextPairDataset(Dataset):
    """Simple dataset that returns (clean, corrupted, label) text triples."""
    def __init__(self, texts, max_words=20):
        self.texts = [" ".join(t.split()[:max_words]) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.texts[idx], 0


def collate_same(batch):
    """Pass identical strings — corruption happens in patched tokenize_plus."""
    clean_list, _, labels = zip(*batch)
    return list(clean_list), list(clean_list), torch.tensor(list(labels))


def perplexity_metric(logits, clean_logits, input_lengths, labels,
                      mean=True, loss=True):
    """CE loss using argmax of clean_logits as pseudo-targets."""
    batch_size = logits.size(0)
    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    count = 0
    for b in range(batch_size):
        seq_len = min(input_lengths[b].item(), logits.size(1))
        if seq_len < 2:
            continue
        targets = clean_logits[b, :seq_len - 1].detach().argmax(dim=-1)
        pred = logits[b, :seq_len - 1].float()
        ce = nn.functional.cross_entropy(pred, targets)
        total_loss = total_loss + ce
        count += 1
    result = total_loss / max(count, 1)
    return result if loss else -result


def patch_tokenize_for_shuffle(eap_module, noise_std=0.05):
    """
    Patch EAP's tokenize_plus so every 2nd call shuffles tokens.
    Call this BEFORE running attribution.
    Returns counter dict so you can reset it.
    """
    from eap.utils import tokenize_plus as _orig

    counter = {"n": 0}

    def patched(model, inputs, max_length=None):
        tokens, attn_mask, input_lengths, n_pos = _orig(model, inputs, max_length)
        counter["n"] += 1
        if counter["n"] % 2 == 0:
            for b in range(tokens.size(0)):
                seq_len = input_lengths[b].item()
                if seq_len > 2:
                    perm = torch.randperm(seq_len - 1, device=tokens.device) + 1
                    tokens[b, 1:seq_len] = tokens[b, perm]
        return tokens, attn_mask, input_lengths, n_pos

    eap_module.tokenize_plus = patched
    return counter


def run_eap_ig_nodes(tl_model, texts, max_words=20, batch_size=1):
    """
    Run EAP-IG node-level attribution using TransformerLens model.
    Returns: raw_scores tensor (n_forward_nodes,)
    """
    from functools import partial
    from eap.graph import Graph
    from eap.attribute_node import attribute_node
    import eap.attribute_node as eap_mod

    # Patch corruption
    counter = patch_tokenize_for_shuffle(eap_mod)

    dataset = TextPairDataset(texts, max_words=max_words)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=collate_same, shuffle=False)

    counter["n"] = 0
    graph = Graph.from_model(tl_model, node_scores=True)
    print(f"Graph: {graph.n_forward} forward nodes")

    attribute_node(
        tl_model, graph, dataloader,
        partial(perplexity_metric, loss=True, mean=True),
        method="EAP", quiet=False,
    )
    return graph.nodes_scores.clone()


def extract_node_scores(raw_scores, n_layers, n_heads):
    """
    Convert raw EAP-IG node scores → dict of component scores.
    Handles the graph index layout: [input, a0.h0, ..., a0.hN, m0, a1.h0, ...].
    """
    params_per_head = 4 * D_MODEL * D_HEAD
    params_per_mlp = 3 * D_MODEL * D_FF

    scores = {}
    idx = 1  # skip input node
    for layer in range(n_layers):
        for head in range(n_heads):
            name = f"a{layer}.h{head}"
            raw = raw_scores[idx].item()
            scores[name] = {
                "type": "head", "layer": layer, "head": head,
                "raw_score": raw,
                "normalized_score": abs(raw) / params_per_head,
                "params": params_per_head,
            }
            idx += 1
        name = f"m{layer}"
        raw = raw_scores[idx].item()
        scores[name] = {
            "type": "mlp", "layer": layer,
            "raw_score": raw,
            "normalized_score": abs(raw) / params_per_mlp,
            "params": params_per_mlp,
        }
        idx += 1
    return scores


def node_scores_to_layer_importance(component_scores, n_layers):
    """
    Aggregate per-component scores → per-layer importance.
    Sum of |score| across all heads + MLP in each layer.
    """
    layer_importance = {}
    for layer in range(n_layers):
        total = 0.0
        for name, info in component_scores.items():
            if info["layer"] == layer:
                total += abs(info["raw_score"])
        layer_importance[layer] = total
    return layer_importance


# ═══════════════════════════════════════════════════════════════
#  RelP node-level (requires RelP-modified TransformerLens)
# ═══════════════════════════════════════════════════════════════

def run_relp_nodes(tl_model, texts, num_samples=500, max_seq_len=128):
    """
    Run RelP sub-component attribution using TransformerLens model.
    RelP = attribution patching with LRP backward pass.

    Returns:
        sub_scores: dict of (layer, hook_name) → scalar score
    """
    from transformer_lens import ActivationCache

    n_layers = tl_model.cfg.n_layers
    loss_fct = nn.CrossEntropyLoss()

    sub_hooks = [
        'attn.hook_q', 'attn.hook_k', 'attn.hook_v', 'attn.hook_z',
        'attn.hook_result',
        'mlp.hook_pre', 'mlp.hook_pre_linear', 'mlp.hook_post',
        'hook_mlp_out',
    ]

    sub_scores = {}
    for layer in range(n_layers):
        for hook in sub_hooks:
            sub_scores[(layer, hook)] = 0.0

    # Cache helpers
    filter_fn = lambda name: "_input" not in name and "attn_in" not in name

    tokens_global = [None]  # mutable container for closure

    def ce_metric(logits):
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = tokens_global[0][:, 1:].contiguous()
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def get_cache_fwd_and_bwd(model, tokens, metric):
        model.reset_hooks()
        cache = {}
        def fwd_hook(act, hook):
            cache[hook.name] = act.detach()
        model.add_hook(filter_fn, fwd_hook, "fwd")

        grad_cache = {}
        def bwd_hook(act, hook):
            grad_cache[hook.name] = act.detach()
        model.add_hook(filter_fn, bwd_hook, "bwd")

        value = metric(model(tokens))
        value.backward()
        model.reset_hooks()
        return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)

    # ── Main loop ──
    from datasets import load_dataset
    pile = load_dataset("NeelNanda/pile-10k", split="train")

    for i in tqdm(range(min(num_samples, len(texts))), desc="RelP attribution"):
        text = texts[i]
        clean_tokens = tl_model.to_tokens(text)[:, :max_seq_len]

        # Token-level corruption
        corrupted_tokens = clean_tokens.clone()
        seq_len = clean_tokens.size(1)
        if seq_len > 2:
            perm = torch.randperm(seq_len - 1, device=clean_tokens.device) + 1
            corrupted_tokens[0, 1:] = clean_tokens[0, perm]

        tokens_global[0] = clean_tokens
        _, clean_cache, clean_grad = get_cache_fwd_and_bwd(
            tl_model, clean_tokens, ce_metric)

        tokens_global[0] = corrupted_tokens
        _, corrupt_cache, corrupt_grad = get_cache_fwd_and_bwd(
            tl_model, corrupted_tokens, ce_metric)

        with torch.no_grad():
            for layer in range(n_layers):
                for hook in sub_hooks:
                    key = f"blocks.{layer}.{hook}"
                    if (key in clean_cache.cache_dict and
                        key in corrupt_cache.cache_dict and
                        key in corrupt_grad.cache_dict):
                        diff = clean_cache[key] - corrupt_cache[key]
                        grad = corrupt_grad[key]
                        score = (grad * diff).abs().sum().item()
                        sub_scores[(layer, hook)] += score

        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Average
    for k in sub_scores:
        sub_scores[k] /= num_samples

    return sub_scores


def relp_to_layer_importance(sub_scores, n_layers):
    """Convert RelP sub-component scores → per-layer importance."""
    layer_imp = {}
    for layer in range(n_layers):
        total = 0.0
        for (l, hook), score in sub_scores.items():
            if l == layer:
                total += score
        layer_imp[layer] = total
    return layer_imp


# ═══════════════════════════════════════════════════════════════
#  Hook-to-weight mapping (for RelP sub-component → matrix)
# ═══════════════════════════════════════════════════════════════

HOOK_TO_WEIGHT = {
    'attn.hook_q':          'self_attn.q_proj',
    'attn.hook_k':          'self_attn.k_proj',
    'attn.hook_v':          'self_attn.v_proj',
    'attn.hook_result':     'self_attn.o_proj',
    'mlp.hook_pre':         'mlp.gate_proj',
    'mlp.hook_pre_linear':  'mlp.up_proj',
    'hook_mlp_out':         'mlp.down_proj',
}

def relp_to_matrix_importance(sub_scores, n_layers):
    """Convert RelP sub-component scores → per-matrix importance dict."""
    importance = {}
    for (layer, hook), score in sub_scores.items():
        if hook in HOOK_TO_WEIGHT:
            matrix_name = HOOK_TO_WEIGHT[hook]
            importance[(layer, matrix_name)] = score
    return importance
