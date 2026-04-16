# attribution_nodes.py — node-level circuit discovery
#
# EAP-IG (full library) and RelP (direct hook-based).
# Now includes optional Gaussian noise on embeddings during corrupted forward.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import N_LAYERS, N_HEADS, D_MODEL, D_HEAD, D_FF, EMBEDDING_NOISE_STD
from corruption import make_tl_embedding_noise_hook


# ═══════════════════════════════════════════════════════════════
#  EAP-IG node-level (unchanged)
# ═══════════════════════════════════════════════════════════════

class TextPairDataset(Dataset):
    def __init__(self, texts, max_words=20):
        self.texts = [" ".join(t.split()[:max_words]) for t in texts]
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.texts[idx], 0


def collate_same(batch):
    clean_list, _, labels = zip(*batch)
    return list(clean_list), list(clean_list), torch.tensor(list(labels))


def perplexity_metric(logits, clean_logits, input_lengths, labels,
                      mean=True, loss=True):
    batch_size = logits.size(0)
    total = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    count = 0
    for b in range(batch_size):
        seq_len = min(input_lengths[b].item(), logits.size(1))
        if seq_len < 2:
            continue
        targets = clean_logits[b, :seq_len - 1].detach().argmax(dim=-1)
        pred = logits[b, :seq_len - 1].float()
        ce = nn.functional.cross_entropy(pred, targets)
        total = total + ce
        count += 1
    result = total / max(count, 1)
    return result if loss else -result


def patch_tokenize_for_shuffle(eap_module):
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
    from functools import partial
    from eap.graph import Graph
    from eap.attribute_node import attribute_node
    import eap.attribute_node as eap_mod

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
    params_per_head = 4 * D_MODEL * D_HEAD
    params_per_mlp = 3 * D_MODEL * D_FF
    scores = {}
    idx = 1
    for layer in range(n_layers):
        for head in range(n_heads):
            raw = raw_scores[idx].item()
            scores[f"a{layer}.h{head}"] = {
                "type": "head", "layer": layer, "head": head,
                "raw_score": raw,
                "normalized_score": abs(raw) / params_per_head,
                "params": params_per_head,
            }
            idx += 1
        raw = raw_scores[idx].item()
        scores[f"m{layer}"] = {
            "type": "mlp", "layer": layer,
            "raw_score": raw,
            "normalized_score": abs(raw) / params_per_mlp,
            "params": params_per_mlp,
        }
        idx += 1
    return scores


def node_scores_to_layer_importance(component_scores, n_layers):
    layer_imp = {}
    for layer in range(n_layers):
        total = 0.0
        for name, info in component_scores.items():
            if info["layer"] == layer:
                total += abs(info["raw_score"])
        layer_imp[layer] = total
    return layer_imp


# ═══════════════════════════════════════════════════════════════
#  RelP node-level (now with proper memory management + noise)
# ═══════════════════════════════════════════════════════════════

def run_relp_nodes(tl_model, texts, num_samples=500, max_seq_len=128,
                    add_noise=True, noise_std=None):
    """Memory-efficient RelP: one layer at a time to avoid OOM."""
    from transformer_lens import ActivationCache
    if noise_std is None:
        noise_std = EMBEDDING_NOISE_STD
    n_layers = tl_model.cfg.n_layers
    loss_fct = nn.CrossEntropyLoss()
    n_samples = min(num_samples, len(texts))
    sub_hooks = [
        'attn.hook_q', 'attn.hook_k', 'attn.hook_v', 'attn.hook_z',
        'mlp.hook_pre', 'mlp.hook_pre_linear', 'mlp.hook_post',
        'hook_mlp_out',
    ]
    sub_scores = {(l, h): 0.0 for l in range(n_layers) for h in sub_hooks}
    tokens_global = [None]

    def ce_metric(logits):
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = tokens_global[0][:, 1:].contiguous()
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    print(f"Pre-tokenizing {n_samples} samples...")
    all_clean, all_corrupt = [], []
    for i in range(n_samples):
        clean = tl_model.to_tokens(texts[i])[:, :max_seq_len]
        corrupt = clean.clone()
        sl = clean.size(1)
        if sl > 2:
            perm = torch.randperm(sl - 1, device=clean.device) + 1
            corrupt[0, 1:] = clean[0, perm]
        all_clean.append(clean.cpu())
        all_corrupt.append(corrupt.cpu())

    for layer in tqdm(range(n_layers), desc="RelP layers"):
        allowed = {f"blocks.{layer}.{h}" for h in sub_hooks}
        layer_filter = lambda name, _a=allowed: name in _a

        for s_idx in range(n_samples):
            clean_tokens = all_clean[s_idx].to(tl_model.cfg.device)
            corrupt_tokens = all_corrupt[s_idx].to(tl_model.cfg.device)

            tl_model.reset_hooks()
            clean_cache, clean_grad = {}, {}
            def clean_fwd(act, hook): clean_cache[hook.name] = act.detach()
            def clean_bwd(act, hook): clean_grad[hook.name] = act.detach()
            tl_model.add_hook(layer_filter, clean_fwd, "fwd")
            tl_model.add_hook(layer_filter, clean_bwd, "bwd")
            tokens_global[0] = clean_tokens
            value = ce_metric(tl_model(clean_tokens))
            value.backward()
            tl_model.reset_hooks()

            corrupt_cache, corrupt_grad = {}, {}
            def corr_fwd(act, hook): corrupt_cache[hook.name] = act.detach()
            def corr_bwd(act, hook): corrupt_grad[hook.name] = act.detach()
            tl_model.add_hook(layer_filter, corr_fwd, "fwd")
            tl_model.add_hook(layer_filter, corr_bwd, "bwd")
            if add_noise:
                noise_hook = make_tl_embedding_noise_hook(std=noise_std)
                tl_model.add_hook('hook_embed', noise_hook, 'fwd')
            tokens_global[0] = corrupt_tokens
            value = ce_metric(tl_model(corrupt_tokens))
            value.backward()
            tl_model.reset_hooks()

            with torch.no_grad():
                for hook in sub_hooks:
                    key = f"blocks.{layer}.{hook}"
                    if (key in clean_cache and key in corrupt_cache
                            and key in corrupt_grad):
                        diff = clean_cache[key] - corrupt_cache[key]
                        grad = corrupt_grad[key]
                        sub_scores[(layer, hook)] += (grad * diff).abs().sum().item()

            del clean_cache, clean_grad, corrupt_cache, corrupt_grad
            del clean_tokens, corrupt_tokens
            tl_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    for k in sub_scores:
        sub_scores[k] /= n_samples
    return sub_scores

def relp_to_layer_importance(sub_scores, n_layers):
    layer_imp = {}
    for layer in range(n_layers):
        total = 0.0
        for (l, hook), score in sub_scores.items():
            if l == layer:
                total += score
        layer_imp[layer] = total
    return layer_imp


# ═══════════════════════════════════════════════════════════════
#  Hook-to-weight mapping
# ═══════════════════════════════════════════════════════════════

HOOK_TO_WEIGHT = {
    'attn.hook_q':         'self_attn.q_proj',
    'attn.hook_k':         'self_attn.k_proj',
    'attn.hook_v':         'self_attn.v_proj',
    'attn.hook_z':         'self_attn.o_proj',  
    'mlp.hook_pre':        'mlp.gate_proj',
    'mlp.hook_pre_linear': 'mlp.up_proj',
    'hook_mlp_out':        'mlp.down_proj'
}


def relp_to_matrix_importance(sub_scores, n_layers):
    """Convert RelP sub-component scores → per-matrix scalar importance."""
    importance = {}
    for (layer, hook), score in sub_scores.items():
        if hook in HOOK_TO_WEIGHT:
            importance[(layer, HOOK_TO_WEIGHT[hook])] = score
    return importance