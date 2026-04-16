# corruption.py — modular corruption for clean vs corrupt pairs

import torch
import numpy as np
from config import EMBEDDING_NOISE_STD


def shuffle_tokens(tokens):
    """Shuffle all non-BOS tokens. Returns cloned, permuted tensor."""
    shuffled = tokens.clone()
    for b in range(shuffled.size(0)):
        seq_len = shuffled.size(1)
        if seq_len > 2:
            perm = torch.randperm(seq_len - 1, device=tokens.device) + 1
            shuffled[b, 1:] = tokens[b, perm]
    return shuffled


def add_embedding_noise(embeddings, std=None):
    """Add Gaussian noise to embedding activations."""
    if std is None:
        std = EMBEDDING_NOISE_STD
    return embeddings + torch.randn_like(embeddings) * std


# ═══════════════════════════════════════════════════════════════
#  Embedding hook — used during corrupted forward pass
# ═══════════════════════════════════════════════════════════════

def make_embedding_noise_hook(std=None):
    """
    Returns (hook_fn, handle_container).

    Usage for HF model:
        hook_fn, container = make_embedding_noise_hook()
        h = model.model.embed_tokens.register_forward_hook(hook_fn)
        # ... corrupted forward ...
        h.remove()

    Usage for TransformerLens:
        hook_fn_tl = make_tl_embedding_noise_hook()
        model.add_hook('hook_embed', hook_fn_tl, 'fwd')
        # ... corrupted forward ...
        model.reset_hooks()
    """
    if std is None:
        std = EMBEDDING_NOISE_STD

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + torch.randn_like(output[0]) * std,) + output[1:]
        return output + torch.randn_like(output) * std

    return hook_fn


def make_tl_embedding_noise_hook(std=None):
    """TransformerLens-style hook (signature: activation, hook)."""
    if std is None:
        std = EMBEDDING_NOISE_STD

    def hook_fn(activation, hook):
        return activation + torch.randn_like(activation) * std

    return hook_fn


def corrupt_tokens(clean_tokens, mode="shuffle"):
    """Token-level corruption only."""
    if mode == "shuffle":
        return shuffle_tokens(clean_tokens)
    raise ValueError(f"Unknown token corruption mode: {mode}")


def corrupt_embeddings(clean_embeddings, mode="noise", noise_std=None):
    """Embedding-level corruption only."""
    if mode == "noise":
        return add_embedding_noise(clean_embeddings, std=noise_std)
    raise ValueError(f"Unknown embedding corruption mode: {mode}")


def make_corrupt_pair(clean_tokens, embed_fn=None, mode="both", noise_std=None):
    """
    Modes:
        'shuffle' -> (clean_tokens, corrupt_tokens)
        'noise'   -> (clean_embs, corrupt_embs)
        'both'    -> (clean_embs, corrupt_embs) with shuffle + noise
    """
    if noise_std is None:
        noise_std = EMBEDDING_NOISE_STD

    if mode == "shuffle":
        return clean_tokens, shuffle_tokens(clean_tokens)
    if embed_fn is None:
        raise ValueError(f"embed_fn required for mode='{mode}'")
    if mode == "noise":
        clean_embs = embed_fn(clean_tokens)
        return clean_embs, add_embedding_noise(clean_embs, std=noise_std)
    if mode == "both":
        shuffled = shuffle_tokens(clean_tokens)
        clean_embs = embed_fn(clean_tokens)
        corrupt_embs = add_embedding_noise(embed_fn(shuffled), std=noise_std)
        return clean_embs, corrupt_embs
    raise ValueError(f"Unknown mode: {mode}")