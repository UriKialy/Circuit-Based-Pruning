# corruption.py — modular corruption for clean vs corrupt pairs
#
# Design: separate functions for each corruption type + a wrapper
# that can call one or both.

import torch
import numpy as np


def shuffle_tokens(tokens):
    """
    Shuffle all non-BOS tokens in-place.
    Destroys positional/syntactic structure, preserves token distribution.

    Args:
        tokens: (batch, seq_len) token ids

    Returns:
        shuffled: (batch, seq_len) — cloned & permuted
    """
    shuffled = tokens.clone()
    for b in range(shuffled.size(0)):
        seq_len = shuffled.size(1)
        if seq_len > 2:
            perm = torch.randperm(seq_len - 1, device=tokens.device) + 1
            shuffled[b, 1:] = tokens[b, perm]
    return shuffled


def add_embedding_noise(embeddings, std=0.05):
    """
    Add Gaussian noise to embedding activations.
    Disrupts continuous representations.

    Args:
        embeddings: (batch, seq_len, d_model)
        std: noise standard deviation

    Returns:
        noisy: (batch, seq_len, d_model)
    """
    noise = torch.randn_like(embeddings) * std
    return embeddings + noise


def corrupt_tokens(clean_tokens, mode="shuffle"):
    """
    Produce corrupted token ids from clean ones.
    Only applies token-level operations.

    Args:
        clean_tokens: (batch, seq_len)
        mode: 'shuffle' — only option at token level

    Returns:
        corrupt: (batch, seq_len)
    """
    if mode == "shuffle":
        return shuffle_tokens(clean_tokens)
    else:
        raise ValueError(f"Unknown token corruption mode: {mode}")


def corrupt_embeddings(clean_embeddings, mode="noise", noise_std=0.05):
    """
    Produce corrupted embeddings from clean ones.
    Only applies embedding-level operations.

    Args:
        clean_embeddings: (batch, seq_len, d_model)
        mode: 'noise' — only option at embedding level

    Returns:
        corrupt: (batch, seq_len, d_model)
    """
    if mode == "noise":
        return add_embedding_noise(clean_embeddings, std=noise_std)
    else:
        raise ValueError(f"Unknown embedding corruption mode: {mode}")


def make_corrupt_pair(clean_tokens, embed_fn=None, mode="both", noise_std=0.05):
    """
    Top-level wrapper that produces clean/corrupt pairs.

    Modes:
        'shuffle'  — token shuffle only, returns (clean_tokens, corrupt_tokens)
        'noise'    — Gaussian noise on embeddings only, returns (clean_embs, corrupt_embs)
        'both'     — shuffle tokens AND add noise to embeddings of the shuffled version

    Args:
        clean_tokens: (batch, seq_len) token ids
        embed_fn:     callable that maps tokens → embeddings (needed for 'noise'/'both')
        mode:         'shuffle', 'noise', or 'both'
        noise_std:    std for Gaussian noise

    Returns:
        If mode == 'shuffle': (clean_tokens, corrupt_tokens)
        If mode == 'noise':   (clean_embs, corrupt_embs)
        If mode == 'both':    (clean_embs, corrupt_embs)
            where corrupt_embs = embed(shuffled_tokens) + noise
    """
    if mode == "shuffle":
        return clean_tokens, shuffle_tokens(clean_tokens)

    if embed_fn is None:
        raise ValueError(f"embed_fn required for mode='{mode}'")

    if mode == "noise":
        clean_embs = embed_fn(clean_tokens)
        corrupt_embs = add_embedding_noise(clean_embs, std=noise_std)
        return clean_embs, corrupt_embs

    if mode == "both":
        shuffled_tokens = shuffle_tokens(clean_tokens)
        clean_embs = embed_fn(clean_tokens)
        corrupt_embs = embed_fn(shuffled_tokens)
        corrupt_embs = add_embedding_noise(corrupt_embs, std=noise_std)
        return clean_embs, corrupt_embs

    raise ValueError(f"Unknown mode: {mode}")
