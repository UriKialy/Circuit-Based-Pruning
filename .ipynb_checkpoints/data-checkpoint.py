# data.py — load pile-10k, C4, calibration data, wikitext

import random
import torch
from datasets import load_dataset


# ═══════════════════════════════════════════════════════════════
#  Raw text loaders (for attribution)
# ═══════════════════════════════════════════════════════════════

def load_pile10k(num_samples, min_words=20):
    """Load filtered texts from pile-10k."""
    pile = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [t.strip() for t in pile["text"]
             if len(t.strip().split()) > min_words]
    return texts[:num_samples]


def load_c4_texts(num_samples, min_words=20):
    """Load filtered texts from C4 validation split, same size as pile-10k."""
    c4 = load_dataset("allenai/c4", "en", split="validation")
    texts = [t.strip() for t in c4["text"]
             if len(t.strip().split()) > min_words]
    return texts[:num_samples]


def load_texts(name, num_samples, min_words=20):
    """Unified text loader — 'pile10k' or 'c4'."""
    if name == "pile10k":
        return load_pile10k(num_samples, min_words)
    elif name == "c4":
        return load_c4_texts(num_samples, min_words)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ═══════════════════════════════════════════════════════════════
#  Calibration loaders (for pruning — matches Wanda protocol)
# ═══════════════════════════════════════════════════════════════

def _sample_sequences(dataset_texts, nsamples, seed, seqlen, tokenizer):
    """Sample random subsequences of length `seqlen` from tokenized texts."""
    random.seed(seed)
    loader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset_texts) - 1)
            enc = tokenizer(dataset_texts[i], return_tensors="pt")
            if enc.input_ids.shape[1] > seqlen:
                break
        start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, start : start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader


def get_calibration_loader(dataset_name, nsamples, seed, seqlen, tokenizer):
    """
    Get calibration dataloader for pruning.
    Returns: list of (input_ids, targets) tuples
    """
    if dataset_name == "pile10k":
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        texts = ds["text"]
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=False)
        texts = ds["text"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return _sample_sequences(texts, nsamples, seed, seqlen, tokenizer)


# ═══════════════════════════════════════════════════════════════
#  WikiText-2 test encoding (for PPL evaluation)
# ═══════════════════════════════════════════════════════════════

def get_wikitext2_testenc(tokenizer):
    """Load and tokenize full WikiText-2 test set."""
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")