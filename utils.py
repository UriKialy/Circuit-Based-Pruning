# utils.py — shared utilities

import gc
import torch
import torch.nn as nn
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, CALIBRATION_SEQLEN
from pruning import find_layers


def load_model(model_name=MODEL_NAME, device_map="auto"):
    """Load HF model in float16 with seqlen set."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16,
        device_map=device_map, low_cpu_mem_usage=True,
    )
    model.eval()
    model.seqlen = CALIBRATION_SEQLEN
    return model


def load_tokenizer(model_name=MODEL_NAME):
    """Load tokenizer with pad_token set."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def check_sparsity(model):
    """Check global sparsity of the model. Returns float."""
    zeros = 0
    total = 0
    for layer in model.model.layers:
        subset = find_layers(layer)
        for name in subset:
            w = subset[name].weight.data
            zeros += (w == 0).sum().item()
            total += w.numel()
    return zeros / total


def check_sparsity_per_layer(model):
    """Check sparsity per layer. Returns dict layer_idx → float."""
    result = {}
    for i, layer in enumerate(model.model.layers):
        subset = find_layers(layer)
        z, t = 0, 0
        for name in subset:
            w = subset[name].weight.data
            z += (w == 0).sum().item()
            t += w.numel()
        result[i] = z / t
    return result


def free_memory():
    """Force garbage collection and empty CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def save_scores(scores, path):
    """Save score dict to pickle."""
    with open(path, "wb") as f:
        pickle.dump(scores, f)
    print(f"Saved scores to {path}")


def load_scores(path):
    """Load score dict from pickle."""
    with open(path, "rb") as f:
        scores = pickle.load(f)
    print(f"Loaded scores from {path}")
    return scores


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory: {(total-free)/1e9:.1f}GB used / {total/1e9:.1f}GB total "
              f"({free/1e9:.1f}GB free)")
