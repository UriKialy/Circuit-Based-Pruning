# utils.py — shared utilities, importing from Wanda where possible

import gc
import sys
import torch
import pickle

sys.path.insert(0, "/workspace/wanda")
from lib.prune import check_sparsity, find_layers

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, CALIBRATION_SEQLEN


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
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def check_sparsity_per_layer(model):
    result = {}
    for i, layer in enumerate(model.model.layers):
        subset = find_layers(layer)
        z = sum((subset[n].weight.data == 0).sum().item() for n in subset)
        t = sum(subset[n].weight.data.numel() for n in subset)
        result[i] = z / t
    return result


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def save_scores(scores, path):
    with open(path, "wb") as f:
        pickle.dump(scores, f)
    print(f"Saved scores to {path}")


def load_scores(path):
    with open(path, "rb") as f:
        scores = pickle.load(f)
    print(f"Loaded scores from {path}")
    return scores


def print_gpu_memory():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory: {(total-free)/1e9:.1f}GB used / {total/1e9:.1f}GB total "
              f"({free/1e9:.1f}GB free)")