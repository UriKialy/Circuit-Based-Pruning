#!/bin/bash
# Run this on RunPod after launching an A100-80GB pod
set -e

cd /workspace

# ── Clone repos ──
git clone https://github.com/UriKialy/EAP-IG.git EAP-IG-main 2>/dev/null || echo "EAP-IG already cloned"
git clone https://github.com/UriKialy/wandaplus.git wandaplus-main 2>/dev/null || echo "wandaplus already cloned"

# ── Install deps ──
pip install torch transformers datasets einops tqdm huggingface_hub accelerate -q
pip install transformer_lens -q
pip install lm-eval -q

# ── Install EAP-IG as package ──
cd /workspace/EAP-IG-main
pip install -e . -q
cd /workspace

# ── Download LLaMA-1 7B (requires HF login) ──
huggingface-cli login
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading LLaMA-1 7B tokenizer...')
AutoTokenizer.from_pretrained('huggyllama/llama-7b')
print('Downloading LLaMA-1 7B model...')
AutoModelForCausalLM.from_pretrained('huggyllama/llama-7b', torch_dtype='float16')
print('Done!')
"

# ── Pre-download datasets ──
python -c "
from datasets import load_dataset
print('Downloading pile-10k...')
load_dataset('NeelNanda/pile-10k', split='train')
print('Downloading C4 validation...')
load_dataset('allenai/c4', 'en', split='validation', streaming=False)
print('Downloading WikiText-2...')
load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
print('Done!')
"

echo "Setup complete."
