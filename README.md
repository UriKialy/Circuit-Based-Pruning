# Circuit-Guided Pruning

**From Interpretability Scores to Sparsity Budgets in Large Language Models**

Circuit-Guided Pruning uses mechanistic interpretability—specifically, circuit discovery via Relevance Patching (RelP)—to allocate non-uniform, layer-wise sparsity budgets for LLM pruning. Instead of pruning every layer equally, we let circuit importance scores decide how much to prune each layer, while Wanda handles the fine-grained weight selection within each layer.

## Key Results (LLaMA-3.2-3B)

| Sparsity | Wanda PPL | Ours PPL | Improvement |
|----------|-----------|----------|-------------|
| 30%      | 8.61      | 8.58     | −0.3%       |
| 50%      | 19.66     | 16.78    | −14.7%      |
| 70%      | 1769.44   | 1072.98  | −39.4%      |

Zero-shot accuracy improves by +1.1 points at 50% sparsity and +2.6 points at 70% across seven benchmarks (BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA).

## Method Overview

1. **Circuit Attribution (RelP):** Score each transformer layer by its causal contribution to language modeling using Relevance Patching on a small calibration set.
2. **Sparsity Allocation:** Convert layer importance scores into per-layer sparsity budgets via an inverse softmax-temperature scheme.
3. **Local Pruning (Wanda):** Within each layer, use Wanda's `|W| · ‖X‖` criterion to select which weights to remove.

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/circuit-guided-pruning.git
```

### Dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0
- TransformerLens (modified, following [RelP](https://github.com/Farnazgh/RelP))
- Hugging Face Transformers
- lm-eval (for zero-shot evaluation)

## Usage

### Step 1: Compute circuit scores

```bash
python compute_scores.py \
    --model meta-llama/Llama-3.2-3B \
    --method relp \
    --n_samples 500 \
    --output scores/llama3.2_3b_relp.pt
```

### Step 2: Prune with circuit-guided allocation

```bash
python prune.py \
    --model meta-llama/Llama-3.2-3B \
    --scores scores/llama3.2_3b_relp.pt \
    --sparsity 0.5 \
    --temperature 5.0 \
    --output models/llama3.2_3b_pruned_50
```

### Step 3: Evaluate

```bash
python evaluate.py \
    --model models/llama3.2_3b_pruned_50 \
    --eval_ppl \
    --eval_zeroshot
```

## Project Structure

```
├── compute_scores.py    # RelP / EAP-IG circuit attribution
├── prune.py             # Circuit-guided pruning with Wanda
├── evaluate.py          # Perplexity & zero-shot evaluation
├── src/
│   ├── attribution/     # RelP and EAP-IG implementations
│   ├── allocation/      # Softmax-temperature sparsity allocation
│   └── pruning/         # Wanda local pruning
├── configs/             # Experiment configurations
└── scripts/             # Ablation & analysis scripts
```

## Citation

```bibtex
@article{kialy2025circuit,
  title={Circuit-Based Pruning: From Interpretability Scores to Sparsity Budgets in Large Language Models},
  author={Kialy, Uri Z.},
  year={2025}
}
```

## Acknowledgments

This work builds on [Wanda](https://github.com/locuslab/wanda), [RelP](https://github.com/Farnazgh/RelP), and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## License

MIT
