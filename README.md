# Circuit-Guided Pruning

**From Interpretability Scores to Sparsity Budgets in Large Language Models**

Circuit-Guided Pruning uses mechanistic interpretability — circuit attribution via **Relevance Patching (RelP)** and **EAP-IG** — to inform LLM pruning. Instead of pruning every layer equally, circuit-importance scores either (i) drive non-uniform per-layer sparsity budgets, with Wanda handling fine-grained weight selection inside each layer, or (ii) replace Wanda's `|W|·‖X‖` criterion entirely.

We validate on two models: **LLaMA-3.2-3B** (primary, all ablations) and **LLaMA-1 7B** ( zero-shot baselines from the Wanda paper).

---

## Headline Results

### LLaMA-3.2-3B — WikiText-2 perplexity

| Sparsity | Wanda (uniform) | Circuit-Guided (ours) | Δ |
|---------:|----------------:|----------------------:|--:|
| 30%      | 8.61            | **8.58**              | −0.3% |
| 50%      | 19.66           | **16.78**             | −14.7% |
| 70%      | 1769.44         | **1072.98**           | −39.4% |

### LLaMA-3.2-3B — Zero-shot accuracy (mean over 7 tasks)

7-task suite: BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OpenBookQA. Evaluated with `lm-eval-harness`.

| Sparsity | Wanda (uniform) | Circuit-Guided (ours) | Δ |
|---------:|----------------:|----------------------:|--:|
| 30%      | 63.3%           | 63.1%                 | −0.2 |
| 50%      | 52.7%           | **53.8%**             | **+1.1** |
| 70%      | 36.0%           | **38.6%**             | **+2.6** |

### LLaMA-1 7B — WikiText-2 perplexity

Evaluated identically to Wanda's setup (seqlen 2048, 166 segments).

| Sparsity | Wanda | Wanda++ | GBLM-Pruner | EAP-IG → Wanda criterion (ours) | RelP → layer budgets (ours) |
|---------:|------:|--------:|------------:|--------------------------------:|----------------------------:|
| 50%      | 7.26  | 7.02    | 7.15        | **7.16**                        | 7.26                        |
| 70%      | 76.17 | 55.52   | 54.60       | 100.72                          | **63.83**                   |

At 70%, RelP-driven non-uniform layer budgets beat Wanda by 16% (63.83 vs 76.17). At 50%, replacing Wanda's `|W|·‖X‖` with EAP-IG weight scores beats Wanda by 1.2% with a simpler signal than Wanda++/GBLM use.

### LLaMA-1 7B — Zero-shot accuracy at 50% sparsity (vs Wanda paper Table 23)

| | BoolQ | RTE | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | **Mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wanda (paper)            | 71.22 | 55.60 | 51.85 | 66.06 | 69.11 | 36.86 | 28.80 | 54.21 |
| **Ours (EAP-IG, α=100)** | 73.06 | 61.01 | 51.69 | 66.93 | 71.68 | 37.12 | 30.00 | **55.93** |

**+1.72 points** absolute over Wanda on the seven shared tasks.

---

## Why It Works — Ablations on LLaMA-3.2-3B

All experiments below live in `notebooks/pruning_by_circuits_llama3_3B_v1.ipynb`.

### 1. The score ranking matters — shuffle ablation 

Randomly permuting the per-layer budget across 5 seeds:

| Sparsity | Uniform | RelP layer (ours) | Shuffled (mean ± std, n=5) |
|---------:|--------:|------------------:|---------------------------:|
| 30%      | 8.61    | **8.61**          | 8.70 ± 0.03                |
| 50%      | 19.68   | **17.89**         | 24.16 ± 1.51               |
| 70%      | 1770    | **1592**          | 3301 ± 2351                |

The specific layer ranking does the work — random non-uniform allocation is *worse* than uniform.

### 2. Don't hand-craft a heuristic — U-shape ablation 

A common heuristic prunes more from the middle layers (U-shape mask). It loses to both uniform Wanda and circuit-guided allocation:

| Sparsity | Uniform | U-shaped | Circuit-Guided |
|---------:|--------:|---------:|---------------:|
| 30%      | 8.61    | 8.68     | **8.61**       |
| 50%      | 19.68   | 22.51    | **17.89**      |
| 70%      | 1770    | 2682     | **1592**       |

Data-driven attribution beats a strong inductive prior.

### 3. Crossover analysis — when does non-uniform start to pay? 

Fine-grained sweep at 5%-point intervals:

| Sparsity     | 25%  | 30%  | 35%  | 40%   | 50%   | 55%   | 60%  | 65%   | 70%  | 80%    | 90%     |
|--------------|-----:|-----:|-----:|------:|------:|------:|-----:|------:|-----:|-------:|--------:|
| Uniform      | 8.22 | 8.62 | 9.35 | 10.65 | 19.68 | 34.73 | 97.6 | 352.8 | 2317 | 73 391 | 334 710 |
| Layer (ours) | **8.21** | **8.58** | **9.20** | **10.27** | **17.89** | **28.51** | **72.2** | 363.0 | **1117** | **12 171** | **27 901** |

Below ~35% the methods tie; above 35% non-uniform pulls ahead and the gap grows with sparsity (≈10× lower PPL at 90%). One reversal at 65% (uniform briefly wins by ~10 points) is interesting and worth a closer look.

### 4. Protection × granularity — what's the right scope? 

We tried four variants of "where do circuit scores act?":

| Variant            | Per-layer budget? | Per-matrix budget? | Cap edge layers? |
|--------------------|:-----------------:|:------------------:|:----------------:|
| Layer-protected    | ✓                 |                    | ✓                |
| Layer-unprotected  | ✓                 |                    |                  |
| Matrix-protected   |                   | ✓                  | ✓                |
| Matrix-unprotected |                   | ✓                  |                  |

Per-layer + unprotected is the simplest and consistently the best, especially at high sparsity:

| Sparsity | Uniform | Layer-prot | **Layer-unprot** | Matrix-prot | Matrix-unprot |
|---------:|--------:|-----------:|-----------------:|------------:|--------------:|
| 30%      | 8.62    | 8.58       | 8.63             | 8.66        | 8.66          |
| 50%      | 19.66   | 17.95      | **16.78**        | 22.74       | 22.03         |
| 70%      | 1769    | 1721       | **1073**         | 1975        | 1340          |

Per-matrix scoring is too noisy at high sparsity; protection caps cost ~5 PPL at 50%.

### 5. Global Wanda vs local Wanda — why per-layer normalization matters 

Wanda's published formulation is local (per-layer). A common naive extension is *global* — pool all weights and threshold globally. Global Wanda collapses on LLaMA-3.2-3B:

| Sparsity | Local Wanda | Global Wanda | Circuit-Guided |
|---------:|------------:|-------------:|---------------:|
| 30%      | 8.61        | 10 140       | **8.62**       |
| 40%      | 10.64       | 15 149       | **10.42**      |
| 50%      | 19.66       | 15 093       | **17.95**      |

Circuit-guided allocation behaves like an explicit per-layer normalizer derived from interpretability — recovering all the benefit of local Wanda while still allocating non-uniformly.

### 6. LoRA recovery — does the gap survive fine-tuning? 

500 steps of LoRA on a small instruction-tuning corpus, 50% sparsity:

| | PPL before | PPL after | Acc before | Acc after |
|---|---:|---:|---:|---:|
| Uniform        | 19.66     | **11.29** | 52.7%     | **60.1%** |
| Circuit-Guided | **17.95** | 11.81     | **52.9%** | 58.8%     |

**Recovery Dynamics:** Circuit-guided pruning dominates the no-fine-tuning regime (+1.71 PPL, +0.2 acc). However, standard LoRA recovery currently closes this gap, allowing uniform methods like Wanda to achieve parity. We suspect network-wide LoRA dilutes the preserved subgraphs, and are actively testing targeted Circuit Fine-Tuning to maintain our edge.

---

## Method

1. **Circuit attribution.** Score each transformer block by its causal contribution to language modelling. We support two scorers:
   - **RelP** (node-level, via a modified TransformerLens) — used for per-layer importance.
   - **EAP-IG** (weight-level, block-by-block on the HF model) — usable either as a layer scorer or directly as a Wanda-replacement criterion.
2. **Sparsity allocation.** Convert importance scores to per-layer sparsity ratios via inverse softmax-temperature: `s_ℓ ∝ exp(−importance_ℓ / T)`. Optimal T depends on target sparsity — T=10 (≈ uniform) at 50%, T=3 (clearly non-uniform) at 70% on LLaMA-1 7B; T=5 throughout for LLaMA-3.2-3B.
3. **Local pruning.** Within each layer, Wanda's `|W| · ‖X‖` criterion picks the weights to remove given the per-layer budget.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── setup.sh                           # RunPod / fresh-pod environment setup
├── config.py                          # model + experiment constants
│
├── attribution_nodes.py               # RelP node-level attribution
├── attribution_weights.py             # EAP-IG weight-level attribution
├── corruption.py                      # token shuffle / Gaussian noise (clean→corrupt)
├── sparsity.py                        # inverse softmax-temperature allocator
├── pruning.py                         # Wanda + non-uniform Wanda variants
├── protection.py                      # protection-budget ablation
├── regional_optimizer.py              # post-pruning regional refinement
├── evaluation.py                      # WikiText-2 perplexity (Wanda eval)
├── eval_zeroshot.py                   # lm-eval harness wrapper
├── data.py, utils.py                  # calibration loaders, model I/O helpers
│
├── run.py                             # main runner (Experiments 1-3)
├── run_full_paper.py                  # paper sweep + ablations from cached scores
├── run_data_efficiency.py             # attribution sample-count sweep
├── run_scoring_comparison.py          # 4-allocator comparison (RelP/EAP × Pile/C4)
│
├── notebooks/
│   ├── pruning_by_circuits_llama3_3B_v1.ipynb   # 3B headline + all ablations
│   ├── pruning_by_circuits_llama3_3B.ipynb      # earlier 3B run
│   └── RelP_circuit_discovery_llama.ipynb       # RelP development notebook
│
├── results/                           # raw JSON results per experiment
│   └── MASTER_RESULTS.json            # consolidated 7B results
└── plots/                             # figures used in the paper
```

---

## Installation

Tested on a single A100-80GB (LLaMA-1 7B) and an RTX PRO 6000 96GB (LLaMA-3.2-3B). Both fit comfortably at fp16; calibration peaks around 35–40 GB on 7B.

```bash
git clone https://github.com/UriKialy/circuit-guided-pruning.git
cd circuit-guided-pruning
bash setup.sh        # clones EAP-IG + Wanda, installs deps, downloads the model
```

Manual install:
```bash
pip install -r requirements.txt
# Then clone the modified TransformerLens / Wanda repos referenced in setup.sh
```

### Dependencies
- Python ≥ 3.9, PyTorch ≥ 2.0
- TransformerLens (modified, following [RelP](https://github.com/Farnazgh/RelP))
- Hugging Face Transformers ≥ 4.41
- lm-eval (zero-shot evaluation)

---

## Reproducing the Tables

**LLaMA-3.2-3B (headline + ablations).** Open `notebooks/pruning_by_circuits_llama3_3B_v1.ipynb` and run cells in order. Sectioning matches the ablation list above:
- Cell 18 — Wanda uniform baseline at 0/10/20/30/50/70%
- Cell 34 — Experiment 1: U-shape vs Uniform vs Circuit
- Cell 38 — Experiment 5: Shuffle ablation (5 seeds)
- Cell 40 — Experiment 7: Fine-grained crossover sweep
- Cell 50 — Headline numbers + 7-task zero-shot
- Cell 54 — Experiment 4: LoRA recovery
- Cell 56 — Experiment 7: Local vs Global vs Circuit Wanda
- Cell 57 — Experiment 8: Protection × Granularity matrix

**LLaMA-1 7B (zero-shot).**
```bash
# 1. Pre-compute attribution scores once (cached to ./scores)
python run.py --experiment 2 --step attribution_only --dataset pile10k    # RelP node scores
python run.py --experiment 1 --step attribution_only --dataset pile10k    # EAP-IG weight scores

# 2a. Main 70% result — RelP non-uniform Wanda  →  PPL 63.83
python run_full_paper.py --only temp_sweep

# 2b. Main 50% result — EAP-IG-as-criterion (uniform)  →  PPL 7.16
python run.py --experiment 1 --sparsity 0.5 --alpha 100

# 3. Ablations
python run_full_paper.py --only ablation_shuffle      # signal verification
python run_full_paper.py --only ablation_protect      # protection ablation
python run_full_paper.py --only crossover             # 25%-70% sweep
python run_data_efficiency.py                         # sample-count sweep
python run_scoring_comparison.py                      # 4-allocator comparison

# 4. Zero-shot evaluation
python eval_zeroshot.py --method eapig --sparsity 0.5 --alpha 100
```

Consolidated 7B numbers live in `results/MASTER_RESULTS.json`.

---

## Acknowledgments

This work builds on [Wanda](https://github.com/locuslab/wanda) (Sun et al.), [RelP](https://github.com/Farnazgh/RelP) (Ghaffari et al.), and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (Nanda et al.). Zero-shot evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
