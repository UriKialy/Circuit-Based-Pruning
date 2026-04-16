# config.py — constants for LLaMA-1 7B experiments

MODEL_NAME = "huggyllama/llama-7b"

# Architecture (LLaMA-1 7B — NO GQA, all heads are full)
N_LAYERS = 32
N_HEADS = 32
N_KV_HEADS = 32
D_MODEL = 4096
D_HEAD = 128      # 4096 / 32
D_FF = 11008

# Linear layers per decoder block
LINEAR_LAYERS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Attribution
NUM_ATTRIBUTION_SAMPLES = 500
MAX_SEQ_LEN = 128
NOISE_STD = 0.05
NUM_IG_STEPS = 10

# Pruning calibration
NUM_CALIBRATION_SAMPLES = 128
CALIBRATION_SEQLEN = 2048

# Evaluation
EVAL_SEQLEN = 2048

# Sparsity allocation
DEFAULT_TEMPERATURE = 5.0

# ═══════════════════════════════════════════════════════════════
#  Experiment 4 — Protection-based pruning
# ═══════════════════════════════════════════════════════════════

# Per-matrix protection multipliers (literature prior)
#   q/k → redundant (multi-head)
#   v/gate/up → neutral
#   o/down → sensitive (residual integrators)
# Change these freely — they just scale the protection % per matrix.
PROTECTION_MULTIPLIERS = {
    "self_attn.q_proj": 0.7,
    "self_attn.k_proj": 0.7,
    "self_attn.v_proj": 1.0,
    "self_attn.o_proj": 1.3,
    "mlp.gate_proj":    1.0,
    "mlp.up_proj":      1.0,
    "mlp.down_proj":    1.3,
}

# Per-layer protection spread for exp 4b (RelP-driven)
#   (min_mult, max_mult) — layer with highest RelP gets max_mult, lowest gets min_mult
#   (1.0, 1.0) disables layer weighting (uniform across layers)
#   (0.5, 1.5) = wide spread / "worst case"
#   (0.8, 1.2) = moderate (default)
LAYER_PROTECTION_SPREAD = (0.8, 1.2)

# Embedding-level Gaussian noise (applied during corrupted forward pass)
EMBEDDING_NOISE_STD = 0.05