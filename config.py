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
