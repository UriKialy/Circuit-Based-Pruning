import torch
import gc

# 1. Collect garbage
gc.collect()

# 2. Clear the PyTorch cache
torch.cuda.empty_cache()

# 3. Reset the peak memory stats (good for benchmarking CFT)
torch.cuda.reset_peak_memory_stats()