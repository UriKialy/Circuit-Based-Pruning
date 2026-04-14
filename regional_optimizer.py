# regional_optimizer.py — Regional Optimization from Wanda++
#
# Minimizes MSE between dense and pruned block outputs.
# Included for future use — NOT called by default.
#
# Adapted from wandaplus-main/lib/wandaplus_wrapper.py

import copy
import random
import torch
import torch.nn as nn


def regional_optimize(block, clean_inps, attention_mask, position_ids,
                      position_embeddings=None,
                      n_iters=5, n_ro_samples=32,
                      lr=3e-7, dtype_ro=torch.float32):
    """
    Regional Optimization: adjust surviving weights to minimize
    the discrepancy between dense and pruned block outputs.

    Args:
        block:               pruned LlamaDecoderLayer (in-place modified)
        clean_inps:          list of (1, seqlen, d_model) on CPU — calibration inputs
        attention_mask:      attention mask
        position_ids:        position ids
        position_embeddings: RoPE tuple or None
        n_iters:             number of RO iterations
        n_ro_samples:        samples per iteration (randomly selected from clean_inps)
        lr:                  learning rate for RMSprop
        dtype_ro:            dtype for optimization (float32 recommended)

    Note:
        This modifies block weights in-place.
        After RO, you should re-apply the pruning mask to restore sparsity.
    """
    device = next(block.parameters()).device

    # Save dense (unpruned) outputs as targets
    dense_block = copy.deepcopy(block)
    dense_block.eval()
    for p in dense_block.parameters():
        p.requires_grad_(False)

    # Build fwd kwargs
    fwd_kwargs = {}
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask.to(device)
    if position_ids is not None:
        fwd_kwargs["position_ids"] = position_ids.to(device)
    if position_embeddings is not None:
        fwd_kwargs["position_embeddings"] = (
            position_embeddings[0].to(device),
            position_embeddings[1].to(device),
        )

    # Collect pruning mask (to re-apply after weight updates)
    masks = {}
    for name, param in block.named_parameters():
        if "weight" in name:
            masks[name] = (param.data == 0).clone()

    # Cast to optimization dtype
    for param in block.parameters():
        param.data = param.data.to(dtype_ro)
    for param in dense_block.parameters():
        param.data = param.data.to(dtype_ro)

    # Optimizer
    weight_params = [p for n, p in block.named_parameters() if "weight" in n]
    optimizer = torch.optim.RMSprop(weight_params, lr=lr)
    loss_fn = nn.MSELoss(reduction="sum")

    for iteration in range(n_iters):
        # Sample subset
        indices = random.sample(range(len(clean_inps)), min(n_ro_samples, len(clean_inps)))
        total_loss = 0.0

        for idx in indices:
            inp = clean_inps[idx].to(device).to(dtype_ro)

            # Dense target
            with torch.no_grad():
                target = dense_block(inp, **fwd_kwargs)
                if isinstance(target, tuple):
                    target = target[0]

            # Pruned output
            optimizer.zero_grad()
            out = block(inp, **fwd_kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = loss_fn(out, target.detach())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Re-apply pruning mask
            with torch.no_grad():
                for name, param in block.named_parameters():
                    if name in masks:
                        param.data[masks[name]] = 0

        if iteration == 0 or iteration == n_iters - 1:
            print(f"    RO iter {iteration}: loss = {total_loss / len(indices):.4f}")

    # Cast back to original dtype
    for param in block.parameters():
        param.data = param.data.half()

    del dense_block
    torch.cuda.empty_cache()
