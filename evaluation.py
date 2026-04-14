# evaluation.py — perplexity evaluation on WikiText-2
#
# Standard Wanda protocol: full test set, non-overlapping segments, seqlen=2048.

import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_wikitext2_testenc
from config import EVAL_SEQLEN


@torch.no_grad()
def eval_perplexity_wikitext2(model, tokenizer, device=torch.device("cuda:0"),
                                max_samples=None):
    """
    Evaluate perplexity on WikiText-2 test set.

    Uses full test set concatenated, split into non-overlapping
    segments of EVAL_SEQLEN tokens (141 segments for seqlen=2048).

    Returns: float perplexity
    """
    testenc = get_wikitext2_testenc(tokenizer)
    seqlen = min(EVAL_SEQLEN, getattr(model.config, "max_position_embeddings", 2048))
    nsamples = testenc.numel() // seqlen
    if max_samples:
        nsamples = min(nsamples, max_samples)

    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    for i in tqdm(range(nsamples), desc="Eval PPL"):
        inputs = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)
        logits = model(inputs).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()
