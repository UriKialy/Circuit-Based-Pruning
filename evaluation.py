# evaluation.py — uses Wanda's actual eval function

import sys
sys.path.insert(0, "/workspace/wanda")

from lib.eval import eval_ppl, eval_ppl_wikitext
from lib.data import get_loaders

import torch


def eval_perplexity_wikitext2(model, tokenizer, device=torch.device("cuda:0")):
    """Exact Wanda PPL evaluation on WikiText-2."""
    from data import get_wikitext2_testenc
    model.seqlen = getattr(model, 'seqlen', 2048)
    testenc = get_wikitext2_testenc(tokenizer)
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testenc, 1, device)
    return ppl