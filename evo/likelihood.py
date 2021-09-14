from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import esm
from typing import Union, List
from functools import partial


from .tokenization import Vocab
from .tensor import batched_iterator
from .align import MSA


@torch.no_grad()
def sequence_logits(
    model: esm.model.ProteinBertModel,
    vocab: Vocab,
    sequence: Union[str, MSA],
    verbose: bool = False,
    mask_positions: bool = True,
    max_tokens: int = 2 ** 14,
) -> torch.Tensor:

    device = next(model.parameters()).device

    tokens = torch.from_numpy(vocab.encode(sequence)).to(device)
    start = int(vocab.prepend_bos)
    end = tokens.size(-1) - int(vocab.append_eos)
    if mask_positions:
        repeat_dims = [1] * tokens.dim()
        tokens = tokens.unsqueeze(0).repeat(end - start, *repeat_dims)
        if tokens.dim() == 2:
            tokens[torch.arange(end - start), torch.arange(start, end)] = vocab.mask_idx
        else:
            tokens[torch.arange(end - start), 0, torch.arange(start, end)] = vocab.mask_idx

        logits = torch.zeros((end - start, len(vocab)), device=device)
        batch_size = max(1, max_tokens // np.prod(tokens.size()[1:]))
        for i, batch in enumerate(
            batched_iterator(tokens, batch_size=batch_size, device=device, verbose=verbose)
        ):
            idx = i * batch_size

            batch_indices = torch.arange(batch.size(0))
            out = model(batch)["logits"]
            if tokens.dim() == 2:
                out = out[
                    batch_indices,
                    batch_indices + start + idx,
                ]
            else:
                out = out[
                    batch_indices,
                    0,
                    batch_indices + start + idx,
                ]
            logits[idx : idx + batch_size] = out
    else:
        logits = model(tokens.unsqueeze(0))["logits"][0, start:end]
    return logits


@torch.no_grad()
def sequence_mutant_scores(
    model: esm.model.ProteinBertModel,
    vocab: Vocab,
    sequence: Union[str, MSA],
    verbose: bool = False,
    mask_positions: bool = True,
    max_tokens: int = 2 ** 14,
) -> torch.Tensor:
    print("hello!")
    logits = sequence_logits(
        model, vocab, sequence, verbose, mask_positions, max_tokens
    ).log_softmax(-1)
    wt_encoded = torch.from_numpy(vocab.encode(sequence))
    if wt_encoded.dim() == 2:
        wt_encoded = wt_encoded[0]
    start = int(vocab.prepend_bos)
    end = wt_encoded.size(-1) - int(vocab.append_eos)
    wt_encoded = wt_encoded[start:end]

    wt_logits = logits[torch.arange(len(wt_encoded)), wt_encoded].unsqueeze(1)
    return logits - wt_logits


@torch.no_grad()
def sequence_pseudo_ppl(
    model: esm.model.ProteinBertModel,
    vocab: Vocab,
    sequence: str,
    mask_positions: bool = True,
    verbose: bool = False,
    max_tokens: int = 2 ** 14,
    reduction: str = "mean",
    log: bool = False,
) -> float:

    device = next(model.parameters()).device
    tokens = torch.from_numpy(vocab.encode(sequence)).to(device)
    start = int(vocab.prepend_bos)
    end = tokens.size(-1) - int(vocab.append_eos)
    residue_tokens = tokens[start:end]

    logits = sequence_logits(
        model,
        vocab,
        sequence,
        mask_positions=mask_positions,
        verbose=verbose,
        max_tokens=max_tokens,
    )

    pseudo_ppl = nn.CrossEntropyLoss(reduction=reduction)(
        logits.view(-1, len(vocab)), residue_tokens
    )
    if not log:
        pseudo_ppl = pseudo_ppl.exp()
    return pseudo_ppl.item()


@torch.no_grad()
def pseudo_ppl(
    model: esm.model.ProteinBertModel,
    alphabet_or_vocab: Union[esm.data.Alphabet, Vocab],
    sequences: List[str],
    mask_positions: bool = True,
    max_tokens: int = 2 ** 14,
    log: bool = False,
):
    if not isinstance(alphabet_or_vocab, Vocab):
        vocab = Vocab.from_esm_alphabet(alphabet_or_vocab)
    else:
        vocab = alphabet_or_vocab

    model = model.cuda().eval()

    compute = partial(
        sequence_pseudo_ppl,
        model,
        vocab,
        max_tokens=max_tokens,
        mask_positions=mask_positions,
        log=log,
    )

    pseudo_ppl = []
    for sequence in tqdm(sequences):
        pppl = compute(sequence)
        pseudo_ppl.append(pppl)
    return torch.tensor(pseudo_ppl, dtype=torch.float)
