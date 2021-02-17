from typing import Dict, Optional
import torch
import tape
import transformers
import numpy as np
import esm
from copy import copy
import logging

logger = logging.getLogger(__name__)


class Vocab(object):
    def __init__(
        self,
        tokens: Dict[str, int],
        bos_token: str = "<cls>",
        eos_token: str = "<sep>",
        unk_token: Optional[str] = None,
        pad_token: str = "<pad>",
        mask_token: Optional[str] = None,
        prepend_bos: bool = True,
        append_eos: bool = True,
    ):
        if prepend_bos and bos_token not in tokens:
            raise KeyError(f"bos token '{bos_token}' not in input tokens.")
        if append_eos and eos_token not in tokens:
            raise KeyError(f"eos token '{eos_token}' not in input tokens.")
        if unk_token is not None and unk_token not in tokens:
            raise KeyError(f"unk token '{unk_token}' not in input tokens.")
        if pad_token not in tokens:
            raise KeyError(f"pad token '{pad_token}' not in input tokens.")
        if mask_token is not None and mask_token not in tokens:
            raise KeyError(f"mask token '{mask_token}' not in input tokens.")

        # prevent modifications to original dictionary from having an effect.
        tokens = copy(tokens)
        for tok in list(tokens.keys()):
            if len(tok) > 1 and tok not in {
                bos_token,
                eos_token,
                unk_token,
                mask_token,
                pad_token,
            }:
                logger.warning(f"Removing non-special token of length > 1: {tok}")
                tokens.pop(tok)

        self.tokens_to_idx = tokens
        self.tokens = list(tokens.keys())
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.bos_idx = tokens[bos_token]
        self.eos_idx = tokens[eos_token]
        self.pad_idx = tokens[pad_token]

        self.allow_unknown = unk_token is not None
        if unk_token is not None:
            self.unk_idx = tokens[unk_token]

        self.allow_mask = mask_token is not None
        if mask_token is not None:
            self.mask_idx = tokens[mask_token]

        self.uint8_symbols = np.sort(
            np.array([tok for tok in self.tokens if len(tok) == 1], dtype="|S1").view(
                np.uint8
            )
        )
        self.numpy_indices = torch.tensor(
            [self.index(chr(tok)) for tok in self.uint8_symbols],
            dtype=np.long,
        )

    def index(self, token: str) -> int:
        return self.tokens_to_idx[token]

    def token(self, index: int) -> str:
        return self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def to_dict(self) -> Dict[str, int]:
        return copy(self.tokens_to_idx)

    def encode(self, sequence: str) -> torch.Tensor:
        locs = np.digitize(
            np.array(list(sequence), dtype="|S1").view(np.uint8),
            self.uint8_symbols,
            right=True,
        )
        indices = self.numpy_indices[locs]
        if self.prepend_bos:
            indices = np.append(self.bos_idx, indices)
        if self.append_eos:
            indices = np.append(indices, self.eos_idx)
        return torch.from_numpy(indices)

    @classmethod
    def from_esm_alphabet(cls, alphabet: esm.data.Alphabet) -> "Vocab":
        return cls(
            tokens=alphabet.tok_to_idx,
            bos_token="<cls>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            prepend_bos=alphabet.prepend_bos,
            append_eos=alphabet.append_eos,
        )

    @classmethod
    def from_tape_tokenizer(cls, tokenizer: tape.tokenizers.TAPETokenizer) -> "Vocab":
        if "<unk>" in tokenizer.vocab:
            unk_token: Optional[str] = "<unk>"
        elif "X" in tokenizer.vocab:
            unk_token = "X"
        else:
            unk_token = None

        return cls(
            tokens=tokenizer.vocab,
            bos_token=tokenizer.start_token,
            eos_token=tokenizer.stop_token,
            unk_token=unk_token,
            pad_token="<pad>",
            mask_token=tokenizer.mask_token,
            prepend_bos=True,
            append_eos=True,
        )

    @classmethod
    def from_huggingface_tokenizer(
        cls, tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    ) -> "Vocab":
        return cls(
            tokens=tokenizer.get_vocab(),
            bos_token=tokenizer.cls_token,
            eos_token=tokenizer.sep_token,
            unk_token=tokenizer.unk_token,
            pad_token=tokenizer.pad_token,
            mask_token=tokenizer.mask_token,
            prepend_bos=tokenizer.cls_token is not None,
            append_eos=tokenizer.sep_token is not None,
        )


def test_encode_sequence():
    sequence = "LFKLGAENIFLGRKAATKEEAIRFAGEQLVKGGYVEPEYVQAMLDREKLTPTYLGESIAVPHGTVEAK"
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    vocab = Vocab.from_esm_alphabet(alphabet)
    batch_converter = alphabet.get_batch_converter()
    _, _, esm_tokens = batch_converter([("", sequence)])
    evo_tokens = vocab.encode(sequence)[None]
    assert (esm_tokens == evo_tokens).all()
