import esm
from pathlib import Path
import argparse
from functools import reduce
import torch
import pandas as pd
from evo.tokenization import Vocab
from evo.parsing import read_first_sequence
from evo.align import MSA
from evo.likelihood import sequence_mutant_scores
from evo.tensor import numpy_seed


@torch.no_grad()
def score_single_sequence(sequence: str, max_tokens: int = 2 ** 14) -> pd.DataFrame:
    outputs = []
    for i in range(1, 6):
        model_name = f"esm1v_t33_650M_UR90S_{i}"
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        vocab = Vocab.from_esm_alphabet(alphabet)
        model = model.eval().cuda()
        scores = sequence_mutant_scores(
            model, vocab, sequence, verbose=True, parallel=True, max_tokens=max_tokens
        )
        scores = scores.reset_index().melt(id_vars=["mut_aa"], value_name=model_name)
        scores.insert(
            0, "mutant", scores["wt_aa"] + scores["Position"].astype(str) + scores["mut_aa"]
        )
        scores = scores.drop(columns=["wt_aa", "Position", "mut_aa"])
        outputs.append(scores)
    return reduce(pd.merge, outputs)


@torch.no_grad()
def score_msa(msa: MSA, max_tokens: int = 2 ** 14) -> pd.DataFrame:
    outputs = []
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    vocab = Vocab.from_esm_alphabet(alphabet)
    model = model.eval().cuda()
    num_samples = 5 if msa.depth > 384 else 1
    for i in range(num_samples):
        with numpy_seed(i):
            sample_msa = msa.sample_weights(384)
        scores = sequence_mutant_scores(
            model, vocab, sample_msa, verbose=True, parallel=True, max_tokens=max_tokens
        )
        scores = scores.reset_index().melt(
            id_vars=["mut_aa"], value_name=f"esm_msa1b_sample{i + 1}"
        )
        scores.insert(
            0, "mutant", scores["wt_aa"] + scores["Position"].astype(str) + scores["mut_aa"]
        )
        scores = scores.drop(columns=["wt_aa", "Position", "mut_aa"])
        outputs.append(scores)
    return reduce(pd.merge, outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--infile", required=True, type=Path, help="Path to fasta or a3m file"
    )
    parser.add_argument("--method", choices=["single", "msa", "both"], default="single")
    parser.add_argument(
        "-o", "--outdir", required=True, type=Path, help="Path to output directory"
    )
    parser.add_argument("--max_tokens", type=int, default=2 ** 14, help="Max tokens per GPU")
    args = parser.parse_args()

    run_single = args.method in ("single", "both")
    run_msa = args.method in ("msa", "both")

    if run_single:
        _, seq = read_first_sequence(args.infile)
        scores = score_single_sequence(seq, max_tokens=args.max_tokens)
        scores.to_csv(args.outdir / f"{args.infile.stem}_esm1v.csv", index=False)

    if run_msa:
        msa = MSA.from_file(args.infile, keep_insertions=True, uppercase=True)
        scores = score_msa(msa, max_tokens=args.max_tokens)
        scores.to_csv(args.outdir / f"{args.infile.stem}_msa_transformer.csv", index=False)
