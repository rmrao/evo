from typing import List, Tuple
from itertools import product
import pandas as pd

_FASTA_VOCAB = "ARNDCQEGHILKMFPSTWYV"


def get_mutants(sequence: str) -> List[str]:
    mutants = []
    for (i, wt), mut in product(enumerate(sequence), _FASTA_VOCAB):
        if wt == mut:
            continue
        mutant = f"{wt}{i + 1}{mut}"
        mutants.append(mutant)
    return mutants


def mutant_to_names(mutant: str) -> Tuple[str, int, str]:
    return mutant[0], int(mutant[1:-1]), mutant[-1]


def make_mutation(sequence: str, mutant: str) -> str:
    if "," in mutant:
        mutants = ",".split(mutant)
        for mutant in mutants:
            sequence = make_mutation(sequence, mutant)
        return sequence
    else:
        wt, pos, mut = mutant_to_names(mutant)
        assert sequence[pos - 1] == wt
        return sequence[:pos - 1] + mut + sequence[pos:]


def create_mutant_df(sequence: str) -> pd.DataFrame:
    names = ["WT"] + get_mutants(sequence)
    sequences = [sequence] + [make_mutation(sequence, mut) for mut in names[1:]]
    return pd.DataFrame({"mutant": names, "sequence": sequences})
