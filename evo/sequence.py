from typing import List, Tuple
import re
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


def make_mutation(sequence: str, mutant: str, start_ind: int = 1) -> str:
    delimiters = [",", r"\+", ":"]
    expression = re.compile("|".join(delimiters))
    if mutant.upper() == "WT":
        return sequence
    if expression.search(mutant):
        mutants = expression.split(mutant)
        for mutant in mutants:
            sequence = make_mutation(sequence, mutant)
        return sequence
    else:
        wt, pos, mut = mutant_to_names(mutant)
        assert sequence[pos - start_ind] == wt
        return sequence[:pos - start_ind] + mut + sequence[pos - start_ind + 1:]


def create_mutant_df(sequence: str) -> pd.DataFrame:
    names = ["WT"] + get_mutants(sequence)
    sequences = [sequence] + [make_mutation(sequence, mut) for mut in names[1:]]
    return pd.DataFrame({"mutant": names, "sequence": sequences})
