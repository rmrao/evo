from typing import Tuple, List, Dict, Optional
import string
from pathlib import Path
from Bio import SeqIO
import subprocess
from .typed import PathLike


def read_sequences(
    filename: PathLike,
    remove_insertions: bool = False,
    remove_gaps: bool = False,
) -> List[Tuple[str, str]]:

    filename = Path(filename)
    if filename.suffix == ".sto":
        form = "stockholm"
    elif filename.suffix in (".fas", ".fasta", ".a3m"):
        form = "fasta"
    else:
        raise ValueError(f"Unknown file format {filename.suffix}")

    translate_dict: Dict[str, Optional[str]] = {}
    if remove_insertions:
        translate_dict.update(dict.fromkeys(string.ascii_lowercase))
    else:
        translate_dict.update(dict(zip(string.ascii_lowercase, string.ascii_uppercase)))

    if remove_gaps:
        translate_dict["-"] = None

    translate_dict["."] = None
    translate_dict["*"] = None
    translation = str.maketrans(translate_dict)

    def process_record(record: SeqIO.SeqRecord) -> Tuple[str, str]:
        description = record.description
        sequence = str(record.seq).translate(translation)
        return description, sequence

    return [process_record(rec) for rec in SeqIO.parse(str(filename), form)]


def count_sequences(seqfile: PathLike) -> int:
    num_seqs = subprocess.check_output(f'grep "^>" -c {seqfile}', shell=True)
    return int(num_seqs)
