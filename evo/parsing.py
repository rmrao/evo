from typing import Tuple, List, Dict, Optional
import string
from pathlib import Path
from Bio import SeqIO
import subprocess
from .typed import PathLike
from .constants import IUPAC_CODES
import numpy as np


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
    "Utility for quickly counting sequences in a fasta/a3m file."
    num_seqs = subprocess.check_output(f'grep "^>" -c {seqfile}', shell=True)
    return int(num_seqs)


def parse_PDB(x, atoms=["N", "CA", "C"], chain=None):
    """
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    xyz, seq, min_resn, max_resn = {}, {}, np.inf, -np.inf
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12 : 12 + 4].strip()
                resi = line[17 : 17 + 3]
                resn = line[22 : 22 + 5].strip()
                x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])
                breakpoint()

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    for resn in range(min_resn, max_resn + 1):
        if resn in seq:
            for k in sorted(seq[resn]):
                seq_.append(IUPAC_CODES.get(seq[resn][k].capitalize(), "X"))
        else:
            seq_.append("X")
        if resn in xyz:
            for k in sorted(xyz[resn]):
                for atom in atoms:
                    if atom in xyz[resn][k]:
                        xyz_.append(xyz[resn][k][atom])
                    else:
                        xyz_.append(np.full(3, np.nan))
        else:
            for atom in atoms:
                xyz_.append(np.full(3, np.nan))

    valid_resn = np.array(sorted(xyz.keys()))
    return np.array(xyz_).reshape(-1, len(atoms), 3), "".join(seq_), valid_resn
