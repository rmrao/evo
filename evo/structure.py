from typing import Optional, Sequence, NamedTuple
import enum
import operator
import numpy as np
from .typed import PathLike
from .constants import IUPAC_CODES


class PDB_SPEC(enum.Enum):
    ID = slice(0, 6)
    RESIDUE = slice(17, 20)
    RESN = slice(22, 27)
    ATOM = slice(12, 16)
    CHAIN = slice(21, 22)


class NotAnAtomLine(Exception):
    """Raised if input line is not an atom line"""
    pass


class AtomLine(NamedTuple):

    ID: str
    RESIDUE: str
    RESN: str
    ATOM: str
    CHAIN: str

    @classmethod
    def from_line(cls, line: str) -> "AtomLine":
        if line[PDB_SPEC.ID.value] == "HETATM" and line[PDB_SPEC.RESIDUE.value] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")
        elif not line[PDB_SPEC.ID.value].startswith("ATOM"):
            raise NotAnAtomLine


class Structure(object):

    def __init__(self):
        pass


    @classmethod
    def from_pdb(cls, path: PathLike, atoms: Sequence[str] = ["N", "CA", "C"], chain: Optional[str] = None) -> "Structure":
        """
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        """

        xyz, seq, min_resn, max_resn = {}, {}, np.inf, -np.inf
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in map(operator.methodcaller("rstrip"), f):

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
