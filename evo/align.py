from typing import List, Tuple, Dict, Optional, Union, TextIO
import re
import numpy as np
from pathlib import Path
from Bio import SeqIO
from .typed import PathLike


class MSA:
    """ Class that represents a multiple sequence alignment.
    """

    def __init__(self, sequences: List[Tuple[str, str]]):
        self.headers = [header for header, _ in sequences]
        self.sequences = [seq for _, seq in sequences]
        self._seqlen = len(self.sequences[0])
        self._depth = len(self.sequences)
        assert all(len(seq) == self._seqlen for seq in self.sequences), \
            "Seqlen Mismatch!"

    def __repr__(self) -> str:
        return (
            f"MSA, L: {self.seqlen}, N: {self.depth}\n"
            f"{self.array}"
        )

    @property
    def array(self) -> np.ndarray:
        if not hasattr(self, "_array"):
            self._array = np.array([list(seq) for seq in self.sequences], dtype="|S1")
        return self._array

    @property
    def dtype(self) -> type:
        return self._array.dtype

    @dtype.setter
    def dtype(self, value: type) -> None:
        assert value in (np.uint8, np.dtype("S1"))
        self._array = self._array.view(value)

    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def is_covered(self) -> np.ndarray:
        match = b"-" if self.dtype == np.dtype("S1") else ord("-")
        if not hasattr(self, "_is_covered"):
            self._is_covered = (self.array[1:] != match).any(0)
        return self._is_covered

    @property
    def coverage(self) -> float:
        if not hasattr(self, "_coverage"):
            self._coverage = self.is_covered.mean()
        return self._coverage

    @classmethod
    def from_stockholm(
        cls,
        stofile: Union[PathLike, TextIO],
        keep_insertions: bool = False,
    ) -> "MSA":

        output = []
        valid_indices = None
        for record in SeqIO.parse(stofile, "stockholm"):
            description = record.description
            sequence = str(record.seq)
            if not keep_insertions:
                if valid_indices is None:
                    valid_indices = [i for i, aa in enumerate(sequence) if aa != "-"]
                sequence = "".join(sequence[idx] for idx in valid_indices)
            output.append((description, sequence))
        return cls(output)

    @classmethod
    def from_fasta(
        cls,
        fasfile: Union[PathLike, TextIO],
    ) -> "MSA":

        output = []
        for record in SeqIO.parse(fasfile, "fasta"):
            description = record.description
            sequence = str(record.seq)
            sequence = re.sub(r"([a-z]|\.|\*)", "", sequence)
            output.append((description, sequence))
        return cls(output)

    @classmethod
    def from_file(
        cls,
        alnfile: PathLike,
        keep_insertions: bool = False,
    ) -> "MSA":
        filename = Path(alnfile)
        if filename.suffix == ".sto":
            return cls.from_stockholm(filename, keep_insertions)
        elif filename.suffix in (".fas", ".fasta", ".a3m"):
            assert not keep_insertions
            return cls.from_fasta(filename)
        else:
            raise ValueError(f"Unknown file format {filename.suffix}")
