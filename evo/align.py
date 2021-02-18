from typing import List, Tuple, Union, Iterator, Sequence, TextIO
import contextlib
import math
import tempfile
import re
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.distance import squareform, pdist
from Bio import SeqIO
from Bio.Seq import Seq
from .typed import PathLike


class MSA:
    """Class that represents a multiple sequence alignment."""

    def __init__(
        self,
        sequences: List[Tuple[str, str]],
        seqid_cutoff: float = 0.2,
    ):
        self.headers = [header for header, _ in sequences]
        self.sequences = [seq for _, seq in sequences]
        self._seqlen = len(self.sequences[0])
        self._depth = len(self.sequences)
        self.seqid_cutoff = seqid_cutoff
        assert all(
            len(seq) == self._seqlen for seq in self.sequences
        ), "Seqlen Mismatch!"

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(self.headers, self.sequences)

    def select(self, indices: Sequence[int], axis: str = "seqs") -> "MSA":
        assert axis in ("seqs", "positions")
        if axis == "seqs":
            data = [(self.headers[idx], self.sequences[idx]) for idx in indices]
            return self.__class__(data)
        else:
            data = [
                (header, "".join(seq[idx] for idx in indices)) for header, seq in self
            ]
            return self.__class__(data)

    def filter_coverage(self, threshold: float, axis: str = "seqs") -> "MSA":
        assert 0 <= threshold <= 1
        assert axis in ("seqs", "positions")
        notgap = self.array != self.gap
        match = notgap.mean(1 if axis == "seqs" else 0)
        indices = np.where(match >= threshold)[0]
        return self.select(indices, axis=axis)

    def hhfilter(
        self,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20.0,
        binary: str = "hhfilter",
    ) -> "MSA":

        with tempfile.TemporaryDirectory(dir="/dev/shm") as tempdirname:
            tempdir = Path(tempdirname)
            fasta_file = tempdir / "input.fasta"
            fasta_file.write_text(
                "\n".join(f">{i}\n{seq}" for i, seq in enumerate(self.sequences))
            )
            output_file = tempdir / "output.fasta"
            command = " ".join(
                [
                    f"{binary}",
                    f"-i {fasta_file}",
                    "-M a3m",
                    f"-o {output_file}",
                    f"-id {seqid}",
                    f"-diff {diff}",
                    f"-cov {cov}",
                    f"-qid {qid}",
                    f"-qsc {qsc}",
                ]
            ).split(" ")
            result = subprocess.run(command, capture_output=True)
            result.check_returncode()
            with output_file.open() as f:
                indices = [int(line[1:].strip()) for line in f if line.startswith(">")]
            return self.select(indices, axis="seqs")

    @property
    def gap(self) -> Union[bytes, int]:
        return b"-" if self.dtype == np.dtype("S1") else ord("-")

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
    def seqid_cutoff(self) -> float:
        return self._seqid_cutoff

    @seqid_cutoff.setter
    def seqid_cutoff(self, value: float) -> None:
        assert 0 <= value <= 1
        if getattr(self, "_seqid_cutoff", None) != value:
            with contextlib.suppress(AttributeError):
                delattr(self, "_weights")
            with contextlib.suppress(AttributeError):
                delattr(self, "_neff")
        self._seqid_cutoff = value

    @property
    def is_covered(self) -> np.ndarray:
        if not hasattr(self, "_is_covered"):
            self._is_covered = (self.array[1:] != self.gap).any(0)
        return self._is_covered

    @property
    def coverage(self) -> float:
        if not hasattr(self, "_coverage"):
            self._coverage = self.is_covered.mean()
        return self._coverage

    @property
    def weights(self) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = 1 / (
                squareform(pdist(self.array, "hamming")) < self.seqid_cutoff
            ).sum(1)
        return self._weights

    def neff(self, normalization: Union[float, str] = "none") -> float:
        if isinstance(normalization, str):
            assert normalization in ("none", "sqrt", "seqlen")
            normalization = {
                "none": 1,
                "sqrt": math.sqrt(self.seqlen),
                "seqlen": self.seqlen,
            }[normalization]
        if not hasattr(self, "_neff"):
            self._neff = self.weights.sum()
        return self._neff / normalization

    @classmethod
    def from_stockholm(
        cls,
        stofile: Union[PathLike, TextIO],
        keep_insertions: bool = False,
        **kwargs,
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
        return cls(output, **kwargs)

    @classmethod
    def from_fasta(
        cls,
        fasfile: Union[PathLike, TextIO],
        **kwargs,
    ) -> "MSA":

        output = []
        for record in SeqIO.parse(fasfile, "fasta"):
            description = record.description
            sequence = str(record.seq)
            sequence = re.sub(r"([a-z]|\.|\*)", "", sequence)
            output.append((description, sequence))
        return cls(output, **kwargs)

    @classmethod
    def from_file(
        cls,
        alnfile: PathLike,
        keep_insertions: bool = False,
        **kwargs,
    ) -> "MSA":
        filename = Path(alnfile)
        if filename.suffix == ".sto":
            return cls.from_stockholm(filename, keep_insertions, **kwargs)
        elif filename.suffix in (".fas", ".fasta", ".a3m"):
            assert not keep_insertions
            return cls.from_fasta(filename, **kwargs)
        else:
            raise ValueError(f"Unknown file format {filename.suffix}")

    def write(self, outfile: PathLike, form: str = "fasta") -> None:
        SeqIO.write(
            (SeqIO.SeqRecord(Seq(seq), description=header) for header, seq in self),
            outfile,
            form,
        )
