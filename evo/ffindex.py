from typing import BinaryIO, Dict, Any
from io import StringIO
import threading
import pandas as pd
from .typed import PathLike
from .align import MSA


class FFindex(object):
    def __init__(self, index_file: PathLike, data_file: PathLike):
        super().__init__()
        self._threadlocal = threading.local()
        self._index_file = index_file
        self._data_file = data_file
        self._index = pd.read_csv(
            index_file, delimiter="\t", names=["Key", "Offset", "Length"]
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> str:
        key, offset, length = self.index.iloc[idx]
        self.data.seek(offset)
        data = self.data.read(length)
        return data.decode()[:-1]

    def __repr__(self) -> str:
        return (
            f"FFindex({self._index_file}, {self._data_file})\n"
            f"# Entries: {len(self):,}"
        )

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    @property
    def data(self) -> BinaryIO:
        if not hasattr(self._threadlocal, "data"):
            self._threadlocal.data = open(self._data_file, "rb")
        return self._threadlocal.data

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_threadlocal"}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self._threadlocal = threading.local()

    def __del__(self):
        if hasattr(self._threadlocal, "data"):
            self._threadlocal.data.close()
            del self._threadlocal.data


class MSAFFindex(object):

    def __init__(self, index_file: PathLike, data_file: PathLike):
        super().__init__()
        self.ffindex = FFindex(index_file, data_file)

    def __repr__(self) -> str:
        return f"MSA{self.ffindex}"

    def __len__(self) -> int:
        return len(self.ffindex)

    def __getitem__(self, idx: int) -> MSA:
        data = self.ffindex[idx]
        _, lines = data.split("\n", maxsplit=1)  # remove first line
        buffer = StringIO(lines)
        return MSA.from_fasta(buffer)
