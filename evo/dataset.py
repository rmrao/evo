from typing import List, Any, Optional, Collection
from pathlib import Path
import torch
import numpy as np
from .typed import PathLike
from .align import MSA


class CollatableDataset(torch.utils.data.Dataset):
    def collater(self, batch: List[Any]) -> Any:
        try:
            return torch.stack(batch, 0)
        except Exception:
            return batch


class BaseWrapperDataset(CollatableDataset):
    """BaseWrapperDataset. Wraps an existing dataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset to wrap.
    """

    def __init__(self, dataset: CollatableDataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def collater(self, batch: List[Any]) -> Any:
        return self.dataset.collater(batch)


class NPZDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        lazy: bool = False,
    ):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.npz")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.name in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = file_list
        self._lazy = lazy

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = np.load(self._file_list[index])
        if not self._lazy:
            item = dict(item)
        return item


class A3MDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of a3m files.
    Args:
        data_file (Union[str, Path]): Path to directory of a3m files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        max_seqs_per_msa: Optional[int] = None,
        sample_method: str = "fast",
    ):
        assert sample_method in ("fast", "best")
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.a3m")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.name in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .a3m files found in {data_file}")

        self._file_list = file_list
        self._max_seqs_per_msa = max_seqs_per_msa
        self._sample_method = sample_method

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        msa = MSA.from_fasta(self._file_list[index])
        if self._max_seqs_per_msa is not None:
            msa = msa.select_diverse(self._max_seqs_per_msa, method=self._sample_method)
        return msa
