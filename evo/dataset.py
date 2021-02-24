from typing import List, Any, Optional, Collection, TypeVar, Sequence, Union, Tuple
import math
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np
from .tensor import collate_tensors
from .typed import PathLike
from .align import MSA
from .tokenization import Vocab


class CollatableDataset(torch.utils.data.Dataset):
    def collater(self, batch: List[Any]) -> Any:
        try:
            return torch.stack(batch, 0)
        except Exception:
            return batch


class CollatableVocabDataset(CollatableDataset):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab


class BaseWrapperDataset(CollatableVocabDataset):
    """BaseWrapperDataset. Wraps an existing dataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset to wrap.
    """

    def __init__(self, dataset: CollatableVocabDataset):
        super().__init__(dataset.vocab)
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

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = sorted(file_list)
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

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .a3m files found in {data_file}")

        self._file_list = sorted(file_list)
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


class MaxTokenBatch(object):
    def __init__(self, max_tokens: int, pad_idx: int):
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.items: List[torch.Tensor] = []
        self.sizes = None

    def can_add_item(self, item: torch.Tensor) -> bool:
        sizes = np.asarray(item.size())
        if self.sizes is not None:
            sizes = np.max([self.sizes, sizes], 0)
        total_tokens = (len(self.items) + 1) * sizes.prod()
        return total_tokens <= self.max_tokens

    def add_item(self, item: torch.Tensor):
        self.items.append(item)
        sizes = np.asarray(item.size())
        if self.sizes is None:
            self.sizes = sizes
        else:
            self.sizes = np.max([self.sizes, sizes], 0)
        if self.num_tokens > self.max_tokens:
            raise RuntimeError("Too many sequences in batch!")

    def finalize(self) -> torch.Tensor:
        return collate_tensors(self.items, constant_value=self.pad_idx)

    @property
    def num_tokens(self) -> int:
        if self.sizes is None:
            return 0
        else:
            return len(self.items) * self.sizes.prod()


BatchOrSequence = TypeVar("BatchOrSequence", MaxTokenBatch, Sequence[MaxTokenBatch])


class AutoBatchingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: CollatableVocabDataset, max_tokens: int, shuffle: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def maybe_make_and_add_batch(
        self,
        batch: Optional[BatchOrSequence],
        item: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> Tuple[BatchOrSequence, bool]:
        if batch is None:
            if isinstance(item, torch.Tensor):
                batch = MaxTokenBatch(  # type: ignore
                    self.max_tokens, self.vocab.pad_idx
                )
            else:
                batch = [  # type: ignore
                    MaxTokenBatch(self.max_tokens, self.vocab.pad_idx) for _ in item
                ]

        if isinstance(batch, MaxTokenBatch):
            can_add = batch.can_add_item(item)  # type: ignore
            if can_add:
                batch.add_item(item)  # type: ignore
        else:
            can_add = batch[0].can_add_item(item[0])  # type: ignore
            if can_add:
                for b, i in zip(batch, item):  # type: ignore
                    b.add_item(i)
        return batch, can_add  # type: ignore

    def __iter__(self):
        indices = np.arange(len(self))

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            worker_rank = dist.get_rank()
        else:
            world_size = 1
            worker_rank = 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            world_size *= worker_info.num_workers
            worker_rank = worker_rank * worker_rank.num_workers + worker_info.id

        chunk_size = math.ceil(len(indices) / world_size)
        indices = indices[chunk_size * worker_rank:chunk_size * (worker_rank + 1)]

        if self.shuffle:
            indices = np.random.permutation(indices)

        batch = None
        for idx in indices:
            items = self.dataset[idx]
            batch, added = self.maybe_make_and_add_batch(batch, items)
            if not added:
                if isinstance(batch, MaxTokenBatch):
                    yield batch.finalize()
                else:
                    yield type(items)(b.finalize() for b in batch)
                batch, added = self.maybe_make_and_add_batch(None, items)
                if not added:
                    breakpoint()
                assert added, "Item size too large to include!"
        if batch:
            if isinstance(batch, MaxTokenBatch):
                yield batch.finalize()
            else:
                yield type(items)(b.finalize() for b in batch)


class RandomCropDataset(BaseWrapperDataset):
    def __init__(self, dataset: CollatableVocabDataset, max_seqlen: int):
        super().__init__(dataset)
        self.max_seqlen = max_seqlen
        self.num_special = int(self.vocab.prepend_bos) + int(self.vocab.append_eos)
        self.max_seqlen_no_special = self.max_seqlen - self.num_special

    def __getitem__(self, idx):
        item = self.dataset[idx]
        seqlen = item.size(-1)
        if seqlen > self.max_seqlen:
            low_idx = int(self.vocab.prepend_bos)
            high_idx = seqlen - int(self.vocab.append_eos)
            start_idx = np.random.randint(low_idx, high_idx)
            end_idx = start_idx + self.max_seqlen_no_special
            item = torch.cat(
                [
                    item[..., :low_idx],
                    item[..., start_idx:end_idx],
                    item[..., high_idx:],
                ],
                -1,
            )
        return item


class SubsampleMSADataset(BaseWrapperDataset):
    def __init__(self, dataset: CollatableVocabDataset, max_tokens: int):
        super().__init__(dataset)
        self.max_tokens = max_tokens

    def __getitem__(self, idx):
        msa = self.dataset[idx]

        num_alignments, seqlen = msa.size()
        max_alignments = self.max_tokens // seqlen
        if max_alignments < num_alignments:
            indices = np.random.randint(1, num_alignments, size=max_alignments - 1)
            indices = np.append(0, indices)
            msa = msa[indices]

        return msa


class MaskedTokenWrapperDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
    ):
        # TODO - add column masking?
        # TODO - add collater
        super().__init__(dataset)
        assert 0 <= mask_prob <= 1
        assert 0 <= random_token_prob <= 1
        assert 0 <= leave_unmasked_prob <= 1

        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob

    def __getitem__(self, idx):
        item = self.dataset[idx]
        random_probs = torch.rand_like(item, dtype=torch.float)
        random_probs[(item == self.vocab.bos_idx) | (item == self.vocab.eos_idx)] = 1
        do_mask = random_probs < self.mask_prob

        tgt = item.masked_fill(~do_mask, self.vocab.pad_idx)
        mask_with_token = random_probs < (
            self.mask_prob * (1 - self.leave_unmasked_prob)
        )
        src = item.masked_fill(mask_with_token, self.vocab.mask_idx)
        mask_with_random = random_probs < (self.mask_prob * self.random_token_prob)
        # TODO - maybe prevent special tokens?
        rand_tokens = torch.randint_like(src, len(self.vocab))
        src[mask_with_random] = rand_tokens[mask_with_random]

        return src, tgt

    @property
    def mask_prob(self) -> float:
        return self._mask_prob

    @property
    def random_token_prob(self) -> float:
        return self._random_token_prob

    @property
    def leave_unmasked_prob(self) -> float:
        return self._leave_unmasked_prob
