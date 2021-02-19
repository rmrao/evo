from typing import Sequence, TypeVar
import torch
import numpy as np


TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)


def collate_tensors(
    sequences: Sequence[TensorLike], constant_value=0, dtype=None
) -> TensorLike:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array
