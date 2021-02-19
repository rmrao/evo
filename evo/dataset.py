from typing import List, Any
import torch


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
