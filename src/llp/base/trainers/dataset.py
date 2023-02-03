from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class DL_sample:
    """
    this sampler doesn't stop at its end of iteration.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.iterator = iter(dataloader)

    def __iter__(self) -> DL_sample:
        return self

    def __next__(self) -> Iterable:
        try:
            return next(self.iterator)
        except StopIteration:
            self.reset()
            return next(self.iterator)

    def __len__(self) -> int:
        return len(self.dataloader)

    def reset(self) -> None:
        self.iterator = iter(self.dataloader)


class SimpleDataset(Dataset):
    def __init__(
        self,
        inputs: dict,
        transforms: Callable[[Tensor], Tensor] | None = None,
        ds_config: DictConfig | None = None,
        **args: Any,
    ) -> None:
        self.inputs = inputs
        self.transforms = transforms
        self.ds_config = DictConfig({}) if ds_config is None else ds_config
        self.keys = []
        for key in args:
            setattr(self, key, args[key])
            self.keys.append(key)

    def __len__(self) -> int:
        key = list(self.inputs.keys())[0]
        return len(self.inputs[key])

    def __getitem__(self, index: int) -> dict:
        data: dict[str, Any] = {}
        data["index"] = index
        for key in self.inputs:
            data[key] = self.inputs[key][index]
        if self.transforms is not None:
            data["inputs"] = self.transforms(data["inputs"])
        for key in self.keys:
            obj = getattr(self, key)
            data[key] = obj[index] if hasattr(obj, "__getitem__") else obj
        return data
