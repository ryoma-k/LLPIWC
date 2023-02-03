from __future__ import annotations

import collections
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from .base.trainers import SimpleDataset


def preprocess_datasets(ds_config: DictConfig) -> None:
    load_mnist(ds_config, download=True)
    load_fashionmnist(ds_config, download=True)
    load_kmnist(ds_config, download=True)
    load_cifar10(ds_config, download=True)


def load_mnist(
    ds_config: DictConfig, download: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    root = Path(ds_config["root"])
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trn_ds = torchvision.datasets.MNIST(
        root / "MNIST", train=True, download=download, transform=transform
    )
    tst_ds = torchvision.datasets.MNIST(
        root / "MNIST", train=False, download=download, transform=transform
    )
    return (
        trn_ds.data.unsqueeze(1).float(),
        trn_ds.targets,
        tst_ds.data.unsqueeze(1).float(),
        tst_ds.targets,
    )


def load_fashionmnist(
    ds_config: DictConfig, download: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    root = Path(ds_config["root"])
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trn_ds = torchvision.datasets.FashionMNIST(
        root / "FashionMNIST", train=True, download=download, transform=transform
    )
    tst_ds = torchvision.datasets.FashionMNIST(
        root / "FashionMNIST", train=False, download=download, transform=transform
    )
    return (
        trn_ds.data.unsqueeze(1).float(),
        trn_ds.targets,
        tst_ds.data.unsqueeze(1).float(),
        tst_ds.targets,
    )


def load_kmnist(
    ds_config: DictConfig, download: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    root = Path(ds_config["root"])
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trn_ds = torchvision.datasets.KMNIST(
        root / "KMNIST", train=True, download=download, transform=transform
    )
    tst_ds = torchvision.datasets.KMNIST(
        root / "KMNIST", train=False, download=download, transform=transform
    )
    return (
        trn_ds.data.unsqueeze(1).float(),
        trn_ds.targets,
        tst_ds.data.unsqueeze(1).float(),
        tst_ds.targets,
    )


def load_cifar10(
    ds_config: DictConfig, download: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    root = Path(ds_config["root"])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trn_ds = torchvision.datasets.CIFAR10(
        root / "CIFAR10", train=True, download=download, transform=transform
    )
    tst_ds = torchvision.datasets.CIFAR10(
        root / "CIFAR10", train=False, download=download, transform=transform
    )
    return (
        torch.from_numpy(trn_ds.data.transpose(0, 3, 1, 2)).float(),
        torch.from_numpy(np.array(trn_ds.targets)),
        torch.from_numpy(tst_ds.data.transpose(0, 3, 1, 2)).float(),
        torch.from_numpy(np.array(tst_ds.targets)),
    )


def make_datasets(
    ds_config: DictConfig, seed: int, train: bool = True
) -> tuple[Dataset, Dataset, Dataset, pd.DataFrame]:
    trn_val_data, trn_val_label, tst_data, tst_label = eval(
        f'load_{ds_config["dataset_name"]}'
    )(ds_config)
    # trn-val split
    val_ratio = ds_config.val_ratio
    val_size = int(len(trn_val_data) * val_ratio)
    trn_size = len(trn_val_data) - val_size
    if ds_config.get("pretrained_weight"):
        df_path = Path(ds_config.pretrained_weight) / "split.csv"
        split_df = pd.read_csv(df_path)
        trn_val_idx = split_df["idx"].values
    else:
        trn_val_idx = np.random.RandomState(seed=seed).permutation(len(trn_val_data))
        split_df = pd.DataFrame(trn_val_idx, columns=["idx"])
        split_df.loc[split_df.index[:trn_size], "split"] = "trn"
        split_df.loc[split_df.index[trn_size:], "split"] = "val"
    trn_data = trn_val_data[trn_val_idx[:trn_size]]
    val_data = trn_val_data[trn_val_idx[trn_size:]]
    trn_label = trn_val_label[trn_val_idx[:trn_size]]
    val_label = trn_val_label[trn_val_idx[trn_size:]]
    # make datasets
    if ds_config.loss_mode == "llpfc":
        DSClass = LLPFCDataset
    else:
        DSClass = MCLLPDataset
    trn_ds = DSClass(
        ds_config=ds_config, inputs={"data": trn_data, "label": trn_label}, ds_type=0
    )
    val_ds = DSClass(
        ds_config=ds_config, inputs={"data": val_data, "label": val_label}, ds_type=1
    )
    tst_ds = DSClass(
        ds_config=ds_config, inputs={"data": tst_data, "label": tst_label}, ds_type=2
    )
    return trn_ds, val_ds, tst_ds, split_df


class MCLLPDataset(SimpleDataset):
    def __init__(self, **kargs: Any) -> None:
        super().__init__(**kargs)
        self.eps = 1e-20
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kldivloss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.is_frozen_weight = False
        self.init_multi_label()

    def __len__(self) -> int:
        return int(self.num_bag)

    def __getitem__(self, index: int) -> dict:
        output: dict[str, Any] = {}
        output["index"] = index
        for key in self.reshaped_inputs:
            output[key] = self.reshaped_inputs[key][index]
        output["labelset"] = output["label"].sort(-1).values
        output["labelset_minus"] = self.get_minus_labelset(output["labelset"])
        for key in self.keys:
            obj = getattr(self, key)
            output[key] = obj[index] if hasattr(obj, "__getitem__") else obj
        return output

    @property
    def num_bag(self) -> int:
        num_bag: int = super().__len__() // self.K
        return num_bag

    @property
    def num_class(self) -> int:
        return self.ds_config.num_class

    @property
    def K(self) -> int:
        return self.ds_config.num_instance

    def init_multi_label(self) -> None:
        self.reshaped_inputs = {}
        for key in self.inputs:
            data = self.inputs[key][: self.num_bag * self.K].reshape(
                self.num_bag, self.K, *self.inputs[key].shape[1:]
            )
            self.reshaped_inputs[key] = data
        weights = (
            F.one_hot(self.reshaped_inputs["label"], self.num_class)
            .float()
            .mean(1, keepdim=True)
            .repeat(1, self.K, 1)
        )
        self.reshaped_inputs["weights_log"] = (weights + self.eps).log()

    def get_minus_labelset(self, labelset: torch.Tensor) -> torch.Tensor:
        cnt = collections.Counter(labelset.numpy())
        num_class = self.num_class
        minus_one_labelset = [
            list(
                itertools.chain.from_iterable(
                    [[i] * (cnt[i] if i != j else cnt[i] - 1) for i in range(num_class)]
                    if cnt[j] > 0
                    else [[-1] * (self.K - 1)]
                )
            )
            for j in range(num_class)
        ]
        return torch.Tensor(minus_one_labelset).long()

    def index2reshaped_index(self, index: torch.Tensor) -> torch.Tensor:
        return index

    def update_weight(
        self, indexes: Tensor, outputs: Tensor, labelset: Tensor
    ) -> Tensor:
        weights_log = self.reshaped_inputs["weights_log"]
        with torch.no_grad():
            if self.is_frozen_weight:
                kl_div = torch.tensor(0.0)
            else:
                # labelset: bs x K
                # outputs: bs x K x classes
                log_outputs = self.log_softmax(outputs).detach().cpu()
                indexes = self.index2reshaped_index(indexes.cpu())
                num_class = log_outputs.shape[-1]
                kl_div: Tensor = self.kldivloss(
                    log_outputs.reshape(-1, num_class),
                    weights_log[indexes].reshape(-1, num_class),
                )
                weights_log[indexes] = log_outputs.reshape_as(weights_log[indexes])
        return kl_div

    @property
    def weights_log(self) -> torch.Tensor:
        return self.reshaped_inputs["weights_log"]

    @weights_log.setter
    def weights_log(self, weights_log: torch.Tensor) -> None:
        self.reshaped_inputs["weights_log"] = weights_log

    def end_epoch(self, epoch: int) -> None:
        pass


class LLPFCDataset(MCLLPDataset):
    def __init__(self, *args: Any, **kargs: Any):
        super().__init__(*args, **kargs)
        self.update_group()

    def __len__(self) -> int:
        return int(self.num_instance)

    def __getitem__(self, index: int) -> dict:
        output: dict[str, Any] = {}
        output["index"] = index
        for key in self.shuffled_group_inputs:
            bag_index, instance_index = index // self.K, index % self.K
            output[key] = self.shuffled_group_inputs[key][bag_index][instance_index][
                None
            ]
        output["labelset"] = self.shuffled_group_inputs["label"][bag_index]
        for key in self.keys:
            obj = getattr(self, key)
            output[key] = obj[index] if hasattr(obj, "__getitem__") else obj
        return output

    @property
    def num_instance(self) -> int:
        num_instance: int = self.num_available_bag * self.K
        return num_instance

    @property
    def num_available_bag(self) -> int:
        num_bag: int = self.num_bag // self.num_class * self.num_class
        return num_bag

    @property
    def num_group(self) -> int:
        num_group: int = self.num_bag // self.num_class
        return num_group

    def update_group(self) -> None:
        self.shuffle_groups()
        self.update_llpfc_params()

    def update_llpfc_params(self) -> None:
        # group_n_c: num_group x C x C
        group_n_c = (
            F.one_hot(
                self.shuffled_group_inputs["label"].reshape(
                    self.num_group, self.num_class, self.K
                ),
                self.num_class,
            )
            .float()
            .sum(2)
        )
        # gammas: g_bs x C x C
        # alphas: g_bs x C
        # sigmas: g_bs x C
        gammas = group_n_c / group_n_c.sum(-1, keepdim=True)
        sigmas = gammas.mean(1)
        alphas = torch.ones_like(sigmas) / self.num_class
        # Ts: g_bs x C x C
        Ts = gammas * alphas[:, :, None] / (sigmas[:, None] + self.eps)
        self.shuffled_group_inputs["T"] = (
            Ts[:, None, None]
            .expand(
                self.num_group, self.num_class, self.K, self.num_class, self.num_class
            )
            .reshape(self.num_available_bag, self.K, self.num_class, self.num_class)
        )
        self.shuffled_group_inputs["noisy_y"] = (
            torch.arange(self.num_class)[None, :, None]
            .expand(self.num_group, self.num_class, self.K)
            .reshape(-1, self.K)
        )

    def shuffle_groups(self) -> None:
        rand_bag_idxs = torch.randperm(self.num_bag)[: self.num_available_bag]
        self.shuffled_group_inputs = {}
        for key in self.inputs:
            data = self.reshaped_inputs[key][rand_bag_idxs]
            self.shuffled_group_inputs[key] = data
        self.rand_bag_idxs = rand_bag_idxs

    def index2reshaped_index(
        self, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rand_bag_idxs[index // self.K], index % self.K

    def end_epoch(self, epoch: int) -> None:
        if epoch % self.ds_config.regroup_epoch == 0:
            self.update_group()
