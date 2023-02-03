from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Summary:
    def __init__(self, log_dir: str | None) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self.counter_epoch: dict[str, int] = {}
        self.counter_total: dict[str, int] = {}
        self.value_sum_epoch: dict[str, float] = {}
        self.last_mean_vs: dict[str, float] = {}
        self.epoch = 0
        self.epoch_values: defaultdict[str, list] = defaultdict(list)

    def add_scalar(
        self, name: str, value: float, num: int = 1, take_average: bool = True
    ) -> None:
        """
        stores scalar value and takes average.
        name: type of scalar, e.g.) "loss"
        value: scalar
        num: if the value has already taken averaged, set this valuable to its original size.
        """
        if take_average:
            self.counter_epoch[name] = self.counter_epoch.get(name, 0) + num
        self.counter_total[name] = self.counter_total.get(name, 0) + 1
        self.value_sum_epoch[name] = self.value_sum_epoch.get(name, 0.0) + value * num
        if self.writer is not None:
            self.writer.add_scalar(name, value, self.counter_total[name])

    def add_text(self, *args: str) -> None:
        if self.writer is not None:
            self.writer.add_text(*args)

    def end_epoch(self, epoch: int) -> dict:
        """
        call this method at the end of the epoch.
        """
        mean_vs = {}
        for key in self.counter_epoch:
            v = self.value_sum_epoch[key]
            c = self.counter_epoch[key]
            if self.writer is not None:
                self.writer.add_scalar("ep/" + key, v / max(c, 1e-12), epoch)
            self.epoch_values[key].append((epoch, v / max(c, 1e-12)))
            mean_vs[key] = v / max(c, 1e-12)
        for key in self.value_sum_epoch:
            if key not in self.counter_epoch:
                mean_vs[key] = self.value_sum_epoch[key]
        self.counter_epoch = {}
        self.value_sum_epoch = {}
        self.last_mean_vs = mean_vs
        self.epoch = epoch
        return mean_vs

    def add_scalar_ep(self, name: str, value: float, epoch: int) -> None:
        """
        call this method when you don't want to calculate value means at ends of epochs.
        """
        if self.writer is not None:
            self.writer.add_scalar("ep/" + name, value, epoch)
        self.value_sum_epoch[name] = value

    def state_dict(self) -> dict:
        return {
            "counter_epoch": self.counter_epoch,
            "counter_total": self.counter_total,
            "value_sum_epoch": self.value_sum_epoch,
            "last_mean_vs": self.last_mean_vs,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)

    def get_epoch_df(self, key: str) -> pd.DataFrame:
        return pd.DataFrame(self.epoch_values[key], columns=["epoch", "values"])
