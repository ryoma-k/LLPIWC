from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.optim import lr_scheduler as lrs
from tqdm import tqdm

from ..utils import Summary
from .evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloaders: dict[str, Any],
        evaluator: Evaluator,
        summary: Summary | None,
        config: DictConfig,
        save: bool = True,
    ) -> None:
        self.model = model
        self.device: torch._C.device = next(iter(self.model.parameters())).device
        self.dataloaders = dataloaders
        self.evaluator = evaluator
        self.summary = summary
        self.state_dict: dict[Any, Any] = dict()
        self.set_args(config)
        self.save = save
        self.iteration, self.ini_iter, self.epoch, self.best_iteration = 0, 0, 0, 0

    def set_args(self, args: DictConfig) -> None:
        self.state_dict.update(args)
        if self.state_dict.get("epochs"):
            self.state_dict["iterations"] = (
                len(self.dataloaders["trn"][0]) * self.state_dict["epochs"]
            )
            self.state_dict["test_freq"] = len(self.dataloaders["trn"][0])
            self.state_dict["scheduler_step_freq"] = len(self.dataloaders["trn"][0])
        else:
            self.state_dict["scheduler_step_freq"] = 1
        self.set_optimizer()
        self.set_scheduler()

    def run_iter_trn(self, *args: Any, **kargs: Any) -> tuple[Tensor, Tensor | None]:
        return self.classify_iter(*args, **kargs)

    def run_iter_tst(self, *args: Any, **kargs: Any) -> tuple[Tensor, Tensor | None]:
        return self.classify_iter(*args, **kargs)

    def classify_iter(self, data: dict, stage: str = "trn") -> tuple[Tensor, Tensor]:
        image = data["img"].to(self.device).float()
        label = data["label"].to(self.device).float()
        output = self.model(image)
        loss = self.evaluator(output, label, mode="class", stage=stage)
        with torch.no_grad():
            self.evaluator.criterion(output, label, stage=stage)
        if stage == "trn":
            self.optimizer[0].zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.state_dict["max_grad_norm"]
            )
            if self.summary is not None:
                self.summary.add_scalar("grad_norm", grad_norm.item())
            self.optimizer[0].step()
        return output, loss

    def run(self) -> None:
        data = None
        for iteration in tqdm(
            range(self.ini_iter, self.state_dict["iterations"]),
            dynamic_ncols=True,
            smoothing=0.01,
        ):
            self.iteration = iteration
            # TRAIN
            self.model.train()
            with torch.set_grad_enabled(True):
                if self.state_dict["scheduler"] != "lr_find" or data is None:
                    data = self.load_data()
                output, loss = self.run_iter_trn(data, stage="trn")
                if (
                    len(self.scheduler) > 0
                    and (iteration + 1) % self.state_dict["scheduler_step_freq"] == 0
                ):
                    self.scheduler[0].step()
                    if self.summary is not None:
                        self.summary.add_scalar(
                            "lr",
                            self.scheduler[0].get_last_lr()[-1],
                            take_average=False,
                        )
            # TEST
            if (iteration + 1) % self.state_dict["test_freq"] == 0:
                self.model.eval()
                with torch.no_grad():
                    dls = self.dataloaders["tst"]
                    for key in dls:
                        for data in tqdm(dls[key], leave=False):
                            output, loss = self.run_iter_tst(data, stage=key)
                self.end_epoch()
                if self.summary is not None:
                    self.summary.end_epoch(self.epoch)
                if self.evaluator.is_best_score():
                    self.save_state_pth("best")
                    self.best_iteration = iteration
                if self.state_dict.get("early_stopping_rounds"):
                    if (
                        iteration - self.best_iteration
                        >= self.state_dict["early_stopping_rounds"]
                        * self.state_dict["test_freq"]
                    ):
                        print(
                            f"\nvalidation score didn't improve for {self.state_dict['early_stopping_rounds']} rounds"
                        )
                        print(f"early stopping at iteration {iteration}")
                        break
            if (iteration + 1) % self.state_dict.get("save_freq", 5000) == 0:
                self.save_state_pth(iteration)

    def load_data(self, stage: str = "trn") -> dict[str, Tensor]:
        data = {}
        dl_datas = [next(dl) for dl in self.dataloaders[stage]]
        if len(dl_datas) == 1:
            data = dl_datas[0]
        else:
            keys = dl_datas[0].keys()
            for key in keys:
                data[key] = torch.cat([dl_data[key] for dl_data in dl_datas], 0)
        return data

    def data2device(
        self, data: dict, keys: list[str], warning: bool = False
    ) -> dict[str, torch.Tensor]:
        use_keys = list(filter(lambda x: x in data.keys(), keys))
        if warning and len(use_keys) != keys:
            print(f"key {set(keys) - set(use_keys)} missing")
        return {key: data[key].to(self.device) for key in use_keys}

    def set_optimizer(self, optim_state: dict | None = None) -> None:
        if optim_state is None:
            optim_state = self.state_dict["optim_state"]
        parameters = [self.model.parameters()]
        self.optimizer = []
        for param in parameters:
            if hasattr(optim, self.state_dict["optimizer"]):
                self.optimizer.append(
                    eval(f'optim.{self.state_dict["optimizer"]}(param, **optim_state)')
                )
            else:
                raise Exception

    def set_scheduler(self, scheduler_state: dict | None = None) -> None:
        self.scheduler: list[lrs._LRScheduler] = []
        if self.state_dict["scheduler"] is None:
            return
        if scheduler_state is None:
            scheduler_state = self.state_dict.get("scheduler_state", {})
        assert isinstance(scheduler_state, (dict, DictConfig))
        for optimizer in self.optimizer:
            if self.state_dict["scheduler"] == "lr_find":
                self.scheduler.append(
                    lrs.LambdaLR(optimizer, lr_lambda=lambda x: float(np.exp(x / 10)))
                )
            elif self.state_dict["scheduler"] == "Lambda":
                lr_lambda: list[str] = scheduler_state["lr_lambda"]
                self.scheduler.append(
                    lrs.LambdaLR(
                        optimizer,
                        **scheduler_state,
                        lr_lambda=[eval(lr_lam) for lr_lam in lr_lambda],
                    )
                )
            else:
                scheduler = eval(f'lrs.{self.state_dict["scheduler"]}')
                self.scheduler.append(scheduler(optimizer, **scheduler_state))

    def end_epoch(self) -> None:
        self.evaluator.end_epoch(self.epoch)
        self.epoch += 1

    def get_state_pth(
        self,
    ) -> dict:
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": [opt.state_dict() for opt in self.optimizer],
            "scheduler": [sch.state_dict() for sch in self.scheduler],
            "summary": self.summary.state_dict() if self.summary is not None else None,
            "iteration": self.iteration,
            "epoch": self.epoch,
        }
        return state_dict

    def save_state_pth(
        self,
        name: str | int,
        state_dict: dict[str, Any] | None = None,
    ) -> None:
        if state_dict is None:
            state_dict = self.get_state_pth()
        torch.save(
            state_dict, Path(self.state_dict["save_path"]) / "pths" / f"{name}.pth"
        )

    def load_resume(self, pths: dict, only_model_load: bool = False) -> None:
        self.model.load_state_dict(pths["model"])
        if only_model_load:
            return
        for i in range(len(self.optimizer)):
            self.optimizer[i].load_state_dict(pths["optimizer"][i])
        for i in range(len(self.scheduler)):
            self.scheduler[i].load_state_dict(pths["scheduler"][i])
        if self.summary is not None:
            self.summary.load_state_dict(pths["summary"])
        self.ini_iter = pths["iteration"] + 1
        self.epoch = pths["epoch"]

    def set_lr(self, lr: float, set_initial: bool = True) -> None:
        for optimizer in self.optimizer:
            for param in optimizer.param_groups:
                param["lr"] = lr
                if set_initial:
                    param["initial_lr"] = lr
        self.set_scheduler()
