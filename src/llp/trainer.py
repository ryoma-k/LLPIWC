from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base.trainers import Trainer


class MCLLPTrainer(Trainer):
    def run_iter_trn(self, *args: Any, **kargs: Any) -> tuple[Tensor, Tensor | None]:
        return self.mcllp_iter(*args, **kargs)

    def run_iter_tst(self, *args: Any, **kargs: Any) -> tuple[Tensor, Tensor | None]:
        return self.mcllp_iter(*args, **kargs)

    def mcllp_iter(
        self,
        data: dict[str, Any],
        stage: str = "trn",
        evaluate: bool = True,
        update_weight: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        inputs = self.data2device(data, ["data"])
        labels = self.data2device(
            data,
            [
                "index",
                "label",
                "labelset",
                "labelset_minus",
                "weights_log",
                "T",
                "noisy_y",
            ],
        )
        outputs = self.model(**inputs)
        with torch.no_grad():
            if evaluate:
                self.evaluator.criterion(outputs, labels, stage=stage)
            if update_weight:
                weight_kl = self.update_weight(
                    labels["index"], outputs, labels["labelset"], stage=stage
                )
                if self.summary is not None:
                    self.summary.add_scalar(f"{stage}/weight_kl", weight_kl.item())
        if stage == "trn":
            if self.state_dict.get("unfreeze_weight_epoch"):
                if (not self.unfreeze_flag) and self.epoch + 1 >= self.state_dict[
                    "unfreeze_weight_epoch"
                ]:
                    self.set_unfreeze_flag(True)
                    self.set_freeze_dataset_weight(False)
        if isinstance(self.loss_mode, list):
            loss: Tensor = torch.zeros(1)
            for loss_mode in self.loss_mode:
                loss = loss + self.evaluator(
                    outputs, labels, mode=loss_mode, stage=stage
                )
        else:
            loss = self.evaluator(outputs, labels, mode=self.loss_mode, stage=stage)
        if stage == "trn":
            self.optimizer[0].zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.state_dict["max_grad_norm"]
            )
            if self.summary is not None:
                self.summary.add_scalar("grad_norm", grad_norm.item())
            self.optimizer[0].step()
        return outputs, loss

    @property
    def unfreeze_flag(self) -> bool:
        return self._unfreeze_flag

    def set_unfreeze_flag(self, flag: bool) -> None:
        self._unfreeze_flag = flag

    def set_args(self, args: DictConfig) -> None:
        super().set_args(args)
        self.loss_mode = self.state_dict["loss_mode"]
        self.set_unfreeze_flag(False)
        self.evaluator.to(self.device)

    def update_weight(
        self, indexes: Tensor, outputs: Tensor, labelset: Tensor, stage: str = "trn"
    ) -> Tensor:
        if stage in ["trn", "init_weight"]:
            dl = self.dataloaders["trn"][0]
        else:
            dl = self.dataloaders["tst"][stage]
        ds = dl.dataset
        kl: Tensor = ds.update_weight(indexes, outputs, labelset)
        return kl

    def final_evaluation(self) -> None:
        best_pth = torch.load(Path(self.state_dict["save_path"]) / "pths/best.pth")
        dataloader = self.dataloaders["tst"]["tst"]
        self.run_dataloader(dataloader, "final_test", best_pth, True, False)
        with open(f'{self.state_dict["save_path"]}/final_score.txt', "w") as f:
            f.write("val : " + str(self.evaluator.best_score) + "\n")
            f.write("tst : " + str(self.evaluator.best_score_tst) + "\n")

    def run_dataloader(
        self,
        dataloader: DataLoader,
        stage: str,
        pth: dict | None = None,
        evaluate: bool = False,
        update_weight: bool = False,
    ) -> None:
        if pth is not None:
            self.load_resume(pth, only_model_load=True)
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, leave=False):
                output, loss = self.run_iter_tst(
                    data, stage=stage, evaluate=evaluate, update_weight=update_weight
                )
        if evaluate and self.summary is not None:
            self.summary.end_epoch(self.epoch)

    def get_state_pth(
        self,
    ) -> dict:
        state_dict = super().get_state_pth()
        state_dict["trn_dataset_weight"] = self.dataloaders["trn"][
            0
        ].dataset.weights_log
        state_dict["val_dataset_weight"] = self.dataloaders["tst"][
            "val"
        ].dataset.weights_log
        state_dict["tst_dataset_weight"] = self.dataloaders["tst"][
            "tst"
        ].dataset.weights_log
        return state_dict

    def load_dataset_weight(self, pths: dict) -> None:
        self.dataloaders["trn"][0].dataset.weights_log = pths["trn_dataset_weight"]
        self.dataloaders["tst"]["val"].dataset.weights_log = pths["val_dataset_weight"]
        self.dataloaders["tst"]["tst"].dataset.weights_log = pths["tst_dataset_weight"]

    def set_freeze_dataset_weight(self, freeze: bool = True) -> None:
        self.dataloaders["trn"][0].dataset.is_frozen_weight = freeze
        self.dataloaders["tst"]["val"].dataset.is_frozen_weight = freeze
        self.dataloaders["tst"]["tst"].dataset.is_frozen_weight = freeze

    def end_epoch(self) -> None:
        super().end_epoch()
        self.dataloaders["trn"][0].dataset.end_epoch(self.epoch)
