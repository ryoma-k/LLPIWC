from __future__ import annotations

from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from torch import Tensor

from ..utils import Summary


class Evaluator:
    def __init__(self, lambdas: DictConfig, summary: Summary | None = None):
        super().__init__()
        self.lambdas = lambdas
        self.summary = summary
        self.best_score = -1.0
        self.best_score_tst = -1.0

    def __call__(
        self,
        *inputs: Tensor | dict[str, Tensor],
        mode: str,
        stage: str,
        lam_mode: str = "",
        note: str = "",
        write: bool = True,
        **args: Any,
    ) -> Tensor:
        _lambda = self.lambdas.get(f"lambda_{mode}{lam_mode}", 1.0)
        try:
            func = eval(f"self.{mode}")
        except AttributeError:
            print(f"mode : {mode} does not exist.")
            raise Exception
        loss_dict: dict[str, Tensor] = func(*inputs, mode=mode, stage=stage, **args)
        loss = loss_dict["loss"]
        if (
            self.lambdas.debug
            and loss.numel() == 1
            and (torch.isnan(loss) or loss == float("inf"))
        ):
            import pdb

            pdb.set_trace()
        loss = loss * _lambda
        if len(lam_mode):
            lam_mode = "_" + lam_mode
        if len(note):
            note = "_" + note
        if write and self.summary is not None:
            for key, value in loss_dict.items():
                self.summary.add_scalar(
                    f"{stage}/{key}_{mode}{lam_mode}{note}", value.item()
                )
        return loss

    def log(
        self, value: Tensor, mode: str, stage: str, note: str = "", **kargs: Any
    ) -> None:
        if self.summary is not None:
            self.summary.add_scalar(f"{stage}/{mode}{note}", value.item(), **kargs)

    def accuracy(self, output: Tensor, label: Tensor, ignore: int = -100) -> Tensor:
        assert output[..., 0].shape == label.shape
        return (output.argmax(-1) == label)[label != ignore].float().mean()

    def criterion(self, outputs: Tensor, targets: dict, stage: str) -> None:
        pass

    def end_epoch(self, epoch: int) -> None:
        pass

    def is_best_score(self) -> bool:
        return False

    def reset_best_scores(self) -> None:
        self.best_score = -1.0
        self.best_score_tst = -1.0

    def to(self, device: torch._C.device) -> None:
        pass
