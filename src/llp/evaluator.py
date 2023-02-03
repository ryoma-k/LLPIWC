from __future__ import annotations

import itertools
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from .base.trainers import Evaluator
from .base.utils import Summary
from .sinkhorn import batch_sinkhorn


class LogFactorial(nn.Module):
    def __init__(self, max_K: int) -> None:
        super().__init__()
        self.max_K = max_K + 1
        log_Ks = torch.arange(self.max_K).log()
        log_Ks[0] = 0
        log_Ks = log_Ks.cumsum(dim=0)
        self.register_buffer("log_Ks", log_Ks)

    def forward(self, Ks: Tensor) -> Tensor:
        return self.log_Ks.expand(*Ks.shape[:-1], self.max_K).gather(-1, Ks)


class MCLLPEvaluator(Evaluator):
    def __init__(self, lambdas: DictConfig, summary: Summary):
        super().__init__(lambdas, summary)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.crossent = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        # perm_idx: H x K-1
        # comb_idx: K x K-1
        if self.lambdas.K <= 8:
            self.perm_idx = torch.tensor(
                list(
                    itertools.permutations(
                        range(self.lambdas.K - 1), self.lambdas.K - 1
                    )
                )
            )
        else:
            self.perm_idx = None
        self.comb_idx = torch.tensor(
            list(itertools.combinations(range(self.lambdas.K), self.lambdas.K - 1))[
                ::-1
            ]
        )
        self.eps = 1e-20
        self.inf = 1e20
        self.logfactorial = LogFactorial(self.lambdas.K)

    def to(self, device: torch._C.device) -> None:
        if self.perm_idx is not None:
            self.perm_idx = self.perm_idx.to(device)
        self.comb_idx = self.comb_idx.to(device)
        self.logfactorial.to(device)

    def calc_weight_other_log(self, weights_log, targets: dict) -> Tensor:
        # weights_log: bs x K x class
        # labelset_minus: bs x class x K-1
        labelset_minus = targets["labelset_minus"]
        inf_fill_mask = (labelset_minus == -1).any(-1).clone()
        labelset_minus[labelset_minus == -1] = 0
        # weights_log: bs x K x class -> bs x K x K x class
        weights_log = weights_log[:, None].expand(-1, self.lambdas.K, -1, -1)
        # weights_log: bs x K x K x class -> bs x K x K-1 x class
        # comb_idx: K x K-1
        weights_log = weights_log.gather(
            dim=2,
            index=self.comb_idx[None, :, :, None].expand(
                len(weights_log), -1, -1, self.lambdas.num_class
            ),
        )
        # weights_log: bs x K x K-1 x class -> bs x K x class x K-1 x class
        weights_log = weights_log[:, :, None].expand(
            -1, -1, self.lambdas.num_class, -1, -1
        )
        # weights_other_log: bs x K x class x K-1 x class -> bs x K x class x K-1 x K-1
        weights_other_log = weights_log.gather(
            dim=4,
            index=labelset_minus[:, None, :, None].expand(
                -1, self.lambdas.K, -1, self.lambdas.K - 1, -1
            ),
        )
        # weights_other_log: bs x K x class x K-1 x K-1 -> bs x K x class x H x K-1
        weights_other_log = weights_other_log.gather(
            dim=3,
            index=self.perm_idx[None, None, None].expand(
                len(weights_log), self.lambdas.K, self.lambdas.num_class, -1, -1
            ),
        )
        # weights_log: bs x K x class x H x K-1 -> bs x K x class x H
        weights_other_log = weights_other_log.sum(-1)
        weights_other_log = weights_other_log.double()
        # weights_log: bs x K x class x H -> bs x K x class
        weights_other_log_max = weights_other_log.amax(-1)
        weights_other_log = (
            weights_other_log - weights_other_log_max[..., None]
        ).exp().sum(-1).log() + weights_other_log_max
        weights_other_log = weights_other_log.float()
        labelset = F.one_hot(labelset_minus, self.lambdas.num_class).float().sum(2)
        factorial_term = self.logfactorial(labelset.long()).sum(-1)
        weights_other_log = weights_other_log - factorial_term[:, None]
        weights_other_log.masked_fill_(inf_fill_mask[:, None], -self.inf)
        return weights_other_log

    def calc_weight_other_log_approx(
        self, weights_log, targets: dict, **kargs: Any
    ) -> Tensor:
        # weights_log: bs x K x class
        # labelset: bs x K
        # weights_log: bs x K x class -> bs x K x K x class
        weights_log = weights_log[:, None].expand(-1, self.lambdas.K, -1, -1)
        # weights_log: bs x K x K x class -> bs x K x K-1 x class
        # comb_idx: K x K-1
        weights_log = weights_log.gather(
            dim=2,
            index=self.comb_idx[None, :, :, None].expand(
                len(weights_log), -1, -1, self.lambdas.num_class
            ),
        )
        weights_log = weights_log.double()
        weights_log_max = weights_log.amax(2)
        weights_log = (weights_log - weights_log_max[:, :, None]).exp().sum(
            2
        ).log() + weights_log_max
        weights_log = weights_log.float()
        p_hat_log = (
            weights_log
            - torch.tensor(self.lambdas.K - 1, device=weights_log.device).log()
        )
        # labelset: bs x 1 x class
        # label_oh: bs x class x class
        labelset = (
            F.one_hot(targets["labelset"], self.lambdas.num_class)
            .float()
            .sum(1, keepdim=True)
        )
        label_oh = F.one_hot(
            torch.arange(self.lambdas.num_class, device=weights_log.device),
            self.lambdas.num_class,
        ).float()
        # n_c: bs x class x class
        n_c = labelset - label_oh
        # weights_other_log: bs x K x class
        n_c = n_c[:, None]
        weights_other_log = (
            n_c.clip(min=0) * p_hat_log[:, :, None]
            - self.logfactorial(n_c.clip(min=0).long())
        ).sum(-1) + self.logfactorial(
            torch.tensor(self.lambdas.K - 1, device=p_hat_log.device)
        )
        weights_other_log.masked_fill_((n_c == -1).any(-1), -self.inf)
        return weights_other_log

    def rc(self, outputs: Tensor, targets: dict, **kargs: Any) -> dict[str, Tensor]:
        # outputs: bs x K x class
        # weights_other_log: bs x K x class x H
        # weights_log: bs x K x class
        weights_other_log = self.calc_weight_other_log(targets["weights_log"], targets)
        weights_log = targets["weights_log"] + weights_other_log
        weights = self.softmax(weights_log)
        loss: Tensor = self.crossent(outputs.transpose(1, 2), weights.transpose(1, 2))
        weights_other_approx_log = self.calc_weight_other_log_approx(
            targets["weights_log"], targets
        )
        weights_approx = self.softmax(targets["weights_log"] + weights_other_approx_log)
        diff_weight: Tensor = (weights - weights_approx).abs().sum(-1).mean()
        return {"loss": loss, "diff_weight": diff_weight}

    def rc_approx(
        self, outputs: Tensor, targets: dict, **kargs: Any
    ) -> dict[str, Tensor]:
        # outputs: bs x K x class
        # weights_other_log: bs x K x class x H
        # weights_log: bs x K x class
        weights_other_log = self.calc_weight_other_log_approx(
            targets["weights_log"], targets
        )
        weights_log = targets["weights_log"] + weights_other_log
        weights = self.softmax(weights_log)
        loss: Tensor = self.crossent(outputs.transpose(1, 2), weights.transpose(1, 2))
        return {"loss": loss}

    def cc(self, outputs: Tensor, targets: dict, **kargs: Any) -> dict[str, Tensor]:
        # outputs: bs x K x class
        # weights_other_log: bs x K x class x H
        # weights_log: bs x K x class
        out_log = self.log_softmax(outputs)
        others_log = self.calc_weight_other_log(out_log, targets)
        outs_log = out_log + others_log
        outs_log = outs_log.double()
        outs_log_max = outs_log.amax(-1)
        loss: Tensor = -(
            (outs_log - outs_log_max[..., None]).exp().sum(-1).log() + outs_log_max
        ).mean()
        loss = loss.float() / self.lambdas.K  # to scale with DLLP
        return {"loss": loss}

    def cc_approx(
        self, outputs: Tensor, targets: dict, **kargs: Any
    ) -> dict[str, Tensor]:
        # outputs: bs x K x class
        # weights_other_log: bs x K x class x H
        # weights_log: bs x K x class
        out_log = self.log_softmax(outputs)
        others_log = self.calc_weight_other_log_approx(out_log, targets)
        outs_log = out_log + others_log
        outs_log = outs_log.double()
        outs_log_max = outs_log.amax(-1)
        loss: Tensor = -(
            (outs_log - outs_log_max[..., None]).exp().sum(-1).log() + outs_log_max
        ).mean()
        loss = loss.float() / self.lambdas.K  # to scale with DLLP
        return {"loss": loss}

    def pll(self, outputs: Tensor, targets: dict, **kargs: Any) -> dict[str, Tensor]:
        weights_log = targets["weights_log"]
        weights = self.softmax(weights_log)
        label_flag = F.one_hot(targets["labelset"], self.lambdas.num_class).any(
            1, keepdim=True
        )
        weights = weights * label_flag
        weights = weights / (weights.sum(-1, keepdim=True) + self.eps)
        loss: Tensor = (
            -(weights * self.log_softmax(outputs)).sum(-1).reshape(-1).mean(0)
        )
        return {"loss": loss}

    def dllp(self, outputs: Tensor, targets: dict, **kargs: Any) -> dict[str, Tensor]:
        labelset = F.one_hot(targets["labelset"], self.lambdas.num_class).float().sum(1)
        labelset = labelset / labelset.sum(-1, keepdim=True)
        out_log = self.log_softmax(outputs).double()
        out_log_max = out_log.amax(1)
        out_log = (
            (out_log - out_log_max[:, None]).exp().sum(1).log()
            + out_log_max
            - torch.tensor(self.lambdas.K).log()
        )
        out_log = out_log.float()
        loss = self.crossent(out_log, labelset)
        return {"loss": loss}

    def dllp_approx(
        self, outputs: Tensor, targets: dict, **kargs: Any
    ) -> dict[str, Tensor]:
        out_log = self.log_softmax(outputs).double()
        out_log_max = out_log.amax(1)
        out_log = (
            (out_log - out_log_max[:, None]).exp().sum(1).log()
            + out_log_max
            - torch.tensor(self.lambdas.K).log()
        )
        out_log = out_log.float()
        n_c = F.one_hot(targets["labelset"], self.lambdas.num_class).float().sum(1)
        dllp = (
            -F.cross_entropy(out_log, n_c / self.lambdas.K, reduction="none")
            * self.lambdas.K
        )
        loss: Tensor = -(
            dllp
            - (self.logfactorial(n_c.clip(min=0).long())).sum(axis=1)
            + self.logfactorial(torch.tensor(self.lambdas.K, device=out_log.device))
        ).sum() / len(n_c)
        return {"loss": loss}

    def ot(self, outputs: Tensor, targets: dict, **kargs: Any) -> dict[str, Tensor]:
        Ps_log = targets["weights_log"]
        Ns = F.one_hot(targets["labelset"], self.lambdas.num_class).float().sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Q = batch_sinkhorn(
                torch.ones(len(Ns), self.lambdas.K, dtype=Ns.dtype, device=Ns.device),
                Ns,
                -Ps_log,
                reg=1,
                numItermax=75,
            )
        loss: Tensor = self.crossent(outputs.transpose(1, 2), Q.transpose(1, 2))
        return {"loss": loss}

    def llpfc(
        self, outputs: Tensor, targets: Tensor, **kargs: Any
    ) -> dict[str, Tensor]:
        # outs: bs x C
        outs = self.softmax(outputs)
        # Ts: bs x C x C
        Ts = targets["T"]
        noisy_labels = targets["noisy_y"].reshape(-1)
        # transformed_output: bs x C
        transformed_output = torch.einsum("bkci,bki->bc", Ts, outs)
        # noisy_labels: g_bs x C
        loss = self.nll((transformed_output + self.eps).log(), noisy_labels)
        return {"loss": loss}

    def supervised(
        self, outputs: Tensor, targets: dict, **kargs: Any
    ) -> dict[str, Tensor]:
        label = targets["label"]
        loss: Tensor = self.crossent(outputs.transpose(1, 2), label)
        return {"loss": loss}

    def criterion(self, outputs: Tensor, targets: dict, stage: str) -> None:
        values: dict[str, Tensor] = {}
        num_sample = outputs.shape[0] * outputs.shape[1]
        values["acc"] = self.accuracy(outputs, targets["label"])
        for key in values:
            self.log(values[key], key, stage, num=num_sample)

    def is_best_score(self) -> bool:
        if self.summary is None:
            return False
        current_score = self.summary.last_mean_vs.get("val/acc", self.inf)
        if self.best_score < current_score:
            self.best_score = current_score
            self.best_score_tst = self.summary.last_mean_vs.get("tst/acc", self.inf)
            return True
        else:
            return False
