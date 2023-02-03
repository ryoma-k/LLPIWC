from __future__ import annotations

import warnings
from typing import Any

import torch


def batch_sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    reg: float,
    numItermax: int = 1000,
    stopThr: float = 1e-9,
    verbose: bool = False,
    take_log: bool = False,
    warn: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, dict]:

    with torch.no_grad():
        # init data
        dim_a = a.shape[1]
        dim_b = b.shape[1]
        bs = a.shape[0]

        if take_log:
            log: dict[str, Any] = {"err": []}

        # u: bs x dim_a
        # v: bs x dim_b
        u = torch.ones(bs, dim_a, dtype=M.dtype, device=M.device) / dim_a
        v = torch.ones(bs, dim_b, dtype=M.dtype, device=M.device) / dim_b

        # K: bs x dim_a x dim_b
        K = torch.exp(M / (-reg))

        Kp = (1 / a)[:, :, None] * K

        err = 1
        for ii in range(numItermax):
            uprev = u
            vprev = v
            # KtransposeU: bs x dim_b
            KtransposeU = torch.einsum("bij,bi->bj", K, u)
            v = b / KtransposeU
            u = 1.0 / torch.einsum("bij,bj->bi", Kp, v)

            if (
                (KtransposeU == 0).any()
                or u.isnan().any()
                or u.isinf().any()
                or v.isnan().any()
                or v.isinf().any()
            ):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                warnings.warn("Warning: numerical errors at iteration %d" % ii)
                u = uprev
                v = vprev
                break
            if ii % 10 == 0:
                # we can speed up the process by checking for the error only all the 10th iterations
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = torch.einsum("bi,bij,bj->bj", u, K, v)
                err = torch.linalg.norm(tmp2 - b, dim=1).max()  # violation of marginal
                if take_log:
                    log["err"].append(err)

                if err < stopThr:
                    break
                if verbose:
                    if ii % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(ii, err))
        else:
            if warn:
                warnings.warn(
                    "Sinkhorn did not converge. You might want to "
                    "increase the number of iterations `numItermax` "
                    "or the regularization parameter `reg`."
                )
        if take_log:
            log["niter"] = ii
            log["u"] = u
            log["v"] = v

        if take_log:
            return u[:, :, None] * K * v[:, None], log
        else:
            return u[:, :, None] * K * v[:, None]


if __name__ == "__main__":
    import numpy as np
    import ot

    a = torch.rand(5, 3)
    b = torch.rand(5, 7)
    M = torch.rand(5, 3, 7) * 10
    reg = 1
    pot_res = ot.sinkhorn(a[0].numpy(), b[0].numpy(), M[0].numpy(), reg=1)
    my_res = batch_sinkhorn(a[0:1], b[0:1], M[0:1], reg=1).numpy()[0]
    print(np.isclose(pot_res, my_res).all())
