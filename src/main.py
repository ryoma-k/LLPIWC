from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from pprint import pprint

import torch
from llp import (
    MCLLPEvaluator,
    MCLLPTrainer,
    make_datasets,
    make_model,
    preprocess_datasets,
)
from llp.base.trainers import DL_sample
from llp.base.utils import Summary
from llp.base.utils import Tools as T
from llp.base.utils import seed_everything
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        usage="python main.py -cp yamls/base.yaml", add_help=True
    )
    parser.add_argument(
        "-cp",
        "--config_path",
        type=str,
        nargs="+",
        default=["config.yaml"],
        help="config path",
    )
    parser.add_argument("-seed", "--seed", type=int)
    parser.add_argument("-nw", "--num_worker", type=int)
    parser.add_argument("-memo", "--memo", type=str)
    parser.add_argument("-multi", "--multi_gpu", action="store_true")
    parse = parser.parse_args()

    # load yaml
    config_name = "_".join([Path(cp).name.split(".")[0] for cp in parse.config_path])
    config = OmegaConf.merge(*[OmegaConf.load(cp) for cp in parse.config_path])
    pprint(config)

    # fix seed
    seed = parse.seed
    if seed is None:
        seed = config.main.get("seed")
    if seed is None:
        seed = random.randint(1, 10000)
    config_file_name_seed = config_name + f"_seed_{seed}"
    seed_everything(seed)
    if parse.memo is not None:
        config_file_name_seed += "_" + parse.memo
    config.main.save_path += "/" + config_file_name_seed
    config.main.seed = seed
    config.trainer.save_path = config.main.save_path
    config.dataset.save_path = config.main.save_path
    if config.trainer.loss_mode == "cc_approx2rc_approx":
        config.trainer.loss_mode = "rc_approx"
        pretrained_path = config.main.save_path.replace("2rc_approx", "")
        if config.dataset.get("pretrained_weight_lr"):
            cc_best_lr = config.dataset.pretrained_weight_lr
            pretrained_path = re.sub("prelr1e-._", "", pretrained_path)
            pretrained_path = re.sub("lr1e-.", f"lr{cc_best_lr}", pretrained_path)
        config.dataset.pretrained_weight = pretrained_path
        print(f"load path {pretrained_path}")
    if config.trainer.loss_mode == "llpfc":
        config.dataset.trn_bs = config.loss.lambdas.K * config.dataset.trn_bs
    T.make_dir(config.main.save_path + "/pths")
    OmegaConf.save(config, config.main.save_path + "/config.yaml")
    print(f"save path {config.dataset.save_path}")

    # tensorboard logger
    summary = Summary(config.main.save_path)
    summary.add_text("hash", T.get_git_hash())
    summary.add_text("diff", T.get_git_diff())
    summary.add_text("config", json.dumps(OmegaConf.to_yaml(config)))

    # make dataset & dataloader
    if parse.num_worker is None:
        num_w = 0 if config.main.get("debug", False) else 3
    else:
        num_w = parse.num_worker
    if config.main.preprocess:
        with T.timer("data load"):
            preprocess_datasets(config.dataset)
        sys.exit(0)
    else:
        with T.timer("data load"):
            trn_ds, val_ds, tst_ds, split_df = make_datasets(config.dataset, seed=seed)
        split_df.to_csv(config.main.save_path + "/split.csv", index=False)
    trn_bs = config.dataset.trn_bs
    val_bs = config.dataset.get("val_bs", trn_bs)
    trn_dl = DL_sample(
        DataLoader(
            trn_ds, batch_size=trn_bs, shuffle=True, num_workers=num_w, drop_last=True
        )
    )
    val_dl = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=num_w)
    tst_dl = DataLoader(tst_ds, batch_size=val_bs, shuffle=False, num_workers=num_w)

    dataloaders = {"trn": [trn_dl], "tst": {"val": val_dl, "tst": tst_dl}}

    evaluator = MCLLPEvaluator(**config.loss, summary=summary)

    # make model
    model = make_model(config.model).cuda()

    # make trainer
    trainer = MCLLPTrainer(model, dataloaders, evaluator, summary, config.trainer)

    # load pretrained weight
    if config.dataset.get("pretrained_weight"):
        pretrained_weight_pth = torch.load(
            config.dataset.pretrained_weight + "/pths/best.pth"
        )
        trainer.load_dataset_weight(pretrained_weight_pth)
        trainer.set_freeze_dataset_weight(True)

    # training
    with T.timer("training"):
        try:
            with torch.autograd.set_detect_anomaly(True):
                trainer.run()
            # trainer.save_state_pth(trainer.iteration)
            trainer.final_evaluation()
        except KeyboardInterrupt:
            torch.cuda.empty_cache()
            trainer.save_state_pth(f"{trainer.iteration}_interrupt")
            print("##### interrupt #####")
    T.make_dir(config.main.save_path + "/csvs")
    for key in summary.epoch_values.keys():
        summary.get_epoch_df(key).to_csv(
            config.main.save_path + "/csvs/" + key.replace("/", "_") + ".csv",
            index=False,
        )
