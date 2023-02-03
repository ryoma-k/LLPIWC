from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np


class Tools:
    @staticmethod
    def normalize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        out: np.ndarray = (value - mean) / std
        return out

    @staticmethod
    def inv_trans(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        out: np.ndarray = value * std + mean
        return out

    @staticmethod
    def txtread(path: str) -> list[str]:
        lines = []
        with open(path) as f:
            for l in f:
                lines.append(l)
        return lines

    @staticmethod
    @contextmanager
    def timer(name: str) -> Iterator:
        t0 = time.time()
        yield
        print(f"[{name}] done in {time.time() - t0:.8f} s")

    @staticmethod
    def flatten(l: Iterable) -> Any:
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from Tools.flatten(el)
            else:
                yield el

    @staticmethod
    def make_dir(path, ignore: bool = True) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if not ignore:
                print("{} already exists.".format(path))

    @staticmethod
    def get_git_hash() -> str:
        cmd = "git rev-parse --short HEAD"
        _hash = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode(
            "utf-8"
        )
        return _hash

    @staticmethod
    def get_git_diff() -> str:
        cmd = "git diff"
        diff = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode(
            "utf-8"
        )
        diff = "\n".join(["\t" + l for l in diff.split("\n")])
        return diff

    @staticmethod
    def make_temporal_commit() -> str:
        diff = Tools.get_git_diff()
        if len(diff):
            cmd = "git add -A"
            subprocess.run(cmd, shell=True, capture_output=True).stdout.decode("utf-8")
            cmd = "git commit -m 'exp commit.'"
            subprocess.run(cmd, shell=True, capture_output=True).stdout.decode("utf-8")
        _hash = Tools.get_git_hash()
        if len(diff):
            cmd = "git reset --soft @{1}"
            subprocess.run(cmd, shell=True, capture_output=True).stdout.decode("utf-8")
        return _hash
