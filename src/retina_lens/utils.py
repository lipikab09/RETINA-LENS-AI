"""Seeding, device selection, logging."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        pref = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(pref)
