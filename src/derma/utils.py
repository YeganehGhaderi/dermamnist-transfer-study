import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make a run reproducible: same seed -> same weights init, same shuffling, same augmentations."""
    random.seed(seed)                  # Python's own random module
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch on CPU
    torch.cuda.manual_seed_all(seed)   # PyTorch on every GPU
    # cuDNN picks the fastest convolution algorithm, and some of them are not deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own, reproducible seed (used for random augmentations)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
