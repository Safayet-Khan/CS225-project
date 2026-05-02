from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA), and optionally enable
    deterministic CuDNN.

    Note that fully deterministic execution may slow training down on
    GPU; toggle via the ``deterministic`` flag in the config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes both to stdout and (optionally) a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class CSVLogger:
    """Append-only CSV logger for training curves.

    Writes one row per call to ``log``.  Columns are inferred from the
    first row's keys; subsequent rows are required to have the same set.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fieldnames: Optional[list[str]] = None

    def log(self, row: Mapping[str, Any]) -> None:
        new_file = not os.path.exists(self.path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        elif set(row.keys()) != set(self._fieldnames):
            raise ValueError(
                f"CSVLogger: row keys {sorted(row.keys())} do not match "
                f"the established schema {sorted(self._fieldnames)}."
            )
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row[k] for k in self._fieldnames})


def save_checkpoint(
    state: Mapping[str, Any],
    path: str,
) -> None:
    """Save a PyTorch checkpoint to ``path`` (creates parent dirs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str, map_location: Any | None = None
) -> Dict[str, Any]:
    """Load a previously saved checkpoint."""
    return torch.load(path, map_location=map_location, weights_only=False)


def make_run_dir(base_dir: str, model_name: str, seed: int) -> str:
    """Return ``<base_dir>/<model_name>/seed<seed>/`` (created)."""
    run_dir = os.path.join(base_dir, model_name, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def dump_json(path: str, obj: Mapping[str, Any]) -> None:
    """Write ``obj`` to ``path`` as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    """Fallback for JSON serialisation of non-trivial values."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serialisable.")


@contextmanager
def evaluating(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """Temporarily switch a model to ``eval`` mode and restore afterwards."""
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()
