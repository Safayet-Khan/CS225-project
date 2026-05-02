from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Per-run metric container
# ---------------------------------------------------------------------------


@dataclass
class BinaryMetrics:
    """Container for the four headline metrics."""

    accuracy: float
    f1: float
    roc_auc: float
    mcc: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Accuracy={self.accuracy:.4f}  "
            f"F1={self.f1:.4f}  "
            f"ROC-AUC={self.roc_auc:.4f}  "
            f"MCC={self.mcc:.4f}"
        )


def compute_binary_metrics(
    logits_or_probs: Sequence[float],
    targets: Sequence[int],
    threshold: float = 0.5,
    is_logits: bool = True,
) -> BinaryMetrics:
    scores = np.asarray(logits_or_probs, dtype=np.float64).reshape(-1)
    y_true = np.asarray(targets, dtype=np.int64).reshape(-1)
    if scores.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: scores {scores.shape} vs targets {y_true.shape}"
        )
    if is_logits:
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        probs = scores
    y_pred = (probs >= threshold).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    # f1 on the positive (abnormal) class -> binary average.
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    if len(np.unique(y_true)) < 2:
        roc = float("nan")
    else:
        roc = roc_auc_score(y_true, probs)

    return BinaryMetrics(
        accuracy=float(acc),
        f1=float(f1),
        roc_auc=float(roc),
        mcc=float(mcc),
    )

@dataclass
class MetricStats:
    """Mean +/- std across a list of :class:`BinaryMetrics`."""

    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    roc_auc_mean: float
    roc_auc_std: float
    mcc_mean: float
    mcc_std: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def pretty(self) -> str:
        return (
            f"Accuracy={self.accuracy_mean:.4f} +/- {self.accuracy_std:.4f}\n"
            f"F1-Score={self.f1_mean:.4f} +/- {self.f1_std:.4f}\n"
            f"ROC-AUC={self.roc_auc_mean:.4f} +/- {self.roc_auc_std:.4f}\n"
            f"MCC={self.mcc_mean:.4f} +/- {self.mcc_std:.4f}"
        )


def aggregate_metrics(metric_list: Iterable[BinaryMetrics]) -> MetricStats:
    """Compute mean +/- std across a list of per-seed metric results."""
    metric_list = list(metric_list)
    if not metric_list:
        raise ValueError("aggregate_metrics: empty input.")
    accs = np.array([m.accuracy for m in metric_list])
    f1s = np.array([m.f1 for m in metric_list])
    aucs = np.array([m.roc_auc for m in metric_list])
    mccs = np.array([m.mcc for m in metric_list])

    return MetricStats(
        accuracy_mean=float(accs.mean()),
        accuracy_std=float(accs.std(ddof=0)),
        f1_mean=float(f1s.mean()),
        f1_std=float(f1s.std(ddof=0)),
        roc_auc_mean=float(np.nanmean(aucs)),
        roc_auc_std=float(np.nanstd(aucs, ddof=0)),
        mcc_mean=float(mccs.mean()),
        mcc_std=float(mccs.std(ddof=0)),
    )


class MetricAccumulator:
    """Accumulate logits and targets across batches for a single run."""

    def __init__(self) -> None:
        self._logits: List[float] = []
        self._targets: List[int] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self._logits.extend(logits.detach().float().cpu().view(-1).tolist())
        self._targets.extend(targets.detach().long().cpu().view(-1).tolist())

    def reset(self) -> None:
        self._logits.clear()
        self._targets.clear()

    def compute(self, threshold: float = 0.5) -> BinaryMetrics:
        return compute_binary_metrics(
            self._logits, self._targets, threshold=threshold, is_logits=True
        )

    @property
    def logits(self) -> List[float]:
        return list(self._logits)

    @property
    def targets(self) -> List[int]:
        return list(self._targets)
