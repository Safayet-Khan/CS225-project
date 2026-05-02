from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(
        self,
        weight_neg: float = 1.0,
        weight_pos: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")
        self.weight_neg = float(weight_neg)
        self.weight_pos = float(weight_pos)
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        sample_weights = (
            self.weight_pos * targets + self.weight_neg * (1.0 - targets)
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        loss = loss * sample_weights
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return (
            f"weight_neg={self.weight_neg:.4f}, "
            f"weight_pos={self.weight_pos:.4f}, "
            f"reduction='{self.reduction}'"
        )


def build_loss(
    class_weights: Tuple[float, float], reduction: str = "mean"
) -> WeightedBCEWithLogitsLoss:
    """Convenience constructor used by the training scripts."""
    w_neg, w_pos = class_weights
    return WeightedBCEWithLogitsLoss(
        weight_neg=w_neg, weight_pos=w_pos, reduction=reduction
    )
