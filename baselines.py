"""
Single-modality baselines.

Per the AudioFuse paper (Table I) and the comparative-fusion report
(Table II), the spectrogram and waveform baselines use identical encoders
to the corresponding branch of the fusion model, followed by a small MLP
classification head.  The head architecture (Dense(192) + ReLU + Dropout
0.5 + sigmoid) is the same as the fusion head, so any difference in
performance is attributable purely to the available modality.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn

from .cnn_branch import WaveformCNN
from .vit_branch import SpectrogramViT


# ---------------------------------------------------------------------------
# Shared classification head
# ---------------------------------------------------------------------------


class ClassificationHead(nn.Module):
    """``Linear -> ReLU -> Dropout -> Linear`` head producing a single logit."""

    def __init__(self, in_features: int, hidden_dim: int = 192,
                 dropout: float = 0.5, num_classes: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)              # raw logits; loss handles sigmoid


# ---------------------------------------------------------------------------
# Spectrogram-only baseline (ViT)
# ---------------------------------------------------------------------------


class SpectrogramBaseline(nn.Module):
    """ViT encoder + classification head (no waveform branch)."""

    def __init__(self, model_cfg: Mapping) -> None:
        super().__init__()
        vit_cfg = model_cfg["vit"]
        head_cfg = model_cfg["fusion_head"]

        self.encoder = SpectrogramViT(
            img_size=vit_cfg["img_size"],
            patch_size=vit_cfg["patch_size"],
            in_channels=vit_cfg["in_channels"],
            embed_dim=vit_cfg["embed_dim"],
            depth=vit_cfg["depth"],
            num_heads=vit_cfg["num_heads"],
            mlp_ratio=vit_cfg["mlp_ratio"],
            dropout=vit_cfg["dropout"],
            attn_dropout=vit_cfg["attn_dropout"],
        )
        self.head = ClassificationHead(
            in_features=vit_cfg["embed_dim"],
            hidden_dim=head_cfg["hidden_dim"],
            dropout=head_cfg["dropout"],
            num_classes=head_cfg["num_classes"],
        )

    def forward(
        self,
        waveform: Optional[torch.Tensor],
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        del waveform                     # unused; kept for a uniform API
        feats = self.encoder(spectrogram)
        return self.head(feats)


# ---------------------------------------------------------------------------
# Waveform-only baseline (1D-CNN)
# ---------------------------------------------------------------------------


class WaveformBaseline(nn.Module):
    """1D-CNN encoder + classification head (no spectrogram branch)."""

    def __init__(self, model_cfg: Mapping) -> None:
        super().__init__()
        cnn_cfg = model_cfg["cnn1d"]
        head_cfg = model_cfg["fusion_head"]

        self.encoder = WaveformCNN(
            in_channels=cnn_cfg["in_channels"],
            channels=tuple(cnn_cfg["channels"]),
            kernel_size=cnn_cfg["kernel_size"],
            stride=cnn_cfg["stride"],
            pool_size=cnn_cfg["pool_size"],
            dense_units=cnn_cfg["dense_units"],
        )
        self.head = ClassificationHead(
            in_features=self.encoder.feature_dim,
            hidden_dim=head_cfg["hidden_dim"],
            dropout=head_cfg["dropout"],
            num_classes=head_cfg["num_classes"],
        )

    def forward(
        self,
        waveform: torch.Tensor,
        spectrogram: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del spectrogram                  # unused; kept for a uniform API
        feats = self.encoder(waveform)
        return self.head(feats)
