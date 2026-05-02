"""
Compact 1D-CNN that constitutes the waveform branch of AudioFuse.

Specification from the paper (Sec. 3.2, Fig. 1):

    Input: raw waveform of shape (B, 1, 110250)   (5 s @ 22050 Hz)

    Block 1: Conv1D(64,  kernel=16, stride=4) + ReLU
             BatchNorm1d + MaxPool1d(4)
    Block 2: Conv1D(128, kernel=16, stride=4) + ReLU
             BatchNorm1d + MaxPool1d(4)
    Block 3: Conv1D(256, kernel=16, stride=4) + ReLU
             BatchNorm1d                         (no pooling, per Fig. 1)
    GlobalAveragePool1d
    Dense(64) -> 64-dim temporal feature vector f_wave

The figure clearly shows the third block is followed by BatchNorm only
(no MaxPool), and we follow the figure rather than the text body which
loosely describes the pattern as uniform across all three blocks.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """Conv1d -> ReLU -> BatchNorm1d -> optional MaxPool1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        pool_size: int | None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size) if pool_size and pool_size > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.pool(x)
        return x


class WaveformCNN(nn.Module):
    """Three-block 1D CNN that produces a 64-dim waveform embedding."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 16,
        stride: int = 4,
        pool_size: int = 4,
        dense_units: int = 64,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(
                f"WaveformCNN expects exactly 3 conv blocks (got {len(channels)})."
            )

        self.block1 = Conv1DBlock(in_channels, channels[0], kernel_size, stride, pool_size)
        self.block2 = Conv1DBlock(channels[0], channels[1], kernel_size, stride, pool_size)
        # Third block: no MaxPool (per Fig. 1 of the paper).
        self.block3 = Conv1DBlock(channels[1], channels[2], kernel_size, stride, pool_size=None)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(channels[2], dense_units)
        self.act = nn.ReLU(inplace=True)

        self._feature_dim = dense_units

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 64-dim temporal feature vector ``f_wave``.

        Args:
            x: ``(B, 1, T)`` waveform tensor.

        Returns:
            ``(B, dense_units)`` float tensor.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).squeeze(-1)        # (B, channels[-1])
        x = self.act(self.dense(x))
        return x
