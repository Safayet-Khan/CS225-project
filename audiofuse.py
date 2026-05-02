"""
AudioFuse and its fusion-strategy variants.

This module implements the four fusion strategies compared in the
report (Sec. III.B), plus the original AudioFuse late-concatenation
variant from the paper:

    LCF : Late Concatenation Fusion (the original AudioFuse).
    EF  : Early Fusion - inject waveform-derived tokens into the ViT.
    TFN : Tensor Fusion Network - outer product of branch embeddings.
    WLF : Weighted Late Fusion - learnable softmax-weighted concatenation.

All variants share the same backbone encoders (SpectrogramViT,
WaveformCNN) and the same classification head (Dense(192) + ReLU +
Dropout(0.5) + final logit) so that any observed performance difference
is attributable to the fusion module alone.

Naming convention used by ``build_model``:
    "spectrogram_baseline", "waveform_baseline",
    "audiofuse_lcf", "early_fusion", "tensor_fusion",
    "weighted_late_fusion".
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import (
    ClassificationHead,
    SpectrogramBaseline,
    WaveformBaseline,
)
from .cnn_branch import WaveformCNN
from .vit_branch import SpectrogramViT


# ---------------------------------------------------------------------------
# LCF: Late Concatenation Fusion (the original AudioFuse)
# ---------------------------------------------------------------------------


class AudioFuseLCF(nn.Module):
    """Original AudioFuse: ``f_fused = [f_spec; f_wave]`` then MLP head.

    f_spec is 192-d, f_wave is 64-d, so f_fused is 256-d.  The fusion
    module itself introduces zero parameters - the classification head
    does the work.
    """

    def __init__(self, model_cfg: Mapping) -> None:
        super().__init__()
        vit_cfg, cnn_cfg, head_cfg = (
            model_cfg["vit"],
            model_cfg["cnn1d"],
            model_cfg["fusion_head"],
        )

        self.spec_encoder = SpectrogramViT(
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
        self.wave_encoder = WaveformCNN(
            in_channels=cnn_cfg["in_channels"],
            channels=tuple(cnn_cfg["channels"]),
            kernel_size=cnn_cfg["kernel_size"],
            stride=cnn_cfg["stride"],
            pool_size=cnn_cfg["pool_size"],
            dense_units=cnn_cfg["dense_units"],
        )

        fused_dim = vit_cfg["embed_dim"] + self.wave_encoder.feature_dim
        self.head = ClassificationHead(
            in_features=fused_dim,
            hidden_dim=head_cfg["hidden_dim"],
            dropout=head_cfg["dropout"],
            num_classes=head_cfg["num_classes"],
        )

        self._fused_dim = fused_dim

    @property
    def fused_dim(self) -> int:
        """Dimensionality of the fused embedding (e.g. for SMOTE)."""
        return self._fused_dim

    def encode(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """Return the 256-d fused embedding without the classification head."""
        f_spec = self.spec_encoder(spectrogram)
        f_wave = self.wave_encoder(waveform)
        return torch.cat([f_spec, f_wave], dim=1)

    def forward(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.encode(waveform, spectrogram))


# ---------------------------------------------------------------------------
# EF: Early Fusion
# ---------------------------------------------------------------------------


class EarlyFusion(nn.Module):
    """Early-fusion variant per Sec. III.B.2 of the report.

    Implementation (faithful to the report):
        1. The waveform passes through a single Conv1D with 64 filters
           and stride 4 to produce a coarse temporal embedding.
        2. That embedding is reshaped/tiled to the spectrogram's patch
           grid (14 x 14) and projected to ``embed_dim`` channels.
        3. The 196 waveform-derived tokens are concatenated **channel-
           wise** with the spectrogram patch tokens and passed through a
           linear mixer back to ``embed_dim`` before the ViT body.
        4. A separate full 1D-CNN branch is also retained to "preserve
           the specialized temporal pathway", and its 64-d output is
           concatenated with the ViT pooled output for the final head.

    The 1D-CNN branch reuses :class:`WaveformCNN` (which is the same
    encoder used in LCF/baselines) so that the backbones remain
    consistent across all variants.
    """

    def __init__(self, model_cfg: Mapping) -> None:
        super().__init__()
        vit_cfg, cnn_cfg, head_cfg = (
            model_cfg["vit"],
            model_cfg["cnn1d"],
            model_cfg["fusion_head"],
        )

        self.embed_dim = vit_cfg["embed_dim"]
        self.grid_size = vit_cfg["img_size"] // vit_cfg["patch_size"]
        self.num_patches = self.grid_size ** 2

        # 1) Coarse temporal embedding, kept lightweight per the report.
        self.early_conv = nn.Conv1d(
            in_channels=cnn_cfg["in_channels"],
            out_channels=64,
            kernel_size=16,
            stride=4,
            padding=0,
        )
        self.early_act = nn.ReLU(inplace=True)
        # Adaptive pool to 196 time steps so we can match the patch grid.
        self.early_to_tokens = nn.AdaptiveAvgPool1d(self.num_patches)
        self.early_proj = nn.Linear(64, self.embed_dim)

        # 2) Spectrogram ViT, but called token-wise so we can mix early.
        self.spec_vit = SpectrogramViT(
            img_size=vit_cfg["img_size"],
            patch_size=vit_cfg["patch_size"],
            in_channels=vit_cfg["in_channels"],
            embed_dim=self.embed_dim,
            depth=vit_cfg["depth"],
            num_heads=vit_cfg["num_heads"],
            mlp_ratio=vit_cfg["mlp_ratio"],
            dropout=vit_cfg["dropout"],
            attn_dropout=vit_cfg["attn_dropout"],
        )
        # Channel-wise mixer after concatenating spec patches and waveform tokens.
        self.token_mixer = nn.Linear(2 * self.embed_dim, self.embed_dim)

        # 3) Full waveform branch for the specialized temporal pathway.
        self.wave_encoder = WaveformCNN(
            in_channels=cnn_cfg["in_channels"],
            channels=tuple(cnn_cfg["channels"]),
            kernel_size=cnn_cfg["kernel_size"],
            stride=cnn_cfg["stride"],
            pool_size=cnn_cfg["pool_size"],
            dense_units=cnn_cfg["dense_units"],
        )

        fused_dim = self.embed_dim + self.wave_encoder.feature_dim
        self.head = ClassificationHead(
            in_features=fused_dim,
            hidden_dim=head_cfg["hidden_dim"],
            dropout=head_cfg["dropout"],
            num_classes=head_cfg["num_classes"],
        )

    def _early_tokens(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map a raw waveform to ``(B, num_patches, embed_dim)`` tokens."""
        x = self.early_act(self.early_conv(waveform))     # (B, 64, T')
        x = self.early_to_tokens(x)                       # (B, 64, num_patches)
        x = x.transpose(1, 2)                             # (B, num_patches, 64)
        x = self.early_proj(x)                            # (B, num_patches, E)
        return x

    def _vit_with_injection(
        self, spectrogram: torch.Tensor, wave_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Run the ViT body with spec+wave tokens fused channel-wise."""
        # Patch-embed the spectrogram and add positional embedding.
        spec_tokens = self.spec_vit.patch_embed(spectrogram)
        spec_tokens = spec_tokens + self.spec_vit.pos_embed
        spec_tokens = self.spec_vit.pos_drop(spec_tokens)

        # Channel-wise concat -> linear mixer back to embed_dim.
        mixed = torch.cat([spec_tokens, wave_tokens], dim=-1)
        mixed = self.token_mixer(mixed)

        for blk in self.spec_vit.blocks:
            mixed = blk(mixed)
        mixed = self.spec_vit.norm(mixed)
        return mixed.mean(dim=1)                          # (B, E)

    def forward(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        wave_tokens = self._early_tokens(waveform)
        f_spec = self._vit_with_injection(spectrogram, wave_tokens)
        f_wave = self.wave_encoder(waveform)
        fused = torch.cat([f_spec, f_wave], dim=1)
        return self.head(fused)


# ---------------------------------------------------------------------------
# TFN: Tensor Fusion Network
# ---------------------------------------------------------------------------


class TensorFusion(nn.Module):
    """Outer-product fusion (Zadeh et al., 2017) on top of the AudioFuse backbones.

    Each branch embedding is augmented with a constant 1 to capture
    unimodal terms (per the original TFN formulation).  The flattened
    outer product is projected down to a 256-d vector before the head.
    The flattened tensor has size 193 * 65 = 12545 elements with the
    default config, leading to roughly 3.2 M extra parameters - matching
    the Table I figure in the report.
    """

    def __init__(self, model_cfg: Mapping) -> None:
        super().__init__()
        vit_cfg, cnn_cfg, head_cfg = (
            model_cfg["vit"],
            model_cfg["cnn1d"],
            model_cfg["fusion_head"],
        )

        self.spec_encoder = SpectrogramViT(
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
        self.wave_encoder = WaveformCNN(
            in_channels=cnn_cfg["in_channels"],
            channels=tuple(cnn_cfg["channels"]),
            kernel_size=cnn_cfg["kernel_size"],
            stride=cnn_cfg["stride"],
            pool_size=cnn_cfg["pool_size"],
            dense_units=cnn_cfg["dense_units"],
        )

        spec_aug = vit_cfg["embed_dim"] + 1
        wave_aug = self.wave_encoder.feature_dim + 1
        flat_dim = spec_aug * wave_aug
        proj_dim = vit_cfg["embed_dim"] + self.wave_encoder.feature_dim   # 256

        self.proj = nn.Linear(flat_dim, proj_dim)
        self.head = ClassificationHead(
            in_features=proj_dim,
            hidden_dim=head_cfg["hidden_dim"],
            dropout=head_cfg["dropout"],
            num_classes=head_cfg["num_classes"],
        )

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        """Append a constant 1 dimension to capture unimodal terms."""
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=1)

    def forward(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        f_spec = self.spec_encoder(spectrogram)
        f_wave = self.wave_encoder(waveform)
        s_aug = self._augment(f_spec)                     # (B, 193)
        w_aug = self._augment(f_wave)                     # (B, 65)
        # Outer product (B, 193, 1) * (B, 1, 65) -> (B, 193, 65).
        outer = s_aug.unsqueeze(2) * w_aug.unsqueeze(1)
        flat = outer.flatten(start_dim=1)                 # (B, 193 * 65)
        fused = self.proj(flat)
        return self.head(fused)


# ---------------------------------------------------------------------------
# WLF: Weighted Late Fusion
# ---------------------------------------------------------------------------


class WeightedLateFusion(nn.Module):
    """Learnable softmax-weighted concatenation (Sec. III.B.4 of the report)."""

    def __init__(self, model_cfg: Mapping) -> None:
        super().__init__()
        vit_cfg, cnn_cfg, head_cfg = (
            model_cfg["vit"],
            model_cfg["cnn1d"],
            model_cfg["fusion_head"],
        )

        self.spec_encoder = SpectrogramViT(
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
        self.wave_encoder = WaveformCNN(
            in_channels=cnn_cfg["in_channels"],
            channels=tuple(cnn_cfg["channels"]),
            kernel_size=cnn_cfg["kernel_size"],
            stride=cnn_cfg["stride"],
            pool_size=cnn_cfg["pool_size"],
            dense_units=cnn_cfg["dense_units"],
        )

        # Two unconstrained logits, softmax'd at forward time.
        self.gate_logits = nn.Parameter(torch.zeros(2))

        fused_dim = vit_cfg["embed_dim"] + self.wave_encoder.feature_dim
        self.head = ClassificationHead(
            in_features=fused_dim,
            hidden_dim=head_cfg["hidden_dim"],
            dropout=head_cfg["dropout"],
            num_classes=head_cfg["num_classes"],
        )

    def gates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(alpha_spec, alpha_wave)`` after a softmax over the logits."""
        alpha = F.softmax(self.gate_logits, dim=0)
        return alpha[0], alpha[1]

    def forward(
        self, waveform: torch.Tensor, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        f_spec = self.spec_encoder(spectrogram)
        f_wave = self.wave_encoder(waveform)
        a_spec, a_wave = self.gates()
        fused = torch.cat([a_spec * f_spec, a_wave * f_wave], dim=1)
        return self.head(fused)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


_MODEL_REGISTRY = {
    "spectrogram_baseline": SpectrogramBaseline,
    "waveform_baseline": WaveformBaseline,
    "audiofuse_lcf": AudioFuseLCF,
    "early_fusion": EarlyFusion,
    "tensor_fusion": TensorFusion,
    "weighted_late_fusion": WeightedLateFusion,
}


def available_models() -> Tuple[str, ...]:
    """Return the registered model names."""
    return tuple(_MODEL_REGISTRY.keys())


def build_model(name: str, model_cfg: Mapping) -> nn.Module:
    """Instantiate a model by name.

    Args:
        name: One of the keys returned by :func:`available_models`.
        model_cfg: The ``model:`` block of the YAML configuration.

    Returns:
        The constructed ``nn.Module``.
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available: {', '.join(sorted(_MODEL_REGISTRY.keys()))}."
        )
    return _MODEL_REGISTRY[name](model_cfg)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
