

from __future__ import annotations

import torch
import torch.nn as nn




class PatchEmbed(nn.Module):
    """Conv2D-based patch embedding (image -> sequence of token embeddings)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size           # 14 for default config
        self.num_patches = self.grid_size ** 2            # 196
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W) -> (B, num_patches, embed_dim)``."""
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            raise ValueError(
                f"Expected input of size {self.img_size}x{self.img_size}, "
                f"got {tuple(x.shape[-2:])}."
            )
        x = self.proj(x)                                  # (B, E, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)                  # (B, N, E)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block, two-GeLU MLP per the AudioFuse figure."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(round(embed_dim * mlp_ratio))
        # Two GeLU activations as drawn in Fig. 1.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-block.
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        # MLP sub-block.
        x = x + self.mlp(self.norm2(x))
        return x



class SpectrogramViT(nn.Module):
    """Wide-and-shallow ViT producing a 192-dim spectrogram embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Learnable positional embedding (no [CLS] token).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return the full token sequence ``(B, N, embed_dim)``.

        Useful for the early-fusion variant which needs to inject extra
        tokens after the patch embedding step.
        """
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled 192-dim spectral feature vector ``f_spec``."""
        tokens = self.forward_tokens(x)
        return tokens.mean(dim=1)              # GlobalAveragePool over tokens
