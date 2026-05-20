import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):

    def __init__(self, seq_len: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, in_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.proj = nn.Linear(patch_len * in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, C = x.shape

        x = x.permute(0, 2, 1, 3).reshape(B * N, L, C)

        patches = x.unfold(1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2)
        patches = patches.reshape(B * N, self.num_patches, self.patch_len * C)

        patches = self.proj(patches)
        patches = self.norm(patches)
        patches = patches + self.pos_embed
        patches = self.dropout(patches)

        return patches
