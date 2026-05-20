import torch
import torch.nn as nn
import math


class PatchTemporalEncoder(nn.Module):

    def __init__(self, d_model: int = 128, n_heads: int = 8, d_ff: int = 256,
                 n_layers: int = 3, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(patches)
        encoded = self.norm(encoded)
        return encoded
