import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HorizonAwareRouting(nn.Module):

    def __init__(self, pred_len: int = 96, d_model: int = 128,
                 n_groups: int = 4, dropout: float = 0.1, n_layers: int = 2):
        super().__init__()
        self.pred_len = pred_len
        self.n_groups = n_groups
        self.d_model = d_model
        self.n_layers = n_layers

        self.horizon_embed = nn.Embedding(n_groups, d_model)

        self.hop_routing = nn.Sequential(
            nn.Linear(d_model, d_ff := d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_layers + 1)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, intermediates: list) -> torch.Tensor:
        B, N, D = intermediates[0].shape
        H = self.pred_len
        device = intermediates[0].device

        h_idx = torch.arange(H, device=device)
        group_idx = (h_idx * self.n_groups) // H
        h_emb = self.horizon_embed(group_idx)
        
        hop_weights = torch.softmax(self.hop_routing(h_emb), dim=-1)
        
        stacked_layers = torch.stack(intermediates, dim=0) 
        
        horizon_res = torch.sum(hop_weights.view(H, self.n_layers + 1, 1, 1, 1) * 
                                stacked_layers.unsqueeze(0), dim=1)
        
        horizon_res = horizon_res.transpose(0, 1)

        h_pe = h_emb.view(1, H, 1, D)
        out = self.layer_norm(horizon_res + h_pe)
        
        return self.dropout(out)
