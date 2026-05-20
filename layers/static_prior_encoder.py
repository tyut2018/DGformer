import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StaticPriorEncoder(nn.Module):

    def __init__(self, num_sites: int, static_dim: int = 8, d_model: int = 128,
                 top_k: int = 10, dropout: float = 0.1):
        super().__init__()
        self.num_sites = num_sites
        self.top_k = min(top_k, num_sites - 1)
        self.d_model = d_model

        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

        self.site_embed = nn.Embedding(num_sites, d_model)
        nn.init.xavier_uniform_(self.site_embed.weight)

    def forward(self, static_features: torch.Tensor = None):
        device = self.site_embed.weight.device

        if static_features is not None:
            static_features = static_features.to(device)
            site_emb = self.static_proj(static_features)
        else:
            idx = torch.arange(self.num_sites, device=device)
            site_emb = self.site_embed(idx)

        Q = self.W_q(site_emb)
        K = self.W_k(site_emb)
        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.d_model)

        scores.fill_diagonal_(float('-inf'))

        if self.top_k < self.num_sites - 1:
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            scores = mask

        A_stat = F.softmax(scores, dim=-1)

        return A_stat, site_emb
