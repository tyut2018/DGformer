import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DualPathSpatioTemporalFusion(nn.Module):

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 dropout: float = 0.1, n_layers: int = 2,
                 graph_dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _FusionBlock(d_model, n_heads, dropout, graph_dropout)
            for _ in range(n_layers)
        ])

    def forward(self, node_repr: torch.Tensor, A_dyn: torch.Tensor, 
                return_intermediate: bool = False) -> torch.Tensor:
        out = node_repr
        intermediates = [node_repr]
        for layer in self.layers:
            out = layer(out, A_dyn)
            if return_intermediate:
                intermediates.append(out)
        
        return intermediates if return_intermediate else out


class _FusionBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 graph_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.graph_dropout = graph_dropout

        self.r_q = nn.Linear(d_model, d_model)
        self.r_k = nn.Linear(d_model, d_model)
        self.r_v = nn.Linear(d_model, d_model)
        self.r_out_proj = nn.Linear(d_model, d_model)

        self.g_linear = nn.Linear(d_model, d_model)
        self.g_out_proj = nn.Linear(d_model, d_model)
        self.g_dropout = nn.Dropout(dropout)

        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.fusion_gate[-1].bias, -2.0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_dyn: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        residual = x

        Q = self.r_q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.r_k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.r_v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        r_out = torch.matmul(attn, V)
        r_out = r_out.transpose(1, 2).contiguous().view(B, N, D)
        r_out = self.r_out_proj(r_out)

        A = A_dyn
        if self.training and self.graph_dropout > 0:
            mask = torch.bernoulli(torch.full_like(A, 1.0 - self.graph_dropout))
            A = A * mask
            A = A / (A.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        g_out = torch.bmm(A, x)
        g_out = self.g_linear(g_out)
        g_out = F.gelu(g_out)
        g_out = self.g_dropout(g_out)
        g_out = self.g_out_proj(g_out)

        fused = torch.cat([r_out, g_out], dim=-1)
        beta = torch.sigmoid(self.fusion_gate(fused))
        out = r_out + beta * g_out

        out = self.norm1(out + residual)

        out = self.norm3(out + self.ffn(self.norm2(out)))

        return out
