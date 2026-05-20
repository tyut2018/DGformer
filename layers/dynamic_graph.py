import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicSimilarityGraph(nn.Module):

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        B, N, D = node_repr.shape

        Q = self.W_q(node_repr)
        K = self.W_k(node_repr)

        Q = Q.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        A_sim = attn.mean(dim=1)
        return A_sim


class ContextConditionedDirectionalGraph(nn.Module):

    def __init__(self, context_dim: int = 16, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.edge_proj = nn.Linear(d_model * 2, 1)

    def forward(self, node_repr: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        B, N, D = node_repr.shape

        if context is not None:
            ctx_emb = self.context_encoder(context)
            node_repr = node_repr + ctx_emb.unsqueeze(1)

        src = node_repr.unsqueeze(2).expand(B, N, N, D)
        dst = node_repr.unsqueeze(1).expand(B, N, N, D)
        edge_feat = torch.cat([src, dst], dim=-1)
        A_flow = self.edge_proj(edge_feat).squeeze(-1)
        A_flow = F.softmax(A_flow, dim=-1)

        return A_flow


class ConditionalDynamicGraph(nn.Module):

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 context_dim: int = 16, top_k: int = 10,
                 dropout: float = 0.1, use_flow: bool = True):
        super().__init__()
        self.top_k = top_k
        self.use_flow = use_flow

        self.sim_graph = DynamicSimilarityGraph(d_model, n_heads, dropout)
        if use_flow:
            self.dir_graph = ContextConditionedDirectionalGraph(context_dim, d_model, dropout)

        n_graphs = 3 if use_flow else 2
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_graphs),
        )

        self.diffusion_steps = 2
        self.diffusion_weight = nn.Parameter(torch.tensor(0.5))

    def _topk_sparsify(self, A: torch.Tensor, k: int) -> torch.Tensor:
        B, N, _ = A.shape
        k = min(k, N - 1)

        diag_mask = torch.eye(N, device=A.device, dtype=torch.bool).unsqueeze(0)
        A = A.masked_fill(diag_mask, float('-inf'))

        topk_vals, topk_idx = torch.topk(A, k, dim=-1)
        sparse_A = torch.full_like(A, float('-inf'))
        sparse_A.scatter_(2, topk_idx, topk_vals)

        sparse_A = F.softmax(sparse_A, dim=-1)
        return sparse_A

    def _multi_hop_diffusion(self, A: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.diffusion_weight)
        A_sq = torch.bmm(A, A)
        A_diff = alpha * A + (1 - alpha) * A_sq
        A_diff = A_diff / (A_diff.sum(dim=-1, keepdim=True) + 1e-8)
        return A_diff

    def forward(self, node_repr: torch.Tensor, A_stat: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        B, N, D = node_repr.shape

        A_sim = self.sim_graph(node_repr)

        A_stat_batch = A_stat.unsqueeze(0).expand(B, -1, -1)

        if self.use_flow:
            A_dir = self.dir_graph(node_repr, context)

        node_mean = node_repr.mean(dim=1)
        gate_logits = self.gate_net(node_mean)
        gates = F.softmax(gate_logits, dim=-1)

        if self.use_flow:
            A_fused = (gates[:, 0:1].unsqueeze(-1) * A_stat_batch +
                       gates[:, 1:2].unsqueeze(-1) * A_sim +
                       gates[:, 2:3].unsqueeze(-1) * A_dir)
        else:
            A_fused = (gates[:, 0:1].unsqueeze(-1) * A_stat_batch +
                       gates[:, 1:2].unsqueeze(-1) * A_sim)

        A_fused = self._multi_hop_diffusion(A_fused)

        A_final = self._topk_sparsify(A_fused, self.top_k)

        return A_final
