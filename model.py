import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.patch_embedding import PatchEmbedding
from layers.temporal_encoder import PatchTemporalEncoder
from layers.static_prior_encoder import StaticPriorEncoder
from layers.dynamic_graph import ConditionalDynamicGraph
from layers.spatio_temporal_fusion import DualPathSpatioTemporalFusion
from layers.horizon_routing import HorizonAwareRouting
from layers.forecast_head import TrendResidualForecastHead


class RevIN(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features, 1))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            out_C = x.shape[-1]
            mean = self._mean[..., :out_C]
            stdev = self._stdev[..., :out_C]
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * stdev + mean
            return x


class DGFormer(nn.Module):

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        num_sites: int = 137,
        in_channels: int = 7,
        out_channels: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 3,
        n_fusion_layers: int = 2,
        static_dim: int = 8,
        top_k_static: int = 10,
        top_k_dynamic: int = 10,
        context_dim: int = 16,
        use_flow_graph: bool = True,
        n_horizon_groups: int = 4,
        dropout: float = 0.1,
        graph_dropout: float = 0.0,
        ablation_no_dynamic_graph: bool = False,
        ablation_no_horizon_routing: bool = False,
        ablation_no_dual_head: bool = False,
        use_revin: bool = False,
        use_skip: bool = False,
        channel_independent: bool = False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_sites = num_sites
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.channel_independent = channel_independent

        self.ablation_no_dynamic_graph = ablation_no_dynamic_graph
        self.ablation_no_horizon_routing = ablation_no_horizon_routing
        self.ablation_no_dual_head = ablation_no_dual_head

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=num_sites, affine=True)

        self.use_skip = use_skip
        if use_skip:
            self.skip_proj = nn.Sequential(
                nn.Linear(seq_len, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, pred_len),
            )
            self.skip_gate = nn.Parameter(torch.zeros(1))

        self.patch_embed = PatchEmbedding(
            seq_len=seq_len, patch_len=patch_len, stride=stride,
            d_model=d_model, in_channels=in_channels, dropout=dropout,
        )
        self.temporal_encoder = PatchTemporalEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_enc_layers, dropout=dropout,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_aggregator = nn.Sequential(
            nn.Linear(num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.static_prior = StaticPriorEncoder(
            num_sites=num_sites, static_dim=static_dim,
            d_model=d_model, top_k=top_k_static, dropout=dropout,
        )

        if not channel_independent:
            if not ablation_no_dynamic_graph:
                self.dynamic_graph = ConditionalDynamicGraph(
                    d_model=d_model, n_heads=n_heads // 2,
                    context_dim=context_dim, top_k=top_k_dynamic,
                    dropout=dropout, use_flow=use_flow_graph,
                )

            self.st_fusion = DualPathSpatioTemporalFusion(
                d_model=d_model, n_heads=n_heads // 2,
                dropout=dropout, n_layers=n_fusion_layers,
                graph_dropout=graph_dropout,
            )

            self.horizon_base_proj = nn.Linear(d_model, pred_len * d_model)
            
            if not ablation_no_horizon_routing:
                self.horizon_routing = HorizonAwareRouting(
                    pred_len=pred_len, d_model=d_model,
                    n_groups=n_horizon_groups, dropout=dropout,
                    n_layers=n_fusion_layers,
                )
                self.routing_gate = nn.Parameter(torch.zeros(1))
        else:
            self.ci_head = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, pred_len * out_channels),
            )

        if not ablation_no_dual_head and not channel_independent:
            ma_kernel = max(3, min(25, pred_len // 12))
            if ma_kernel % 2 == 0:
                ma_kernel += 1
            self.forecast_head = TrendResidualForecastHead(
                d_model=d_model, pred_len=pred_len,
                num_sites=num_sites, out_channels=out_channels,
                dropout=dropout, use_horizon_repr=True,
                ma_kernel_size=ma_kernel,
            )
        else:
            self.single_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, out_channels),
            )

        self.has_context = context_dim > 0 and not ablation_no_dynamic_graph and not channel_independent
        if self.has_context:
            self.context_pool = nn.Sequential(
                nn.Linear(d_model, context_dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor,
                static_features: torch.Tensor = None,
                weather_context: torch.Tensor = None) -> dict:
        B, L, N, C = x.shape
        D = self.d_model

        if self.use_revin:
            x = self.revin(x, mode='norm')

        last_val = x[:, -1:, :, :]
        x = x - last_val
        last_val_out = last_val[:, :, :, :self.out_channels]

        if self.use_skip:
            skip_in = x[:, :, :, 0:1].permute(0, 2, 3, 1)
            skip_out = self.skip_proj(skip_in)
            skip_out = skip_out.permute(0, 3, 1, 2)

        patches = self.patch_embed(x)
        encoded = self.temporal_encoder(patches)
        encoded_flat = encoded.reshape(B * N, -1)
        node_repr = self.patch_aggregator(encoded_flat)
        node_repr = node_repr.view(B, N, D)

        A_stat, site_emb = self.static_prior(static_features)
        node_repr = node_repr + site_emb.unsqueeze(0)

        if self.channel_independent:
            H = self.pred_len
            C_out = self.out_channels
            pred = self.ci_head(node_repr)
            pred = pred.view(B, N, H, C_out).permute(0, 2, 1, 3)
            A_dyn = A_stat.unsqueeze(0).expand(B, -1, -1)
            output = {
                'prediction': pred,
                'trend': pred,
                'residual': torch.zeros_like(pred),
            }
        else:
            if self.ablation_no_dynamic_graph:
                A_dyn = A_stat.unsqueeze(0).expand(B, -1, -1)
            else:
                if weather_context is None and self.has_context:
                    ctx = self.context_pool(node_repr.mean(dim=1))
                else:
                    ctx = weather_context
                A_dyn = self.dynamic_graph(node_repr, A_stat, ctx)

            intermediates = self.st_fusion(node_repr, A_dyn, return_intermediate=True)
            fused_repr = intermediates[-1]

            H, D = self.pred_len, self.d_model
            base_proj = self.horizon_base_proj(fused_repr)
            horizon_repr = base_proj.view(B, N, H, D).permute(0, 2, 1, 3)

            if not self.ablation_no_horizon_routing:
                routing_res = self.horizon_routing(intermediates)
                gate = torch.tanh(self.routing_gate) 
                horizon_repr = horizon_repr + gate * routing_res

        if not self.channel_independent:
            if self.ablation_no_dual_head:
                y_hat = self.single_head(horizon_repr)
                output = {
                    'prediction': y_hat,
                    'trend': y_hat,
                    'residual': torch.zeros_like(y_hat),
                }
            else:
                output = self.forecast_head(
                    horizon_repr, spatial_repr=fused_repr)

        if self.use_skip:
            gate = torch.sigmoid(self.skip_gate)
            output['prediction'] = output['prediction'] + gate * skip_out
            output['trend'] = output['trend'] + gate * skip_out

        output['prediction'] = output['prediction'] + last_val_out
        output['trend'] = output['trend'] + last_val_out

        if self.use_revin:
            output['prediction'] = self.revin(output['prediction'], mode='denorm')
            output['trend'] = self.revin(output['trend'], mode='denorm')
            output['residual'] = self.revin(output['residual'], mode='denorm')

        output['A_dynamic'] = A_dyn

        return output


def build_model(config: dict) -> DGFormer:
    return DGFormer(**config)
