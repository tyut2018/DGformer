import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgPool(nn.Module):

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size - 1
        B, H, N, D = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N * D, 1, H)
        x_pad = F.pad(x_flat, (pad, 0), mode='replicate')
        smoothed = F.avg_pool1d(x_pad, self.kernel_size, stride=1)
        smoothed = smoothed.reshape(B, N, D, H).permute(0, 3, 1, 2)
        return smoothed


class TrendResidualForecastHead(nn.Module):

    def __init__(self, d_model: int = 128, pred_len: int = 96,
                 num_sites: int = 137, out_channels: int = 1,
                 dropout: float = 0.1, use_horizon_repr: bool = True,
                 ma_kernel_size: int = 25):
        super().__init__()
        self.pred_len = pred_len
        self.num_sites = num_sites
        self.out_channels = out_channels
        self.use_horizon_repr = use_horizon_repr
        self.d_model = d_model

        self.moving_avg = MovingAvgPool(kernel_size=ma_kernel_size)

        self.trend_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_channels),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_channels),
        )

        if not use_horizon_repr:
            self.horizon_proj = nn.Linear(d_model, pred_len * d_model)

    def forward(self, horizon_repr: torch.Tensor,
                spatial_repr: torch.Tensor = None) -> dict:
        B = horizon_repr.shape[0]

        if not self.use_horizon_repr:
            N, D = horizon_repr.shape[1], horizon_repr.shape[2]
            horizon_repr = self.horizon_proj(horizon_repr)
            horizon_repr = horizon_repr.view(B, N, self.pred_len, D)
            horizon_repr = horizon_repr.permute(0, 2, 1, 3)

        H = horizon_repr.shape[1]
        N = horizon_repr.shape[2]

        trend_input = self.moving_avg(horizon_repr)
        resid_input = horizon_repr - trend_input

        if spatial_repr is not None:
            spatial_exp = spatial_repr.unsqueeze(1).expand(B, H, N, -1)
            resid_input = resid_input + spatial_exp

        y_trend = self.trend_head(trend_input)
        y_residual = self.residual_head(resid_input)

        y_hat = y_trend + y_residual

        return {
            'prediction': y_hat,
            'trend': y_trend,
            'residual': y_residual,
        }

    @staticmethod
    def spectral_separation_loss(trend: torch.Tensor,
                                 residual: torch.Tensor) -> torch.Tensor:
        trend_fft = torch.fft.rfft(trend, dim=1).abs()
        resid_fft = torch.fft.rfft(residual, dim=1).abs()
        trend_spec = trend_fft / (trend_fft.sum(dim=1, keepdim=True) + 1e-8)
        resid_spec = resid_fft / (resid_fft.sum(dim=1, keepdim=True) + 1e-8)
        overlap = torch.min(trend_spec, resid_spec).sum(dim=1).mean()
        return overlap
