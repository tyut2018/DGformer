import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true, eps=1e-8):
    min_len = min(pred.size, true.size)
    pred_flat = pred.flatten()[:min_len]
    true_flat = true.flatten()[:min_len]
    mask = np.abs(true_flat) > 0.01
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((pred_flat[mask] - true_flat[mask]) / (np.abs(true_flat[mask]) + eps))) * 100

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = pred.flatten()
    true = true.flatten()
    return {
        'mae': MAE(pred, true),
        'mse': MSE(pred, true),
        'rmse': RMSE(pred, true),
        'mape': MAPE(pred, true),
    }

def compute_per_horizon_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    H = pred.shape[1]
    results = {}
    for h in range(H):
        p = pred[:, h, :, :].flatten()
        t = true[:, h, :, :].flatten()
        results[h] = {
            'mse': MSE(p, t),
            'mae': MAE(p, t),
        }
    return results

def skill_score(pred_mae: float, persistence_mae: float) -> float:
    if persistence_mae < 1e-8:
        return 0.0
    return 1.0 - pred_mae / persistence_mae


class ForecastLoss(nn.Module):

    def __init__(self, alpha_freq: float = 0.1, beta_decomp: float = 0.05,
                 gamma_spectral: float = 0.1, huber_delta: float = 1.0,
                 loss_type: str = 'huber'):
        super().__init__()
        self.alpha_freq = alpha_freq
        self.beta_decomp = beta_decomp
        self.gamma_spectral = gamma_spectral
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.main_loss = nn.MSELoss()
        else:
            self.main_loss = nn.HuberLoss(delta=huber_delta)

    def forward(self, output: dict, target: torch.Tensor) -> dict:
        pred = output['prediction']
        trend = output['trend']
        residual = output['residual']

        loss_main = self.main_loss(pred, target)

        loss_freq = self._freq_loss(pred, target)

        loss_decomp = self._smoothness_loss(trend)

        loss_spectral = self._spectral_separation(trend, residual)

        total = (loss_main
                 + self.alpha_freq * loss_freq
                 + self.beta_decomp * loss_decomp
                 + self.gamma_spectral * loss_spectral)

        return {
            'total': total,
            'main': loss_main,
            'freq': loss_freq,
            'decomp': loss_decomp,
            'spectral_sep': loss_spectral,
        }

    def _freq_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        loss = F.l1_loss(pred_fft.abs(), target_fft.abs())
        return loss

    def _smoothness_loss(self, trend: torch.Tensor) -> torch.Tensor:
        if trend.shape[1] < 3:
            return torch.tensor(0.0, device=trend.device)
        diff2 = trend[:, 2:] - 2 * trend[:, 1:-1] + trend[:, :-2]
        return diff2.abs().mean()

    @staticmethod
    def _spectral_separation(trend: torch.Tensor,
                             residual: torch.Tensor) -> torch.Tensor:
        if trend.shape[1] < 2:
            return torch.tensor(0.0, device=trend.device)
        trend_fft = torch.fft.rfft(trend, dim=1).abs()
        resid_fft = torch.fft.rfft(residual, dim=1).abs()
        trend_spec = trend_fft / (trend_fft.sum(dim=1, keepdim=True) + 1e-8)
        resid_spec = resid_fft / (resid_fft.sum(dim=1, keepdim=True) + 1e-8)
        overlap = torch.min(trend_spec, resid_spec).sum(dim=1).mean()
        return overlap


class RampLoss(nn.Module):

    def __init__(self, gamma: float = 5.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] < 2:
            return F.l1_loss(pred, target)

        ramps = (target[:, 1:] - target[:, :-1]).abs()
        max_ramp = ramps.max().clamp(min=1e-6)
        weights = 1.0 + self.gamma * ramps / max_ramp

        errors = (pred[:, 1:] - target[:, 1:]).abs()
        loss = (weights * errors).mean()
        return loss


class ModelEMA:

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class EarlyStopping:

    def __init__(self, patience: int = 15, delta: float = 1e-5,
                 save_path: str = 'checkpoints', no_save: bool = False):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.no_save = no_save
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

        os.makedirs(save_path, exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"[EarlyStopping] Triggered after {self.counter} epochs")
        else:
            self.best_loss = val_loss
            self._save_checkpoint(model)
            self.counter = 0

    def _save_checkpoint(self, model: nn.Module):
        self.best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        if self.no_save:
            return
        path = os.path.join(self.save_path, 'best_model.pt')
        torch.save(model.state_dict(), path)

    def load_best(self, model: nn.Module):
        if self.no_save:
            if self.best_state_dict is None:
                raise RuntimeError('No best model state recorded in memory.')
            model.load_state_dict(self.best_state_dict)
            return model
        path = os.path.join(self.save_path, 'best_model.pt')
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]
