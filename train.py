import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

from model import DGFormer
from data_loader import RenewableEnergyDataProvider, generate_synthetic_data
from utils import (
    setup_seed, compute_metrics, compute_per_horizon_metrics,
    ForecastLoss, RampLoss, ModelEMA, EarlyStopping, WarmupCosineScheduler
)


def parse_args():
    parser = argparse.ArgumentParser(description='DGFormer Training')

    parser.add_argument('--data', type=str, default='synthetic')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--target_channel', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=0)

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_enc_layers', type=int, default=3)
    parser.add_argument('--n_fusion_layers', type=int, default=2)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--top_k_static', type=int, default=10)
    parser.add_argument('--top_k_dynamic', type=int, default=10)
    parser.add_argument('--context_dim', type=int, default=16)
    parser.add_argument('--static_dim', type=int, default=8)
    parser.add_argument('--n_horizon_groups', type=int, default=4)
    parser.add_argument('--graph_dropout', type=float, default=0.0)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--es_metric', type=str, default='mse', choices=['mse', 'mae'])
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--alpha_freq', type=float, default=0.1)
    parser.add_argument('--beta_decomp', type=float, default=0.05)
    parser.add_argument('--gamma_spectral', type=float, default=0.1)
    parser.add_argument('--use_ramp_loss', action='store_true')
    parser.add_argument('--ramp_weight', type=float, default=0.1)

    parser.add_argument('--no_dynamic_graph', action='store_true')
    parser.add_argument('--no_flow_graph', action='store_true')
    parser.add_argument('--no_horizon_routing', action='store_true')
    parser.add_argument('--no_dual_head', action='store_true')
    parser.add_argument('--dense_graph', action='store_true')
    parser.add_argument('--channel_independent', action='store_true')

    parser.add_argument('--use_revin', action='store_true')
    parser.add_argument('--use_skip', action='store_true')
    parser.add_argument('--loss_type', type=str, default='huber', choices=['huber', 'mse'])
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--no_save_model', action='store_true')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, ramp_loss_fn=None, ramp_weight=0.1,
                    static_features=None, ema=None):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(x, static_features=static_features)
                loss_dict = criterion(output, y)
                loss = loss_dict['total']
                if ramp_loss_fn is not None:
                    loss = loss + ramp_weight * ramp_loss_fn(output['prediction'], y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
        else:
            output = model(x, static_features=static_features)
            loss_dict = criterion(output, y)
            loss = loss_dict['total']
            if ramp_loss_fn is not None:
                loss = loss + ramp_weight * ramp_loss_fn(output['prediction'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, static_features=None):
    model.eval()
    total_loss = 0
    n_batches = 0
    preds, trues = [], []

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        output = model(x, static_features=static_features)
        loss_dict = criterion(output, y)
        total_loss += loss_dict['total'].item()
        n_batches += 1

        pred = output['prediction'].cpu().numpy()
        true = y.cpu().numpy()
        preds.append(pred)
        trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    metrics = compute_metrics(preds, trues)
    metrics['loss'] = total_loss / max(n_batches, 1)

    horizon_metrics = compute_per_horizon_metrics(preds, trues)

    return metrics, horizon_metrics, preds, trues


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    if args.data == 'synthetic':
        synth_path = os.path.join(args.save_dir, 'synthetic_data.npz')
        os.makedirs(args.save_dir, exist_ok=True)
        generate_synthetic_data(T=5000, N=20, C=7, save_path=synth_path)
        args.data = synth_path

    data_provider = RenewableEnergyDataProvider(
        data_path=args.data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        target_channel=args.target_channel,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        max_len=args.max_len,
    )

    static_features = None
    if data_provider.static_features is not None:
        static_features = torch.FloatTensor(data_provider.static_features).to(device)
        args.static_dim = static_features.shape[1]

    model_config = {
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'num_sites': data_provider.num_sites,
        'in_channels': data_provider.in_channels,
        'out_channels': 1,
        'patch_len': args.patch_len,
        'stride': args.stride,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'n_heads': args.n_heads,
        'n_enc_layers': args.n_enc_layers,
        'n_fusion_layers': args.n_fusion_layers,
        'static_dim': args.static_dim,
        'top_k_static': min(args.top_k_static, data_provider.num_sites - 1),
        'top_k_dynamic': min(args.top_k_dynamic, data_provider.num_sites - 1),
        'context_dim': args.context_dim,
        'use_flow_graph': not args.no_flow_graph,
        'n_horizon_groups': args.n_horizon_groups,
        'dropout': args.dropout,
        'graph_dropout': args.graph_dropout,
        'ablation_no_dynamic_graph': args.no_dynamic_graph,
        'ablation_no_horizon_routing': args.no_horizon_routing,
        'ablation_no_dual_head': args.no_dual_head,
        'use_revin': args.use_revin,
        'use_skip': args.use_skip,
        'channel_independent': args.channel_independent,
    }

    if args.dense_graph:
        model_config['top_k_dynamic'] = data_provider.num_sites - 1

    model = DGFormer(**model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] DGFormer | Parameters: {n_params:,}")
    print(f"[Model] Config: {json.dumps(model_config, indent=2)}")

    if args.no_dynamic_graph:
        print("[Ablation] Dynamic graph DISABLED")
    if args.no_horizon_routing:
        print("[Ablation] Horizon-aware routing DISABLED")
    if args.no_dual_head:
        print("[Ablation] Dual forecast head DISABLED")

    criterion = ForecastLoss(
        alpha_freq=args.alpha_freq,
        beta_decomp=args.beta_decomp,
        gamma_spectral=args.gamma_spectral,
        loss_type=args.loss_type,
    )

    ramp_loss_fn = RampLoss() if args.use_ramp_loss else None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    early_stopping = EarlyStopping(
        patience=args.patience, 
        save_path=args.save_dir,
        no_save=args.no_save_model
    )

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        print(f"[EMA] Enabled with decay={args.ema_decay}")

    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs | LR={args.lr} | Batch={args.batch_size}")
    print(f"Task: seq_len={args.seq_len} -> pred_len={args.pred_len}")
    print(f"Sites: {data_provider.num_sites} | Features: {data_provider.in_channels}")
    print(f"{'='*60}\n")

    best_val_mae = float('inf')
    train_history = []

    for epoch in range(args.epochs):
        t_start = time.time()

        train_loss = train_one_epoch(
            model, data_provider.train_loader, optimizer, criterion, device,
            scaler=scaler, ramp_loss_fn=ramp_loss_fn, ramp_weight=args.ramp_weight,
            static_features=static_features, ema=ema)

        if ema is not None:
            ema.apply(model)
        val_metrics, val_horizon, _, _ = evaluate(
            model, data_provider.val_loader, criterion, device,
            static_features=static_features)
        if ema is not None:
            ema.restore(model)

        scheduler.step()

        elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[0]['lr']

        log_str = (
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            log_str += " ★"

        print(log_str)

        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'lr': current_lr,
        })

        if ema is not None:
            ema.apply(model)
        early_stopping(val_metrics[args.es_metric], model)
        if ema is not None:
            ema.restore(model)
        if early_stopping.early_stop:
            break

    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")

    model = early_stopping.load_best(model)

    test_metrics, test_horizon, test_preds, test_trues = evaluate(
        model, data_provider.test_loader, criterion, device,
        static_features=static_features)

    print(f"\nTest Results (normalized):")
    print(f"  MSE:  {test_metrics['mse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")

    def _to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return obj

    results = _to_serializable({
        'config': model_config,
        'args': vars(args),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_val_mae': float(best_val_mae),
        'train_history': train_history,
    })

    results_path = os.path.join(args.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Results -> {results_path}")

    np.savez(
        os.path.join(args.save_dir, 'predictions.npz'),
        pred=test_preds, true=test_trues)
    print(f"[Saved] Predictions -> {os.path.join(args.save_dir, 'predictions.npz')}")

    return test_metrics


if __name__ == '__main__':
    main()
