import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict


class MultiSiteTimeSeriesDataset(Dataset):

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int,
                 target_channel: int = 0, time_enc: np.ndarray = None):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.time_enc = time_enc

        T, N, C = self.data.shape
        self.target_channel = target_channel if target_channel >= 0 else C + target_channel
        self.num_samples = T - seq_len - pred_len + 1

        assert self.num_samples > 0, (
            f"Not enough data: T={T}, seq_len={seq_len}, pred_len={pred_len}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        x = self.data[s_begin:s_end]
        y = self.data[s_end:r_end, :, self.target_channel:self.target_channel+1]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.time_enc is not None:
            te = torch.from_numpy(self.time_enc[s_begin:s_end].astype(np.float32))
            return x, y, te

        return x, y


class RenewableEnergyDataProvider:

    def __init__(
        self,
        data_path: str,
        seq_len: int = 96,
        pred_len: int = 96,
        batch_size: int = 32,
        target_channel: int = -1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        num_workers: int = 4,
        scale: bool = True,
        max_len: int = 0,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.target_channel = target_channel
        self.scale = scale

        raw_data, static_features = self._load_data(data_path)
        if max_len > 0 and raw_data.shape[0] > max_len:
            raw_data = raw_data[:max_len]
        T, N, C = raw_data.shape
        self.num_sites = N
        self.in_channels = C
        self.static_features = static_features

        print(f"[Data] Loaded: T={T}, N={N}, C={C}")
        print(f"[Data] seq_len={seq_len}, pred_len={pred_len}")

        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))

        train_data = raw_data[:train_end]
        val_data = raw_data[train_end - seq_len:val_end]
        test_data = raw_data[val_end - seq_len:]

        if scale:
            self.mean, self.std = self._compute_stats(train_data)
            train_data = self._normalize(train_data)
            val_data = self._normalize(val_data)
            test_data = self._normalize(test_data)
            print(f"[Data] Normalized (train stats): mean shape={self.mean.shape}")

        self.train_dataset = MultiSiteTimeSeriesDataset(
            train_data, seq_len, pred_len, target_channel)
        self.val_dataset = MultiSiteTimeSeriesDataset(
            val_data, seq_len, pred_len, target_channel)
        self.test_dataset = MultiSiteTimeSeriesDataset(
            test_data, seq_len, pred_len, target_channel)

        print(f"[Data] Samples: train={len(self.train_dataset)}, "
              f"val={len(self.val_dataset)}, test={len(self.test_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    def _load_data(self, data_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        static_features = None

        if data_path.endswith('.npz'):
            f = np.load(data_path, allow_pickle=True)
            if 'data' in f:
                raw_data = f['data']
            elif 'power' in f:
                raw_data = f['power']
            else:
                key = list(f.keys())[0]
                raw_data = f[key]
                print(f"[Data] Using key '{key}' from npz")

            if 'static' in f:
                static_features = f['static']

        elif data_path.endswith('.npy'):
            raw_data = np.load(data_path)

        elif data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < len(df.columns):
                dropped = set(df.columns) - set(numeric_cols)
                print(f"[Data] CSV: dropped non-numeric columns: {dropped}")
            df = df[numeric_cols]
            raw_data = df.values.astype(np.float32)
            if raw_data.ndim == 2:
                raw_data = raw_data[:, :, np.newaxis]

        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        if raw_data.ndim == 2:
            raw_data = raw_data[:, :, np.newaxis]

        return raw_data.astype(np.float32), static_features

    def _compute_stats(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8
        return mean, std

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor,
                          channel: int = None) -> torch.Tensor:
        device = data.device
        if channel is not None:
            mean = torch.tensor(self.mean[0, :, channel], device=device).reshape(1, 1, -1, 1)
            std = torch.tensor(self.std[0, :, channel], device=device).reshape(1, 1, -1, 1)
        else:
            mean = torch.tensor(self.mean, device=device).unsqueeze(0)
            std = torch.tensor(self.std, device=device).unsqueeze(0)
        return data * std + mean


def generate_synthetic_data(
    T: int = 5000, N: int = 20, C: int = 7,
    save_path: str = None
) -> np.ndarray:
    np.random.seed(42)
    t = np.arange(T).astype(np.float32)

    diurnal = np.sin(2 * np.pi * t / 24)
    seasonal = np.sin(2 * np.pi * t / (24 * 365))

    locs = np.random.rand(N, 2) * 100
    dist = np.sqrt(((locs[:, None] - locs[None, :]) ** 2).sum(-1))
    corr = np.exp(-dist / 30)

    data = np.zeros((T, N, C), dtype=np.float32)
    for c in range(C):
        noise = np.random.randn(T, N) @ corr.T * 0.3
        if c == 0:
            base = (diurnal[:, None] * 0.4 + seasonal[:, None] * 0.2 + 0.5)
            power = base + noise
            events = np.random.rand(T) < 0.02
            power[events] *= np.random.uniform(0.1, 0.5, size=(events.sum(), N))
            data[:, :, c] = np.clip(power, 0, 1)
        else:
            data[:, :, c] = (diurnal[:, None] * (0.1 + 0.05 * c) +
                             noise + np.random.randn(T, N) * 0.1)

    static = np.column_stack([
        locs,
        np.random.rand(N) * 500,
        np.random.rand(N) * 100 + 50,
        np.random.rand(N, 4),
    ]).astype(np.float32)

    if save_path:
        np.savez(save_path, data=data, static=static)
        print(f"[Synthetic] Saved to {save_path}: data {data.shape}, static {static.shape}")

    return data
