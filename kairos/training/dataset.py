"""PyTorch dataset for K-line + exogenous factors (market-agnostic).

Reads pickle files produced by :mod:`kairos.data.prepare_dataset`:

    dataset_path/
        train_data.pkl    # {symbol: DataFrame[open, high, low, close, vol, amt]}
        val_data.pkl
        test_data.pkl
        exog_train.pkl    # {symbol: DataFrame[<EXOG_COLS>]} aligned by datetime index
        exog_val.pkl
        exog_test.pkl
        meta.json         # optional, produced by kairos-prepare; records the
                          #   market / freq / exog schema

The dataset itself makes no instrument-specific assumptions: time features and
standardisation are applied identically to spot and perpetual-swap crypto bars.
The pickle layout is shared so a single trainer works for both.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kairos.training.config import TrainConfig


class KronosSequenceDataset(Dataset):
    """Sliding-window dataset for Kronos training on any K-line market.

    The historical name :class:`AShareKronosDataset` is kept as an alias at
    the bottom of the module for backward compatibility.
    """

    def __init__(self, split: str, cfg: TrainConfig):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.split = split
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.window = cfg.lookback_window + cfg.predict_window + 1

        root = Path(cfg.dataset_path)
        with open(root / f"{split}_data.pkl", "rb") as f:
            self.data: dict[str, pd.DataFrame] = pickle.load(f)

        exog_file = root / f"exog_{split}.pkl"
        if cfg.use_exog and exog_file.exists():
            with open(exog_file, "rb") as f:
                self.exog: dict[str, pd.DataFrame] = pickle.load(f)
        else:
            self.exog = {}

        self.indices: list[tuple[str, int]] = []
        for sym, df in list(self.data.items()):
            df = df.reset_index()
            if "datetime" not in df.columns:
                df = df.rename(columns={df.columns[0]: "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            n = len(df) - self.window + 1
            if n <= 0:
                continue
            df["minute"] = df["datetime"].dt.minute
            df["hour"] = df["datetime"].dt.hour
            df["weekday"] = df["datetime"].dt.weekday
            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            self.data[sym] = df
            for i in range(n):
                self.indices.append((sym, i))

        limit = cfg.n_train_iter if split == "train" else cfg.n_val_iter
        self.n_samples = min(limit, len(self.indices))
        print(f"[{split.upper()}] pool={len(self.indices)}, using {self.n_samples}/epoch.")

    def set_epoch_seed(self, epoch: int) -> None:
        self.rng.seed(self.cfg.seed + epoch)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sym, start = self.indices[self.rng.randint(0, len(self.indices) - 1)]
        df = self.data[sym]
        end = start + self.window
        win = df.iloc[start:end]

        x = win[self.cfg.feature_list].values.astype(np.float32)
        x_stamp = win[self.cfg.time_feature_list].values.astype(np.float32)

        past = x[: self.cfg.lookback_window]
        mu, sd = past.mean(0), past.std(0)
        x = np.clip((x - mu) / (sd + 1e-5), -self.cfg.clip, self.cfg.clip)

        if self.cfg.use_exog and sym in self.exog:
            edf = self.exog[sym]
            if "datetime" in edf.columns:
                ei = edf.set_index("datetime")
            else:
                ei = edf
            dates = pd.to_datetime(win["datetime"].values)
            exog_win = (
                ei.reindex(dates)
                .fillna(0.0)
                .values.astype(np.float32)
            )
            if exog_win.shape[1] != self.cfg.n_exog:
                # shape mismatch → pad or truncate to match config
                target = self.cfg.n_exog
                if exog_win.shape[1] < target:
                    pad = np.zeros((exog_win.shape[0], target - exog_win.shape[1]),
                                   dtype=np.float32)
                    exog_win = np.concatenate([exog_win, pad], axis=1)
                else:
                    exog_win = exog_win[:, :target]
        else:
            exog_win = np.zeros((self.window, self.cfg.n_exog), dtype=np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_stamp),
            torch.from_numpy(exog_win),
        )


# Backwards-compatible alias. Historical call sites (training scripts,
# external notebooks) import ``AShareKronosDataset`` by name; keep it working
# so the crypto rollout stays a non-breaking change.
AShareKronosDataset = KronosSequenceDataset


__all__ = ["KronosSequenceDataset", "AShareKronosDataset"]
