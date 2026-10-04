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

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kairos.data.contracts import (
    MAIN_COLUMNS, TIME_COLUMNS, bar_delta, contiguous_starts, log_return_targets,
    normalize_window, validate_exog_frame, validate_main_frame,
)
from kairos.data.features import exog_cols_for
from kairos.training.config import TrainConfig


class KronosSequenceDataset(Dataset):
    """Sliding-window dataset for Kronos training on any K-line market.

    The historical name :class:`AShareKronosDataset` is kept as an alias at
    the bottom of the module for backward compatibility.
    """

    def __init__(self, split: str, cfg: TrainConfig, include_targets: bool = False):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.split = split
        self.cfg = cfg
        self.include_targets = include_targets
        self.window = cfg.lookback_window + cfg.predict_window + 1
        if cfg.lookback_window <= 0 or cfg.predict_window < 0:
            raise ValueError("lookback_window must be positive and predict_window nonnegative")
        if include_targets and not 0 < cfg.return_horizon <= cfg.predict_window + 1:
            raise ValueError("return_horizon exceeds the available future bars")
        if list(cfg.feature_list) != MAIN_COLUMNS:
            raise ValueError("feature_list must match the canonical main channel order")
        if list(cfg.time_feature_list) != TIME_COLUMNS:
            raise ValueError("time_feature_list must match the model's fixed time channel order")
        bar_delta(cfg.freq)
        columns = exog_cols_for(cfg.market)
        if cfg.n_exog != len(columns):
            raise ValueError("n_exog must match the exogenous schema")

        root = Path(cfg.dataset_path)
        meta_file = root / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            if "freq" in meta and bar_delta(meta["freq"]) != bar_delta(cfg.freq):
                raise ValueError("dataset frequency does not match cfg.freq")
            if "market" in meta and meta["market"] != cfg.market:
                raise ValueError("dataset market does not match cfg.market")
            if "exog_cols" in meta and meta["exog_cols"] != columns:
                raise ValueError("dataset exogenous schema does not match cfg.market")
        with open(root / f"{split}_data.pkl", "rb") as f:
            self.data: dict[str, pd.DataFrame] = pickle.load(f)

        exog_file = root / f"exog_{split}.pkl"
        if cfg.use_exog:
            if not exog_file.exists():
                raise ValueError(f"required exogenous dataset is missing: {exog_file}")
            with open(exog_file, "rb") as f:
                self.exog: dict[str, pd.DataFrame] = pickle.load(f)
        else:
            self.exog = {}

        self.indices: list[tuple[str, int]] = []
        # Stable symbol order gives every rank the same global window mapping.
        for sym in sorted(self.data):
            main = validate_main_frame(self.data[sym])
            if cfg.use_exog:
                if sym not in self.exog:
                    raise ValueError(f"required exogenous data is missing for {sym}")
                self.exog[sym] = validate_exog_frame(main, self.exog[sym], columns)
            starts = contiguous_starts(main.index, self.window, cfg.freq)
            df = main.reset_index()
            df["minute"] = df["datetime"].dt.minute
            df["hour"] = df["datetime"].dt.hour
            df["weekday"] = df["datetime"].dt.weekday
            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            self.data[sym] = df
            self.indices.extend((sym, int(i)) for i in starts)

        limit = cfg.n_train_iter if split == "train" else cfg.n_val_iter
        if not isinstance(limit, (int, np.integer)) or limit < 0:
            raise ValueError("sample limits must be nonnegative integers")
        self.n_samples = min(limit, len(self.indices))
        self.set_epoch_seed(0)
        print(f"[{split.upper()}] pool={len(self.indices)}, using {self.n_samples}/epoch.")

    def set_epoch_seed(self, epoch: int) -> None:
        """Select one distinct subset per epoch, independent of call/worker order."""
        seed = self.cfg.seed + (epoch if self.split == "train" else 0)
        self._sample_indices = np.random.default_rng(seed).choice(
            len(self.indices), size=self.n_samples, replace=False,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if index < 0 or index >= self.n_samples:
            raise IndexError(index)
        sym, start = self.indices[self._sample_indices[index]]
        df = self.data[sym]
        end = start + self.window
        win = df.iloc[start:end]

        x = normalize_window(win[self.cfg.feature_list].values,
                             self.cfg.lookback_window, self.cfg.clip)
        x_stamp = win[self.cfg.time_feature_list].values.astype(np.float32)

        if self.cfg.use_exog:
            exog_win = self.exog[sym].iloc[start:end].to_numpy(dtype=np.float32, copy=True)
        else:
            exog_win = np.zeros((self.window, self.cfg.n_exog), dtype=np.float32)

        sample = (
            torch.from_numpy(x),
            torch.from_numpy(x_stamp),
            torch.from_numpy(exog_win),
        )
        if self.include_targets:
            targets = log_return_targets(win["close"].to_numpy(),
                                         self.cfg.lookback_window - 1,
                                         self.cfg.return_horizon)
            return (*sample, torch.from_numpy(targets))
        return sample


# Backwards-compatible alias. Historical call sites (training scripts,
# external notebooks) import ``AShareKronosDataset`` by name; keep it working
# so the crypto rollout stays a non-breaking change.
AShareKronosDataset = KronosSequenceDataset


__all__ = ["KronosSequenceDataset", "AShareKronosDataset"]
