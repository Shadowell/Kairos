"""Centralized training configuration.

Keep hyperparameters in one place and edit the instance (or pass overrides via
CLI). Works for both tokenizer and predictor training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    # ---------------- Data & Features ----------------
    dataset_path: str = "./finetune/data/processed_datasets"
    lookback_window: int = 90
    predict_window: int = 10
    max_context: int = 512
    feature_list: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "vol", "amt"]
    )
    time_feature_list: List[str] = field(
        default_factory=lambda: ["minute", "hour", "weekday", "day", "month"]
    )
    n_exog: int = 32  # must match kairos.data.features.EXOG_COLS length

    # ---------------- Training ----------------
    seed: int = 100
    clip: float = 5.0
    epochs: int = 15
    log_interval: int = 100
    batch_size: int = 50
    n_train_iter: int = 50000   # 1000 step * batch 50
    n_val_iter: int = 10000     # 200 step * batch 50
    tokenizer_learning_rate: float = 2e-4
    predictor_learning_rate: float = 5e-6
    warmup_pct: float = 0.1     # OneCycleLR 的 pct_start
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.05
    accumulation_steps: int = 1
    num_workers: int = 2
    patience: int = 3           # 连续 N 个 epoch val 不降就停

    # ---------------- Logging / Checkpoints ----------------
    save_path: str = "./artifacts/checkpoints"
    tokenizer_save_folder_name: str = "tokenizer"
    predictor_save_folder_name: str = "predictor"
    use_comet: bool = False
    comet_api_key: str = ""
    comet_project: str = "kairos"
    comet_workspace: str = ""

    # ---------------- Pretrained ----------------
    pretrained_tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base"
    pretrained_predictor_path: str = "NeoQuasar/Kronos-small"

    # ---------------- Exogenous / Return head ----------------
    use_exog: bool = True
    use_return_head: bool = True
    return_horizon: int = 5
    n_quantiles: int = 9
    ce_weight: float = 0.5
    quantile_weight: float = 2.0
    unfreeze_last_n: int = 1
