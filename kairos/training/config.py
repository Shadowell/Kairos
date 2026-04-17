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
    #: Which market produced the dataset. Used by the feature builder to
    #: pick the right 8 market-specific exog columns and by training-side
    #: heuristics (e.g. default ``return_horizon``). Must match what was
    #: passed to ``kairos-prepare --market``; falls back to the value in
    #: ``<dataset_path>/meta.json`` if present.
    market: str = "ashare"
    #: Native bar frequency of the dataset (``daily`` / ``1min`` / ...).
    #: Mostly a documentation hint right now, but the backtest script uses
    #: it to pick a reasonable cross-sectional aggregation rule.
    freq: str = "daily"
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


# ---------------------------------------------------------------------------
# Market-aware presets
# ---------------------------------------------------------------------------
# The A-share daily baseline is the status quo; crypto-1min inherits most of
# it but shifts a few knobs that differ meaningfully at minute cadence:
#   - ``return_horizon`` bumps to 30 bars (~30 minutes) so the regression
#     head targets a window large enough to wash out microstructure noise.
#   - ``freq`` is flagged so the backtest can aggregate by calendar minute
#     instead of trading day.
#   - ``predict_window`` is widened to match the longer return horizon.
_PRESETS: dict[str, dict] = {
    "ashare-daily": {
        "market": "ashare",
        "freq": "daily",
    },
    "crypto-1min": {
        "market": "crypto",
        "freq": "1min",
        "lookback_window": 256,
        "predict_window": 32,
        "return_horizon": 30,
        # Crypto bars are noisier per unit time; a slightly larger CE weight
        # biases the model toward faithfully reproducing the token
        # distribution before leaning on the quantile regression head.
        "ce_weight": 0.7,
        "quantile_weight": 1.5,
    },
}


def preset_for(name: str) -> dict:
    """Return a parameter dict for the named preset.

    Usage::

        cfg = TrainConfig(**preset_for("crypto-1min"),
                          dataset_path="./finetune/data/crypto_1min")

    Unknown names raise ``KeyError`` so typos surface immediately. This is
    a deliberately thin layer: it exists so downstream configs can pick up
    a reasonable default without copy-pasting the same overrides.
    """

    try:
        return dict(_PRESETS[name])
    except KeyError as e:
        raise KeyError(
            f"Unknown preset {name!r}; available: {list(_PRESETS)}"
        ) from e


def available_presets() -> list[str]:
    """Return the list of preset names known to :func:`preset_for`."""
    return list(_PRESETS)
