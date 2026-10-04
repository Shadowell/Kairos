"""Fixed training-only scale estimates for raw log-return horizon losses."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from kairos.training.config import TrainConfig
from kairos.training.dataset import KronosSequenceDataset


_METHODS = {"uniform", "inverse_volatility"}
_CONFIG_KEYS = {"return_loss_weighting", "return_loss_weights", "return_scale_samples",
                "return_scale_floor"}


def _validate_settings(method, samples, floor, horizon) -> None:
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError("return_loss_weighting must be uniform or inverse_volatility")
    if type(samples) is not int or samples < 1:
        raise ValueError("return_scale_samples must be a positive integer")
    if type(floor) not in (float, int) or not math.isfinite(floor) or floor <= 0:
        raise ValueError("return_scale_floor must be finite and positive")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("return_horizon must be a positive integer")


def _validate_float32_weights(weights: np.ndarray) -> None:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        training_weights = weights.astype(np.float32)
    if not np.isfinite(training_weights).all() or (training_weights <= 0).any():
        raise ValueError(
            "return_loss_weights must remain finite and positive in float32; "
            "increase return_scale_floor"
        )


def estimate_horizon_weights(dataset: KronosSequenceDataset | None, cfg: TrainConfig) -> dict:
    """Estimate from global valid training indices, independent of epoch/rank.

    No loader sampling, normalized tensors or validation/test files are read.
    The fixed cap samples the entire continuous-window pool without replacement.
    """
    method = cfg.return_loss_weighting
    _validate_settings(method, cfg.return_scale_samples, cfg.return_scale_floor, cfg.return_horizon)
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError("return scale sampling requires a nonnegative integer seed")
    result = {
        "method": method, "weights": [1.] * cfg.return_horizon, "scales": None,
        "effective_samples": 0, "max_samples": cfg.return_scale_samples,
        "floor": cfg.return_scale_floor, "seed": cfg.seed, "source_split": None,
        "pool_size": 0,
    }
    if method == "uniform":
        return result
    if dataset is None or dataset.split != "train":
        raise ValueError("horizon scales must be estimated from the training dataset")
    if dataset.cfg.lookback_window != cfg.lookback_window or dataset.cfg.return_horizon != cfg.return_horizon:
        raise ValueError("training dataset and horizon weighting configuration disagree")
    pool_size = len(dataset.indices)
    if pool_size == 0:
        raise ValueError("cannot estimate horizon scales from an empty training window pool")
    count = min(cfg.return_scale_samples, pool_size)
    selected = np.random.default_rng(cfg.seed).choice(pool_size, size=count, replace=False)
    targets = np.empty((count, cfg.return_horizon), dtype=np.float64)
    for row, index in enumerate(selected):
        symbol, start = dataset.indices[int(index)]
        anchor = start + cfg.lookback_window - 1
        prices = dataset.data[symbol]["close"].iloc[anchor:anchor + cfg.return_horizon + 1].to_numpy(dtype=np.float64)
        if (len(prices) != cfg.return_horizon + 1 or not np.isfinite(prices).all()
                or (prices <= 0).any()):
            raise ValueError("training window contains invalid raw close targets")
        targets[row] = np.log(prices[1:]) - np.log(prices[0])
    scales = targets.std(axis=0, dtype=np.float64, ddof=0)
    denominators = np.maximum(scales, cfg.return_scale_floor)
    # Algebraically inverse volatility / mean(inverse volatility), avoiding
    # overflow for very small but valid floors.
    weights = denominators.min() / denominators
    weights /= weights.mean()
    _validate_float32_weights(weights)
    result.update(weights=weights.tolist(), scales=scales.tolist(), effective_samples=count,
                  source_split="train", pool_size=pool_size)
    return result


def return_loss_metadata(training_config: Mapping, horizon: int,
                         audit: dict | None = None, *, allow_historical: bool = False) -> dict:
    """Validate loss provenance without interpreting old v2 bundles as new runs."""
    present = _CONFIG_KEYS.intersection(training_config)
    if not present and allow_historical:
        if audit is not None:
            raise ValueError("return_loss metadata exists without its training configuration")
        return {"method": "historical_uniform", "weights": [1.] * horizon,
                "scales": None, "effective_samples": None, "source_split": None,
                "provenance": "v2_bundle_without_horizon_weighting_metadata"}
    if present != _CONFIG_KEYS:
        raise ValueError("training_config is missing return loss weighting fields")
    method = training_config["return_loss_weighting"]
    samples, floor = training_config["return_scale_samples"], training_config["return_scale_floor"]
    _validate_settings(method, samples, floor, horizon)
    if type(training_config.get("seed")) is not int or training_config["seed"] < 0:
        raise ValueError("return_loss training_config must include a nonnegative integer seed")
    raw_weights = training_config["return_loss_weights"]
    if not isinstance(raw_weights, list) or len(raw_weights) != horizon:
        raise ValueError("return_loss_weights must be resolved before saving the checkpoint")
    if any(type(weight) not in (int, float) for weight in raw_weights):
        raise ValueError("return_loss_weights must contain numbers")
    weights = np.asarray(raw_weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.mean(), 1., rtol=1e-6):
        raise ValueError("return_loss_weights must be finite, positive and have mean 1")
    _validate_float32_weights(weights)
    if method == "uniform" and not np.array_equal(weights, np.ones(horizon)):
        raise ValueError("uniform return_loss_weights must all equal 1")
    if audit is None:
        if method != "uniform":
            raise ValueError("inverse_volatility weighting requires training scale provenance")
        return {"method": method, "weights": weights.tolist(), "scales": None,
                "effective_samples": 0, "max_samples": samples, "floor": floor,
                "seed": training_config["seed"], "source_split": None, "pool_size": 0}
    if not isinstance(audit, dict):
        raise ValueError("return_loss metadata must be an object")
    if (audit.get("method") != method or audit.get("weights") != raw_weights
            or audit.get("floor") != floor or audit.get("max_samples") != samples
            or audit.get("seed") != training_config["seed"]):
        raise ValueError("return_loss metadata disagrees with training_config")
    count, pool = audit.get("effective_samples"), audit.get("pool_size")
    if type(count) is not int or type(pool) is not int or count < 0 or pool < 0:
        raise ValueError("return_loss sample counts must be nonnegative integers")
    if method == "uniform":
        if count != 0 or pool != 0 or audit.get("scales") is not None or audit.get("source_split") is not None:
            raise ValueError("uniform weighting must not claim training scale estimates")
    else:
        if audit.get("source_split") != "train" or count < 1 or count != min(pool, samples):
            raise ValueError("inverse_volatility scales must come from the sampled training pool")
        raw_scales = audit.get("scales")
        if (not isinstance(raw_scales, list) or len(raw_scales) != horizon
                or any(type(value) not in (int, float) for value in raw_scales)):
            raise ValueError("return_loss scales must have one value per horizon")
        scales = np.asarray(raw_scales, dtype=np.float64)
        if not np.isfinite(scales).all() or (scales < 0).any():
            raise ValueError("return_loss scales must be finite and nonnegative")
        denominators = np.maximum(scales, floor)
        expected = denominators.min() / denominators
        expected /= expected.mean()
        if not np.allclose(weights, expected, rtol=1e-6, atol=0.):
            raise ValueError("return_loss weights do not match the recorded training scales")
    return dict(audit)
