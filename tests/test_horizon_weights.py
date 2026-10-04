"""Training-only horizon scales, fixed weighting and v2 bundle compatibility."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from kairos.models.kronos_ext import QuantileReturnHead
from kairos.training.config import TrainConfig
from kairos.training.dataset import KronosSequenceDataset


def _training_dataset(root, **overrides):
    root.mkdir(parents=True, exist_ok=True)
    t = np.arange(96, dtype=np.float64)
    close = np.exp(np.cumsum(0.002 * np.sin(t / 3) + 0.0003 * t))
    index = pd.date_range("2026-01-01", periods=len(t), freq="min", name="datetime")
    # One gap makes windows spanning the missing bar ineligible for estimation.
    index = index[:40].append(index[40:] + pd.Timedelta(minutes=1))
    frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "vol": 1., "amt": close}, index=index)
    pd.to_pickle({"TEST/USDT": frame}, root / "train_data.pkl")
    (root / "meta.json").write_text(json.dumps({"freq": "1min", "market": "crypto"}))
    cfg = TrainConfig(dataset_path=str(root), use_exog=False, lookback_window=4,
                      predict_window=3, return_horizon=3, n_quantiles=3, n_train_iter=2,
                      **overrides)
    return cfg, KronosSequenceDataset("train", cfg, include_targets=True)


def test_inverse_volatility_uses_contiguous_raw_training_pool(tmp_path):
    from kairos.training.horizon_weights import estimate_horizon_weights
    cfg, dataset = _training_dataset(tmp_path, return_scale_samples=7)
    result = estimate_horizon_weights(dataset, cfg)
    chosen = np.random.default_rng(cfg.seed).choice(len(dataset.indices), size=7, replace=False)
    targets = []
    for i in chosen:
        symbol, start = dataset.indices[i]
        anchor = start + cfg.lookback_window - 1
        close = dataset.data[symbol].close.to_numpy(dtype=np.float64)
        targets.append(np.log(close[anchor + 1:anchor + 4]) - np.log(close[anchor]))
    scales = np.std(np.asarray(targets), axis=0, ddof=0, dtype=np.float64)
    expected = 1. / np.maximum(scales, cfg.return_scale_floor)
    expected /= expected.mean()
    assert result["method"] == "inverse_volatility"
    assert result["effective_samples"] == 7 > len(dataset)
    assert result["source_split"] == "train"
    np.testing.assert_allclose(result["scales"], scales, rtol=1e-13)
    np.testing.assert_allclose(result["weights"], expected, rtol=1e-13)


def test_validation_test_files_and_epoch_order_cannot_change_weights(tmp_path):
    from kairos.training.horizon_weights import estimate_horizon_weights
    cfg, dataset = _training_dataset(tmp_path, return_scale_samples=11)
    first = estimate_horizon_weights(dataset, cfg)
    for name in ("val_data.pkl", "test_data.pkl"):
        (tmp_path / name).write_bytes(b"invalid non-training data must never be read")
    dataset.set_epoch_seed(1000)
    second = estimate_horizon_weights(dataset, cfg)
    assert first == second
    duplicate_rank_dataset = KronosSequenceDataset("train", replace(cfg), include_targets=True)
    assert first == estimate_horizon_weights(duplicate_rank_dataset, cfg)


def test_constant_targets_and_explicit_uniform_have_unit_weights(tmp_path):
    from kairos.training.horizon_weights import estimate_horizon_weights
    cfg, dataset = _training_dataset(tmp_path)
    for frame in dataset.data.values():
        frame["close"] = 3.
    assert estimate_horizon_weights(dataset, cfg)["weights"] == [1., 1., 1.]
    uniform = estimate_horizon_weights(None, replace(cfg, return_loss_weighting="uniform"))
    assert uniform["weights"] == [1., 1., 1.]
    assert uniform["effective_samples"] == 0
    assert uniform["scales"] is None


@pytest.mark.parametrize("overrides", [
    {"return_scale_floor": 0.}, {"return_scale_floor": float("nan")},
    {"return_scale_samples": 0}, {"return_scale_samples": 1.5},
    {"return_loss_weighting": "automatic"},
])
def test_invalid_weight_estimation_settings_fail(tmp_path, overrides):
    from kairos.training.horizon_weights import estimate_horizon_weights
    cfg, dataset = _training_dataset(tmp_path, **overrides)
    with pytest.raises(ValueError):
        estimate_horizon_weights(dataset, cfg)


def test_estimator_rejects_validation_dataset_or_empty_training_pool(tmp_path):
    from kairos.training.horizon_weights import estimate_horizon_weights
    cfg, dataset = _training_dataset(tmp_path)
    dataset.split = "val"
    with pytest.raises(ValueError, match="train"):
        estimate_horizon_weights(dataset, cfg)
    dataset.split = "train"
    dataset.indices = []
    with pytest.raises(ValueError, match="empty"):
        estimate_horizon_weights(dataset, cfg)


def test_horizon_weighting_balances_short_and_long_scale_without_changing_units():
    pred = torch.zeros(1, 1, 2, 3, requires_grad=True)
    target = torch.tensor([[[1., 10.]]])
    quantiles = torch.tensor([.1, .5, .9])
    weights = torch.tensor([1., .1])
    loss = QuantileReturnHead.pinball_loss(pred, target, quantiles, horizon_weights=weights)
    normalized = weights / weights.mean()
    expected = (target * normalized).mean() * .5
    assert torch.allclose(loss, expected)
    # Each horizon contributes equally after rescaling its loss, with raw targets unchanged.
    assert torch.allclose(target[..., 0] * normalized[0], target[..., 1] * normalized[1])
    masked = QuantileReturnHead.pinball_loss(pred, target, quantiles,
                                            mask=torch.ones(1, 1), horizon_weights=weights * 3)
    assert torch.allclose(loss, masked)
    unweighted = QuantileReturnHead.pinball_loss(pred, target, quantiles)
    unit = QuantileReturnHead.pinball_loss(pred, target, quantiles, horizon_weights=torch.ones(2))
    assert torch.equal(unweighted, unit)


@pytest.mark.parametrize("weights", [torch.tensor([1.]), torch.tensor([1., 0.]),
                                     torch.tensor([1., -1.]), torch.tensor([1., float("inf")])])
def test_pinball_rejects_invalid_horizon_weights(weights):
    with pytest.raises(ValueError, match="horizon_weights"):
        QuantileReturnHead.pinball_loss(torch.zeros(1, 1, 2, 3), torch.ones(1, 1, 2),
                                       torch.tensor([.1, .5, .9]), horizon_weights=weights)


def test_finite_large_weights_are_normalized_before_model_dtype_conversion():
    pred, target = torch.zeros(1, 1, 2, 3), torch.ones(1, 1, 2)
    quantiles = torch.tensor([.1, .5, .9])
    large = QuantileReturnHead.pinball_loss(
        pred, target, quantiles, horizon_weights=torch.tensor([1e300, 5e299], dtype=torch.float64))
    regular = QuantileReturnHead.pinball_loss(pred, target, quantiles,
                                            horizon_weights=torch.tensor([2., 1.]))
    assert torch.equal(large, regular)


def test_ddp_resolves_once_on_rank_zero_and_broadcasts_identical_weights(tmp_path, monkeypatch):
    from kairos.training import train_predictor as trainer
    from copy import deepcopy
    cfg, dataset = _training_dataset(tmp_path)
    shared = []
    def broadcast(objects, src):
        assert src == 0
        if objects[0] is not None:
            shared[:] = deepcopy(objects)
        else:
            objects[:] = deepcopy(shared)
    monkeypatch.setattr(trainer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer.dist, "broadcast_object_list", broadcast)
    rank_zero = trainer._resolve_return_loss(dataset, cfg)
    monkeypatch.setattr(trainer.dist, "get_rank", lambda: 1)
    other_cfg = replace(cfg, return_loss_weights=None)
    rank_one = trainer._resolve_return_loss(None, other_cfg)
    assert rank_zero == rank_one
    assert cfg.return_loss_weights == other_cfg.return_loss_weights == rank_zero["weights"]


def test_batch_loss_uses_the_same_resolved_weights_in_train_and_validation():
    from kairos.training.train_predictor import _batch_loss
    from tests.test_training_contract import _tiny_model, _RecordingTokenizer
    cfg = TrainConfig(lookback_window=3, return_horizon=2, n_quantiles=3,
                      ce_weight=0., quantile_weight=2., return_loss_weights=[1.5, .5])
    model = _tiny_model()
    batch = (torch.zeros(2, 6, 6), torch.ones(2, 6, 5), torch.zeros(2, 6, 32),
             torch.tensor([[0.1, 0.3], [-0.2, -0.1]]))
    for training in (True, False):
        model.train(training)
        seen = []
        hook = model.return_head.register_forward_hook(lambda _, args, result: seen.append(result))
        loss, _, pin = _batch_loss(model, _RecordingTokenizer(), batch, torch.device("cpu"), cfg)
        hook.remove()
        expected = QuantileReturnHead.pinball_loss(
            seen[0][:, 2:3], batch[3][:, None], torch.tensor([.1, .5, .9]),
            horizon_weights=torch.tensor(cfg.return_loss_weights))
        assert torch.allclose(pin, expected)
        assert torch.allclose(loss, 2 * expected)


def test_bundle_records_training_scale_audit_and_checks_weights(tmp_path):
    from kairos.training.horizon_weights import estimate_horizon_weights
    from kairos.training.artifacts import load_manifest, save_checkpoint
    from tests.test_training_contract import _SavedObject
    cfg, dataset = _training_dataset(tmp_path / "data", return_scale_samples=13)
    # Bundle provenance also records validation bytes; scale estimation does not read them.
    (tmp_path / "data" / "val_data.pkl").write_bytes(b"not used to fit scales")
    audit = estimate_horizon_weights(dataset, cfg)
    cfg.return_loss_weights = audit["weights"]
    checkpoint = tmp_path / "checkpoint"
    manifest = save_checkpoint(_SavedObject(), _SavedObject(), cfg, checkpoint, return_loss=audit)
    assert manifest["target"] == "log_return"
    assert manifest["return_loss"] == audit
    assert manifest["training_config"]["return_loss_weights"] == audit["weights"]
    assert manifest["return_loss"]["effective_samples"] == 13
    path = checkpoint / "kairos_manifest.json"
    manifest["return_loss"]["weights"] = [1., 1., 1.]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="return_loss"):
        load_manifest(checkpoint)


def test_old_v2_bundle_is_explicit_historical_uniform(tmp_path):
    from kairos.training.artifacts import load_manifest, save_checkpoint
    from tests.test_training_contract import _config, _SavedObject
    cfg = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(_SavedObject(), _SavedObject(), cfg, checkpoint)
    path = checkpoint / "kairos_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["return_loss"]
    for field in ("return_loss_weighting", "return_loss_weights", "return_scale_floor", "return_scale_samples"):
        del manifest["training_config"][field]
    old_bytes = json.dumps(manifest)
    path.write_text(old_bytes)
    loaded = load_manifest(checkpoint)
    assert loaded["return_loss"]["method"] == "historical_uniform"
    assert loaded["return_loss"]["weights"] == [1.] * cfg.return_horizon
    assert loaded["return_loss"]["effective_samples"] is None
    assert path.read_text() == old_bytes


def test_training_estimates_once_and_freezes_weights_across_epochs(tmp_path, monkeypatch):
    from kairos.training import train_predictor as trainer
    cfg, _ = _training_dataset(tmp_path / "data", return_scale_samples=5)
    data = tmp_path / "data"
    (data / "val_data.pkl").write_bytes((data / "train_data.pkl").read_bytes())
    cfg.epochs, cfg.batch_size, cfg.num_workers, cfg.n_val_iter = 2, 1, 0, 2
    calls, seen = [], []
    original = trainer.estimate_horizon_weights
    def estimate(dataset, config):
        calls.append(dataset.split)
        return original(dataset, config)
    def loss(model, tokenizer, batch, device, config):
        seen.append(tuple(config.return_loss_weights))
        value = model.weight.square().mean()
        return value, value, value
    monkeypatch.setattr(trainer, "estimate_horizon_weights", estimate)
    monkeypatch.setattr(trainer, "_batch_loss", loss)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    result = trainer._train(torch.nn.Linear(1, 1), None, torch.device("cpu"), cfg,
                            tmp_path / "run", 0, 1)
    assert calls == ["train"]
    assert len(seen) == 8  # two train and two validation batches in each epoch
    assert set(seen) == {tuple(result["return_loss"]["weights"])}


def test_training_estimate_rejects_weights_that_underflow_in_float32(tmp_path):
    from kairos.training.horizon_weights import estimate_horizon_weights
    frames = {}
    for symbol, future_return in [("A", .1), ("B", .2)]:
        close = np.array([100., 100., 100., 100. * np.exp(future_return)])
        frames[symbol] = pd.DataFrame({
            "open": close, "high": close, "low": close, "close": close,
            "vol": 1., "amt": close,
        }, index=pd.date_range("2026-01-01", periods=4, freq="min", name="datetime"))
    pd.to_pickle(frames, tmp_path / "train_data.pkl")
    cfg = TrainConfig(dataset_path=str(tmp_path), use_exog=False,
                      lookback_window=2, predict_window=1, return_horizon=2,
                      return_scale_floor=1e-60)
    dataset = KronosSequenceDataset("train", cfg, include_targets=True)
    assert len(dataset.indices) == 2
    with pytest.raises(ValueError, match="float32.*increase return_scale_floor"):
        estimate_horizon_weights(dataset, cfg)
    # Raising the floor recovers a usable positive vector without changing units.
    result = estimate_horizon_weights(dataset, replace(cfg, return_scale_floor=1e-6))
    assert (np.asarray(result["weights"], dtype=np.float32) > 0).all()


def test_metadata_rejects_float64_positive_weights_that_underflow_in_float32():
    from kairos.training.horizon_weights import return_loss_metadata
    weights = [2., 4e-59]
    config = {"return_loss_weighting": "inverse_volatility", "return_loss_weights": weights,
              "return_scale_samples": 4096, "return_scale_floor": 1e-60, "seed": 100}
    audit = {"method": "inverse_volatility", "weights": weights, "scales": [0., .05],
             "effective_samples": 2, "max_samples": 4096, "floor": 1e-60,
             "seed": 100, "source_split": "train", "pool_size": 2}
    assert (np.asarray(weights, dtype=np.float64) > 0).all()
    with pytest.raises(ValueError, match="float32.*increase return_scale_floor"):
        return_loss_metadata(config, 2, audit)
