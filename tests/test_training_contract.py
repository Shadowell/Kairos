"""Offline regressions for the version 2 learning and checkpoint contract."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kairos.models.kronos_ext import QuantileReturnHead
from kairos.training.config import TrainConfig


def test_pinball_all_ones_mask_matches_unmasked_and_gradient():
    pred = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3).requires_grad_()
    target = torch.ones(2, 2, 2)
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    plain = QuantileReturnHead.pinball_loss(pred, target, quantiles)
    masked = QuantileReturnHead.pinball_loss(pred, target, quantiles, torch.ones(2, 2))
    assert torch.allclose(plain, masked)
    assert torch.allclose(torch.autograd.grad(plain, pred)[0],
                          torch.autograd.grad(masked, pred)[0])


@pytest.mark.parametrize("mask", [torch.zeros(2, 2), torch.full((2, 2), -1.),
                                 torch.full((2, 2), float("nan")), torch.ones(2, 3)])
def test_pinball_rejects_invalid_mask(mask):
    with pytest.raises(ValueError, match="mask"):
        QuantileReturnHead.pinball_loss(torch.zeros(2, 2, 2, 3),
                                       torch.ones(2, 2, 2),
                                       torch.tensor([0.1, 0.5, 0.9]), mask)


def _tiny_model():
    from kairos.models.kronos_ext import KronosWithExogenous
    return KronosWithExogenous(
        s1_bits=2, s2_bits=2, n_layers=1, d_model=16, n_heads=4, ff_dim=32,
        ffn_dropout_p=0., attn_dropout_p=0., resid_dropout_p=0.,
        token_dropout_p=0., learn_te=True, return_horizon=2, n_quantiles=3,
    )


def _tiny_tokenizer():
    from kairos.vendor.kronos import KronosTokenizer
    return KronosTokenizer(
        d_in=6, d_model=16, n_heads=4, ff_dim=32, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0., attn_dropout_p=0., resid_dropout_p=0., s1_bits=2, s2_bits=2,
        beta=0.05, gamma0=1., gamma=1., zeta=1., group_size=2,
    ).eval()


def _tiny_dataset(root):
    import numpy as np
    import pandas as pd
    from kairos.data.features import exog_cols_for
    root = Path(root)
    root.mkdir(parents=True)
    index = pd.date_range("2026-01-01", periods=16, freq="min", name="datetime")
    close = np.exp(np.arange(16) * 0.01)
    main = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "vol": np.ones(16), "amt": close}, index=index)
    exog = pd.DataFrame(0., index=index, columns=exog_cols_for("crypto"))
    for split in ("train", "val", "test"):
        pd.to_pickle({"TEST/USDT": main}, root / f"{split}_data.pkl")
        pd.to_pickle({"TEST/USDT": exog}, root / f"exog_{split}.pkl")
    (root / "meta.json").write_text(json.dumps({
        "market": "crypto", "market_type": "spot", "freq": "1min",
        "feature_cols": list(main.columns), "exog_cols": list(exog.columns),
    }))
    return main, exog


def test_validation_logits_do_not_see_future_tokens():
    torch.manual_seed(7)
    model = _tiny_model().eval()
    ids = torch.zeros(1, 6, dtype=torch.long)
    changed = ids.clone()
    changed[:, 3:] = 3
    with torch.no_grad():
        before = model(ids, ids, use_teacher_forcing=True, s1_targets=ids)
        after = model(changed, changed, use_teacher_forcing=True, s1_targets=changed)
    for original, perturbed in zip(before, after):
        assert torch.allclose(original[:, :3], perturbed[:, :3], atol=1e-6)


def test_incremental_dependency_decode_preserves_complete_history():
    from kairos.vendor.kronos.module import MultiHeadCrossAttentionWithRoPE
    model = _tiny_model().eval()
    attention = model.dep_layer.cross_attn
    upstream = MultiHeadCrossAttentionWithRoPE(16, 4).eval()
    upstream.load_state_dict(attention.state_dict())
    query, history = torch.randn(1, 1, 16), torch.randn(1, 5, 16)
    with torch.no_grad():
        assert torch.allclose(attention(query, history, history),
                              upstream(query, history, history))


class _SavedObject:
    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text('{"tiny": true}')
        (directory / "model.safetensors").write_bytes(b"offline tiny weights")


def _config(tmp_path):
    cfg = TrainConfig(dataset_path=str(tmp_path / "data"), save_path=str(tmp_path),
                      return_loss_weighting="uniform", return_loss_weights=[1.] * 30)
    cfg.run_id = "offline-test"
    cfg.comet_api_key = "MUST_NOT_BE_SAVED"
    data = Path(cfg.dataset_path)
    data.mkdir()
    for name in ("train_data.pkl", "val_data.pkl", "exog_train.pkl", "exog_val.pkl"):
        (data / name).write_bytes(b"deterministic fixture")
    (data / "meta.json").write_text(json.dumps({"market": "crypto", "freq": "1min"}))
    return cfg


def test_run_directory_refuses_collision_and_legacy_bundle(tmp_path):
    from kairos.training.artifacts import create_run_dir, load_manifest
    cfg = _config(tmp_path)
    run = create_run_dir(cfg)
    assert run == tmp_path / "predictor" / "offline-test"
    with pytest.raises(FileExistsError):
        create_run_dir(cfg)
    with pytest.raises(ValueError, match="contract|manifest|legacy"):
        load_manifest(run)


def test_bundle_binds_tokenizer_and_redacts_config(tmp_path):
    from kairos.training.artifacts import load_manifest, save_checkpoint
    cfg = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(_SavedObject(), _SavedObject(), cfg, checkpoint)
    manifest = load_manifest(checkpoint)
    assert manifest["contract_version"] == 2
    assert manifest["target"] == "log_return"
    assert manifest["tokenizer_path"] == "tokenizer"
    assert set(manifest["tokenizer_files"]) == {"config.json", "model.safetensors"}
    assert "MUST_NOT_BE_SAVED" not in (checkpoint / "kairos_manifest.json").read_text()
    assert set(manifest["dataset_hashes"]) >= {"train_data.pkl", "val_data.pkl", "meta.json"}
    (checkpoint / "tokenizer" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash|checksum"):
        load_manifest(checkpoint)


def test_manifest_rejects_wrong_target_and_missing_fields(tmp_path):
    from kairos.training.artifacts import load_manifest, save_checkpoint
    cfg = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(_SavedObject(), _SavedObject(), cfg, checkpoint)
    path = checkpoint / "kairos_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["target"] = "normalized_difference"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target"):
        load_manifest(checkpoint)
    manifest["target"] = "log_return"
    del manifest["feature_cols"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="feature_cols"):
        load_manifest(checkpoint)


class _RecordingTokenizer:
    def encode(self, x, half=True):
        self.seen = x.clone()
        ids = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        return ids, ids


def test_joint_loss_uses_only_anchor_and_raw_dataset_target():
    from kairos.training.train_predictor import _batch_loss
    cfg = TrainConfig(lookback_window=3, return_horizon=2, n_quantiles=3,
                      ce_weight=0., quantile_weight=2., return_loss_weighting="uniform",
                      return_loss_weights=[1., 1.])
    model = _tiny_model().eval()
    tokenizer = _RecordingTokenizer()
    x = torch.zeros(2, 6, 6)
    target = torch.tensor([[0.1, 0.3], [-0.2, -0.1]])
    batch = (x, torch.ones(2, 6, 5), torch.zeros(2, 6, 32), target)
    seen = []
    hook = model.return_head.register_forward_hook(lambda _, args, result: seen.append(result))
    loss, _, pin = _batch_loss(model, tokenizer, batch, torch.device("cpu"), cfg)
    hook.remove()
    expected = QuantileReturnHead.pinball_loss(
        seen[0][:, 2:3], target[:, None], torch.tensor([0.1, 0.5, 0.9]))
    assert seen[0].shape[1] == cfg.lookback_window
    assert tokenizer.seen.shape[1] == cfg.lookback_window + 1
    assert torch.allclose(pin, expected)
    assert torch.allclose(loss, 2 * expected)


class _EpochDataset(TensorDataset):
    def set_epoch_seed(self, epoch):
        self.epoch = epoch


@pytest.mark.parametrize("rank", [0, 1])
def test_training_accumulates_tail_and_selects_joint_validation(tmp_path, monkeypatch, rank):
    from kairos.training import train_predictor as trainer
    cfg = TrainConfig(epochs=5, batch_size=1, accumulation_steps=2, patience=1,
                      num_workers=0, return_horizon=2, n_quantiles=3,
                      return_loss_weighting="uniform")
    dataset = _EpochDataset(torch.zeros(3, 1))
    loader = DataLoader(dataset, batch_size=1)
    monkeypatch.setattr(trainer, "_make_loaders", lambda *args: (loader, loader, dataset, dataset))
    model = torch.nn.Linear(1, 1)
    history = []
    def losses(model, tokenizer, batch, device, cfg):
        # CE improves but the joint objective worsens after the first epoch.
        if model.training:
            loss = model.weight.square().mean()
        else:
            loss = torch.tensor(1. if len(history) < 3 else 2.)
            history.append(float(loss))
        return loss, torch.tensor(0.), loss
    monkeypatch.setattr(trainer, "_batch_loss", losses)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    steps = []
    original = torch.optim.AdamW.step
    def step(self, *args, **kwargs):
        steps.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    result = trainer._train(model, None, torch.device("cpu"), cfg, tmp_path, rank, 2)
    assert result["stopped_epoch"] == 2
    assert result["best_val_loss"] == pytest.approx(1.)
    assert len(steps) == 4  # ceil(3 / 2) updates per epoch, including the tail


def test_accumulation_matches_larger_batch_with_short_final_batch(tmp_path, monkeypatch):
    from kairos.training import train_predictor as trainer
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    dataset = _EpochDataset(torch.tensor([[1.], [2.], [3.], [4.], [5.]]))
    def loaders(cfg, *args):
        loader = DataLoader(dataset, batch_size=cfg.batch_size)
        return loader, loader, dataset, dataset
    monkeypatch.setattr(trainer, "_make_loaders", loaders)
    def losses(model, tokenizer, batch, device, cfg):
        loss = (model(batch[0]) - 0.1).square().mean()
        return loss, loss, loss
    monkeypatch.setattr(trainer, "_batch_loss", losses)
    weights = []
    for batch_size, accumulation in [(2, 2), (4, 1)]:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.01)
        cfg = TrainConfig(epochs=1, batch_size=batch_size, accumulation_steps=accumulation,
                          predictor_learning_rate=0.01, adam_weight_decay=0.,
                          return_loss_weighting="uniform")
        trainer._train(model, None, torch.device("cpu"), cfg, tmp_path, 0, 1)
        weights.append(model.weight.detach().clone())
    assert torch.allclose(weights[0], weights[1], atol=1e-7)


def test_empty_loader_fails_before_scheduler(tmp_path, monkeypatch):
    from kairos.training import train_predictor as trainer
    cfg = TrainConfig()
    dataset = _EpochDataset(torch.empty(0, 1))
    loader = DataLoader(dataset, batch_size=1)
    monkeypatch.setattr(trainer, "_make_loaders", lambda *args: (loader, loader, dataset, dataset))
    with pytest.raises(ValueError, match="empty|Empty"):
        trainer._train(torch.nn.Linear(1, 1), None, torch.device("cpu"), cfg, tmp_path, 0, 1)


def test_real_tiny_training_bundle_reloads_exact_models(tmp_path):
    from kairos.models import KronosWithExogenous
    from kairos.training.artifacts import create_run_dir, load_manifest
    from kairos.training.train_predictor import _train
    from kairos.vendor.kronos import KronosTokenizer
    _tiny_dataset(tmp_path / "data")
    cfg = TrainConfig(
        dataset_path=str(tmp_path / "data"), save_path=str(tmp_path / "runs"),
        lookback_window=4, predict_window=2, return_horizon=2, n_quantiles=3,
        epochs=1, batch_size=2, accumulation_steps=2, n_train_iter=5, n_val_iter=3,
        num_workers=0,
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model, tokenizer = _tiny_model(), _tiny_tokenizer()
        run = create_run_dir(cfg)
        result = _train(model, tokenizer, torch.device("cpu"), cfg, run, 0, 1)
        assert result["stopped_epoch"] == 1
        checkpoint = run / "checkpoints" / "best_model"
        manifest = load_manifest(checkpoint)
        restored = KronosWithExogenous.from_pretrained(checkpoint)
        restored_tokenizer = KronosTokenizer.from_pretrained(checkpoint / manifest["tokenizer_path"])
        for name, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name])
        for name, value in tokenizer.state_dict().items():
            assert torch.equal(value, restored_tokenizer.state_dict()[name])
    finally:
        torch.set_num_threads(threads)


def test_source_without_git_remains_explicitly_unknown(tmp_path, monkeypatch):
    from kairos.training import artifacts
    cfg = _config(tmp_path)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr(artifacts.subprocess, "check_output", unavailable)
    manifest = artifacts.save_checkpoint(_SavedObject(), _SavedObject(), cfg, tmp_path / "checkpoint")
    assert manifest["source_sha"] is None
    assert manifest["source_dirty"] is None


def test_bundle_rejects_reordered_time_feature_slots(tmp_path):
    from kairos.training.artifacts import load_manifest, save_checkpoint
    cfg = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(_SavedObject(), _SavedObject(), cfg, checkpoint)
    path = checkpoint / "kairos_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training_config"]["time_feature_list"].reverse()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="time_feature_list"):
        load_manifest(checkpoint)
    cfg.time_feature_list.reverse()
    with pytest.raises(ValueError, match="time_feature_list"):
        save_checkpoint(_SavedObject(), _SavedObject(), cfg, tmp_path / "reordered")


def test_checkpoint_accepts_equivalent_frequency_aliases(tmp_path):
    from kairos.training.artifacts import save_checkpoint
    cfg = _config(tmp_path)
    cfg.freq = "60min"
    (Path(cfg.dataset_path) / "meta.json").write_text(json.dumps({"market": "crypto", "freq": "1h"}))
    result = save_checkpoint(_SavedObject(), _SavedObject(), cfg, tmp_path / "checkpoint")
    assert result["freq"] == "60min"
