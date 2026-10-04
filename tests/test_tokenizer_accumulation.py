"""Integrate tokenizer accumulation without reverting the v2 predictor contract."""
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
import time
from unittest.mock import patch
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from kairos.data.features import EXOG_COLS
from kairos.training import train_tokenizer
from kairos.training.config import TrainConfig


class _Tokenizer(nn.Module):
    # Additive loss isolates sample weighting; real BSQ batch entropy remains
    # chunk-dependent, as in the existing tokenizer training contract.
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.4))
        self.seen = []

    def forward(self, x):
        if self.training:
            self.seen.extend(x[:, 0, 0].tolist())
        z = self.weight * x
        return (z, z), self.weight.square() * 0.1, None, None

    def save_pretrained(self, path):
        pass

class _RecordingWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.sync_enabled = True
        self.events = []
        module.weight.register_hook(self._backward)

    def _backward(self, gradient):
        self.events.append(("backward", self.sync_enabled))
        return gradient

    @contextmanager
    def no_sync(self):
        self.sync_enabled = False
        try:
            yield
        finally:
            self.sync_enabled = True

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            self.events.append(("forward", self.sync_enabled))
        return self.module(*args, **kwargs)

class _Dataset:
    def __init__(self):
        self.seeds = []

    def set_epoch_seed(self, epoch):
        self.seeds.append(epoch)

class _Loader(list):
    @property
    def sampler(self):
        return self

    def set_epoch(self, epoch):
        pass

class _AdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates = 0
        self.gradients = []

    def step(self, *args, **kwargs):
        self.updates += 1
        self.gradients.append(self.param_groups[0]["params"][0].grad.detach().clone())
        return super().step(*args, **kwargs)

class _OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self, *args, **kwargs):
        self.steps_per_epoch = kwargs["steps_per_epoch"]
        self.updates = -1  # LRScheduler performs an initial step in __init__.
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.updates += 1
        return super().step(*args, **kwargs)

def _batch(values):
    values = torch.tensor(values, dtype=torch.float32)
    price = values[:, None] * torch.tensor([1.0, 1.5, 2.0])[None, :]
    volume = torch.full_like(price, 0.1)
    close = price * 1.01
    x = torch.stack((price, price * 1.02, price * 0.98, close,
                     volume, close * volume), dim=-1)
    return x, torch.zeros(len(values), 3, 5), torch.zeros(len(values), 3, len(EXOG_COLS))

@contextmanager
def _collectives(distributed):
    if distributed:
        yield
    else:
        with patch.object(dist, "all_reduce"), patch.object(dist, "barrier"):
            yield

@pytest.mark.parametrize("accum", [1, 2, 3, 8])
def test_tokenizer_chunks_cover_all_samples_with_weighted_gradients(tmp_path, accum):
    batches = [_batch([0.1, 0.2, 0.3, 0.4, 0.5]), _batch([0.2, 0.4, 0.6])]
    actual, opt, sch, _ = _run("tokenizer", batches, accum, tmp_path)
    expected, reference_opt, _, _ = _run("tokenizer", batches, 1, tmp_path)
    assert actual.module.seen == expected.module.seen
    assert opt.updates == sch.updates == sch.steps_per_epoch == len(batches)
    torch.testing.assert_close(actual.module.weight, expected.module.weight)
    torch.testing.assert_close(torch.stack(opt.gradients), torch.stack(reference_opt.gradients))
    expected_sync = [i == min(accum, len(batch[0])) - 1 for batch in batches
                     for i in range(min(accum, len(batch[0])))]
    assert actual.events == [(phase, sync) for sync in expected_sync
                             for phase in ("forward", "backward")]

def _run(stage, batches, accum, save_dir, *, rank=0, world=1, model=None,
         distributed=False, epochs=1, warmup_pct=0.3, validation_batches=None):
    trainer = train_tokenizer
    module = _Tokenizer()
    wrapped = model if model is not None else _RecordingWrapper(module)
    train_data, val_data = _Dataset(), _Dataset()
    loaders = (_Loader(batches), _Loader([batches[0]] if validation_batches is None else validation_batches), train_data, val_data)
    cfg = TrainConfig(epochs=epochs, accumulation_steps=accum, use_return_head=False,
                      ce_weight=1.0, predictor_learning_rate=0.03,
                      tokenizer_learning_rate=0.03, warmup_pct=warmup_pct,
                      patience=0, log_interval=1000)
    optimizers, schedulers = [], []

    def optimizer(*args, **kwargs):
        opt = _AdamW(*args, **kwargs)
        optimizers.append(opt)
        return opt

    def scheduler(*args, **kwargs):
        sch = _OneCycleLR(*args, **kwargs)
        schedulers.append(sch)
        return sch

    with patch.object(trainer, "_make_loaders", return_value=loaders), \
            patch.object(torch.optim, "AdamW", side_effect=optimizer), \
            patch.object(torch.optim.lr_scheduler, "OneCycleLR", side_effect=scheduler):
        with _collectives(distributed):
            wrapped.training_summary = trainer._train(wrapped, torch.device("cpu"), cfg, save_dir, rank, world)
    return wrapped, optimizers[0], schedulers[0], train_data


@pytest.mark.parametrize("accum", [0, -1])
def test_tokenizer_rejects_nonpositive_accumulation(tmp_path, accum):
    with pytest.raises(ValueError, match="accumulation_steps"):
        _run("tokenizer", [_batch([0.1])], accum, tmp_path)


def test_tokenizer_shares_epoch_seed_across_ranks(tmp_path):
    _, _, _, dataset = _run("tokenizer", [_batch([0.1, 0.2])], 1, tmp_path,
                            rank=1, world=2, epochs=2)
    assert dataset.seeds == [0, 1]


def _ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=Path(directory, "rendezvous").as_uri(),
                            timeout=timedelta(seconds=20))
    try:
        module = _Tokenizer()
        model = DistributedDataParallel(module)
        calls = []
        def comm_hook(state, bucket):
            calls.append(1)
            return default_hooks.allreduce_hook(None, bucket)
        model.register_comm_hook(None, comm_hook)
        batches = [_batch([0.03 * (i+1+rank), 0.04 * (i+1+rank), 0.05 * (i+1+rank)]) for i in range(3)]
        _, opt, sch, _ = _run("tokenizer", batches, 2, Path(directory), rank=rank,
                              world=2, model=model, distributed=True)
        assert len(calls) == opt.updates == sch.updates == 3
        gathered = [torch.empty_like(module.weight) for _ in range(2)]
        dist.all_gather(gathered, module.weight)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo is unavailable")
def test_tokenizer_two_rank_ddp_synchronizes_once_per_batch(tmp_path):
    processes = mp.spawn(_ddp_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 45
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Two-rank gradient accumulation exceeded 45 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)



def test_tokenizer_best_metric_is_finite_on_nonzero_rank(tmp_path):
    wrapped, _, _, _ = _run("tokenizer", [_batch([0.1, 0.2])], 1, tmp_path,
                            rank=1, world=2, epochs=2)
    assert torch.isfinite(torch.tensor(wrapped.training_summary["best_val_recon"]))


def test_tokenizer_loaders_do_not_require_unused_exogenous_sidecars(tmp_path):
    from tests.test_training_contract import _tiny_dataset
    _tiny_dataset(tmp_path / "data")
    for path in (tmp_path / "data").glob("exog_*.pkl"):
        path.unlink()
    cfg = TrainConfig(dataset_path=str(tmp_path / "data"), lookback_window=3,
                      predict_window=2, batch_size=2, num_workers=0)
    train, val, _, _ = train_tokenizer._make_loaders(cfg, 0, 1)
    assert len(train) > 0 and len(val) > 0
    assert cfg.use_exog is True


@pytest.mark.parametrize("empty_train", [True, False])
def test_tokenizer_rejects_empty_loaders_before_saving_best(tmp_path, empty_train):
    with pytest.raises(ValueError, match="empty"):
        _run("tokenizer", [] if empty_train else [_batch([0.1])], 1, tmp_path,
             validation_batches=[])


def test_tokenizer_rejects_nonfinite_validation_metric(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        _run("tokenizer", [_batch([0.1])], 1, tmp_path,
             validation_batches=[_batch([float("nan")])])
