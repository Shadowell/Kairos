"""Fine-tune the Kronos predictor **with exogenous channel + quantile return head**.

This is the Kairos flagship trainer. It implements:
  * method A — exogenous bypass projection fused into token embedding
  * method C — quantile return head with pinball loss
  * progressive unfreeze (last N transformer blocks + new heads only)

Launch::

    torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from time import gmtime, strftime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kairos.models import KronosWithExogenous
from kairos.training.config import TrainConfig, preset_for
from kairos.training.artifacts import create_run_dir, hash_training_data, save_checkpoint
from kairos.training.dataset import KronosSequenceDataset
from kairos.utils import (
    cleanup_ddp,
    format_time,
    get_model_size,
    set_seed,
    setup_ddp,
)
from kairos.vendor.kronos import KronosTokenizer


def _make_loaders(cfg: TrainConfig, rank: int, world: int):
    train = KronosSequenceDataset("train", cfg, include_targets=True)
    val = KronosSequenceDataset("val", cfg, include_targets=True)
    t_sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                   seed=cfg.seed)
    # Validation covers each sample exactly once, without padding duplicates.
    v_sampler = range(rank, len(val), world)
    t_loader = DataLoader(train, batch_size=cfg.batch_size, sampler=t_sampler,
                          num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    v_loader = DataLoader(val, batch_size=cfg.batch_size, sampler=v_sampler,
                          num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    return t_loader, v_loader, train, val


def _unwrap(model):
    return model.module if isinstance(model, DDP) else model


def _batch_loss(model, tokenizer, batch, device, cfg: TrainConfig):
    """The same teacher-forced CE + anchored log-return loss in both phases."""
    x, stamp, exog, targets = (value.to(device, non_blocking=True) for value in batch)
    history = cfg.lookback_window
    if x.shape[1] <= history or targets.shape != (x.shape[0], cfg.return_horizon):
        raise ValueError("batch does not contain history, next-token label and return targets")
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x[:, :history + 1], half=True)
    s1_target, s2_target = s1[:, 1:], s2[:, 1:]
    s1_logits, s2_logits, q_pred = model(
        s1[:, :-1], s2[:, :-1], stamp=stamp[:, :history],
        exog=exog[:, :history] if cfg.use_exog else None,
        use_teacher_forcing=True, s1_targets=s1_target,
    )
    raw_model = _unwrap(model)
    ce, _, _ = raw_model.head.compute_loss(s1_logits, s2_logits, s1_target, s2_target)
    if q_pred is None:
        raise ValueError("version 2 predictor training requires a quantile return head")
    quantiles = torch.linspace(0.1, 0.9, cfg.n_quantiles, device=device)
    pin = raw_model.return_head.pinball_loss(
        q_pred[:, history - 1:history], targets[:, None, :], quantiles,
    )
    return cfg.ce_weight * ce + cfg.quantile_weight * pin, ce, pin


def _train(model, tokenizer, device, cfg: TrainConfig, save_dir: Path,
           rank: int, world: int, dataset_hashes: dict[str, str] | None = None):
    t0 = time.time()
    if cfg.epochs < 1 or cfg.accumulation_steps < 1:
        raise ValueError("epochs and accumulation_steps must be positive")
    t_loader, v_loader, t_ds, v_ds = _make_loaders(cfg, rank, world)
    train_size = torch.tensor(len(t_loader), device=device)
    val_size = torch.tensor(len(v_loader), device=device)
    if dist.is_initialized():
        dist.all_reduce(train_size, op=dist.ReduceOp.MIN)
        dist.all_reduce(val_size)
    if train_size == 0 or val_size == 0:
        raise ValueError("empty training or validation loader; check contiguous windows and batch settings")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=cfg.predictor_learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
    )
    steps_per_epoch = math.ceil(len(t_loader) / cfg.accumulation_steps)
    total_steps = cfg.epochs * steps_per_epoch
    # Tiny CPU runs may not contain enough updates for two nonempty phases.
    if cfg.warmup_pct * total_steps > 1 and (1 - cfg.warmup_pct) * total_steps > 1:
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.predictor_learning_rate, total_steps=total_steps,
            pct_start=cfg.warmup_pct, div_factor=10,
        )
    else:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)

    best = float("inf")
    step_g = 0
    patience = getattr(cfg, "patience", 0)
    bad_epochs = 0

    for ep in range(cfg.epochs):
        ep_t0 = time.time()
        model.train()
        if hasattr(t_loader.sampler, "set_epoch"):
            t_loader.sampler.set_epoch(ep)
        t_ds.set_epoch_seed(ep)
        v_ds.set_epoch_seed(0)
        opt.zero_grad(set_to_none=True)
        group_samples = 0
        for i, batch in enumerate(t_loader):
            update = (i + 1) % cfg.accumulation_steps == 0 or i + 1 == len(t_loader)
            sync = model.no_sync() if isinstance(model, DDP) and not update else nullcontext()
            with sync:
                loss, ce, pin = _batch_loss(model, tokenizer, batch, device, cfg)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                samples = batch[0].shape[0]
                (loss * samples).backward()
            group_samples += samples
            if update:
                for parameter in params:
                    if parameter.grad is not None:
                        parameter.grad.div_(group_samples)
                torch.nn.utils.clip_grad_norm_(params, max_norm=3.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
                group_samples = 0

            if rank == 0 and (step_g + 1) % cfg.log_interval == 0:
                print(f"[ep {ep+1}/{cfg.epochs} step {i+1}/{len(t_loader)}] "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"loss={loss.item():.4f} ce={ce.item():.4f} pinball={pin.item():.4f}")
            step_g += 1

        # --- validation ---
        model.eval()
        loss_sum, count = 0.0, 0
        with torch.no_grad():
            for batch in v_loader:
                # Ranks may have different validation batch counts, so avoid
                # DDP forward collectives and reduce only final sample totals.
                loss, _, _ = _batch_loss(_unwrap(model), tokenizer, batch, device, cfg)
                samples = batch[0].shape[0]
                loss_sum += loss.item() * samples
                count += samples
        totals = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(totals)
        val = (totals[0] / totals[1]).item()
        if not math.isfinite(val):
            raise ValueError("non-finite validation loss")

        improved = val < best - 1e-4
        if improved:
            best = val
            bad_epochs = 0
        else:
            bad_epochs += 1
        if rank == 0:
            print(f"--- ep {ep+1}: val_loss={val:.4f} "
                  f"({format_time(time.time() - ep_t0)} / total {format_time(time.time() - t0)}) ---")
            if improved:
                save = save_dir / "checkpoints" / "best_model"
                manifest = save_checkpoint(_unwrap(model), tokenizer, cfg, save,
                                           dataset_hashes=dataset_hashes)
                if manifest is not None:
                    dataset_hashes = manifest["dataset_hashes"]
                print(f"[save] best → {save} (val_loss={val:.4f})")
            else:
                print(f"[patience] {bad_epochs}/{patience} epochs without improvement")

        if dist.is_initialized():
            dist.barrier()

        if patience > 0 and bad_epochs >= patience:
            if rank == 0:
                print(f"[early-stop] val did not improve for {patience} epochs; stopping at ep {ep+1}")
            break

    return {"best_val_loss": best, "stopped_epoch": ep + 1}


def main():
    preset_name = os.environ.get("KAIROS_PRESET")
    cfg = TrainConfig(**preset_for(preset_name)) if preset_name else TrainConfig()
    ds_override = os.environ.get("KAIROS_DATASET")
    if ds_override:
        cfg.dataset_path = ds_override
    # Smoke-test overrides for CPU / laptop runs. Activate with KAIROS_SMOKE=1.
    if os.environ.get("KAIROS_SMOKE") == "1":
        cfg.epochs = 1
        cfg.batch_size = 4
        cfg.num_workers = 0
        cfg.log_interval = 5
        cfg.unfreeze_last_n = 1
        # 200 samples / batch 4 = 50 steps/epoch. OneCycleLR 的分段边界
        # 对 total_steps < ~20 会触发除零（pct_start * total_steps 被 int()
        # 截成 0），这里留够余量。
        cfg.n_train_iter = 200
        cfg.n_val_iter = 40
        cfg.warmup_pct = 0.2
    # Generic env overrides — handy for shared-GPU boxes where the default
    # batch size OOMs, or for quick hyper-param sweeps without editing code.
    # Values are parsed as int/float on a best-effort basis; unrecognised keys
    # are ignored so typos fall back to the preset default.
    _env_overrides = {
        "KAIROS_BATCH_SIZE": ("batch_size", int),
        "KAIROS_ACCUM_STEPS": ("accumulation_steps", int),
        "KAIROS_NUM_WORKERS": ("num_workers", int),
        "KAIROS_EPOCHS": ("epochs", int),
        "KAIROS_N_TRAIN_ITER": ("n_train_iter", int),
        "KAIROS_N_VAL_ITER": ("n_val_iter", int),
        "KAIROS_LR": ("predictor_learning_rate", float),
        "KAIROS_UNFREEZE_LAST_N": ("unfreeze_last_n", int),
        "KAIROS_LOG_INTERVAL": ("log_interval", int),
    }
    for env_key, (attr, caster) in _env_overrides.items():
        val = os.environ.get(env_key)
        if val is None or val == "":
            continue
        try:
            setattr(cfg, attr, caster(val))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{env_key}={val!r} is not a valid {caster.__name__}"
            ) from e
    pred_override = os.environ.get("KAIROS_PRETRAINED_PREDICTOR")
    if pred_override:
        cfg.pretrained_predictor_path = pred_override
    for env, attribute in (("KAIROS_PRETRAINED_TOKENIZER", "pretrained_tokenizer_path"),
                           ("KAIROS_SAVE_PATH", "save_path"), ("KAIROS_RUN_ID", "run_id")):
        if os.environ.get(env):
            setattr(cfg, attribute, os.environ[env])
    if not cfg.use_return_head:
        raise ValueError("version 2 predictor training requires use_return_head=True")
    if not 0 < cfg.warmup_pct < 1:
        raise ValueError("warmup_pct must be between 0 and 1")
    rank, world, local = setup_ddp() if int(os.environ.get("WORLD_SIZE", "1")) > 1 else (0, 1, 0)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    set_seed(cfg.seed, rank)

    reservation = [None, None]
    if rank == 0:
        try:
            reservation[0] = str(create_run_dir(cfg))
        except (OSError, ValueError) as exc:
            reservation[1] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(reservation, src=0)
    if reservation[1] is not None:
        cleanup_ddp()
        raise RuntimeError(f"Unable to reserve a new run directory: {reservation[1]}")
    save_dir = Path(reservation[0])
    cfg.run_id = save_dir.name

    # The tokenizer is explicit; unrelated experiments must not change this run.
    tok_src = cfg.pretrained_tokenizer_path
    if rank == 0:
        print(f"[tokenizer] loading {tok_src}")
    tokenizer = KronosTokenizer.from_pretrained(tok_src).eval().to(device)

    model = KronosWithExogenous.from_kronos_pretrained(
        cfg.pretrained_predictor_path,
        n_exog=cfg.n_exog,
        use_return_head=cfg.use_return_head,
        return_horizon=cfg.return_horizon,
        n_quantiles=cfg.n_quantiles,
    ).to(device)
    model.freeze_backbone(unfreeze_last_n=cfg.unfreeze_last_n)
    if dist.is_initialized():
        ddp_kwargs = dict(find_unused_parameters=True)
        if use_cuda:
            ddp_kwargs["device_ids"] = [local]
        model = DDP(model, **ddp_kwargs)

    if rank == 0:
        print("Predictor size:", get_model_size(_unwrap(model)))

    summary = {"start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()), "world_size": world}
    try:
        hashes = hash_training_data(cfg) if rank == 0 else None
        summary["final_result"] = _train(model, tokenizer, device, cfg, save_dir, rank, world,
                                          dataset_hashes=hashes)
        if rank == 0:
            (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
