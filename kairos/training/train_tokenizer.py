"""Fine-tune the Kronos BSQ tokenizer on crypto K-line data.

The tokenizer only needs ``x`` (the 6-dim OHLCV+amount window) — time stamps
and the 32-dim exog features produced by ``kairos-prepare`` are ignored here.
We still reuse :class:`KronosSequenceDataset` so the pickle layout is
identical to the predictor trainer.

Launch (GPU)::

    torchrun --standalone --nproc_per_node=1 -m kairos.training.train_tokenizer

Launch (CPU smoke test, from a laptop)::

    MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
        KAIROS_SMOKE=1 KAIROS_PRESET=crypto-1min \
        KAIROS_DATASET=./finetune/data/crypto_1min_btc_eth \
        python -m kairos.training.train_tokenizer

The env-var overrides mirror ``train_predictor`` so the same wrapper scripts
can drive both stages.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from time import gmtime, strftime

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kairos.training.config import TrainConfig, preset_for
from kairos.training.dataset import AShareKronosDataset
from kairos.utils import (
    cleanup_ddp,
    format_time,
    get_model_size,
    set_seed,
    setup_ddp,
)
from kairos.vendor.kronos import KronosTokenizer


def _make_loaders(cfg: TrainConfig, rank: int, world: int):
    train = AShareKronosDataset("train", cfg)
    val = AShareKronosDataset("val", cfg)
    t_sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True)
    v_sampler = DistributedSampler(val, num_replicas=world, rank=rank, shuffle=False)
    t_loader = DataLoader(train, batch_size=cfg.batch_size, sampler=t_sampler,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    v_loader = DataLoader(val, batch_size=cfg.batch_size, sampler=v_sampler,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return t_loader, v_loader, train, val


def _train(model, device, cfg: TrainConfig, save_dir: Path, rank: int, world: int):
    t0 = time.time()
    t_loader, v_loader, t_ds, v_ds = _make_loaders(cfg, rank, world)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=cfg.tokenizer_learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
    )
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.tokenizer_learning_rate,
        steps_per_epoch=len(t_loader), epochs=cfg.epochs,
        pct_start=getattr(cfg, "warmup_pct", 0.03), div_factor=10,
    )

    patience = getattr(cfg, "patience", 0)
    bad_epochs = 0
    best = float("inf")
    step_g = 0

    for ep in range(cfg.epochs):
        ep_t0 = time.time()
        model.train()
        t_loader.sampler.set_epoch(ep)
        t_ds.set_epoch_seed(ep * 10000 + rank)
        v_ds.set_epoch_seed(0)

        for i, (x, _stamp, _exog) in enumerate(t_loader):
            x = x.to(device, non_blocking=True)
            accum = max(1, cfg.accumulation_steps)
            total_loss = 0.0
            size = x.shape[0] // accum
            if size == 0:
                size = x.shape[0]
                accum = 1
            for j in range(accum):
                chunk = x[j * size : (j + 1) * size]
                if chunk.size(0) == 0:
                    continue
                (z_pre, z), bsq_loss, _, _ = model(chunk)
                recon = F.mse_loss(z_pre, chunk) + F.mse_loss(z, chunk)
                loss = (recon + bsq_loss) / 2
                total_loss += loss.item()
                (loss / accum).backward()

            torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
            opt.step(); sch.step(); opt.zero_grad()

            if rank == 0 and (step_g + 1) % cfg.log_interval == 0:
                print(f"[ep {ep+1}/{cfg.epochs} step {i+1}/{len(t_loader)}] "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"loss={total_loss / accum:.4f}")
            step_g += 1

        # --- validation: MSE of the full-codebook reconstruction ---
        model.eval()
        loss_sum, count = 0.0, 0
        with torch.no_grad():
            for x, _stamp, _exog in v_loader:
                x = x.to(device, non_blocking=True)
                (_, z), _, _, _ = model(x)
                loss_sum += F.mse_loss(z, x).item() * x.size(0)
                count += x.size(0)
        ls = torch.tensor(loss_sum, device=device)
        cn = torch.tensor(count, device=device)
        dist.all_reduce(ls); dist.all_reduce(cn)
        val = (ls / cn).item() if cn.item() else 0.0

        improved = val < best - 1e-6
        if rank == 0:
            print(f"--- ep {ep+1}: val_recon={val:.6f} "
                  f"({format_time(time.time() - ep_t0)} / total {format_time(time.time() - t0)}) ---")
            if improved:
                best = val
                save = save_dir / "checkpoints" / "best_model"
                (model.module if hasattr(model, "module") else model).save_pretrained(str(save))
                print(f"[save] best → {save} (val_recon={val:.6f})")

        # 广播 improved 标记给所有 rank，统一做早停决策
        flag = torch.tensor([1 if improved else 0], device=device)
        dist.all_reduce(flag)
        improved_any = flag.item() > 0
        if improved_any:
            bad_epochs = 0
        else:
            bad_epochs += 1
            if rank == 0:
                print(f"[patience] {bad_epochs}/{patience} epochs without improvement")
        dist.barrier()

        if patience > 0 and bad_epochs >= patience:
            if rank == 0:
                print(f"[early-stop] val did not improve for {patience} epochs; stopping at ep {ep+1}")
            break

    return {"best_val_recon": best, "stopped_epoch": ep + 1}


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
        # 200 samples / batch 4 = 50 steps/epoch. OneCycleLR 的分段边界
        # 对 total_steps < ~20 会触发除零（pct_start * total_steps 被 int()
        # 截成 0），这里留够余量。
        cfg.n_train_iter = 200
        cfg.n_val_iter = 40
        cfg.warmup_pct = 0.2

    # Generic env overrides — identical names as train_predictor so a single
    # bootstrap script can drive both stages.
    _env_overrides = {
        "KAIROS_BATCH_SIZE": ("batch_size", int),
        "KAIROS_ACCUM_STEPS": ("accumulation_steps", int),
        "KAIROS_NUM_WORKERS": ("num_workers", int),
        "KAIROS_EPOCHS": ("epochs", int),
        "KAIROS_N_TRAIN_ITER": ("n_train_iter", int),
        "KAIROS_N_VAL_ITER": ("n_val_iter", int),
        "KAIROS_LR": ("tokenizer_learning_rate", float),
        "KAIROS_LOG_INTERVAL": ("log_interval", int),
        "KAIROS_PATIENCE": ("patience", int),
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

    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("请用 torchrun 启动此脚本 (或手动设 RANK/WORLD_SIZE/LOCAL_RANK)")
    rank, world, local = setup_ddp()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    set_seed(cfg.seed, rank)

    save_dir = Path(cfg.save_path) / cfg.tokenizer_save_folder_name
    if rank == 0:
        (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if rank == 0:
        print(f"[tokenizer] loading {cfg.pretrained_tokenizer_path}")
    model = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path).to(device)

    ddp_kwargs = dict(find_unused_parameters=False)
    if use_cuda:
        ddp_kwargs["device_ids"] = [local]
    model = DDP(model, **ddp_kwargs)

    if rank == 0:
        print("Tokenizer size:", get_model_size(model.module))

    summary = {"start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()), "world_size": world}
    summary["final_result"] = _train(model, device, cfg, save_dir, rank, world)
    if rank == 0:
        (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cleanup_ddp()


if __name__ == "__main__":
    main()
