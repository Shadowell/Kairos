"""Fine-tune the Kronos BSQ tokenizer on A-share K-line data.

Launch::

    torchrun --standalone --nproc_per_node=1 -m kairos.training.train_tokenizer
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

from kairos.training.config import TrainConfig
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.tokenizer_learning_rate,
        weight_decay=cfg.adam_weight_decay,
    )
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.tokenizer_learning_rate,
        steps_per_epoch=len(t_loader), epochs=cfg.epochs,
        pct_start=0.03, div_factor=10,
    )

    best = float("inf")
    global_step = 0
    for ep in range(cfg.epochs):
        ep_t0 = time.time()
        model.train()
        t_loader.sampler.set_epoch(ep)
        t_ds.set_epoch_seed(ep * 10000 + rank)
        v_ds.set_epoch_seed(0)

        for i, (x, _stamp, _exog) in enumerate(t_loader):
            x = x.to(device, non_blocking=True)
            total_loss = 0.0
            for j in range(cfg.accumulation_steps):
                size = x.shape[0] // cfg.accumulation_steps
                chunk = x[j * size : (j + 1) * size]
                (z_pre, z), bsq_loss, _, _ = model(chunk)
                recon = F.mse_loss(z_pre, chunk) + F.mse_loss(z, chunk)
                loss = (recon + bsq_loss) / 2
                total_loss += loss.item()
                (loss / cfg.accumulation_steps).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step(); sch.step(); opt.zero_grad()

            if rank == 0 and (global_step + 1) % cfg.log_interval == 0:
                print(f"[ep {ep+1}/{cfg.epochs} step {i+1}/{len(t_loader)}] "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"loss={total_loss / cfg.accumulation_steps:.4f}")
            global_step += 1

        # --- validation ---
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

        if rank == 0:
            print(f"--- ep {ep+1}: val_recon={val:.4f} "
                  f"({format_time(time.time() - ep_t0)} / total {format_time(time.time() - t0)}) ---")
            if val < best:
                best = val
                save = save_dir / "checkpoints" / "best_model"
                model.module.save_pretrained(str(save))
                print(f"[save] best → {save}")
        dist.barrier()

    return {"best_val_loss": best}


def main():
    cfg = TrainConfig()
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("请用 torchrun 启动此脚本")
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")
    set_seed(cfg.seed, rank)

    save_dir = Path(cfg.save_path) / cfg.tokenizer_save_folder_name
    if rank == 0:
        (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path).to(device)
    model = DDP(model, device_ids=[local], find_unused_parameters=False)
    if rank == 0:
        print("Tokenizer size:", get_model_size(model.module))

    summary = {"start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()), "world_size": world}
    summary["final_result"] = _train(model, device, cfg, save_dir, rank, world)
    if rank == 0:
        (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cleanup_ddp()


if __name__ == "__main__":
    main()
