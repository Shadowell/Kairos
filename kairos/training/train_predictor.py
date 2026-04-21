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
import os
import time
from pathlib import Path
from time import gmtime, strftime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kairos.models import KronosWithExogenous
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


def _train(model, tokenizer, device, cfg: TrainConfig, save_dir: Path,
           rank: int, world: int):
    t0 = time.time()
    t_loader, v_loader, t_ds, v_ds = _make_loaders(cfg, rank, world)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=cfg.predictor_learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
    )
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.predictor_learning_rate,
        steps_per_epoch=len(t_loader), epochs=cfg.epochs,
        pct_start=getattr(cfg, "warmup_pct", 0.03), div_factor=10,
    )

    quantiles = None
    if cfg.use_return_head:
        quantiles = torch.linspace(0.1, 0.9, cfg.n_quantiles, device=device)

    best = float("inf")
    step_g = 0
    close_idx = 3  # feature_list order: open, high, low, close, vol, amt
    patience = getattr(cfg, "patience", 0)
    bad_epochs = 0

    for ep in range(cfg.epochs):
        ep_t0 = time.time()
        model.train()
        t_loader.sampler.set_epoch(ep)
        t_ds.set_epoch_seed(ep * 10000 + rank)
        v_ds.set_epoch_seed(0)

        for i, (x, stamp, exog) in enumerate(t_loader):
            x = x.to(device, non_blocking=True)
            stamp = stamp.to(device, non_blocking=True)
            exog = exog.to(device, non_blocking=True) if cfg.use_exog else None

            with torch.no_grad():
                s1_ids, s2_ids = tokenizer.encode(x, half=True)

            s1_in, s2_in = s1_ids[:, :-1], s2_ids[:, :-1]
            s1_tg, s2_tg = s1_ids[:, 1:], s2_ids[:, 1:]
            stamp_in = stamp[:, :-1, :]
            exog_in = exog[:, :-1, :] if exog is not None else None

            s1_logits, s2_logits, q_pred = model(
                s1_in, s2_in, stamp=stamp_in, exog=exog_in,
                use_teacher_forcing=True, s1_targets=s1_tg,
            )
            ce, _, _ = model.module.head.compute_loss(s1_logits, s2_logits, s1_tg, s2_tg)
            loss = cfg.ce_weight * ce

            if cfg.use_return_head and q_pred is not None:
                close_n = x[:, :, close_idx]  # normalized close
                T = close_n.size(1)
                h = cfg.return_horizon
                targets = []
                for k in range(h):
                    rolled = torch.roll(close_n, shifts=-(k + 1), dims=1)
                    targets.append(rolled - close_n)
                target = torch.stack(targets, dim=-1)          # [B, T, h]
                valid = T - h
                target = target[:, :valid]
                q_valid = q_pred[:, :valid]
                mask = torch.ones_like(target[..., 0])
                pin = model.module.return_head.pinball_loss(q_valid, target, quantiles, mask)
                loss = loss + cfg.quantile_weight * pin

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=3.0)
            opt.step(); sch.step()

            if rank == 0 and (step_g + 1) % cfg.log_interval == 0:
                print(f"[ep {ep+1}/{cfg.epochs} step {i+1}/{len(t_loader)}] "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"loss={loss.item():.4f} ce={ce.item():.4f}")
            step_g += 1

        # --- validation ---
        model.eval()
        loss_sum, count = 0.0, 0
        with torch.no_grad():
            for x, stamp, exog in v_loader:
                x = x.to(device); stamp = stamp.to(device)
                exog = exog.to(device) if cfg.use_exog else None
                s1_ids, s2_ids = tokenizer.encode(x, half=True)
                s1_l, s2_l, _ = model(
                    s1_ids[:, :-1], s2_ids[:, :-1],
                    stamp=stamp[:, :-1],
                    exog=exog[:, :-1] if exog is not None else None,
                )
                ce, _, _ = model.module.head.compute_loss(
                    s1_l, s2_l, s1_ids[:, 1:], s2_ids[:, 1:])
                loss_sum += ce.item(); count += 1
        ls = torch.tensor(loss_sum, device=device); cn = torch.tensor(count, device=device)
        dist.all_reduce(ls); dist.all_reduce(cn)
        val = (ls / cn).item() if cn.item() else 0.0

        improved = val < best - 1e-4
        if rank == 0:
            print(f"--- ep {ep+1}: val_ce={val:.4f} "
                  f"({format_time(time.time() - ep_t0)} / total {format_time(time.time() - t0)}) ---")
            if improved:
                best = val
                save = save_dir / "checkpoints" / "best_model"
                model.module.save_pretrained(str(save))
                print(f"[save] best → {save} (val_ce={val:.4f})")
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
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("请用 torchrun 启动此脚本")
    rank, world, local = setup_ddp()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    set_seed(cfg.seed, rank)

    save_dir = Path(cfg.save_path) / cfg.predictor_save_folder_name
    if rank == 0:
        (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # tokenizer: prefer fine-tuned if available, else the public checkpoint
    tok_path = Path(cfg.save_path) / cfg.tokenizer_save_folder_name / "checkpoints" / "best_model"
    tok_src = str(tok_path) if tok_path.exists() else cfg.pretrained_tokenizer_path
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
    ddp_kwargs = dict(find_unused_parameters=True)
    if use_cuda:
        ddp_kwargs["device_ids"] = [local]
    model = DDP(model, **ddp_kwargs)

    if rank == 0:
        print("Predictor size:", get_model_size(model.module))

    summary = {"start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()), "world_size": world}
    summary["final_result"] = _train(model, tokenizer, device, cfg, save_dir, rank, world)
    if rank == 0:
        (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cleanup_ddp()


if __name__ == "__main__":
    main()
