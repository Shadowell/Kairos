"""Reconstruction benchmark for Kronos BSQ tokenizers.

The tokenizer's job is to encode a 6-dim OHLCV window into two streams of
discrete tokens (s1 + s2) and decode them back to continuous values. We
care about three things:

1. **Reconstruction fidelity** — MSE / MAE between the decoded series and
   the standardised input, both overall and per channel (open, high, low,
   close, vol, amt).
2. **Codebook utilisation** — how many of the 2^s1_bits / 2^s2_bits slots
   are actually used, and the Shannon entropy of the observed token
   distribution. A tokenizer that collapses onto a handful of codes is
   useless regardless of MSE.
3. **BSQ regularisation** — the loss term that the tokenizer minimises
   at training time; reported for reference.

Usage::

    python -m kairos.training.eval_tokenizer \
        --ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
        --preset crypto-1min \
        --dataset-path ./finetune/data/crypto_1min_btc_eth \
        --out artifacts/tokenizer_eval_finetuned.json

    python -m kairos.training.eval_tokenizer \
        --baseline --preset crypto-1min \
        --dataset-path ./finetune/data/crypto_1min_btc_eth \
        --out artifacts/tokenizer_eval_baseline.json

Both runs produce the same JSON schema so ``jq`` / diffing the two files
is the natural way to quantify the fine-tune's effect.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from kairos.training.config import TrainConfig, preset_for
from kairos.vendor.kronos import KronosTokenizer


def _load_dataset_meta(dataset_path: str | Path) -> dict:
    path = Path(dataset_path) / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _prep_windows(
    df: pd.DataFrame, cfg: TrainConfig, stride: int, per_symbol_limit: int | None
) -> List[np.ndarray]:
    """Slice ``df`` into standardised lookback windows ready for the tokenizer."""
    df = df.reset_index()
    if "datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])

    x = df[cfg.feature_list].values.astype(np.float32)
    total = len(df)
    window = cfg.lookback_window + 1  # +1 so target shapes match training
    step = max(1, int(stride))
    starts = list(range(0, total - window + 1, step))
    if per_symbol_limit and per_symbol_limit > 0 and len(starts) > per_symbol_limit:
        idx = np.linspace(0, len(starts) - 1, per_symbol_limit).astype(int)
        starts = [starts[i] for i in idx]

    windows = []
    for s in starts:
        w = x[s : s + window]
        past = w[: cfg.lookback_window]
        mu, sd = past.mean(0), past.std(0)
        w_norm = np.clip((w - mu) / (sd + 1e-5), -cfg.clip, cfg.clip)
        windows.append(w_norm.astype(np.float32))
    return windows


def _entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@torch.no_grad()
def run_eval(
    ckpt_path: str | None,
    cfg: TrainConfig,
    batch_size: int = 64,
    max_symbols: int | None = None,
    device: str | None = None,
    use_baseline: bool = False,
    stride: int = 1,
    per_symbol_limit: int | None = None,
) -> Dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device_t}")

    if use_baseline or ckpt_path is None:
        src = cfg.pretrained_tokenizer_path
        print(f"[load] baseline tokenizer: {src}")
    else:
        src = ckpt_path
        print(f"[load] fine-tuned tokenizer: {src}")
    tok = KronosTokenizer.from_pretrained(src).eval().to(device_t)

    s1_bits = getattr(tok, "s1_bits")
    s2_bits = getattr(tok, "s2_bits")
    s1_vocab = 2 ** s1_bits
    s2_vocab = 2 ** s2_bits
    print(f"[tok] d_in={tok.d_in} s1_bits={s1_bits} (vocab {s1_vocab}) "
          f"s2_bits={s2_bits} (vocab {s2_vocab})")

    # Dataset
    root = Path(cfg.dataset_path)
    with open(root / "test_data.pkl", "rb") as f:
        test_data: Dict[str, pd.DataFrame] = pickle.load(f)

    symbols = list(test_data.keys())
    if max_symbols:
        symbols = symbols[:max_symbols]
    print(f"[data] {len(symbols)} symbols")

    # Aggregation accumulators
    sum_sq_full = np.zeros(len(cfg.feature_list), dtype=np.float64)
    sum_abs_full = np.zeros(len(cfg.feature_list), dtype=np.float64)
    sum_sq_pre = np.zeros(len(cfg.feature_list), dtype=np.float64)
    bsq_loss_sum = 0.0
    n_elems = 0
    n_windows = 0

    s1_counts = np.zeros(s1_vocab, dtype=np.int64)
    s2_counts = np.zeros(s2_vocab, dtype=np.int64)

    buf: List[np.ndarray] = []

    def flush():
        nonlocal bsq_loss_sum, n_elems, n_windows
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device_t)
        (z_pre, z), bsq_loss, _, z_indices = tok(x)

        # z, z_pre are [B, T, d_in]; difference in standardised space
        diff_full = (z - x).detach().cpu().numpy()
        diff_pre = (z_pre - x).detach().cpu().numpy()
        sum_sq_full[:] = sum_sq_full + (diff_full ** 2).sum(axis=(0, 1))
        sum_abs_full[:] = sum_abs_full + np.abs(diff_full).sum(axis=(0, 1))
        sum_sq_pre[:] = sum_sq_pre + (diff_pre ** 2).sum(axis=(0, 1))
        n_elems += diff_full.shape[0] * diff_full.shape[1]
        n_windows += diff_full.shape[0]
        bsq_loss_sum += float(bsq_loss.item()) * diff_full.shape[0]

        # Codebook usage — use half-encode so we get the (s1, s2) tuple directly
        s1_idx, s2_idx = tok.encode(x, half=True)
        s1_flat = s1_idx.detach().cpu().numpy().reshape(-1)
        s2_flat = s2_idx.detach().cpu().numpy().reshape(-1)
        # np.bincount is ~2x faster than np.add.at for this shape
        s1_counts[:] = s1_counts + np.bincount(s1_flat, minlength=s1_vocab)[:s1_vocab]
        s2_counts[:] = s2_counts + np.bincount(s2_flat, minlength=s2_vocab)[:s2_vocab]
        buf.clear()

    for si, sym in enumerate(symbols):
        windows = _prep_windows(test_data[sym], cfg, stride, per_symbol_limit)
        print(f"[sym {si + 1}/{len(symbols)}] {sym}: {len(windows)} windows")
        for w in windows:
            buf.append(w)
            if len(buf) >= batch_size:
                flush()
        flush()

    if n_elems == 0:
        raise RuntimeError("No evaluation windows collected — check dataset_path / lookback_window / stride")

    feat = cfg.feature_list
    per_ch_mse = (sum_sq_full / max(n_elems, 1)).tolist()
    per_ch_mae = (sum_abs_full / max(n_elems, 1)).tolist()
    per_ch_mse_pre = (sum_sq_pre / max(n_elems, 1)).tolist()

    s1_used = int((s1_counts > 0).sum())
    s2_used = int((s2_counts > 0).sum())
    s1_entropy = _entropy_bits(s1_counts)
    s2_entropy = _entropy_bits(s2_counts)

    report = {
        "source": "baseline" if use_baseline or ckpt_path is None else "finetuned",
        "tokenizer_path": src,
        "dataset_path": str(cfg.dataset_path),
        "market": cfg.market,
        "freq": cfg.freq,
        "n_symbols": len(symbols),
        "n_windows": int(n_windows),
        "n_elements": int(n_elems),
        "lookback_window": cfg.lookback_window,
        "tokenizer_spec": {
            "s1_bits": int(s1_bits),
            "s2_bits": int(s2_bits),
            "s1_vocab": int(s1_vocab),
            "s2_vocab": int(s2_vocab),
            "d_in": int(tok.d_in),
            "d_model": int(tok.d_model),
        },
        "metrics": {
            "recon_mse_full": float(np.mean(per_ch_mse)),
            "recon_mae_full": float(np.mean(per_ch_mae)),
            "recon_mse_pre_s1_only": float(np.mean(per_ch_mse_pre)),
            "bsq_loss_mean": float(bsq_loss_sum / max(n_windows, 1)),
            "per_channel_mse": {f: float(v) for f, v in zip(feat, per_ch_mse)},
            "per_channel_mae": {f: float(v) for f, v in zip(feat, per_ch_mae)},
        },
        "codebook": {
            "s1": {
                "vocab_size": int(s1_vocab),
                "unique_used": s1_used,
                "utilization": s1_used / s1_vocab,
                "entropy_bits": s1_entropy,
                "entropy_max_bits": float(math.log2(s1_vocab)),
            },
            "s2": {
                "vocab_size": int(s2_vocab),
                "unique_used": s2_used,
                "utilization": s2_used / s2_vocab,
                "entropy_bits": s2_entropy,
                "entropy_max_bits": float(math.log2(s2_vocab)),
            },
        },
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=False, default=None,
                    help="path to fine-tuned tokenizer best_model dir; omit / use --baseline for the pretrained one")
    ap.add_argument("--baseline", action="store_true",
                    help="evaluate the public NeoQuasar/Kronos-Tokenizer-base instead")
    ap.add_argument("--out", default="artifacts/tokenizer_eval.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--market", default=None,
                    help="override market preset; default: read from dataset meta.json")
    ap.add_argument("--dataset-path", default=None,
                    help="override TrainConfig.dataset_path (path to the prepared bundle)")
    ap.add_argument("--preset", default=None,
                    help="named preset from kairos.training.config.preset_for (e.g. 'crypto-1min')")
    ap.add_argument("--stride", type=int, default=1,
                    help="only take one window every N bars (CPU smoke friendly)")
    ap.add_argument("--per-symbol-limit", type=int, default=0,
                    help=">0 means at most N windows per symbol (evenly spaced)")
    args = ap.parse_args()

    overrides: dict = {}
    if args.preset:
        overrides.update(preset_for(args.preset))
    if args.dataset_path:
        overrides["dataset_path"] = args.dataset_path
    if args.market:
        overrides["market"] = args.market
    cfg = TrainConfig(**overrides) if overrides else TrainConfig()

    meta = _load_dataset_meta(cfg.dataset_path)
    if meta:
        if "market" in meta and "market" not in overrides:
            cfg.market = meta["market"]
        if "freq" in meta and "freq" not in overrides:
            cfg.freq = meta["freq"]

    report = run_eval(
        args.ckpt, cfg,
        batch_size=args.batch_size,
        max_symbols=args.max_symbols,
        use_baseline=args.baseline,
        stride=args.stride,
        per_symbol_limit=args.per_symbol_limit or None,
    )

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {out_p}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
