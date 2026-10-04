"""Evaluate raw log-return forecasts at the final visible history bar.

Cross-sectional IC is computed across distinct symbols at an exact UTC time,
then averaged within date/hour buckets. Pooled and per-symbol correlations are
separate diagnostics and never substituted for cross-sectional evidence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from kairos.data.contracts import (
    MAIN_COLUMNS, bar_delta, contiguous_starts, log_return_targets,
    validate_exog_frame, validate_main_frame,
)
from kairos.training.config import TrainConfig, preset_for
from kairos.vendor.kronos import Kronos, KronosPredictor, KronosTokenizer


_BUCKET_ALIASES = {
    "auto": "date", "date": "date", "day": "date", "daily": "date",
    "hour": "hour", "hourly": "hour", "minute": "minute",
    "minutely": "minute", "none": "none", "pool": "none",
}


def _load_dataset_meta(dataset_path: str | Path) -> dict:
    path = Path(dataset_path) / "meta.json"
    if not path.exists():
        return {}
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("dataset meta.json must contain an object")
    return meta


def _bucket_label(date: pd.Timestamp, bucket: str) -> pd.Timestamp | str:
    if bucket == "date":
        return date.normalize()
    if bucket == "hour":
        return date.floor("h")
    if bucket == "minute":
        return date.floor("min")
    return "all_timestamps"


def _number(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _correlations(frame: pd.DataFrame, score: str, target: str) -> dict:
    pairs = frame[[score, target]].replace([np.inf, -np.inf], np.nan).dropna()
    result = dict(pearson=None, pearson_p=None, spearman=None, spearman_p=None,
                  n=len(pairs), hit_rate=None)
    if len(pairs):
        result["hit_rate"] = float(((pairs[score] > 0) == (pairs[target] > 0)).mean())
    if len(pairs) >= 2 and pairs[score].nunique() > 1 and pairs[target].nunique() > 1:
        p, s = pearsonr(pairs[score], pairs[target]), spearmanr(pairs[score], pairs[target])
        result.update(pearson=_number(p.statistic), pearson_p=_number(p.pvalue),
                      spearman=_number(s.statistic), spearman_p=_number(s.pvalue))
    return result


def summarize_records(records: list[dict], horizons: List[int], aggregation: str = "date") -> dict:
    """Separate exact-time cross sections, symbol time series and pooled pairs."""
    bucket = _BUCKET_ALIASES.get(aggregation.lower())
    if bucket is None:
        raise ValueError("aggregation must be auto, date, hour, minute or none")
    columns = ["symbol", "date"] + [f"{kind}_h{h}" for h in horizons for kind in ("score", "ret")]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate symbol/timestamp predictions cannot form an IC cross section")
    report = {
        "n_records": len(frame), "n_symbols": int(frame["symbol"].nunique()),
        "date_range": [str(frame.date.min()), str(frame.date.max())] if len(frame) else [None, None],
        "pooled": {}, "time_series": {}, "cross_sectional": {}, "by_date_mean": {},
        "methodology": {
            "version": 2, "target": "log_return", "anchor": "last_visible_history_bar",
            "cross_section": "exact_utc_timestamp", "minimum_distinct_symbols": 3,
            "aggregation": bucket, "bucket_weighting": "equal_valid_timestamps",
            "summary_weighting": "equal_valid_buckets", "icir_ddof": 1,
            "p_values": "naive_iid_diagnostics_only; overlapping returns are dependent",
            "legacy_aliases": {"overall": "pooled", "by_date_mean": "cross_sectional.summary"},
        },
    }
    for h in horizons:
        key, score, target = f"h{h}", f"score_h{h}", f"ret_h{h}"
        report["pooled"][key] = _correlations(frame, score, target)
        for symbol, group in frame.groupby("symbol", sort=True):
            report["time_series"].setdefault(str(symbol), {})[key] = _correlations(group, score, target)
        points = []
        for date, group in frame.groupby("date", sort=True):
            finite = group.replace([np.inf, -np.inf], np.nan).dropna(subset=[score, target])
            if finite.symbol.nunique() < 3:
                continue
            corr = _correlations(finite, score, target)
            if corr["pearson"] is not None and corr["spearman"] is not None:
                points.append({"timestamp": str(date), "bucket": str(_bucket_label(date, bucket)),
                               "ic": corr["pearson"], "rank_ic": corr["spearman"],
                               "n_symbols": int(finite.symbol.nunique())})
        buckets = []
        if points:
            for label, group in pd.DataFrame(points).groupby("bucket", sort=True):
                buckets.append({"bucket": label, "ic": float(group.ic.mean()),
                                "rank_ic": float(group.rank_ic.mean()), "n_timestamps": len(group),
                                "n_symbols_mean": float(group.n_symbols.mean())})
        ics = np.array([item["ic"] for item in buckets])
        ranks = np.array([item["rank_ic"] for item in buckets])
        sd = float(ics.std(ddof=1)) if len(ics) > 1 else None
        mean = float(ics.mean()) if len(ics) else None
        summary = {"ic": mean, "rank_ic": float(ranks.mean()) if len(ranks) else None,
                   "icir": mean / sd if sd is not None and sd > 1e-12 else None,
                   "ic_std": sd, "n_dates": len(buckets), "n_buckets": len(buckets),
                   "n_timestamps": len(points), "bucket": bucket,
                   "n_symbols_mean": float(np.mean([x["n_symbols"] for x in points])) if points else None}
        report["cross_sectional"][key] = {"summary": summary, "timestamps": points, "buckets": buckets}
        report["by_date_mean"][key] = summary.copy()
    report["overall"] = report["pooled"]
    return report


def _load_trained_predictor(path: str, device: str, tokenizer_path: str | None):
    from kairos.inference import KairosPredictor
    return KairosPredictor.from_checkpoint(path, device=device, tokenizer_path=tokenizer_path)


def _load_baseline_predictor(cfg: TrainConfig, device: str, tokenizer_path: str | None):
    # Never inherit a local fine-tuned tokenizer by accident for the original baseline.
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path or cfg.pretrained_tokenizer_path)
    model = Kronos.from_pretrained(cfg.pretrained_predictor_path)
    return KronosPredictor(model.eval(), tokenizer.eval(), device=device,
                           max_context=cfg.max_context, clip=cfg.clip)


@contextmanager
def _seeded(seed: int):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _validate_dataset_meta(meta: dict, manifest: dict, *, require_provenance: bool = False) -> None:
    if "schema_version" in meta and (type(meta["schema_version"]) is not int or meta["schema_version"] != 2):
        raise ValueError("unsupported dataset schema_version; expected 2 or an unversioned legacy manifest")
    if require_provenance:
        required = ["market", "freq", "exog_cols"]
        if manifest.get("market_type") is not None:
            required.append("market_type")
        missing = [key for key in required if key not in meta or meta[key] is None]
        if missing:
            raise ValueError(f"dataset meta.json missing {missing}; repack the dataset with kairos-prepare")
    for key in ("market", "market_type", "feature_cols", "exog_cols"):
        if key in meta and key in manifest and meta[key] != manifest[key]:
            raise ValueError(f"dataset {key} differs from model contract")
    if "freq" in meta and bar_delta(meta["freq"]) != bar_delta(manifest["freq"]):
        raise ValueError("dataset freq differs from model contract")


def _baseline_scores(predictor, main: list[pd.DataFrame], horizon: int, freq: str) -> np.ndarray:
    delta = bar_delta(freq)
    future = [pd.Series(pd.date_range(df.index[-1] + delta, periods=horizon, freq=delta)) for df in main]
    predictions = predictor.predict_batch(
        [df.rename(columns={"vol": "volume", "amt": "amount"}) for df in main],
        [pd.Series(df.index) for df in main], future,
        pred_len=horizon, sample_count=1, verbose=False,
    )
    if len(predictions) != len(main):
        raise ValueError("baseline prediction batch length mismatch")
    scores = []
    for df, prediction in zip(main, predictions):
        close = prediction["close"].to_numpy(dtype=np.float64)
        if close.shape != (horizon,) or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("baseline must generate finite positive close prices at every horizon")
        scores.append(np.log(close) - np.log(float(df.close.iloc[-1])))
    return np.stack(scores)


def _seed_summary(runs: list[dict], horizons: List[int]) -> dict:
    summary = {"pooled": {}, "cross_sectional": {}}
    for h in horizons:
        key = f"h{h}"
        for section, metrics in (("pooled", ("pearson", "spearman", "hit_rate")),
                                 ("cross_sectional", ("ic", "rank_ic", "icir"))):
            summary[section][key] = {}
            for metric in metrics:
                values = [run[section][key]["summary"][metric] if section == "cross_sectional"
                          else run[section][key][metric] for run in runs]
                values = [value for value in values if value is not None]
                summary[section][key][metric] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "n_seeds": len(values),
                }
    return summary


@torch.no_grad()
def run_backtest(
    ckpt_path: str | None, cfg: TrainConfig, horizons: List[int] = (1, 5),
    batch_size: int = 64, max_symbols: int | None = None, device: str | None = None,
    use_baseline: bool = False, aggregation: str = "auto", stride: int = 1,
    per_symbol_limit: int | None = None, tokenizer_path: str | None = None,
    seed: int = 100, seeds: List[int] | None = None,
) -> Dict:
    """Evaluate a versioned bundle or actual original-Kronos sampled forecasts."""
    if batch_size <= 0 or stride <= 0 or (max_symbols is not None and max_symbols <= 0):
        raise ValueError("batch_size, stride and max_symbols must be positive")
    if per_symbol_limit is not None and per_symbol_limit < 0:
        raise ValueError("per_symbol_limit cannot be negative")
    if aggregation.lower() not in _BUCKET_ALIASES:
        raise ValueError("aggregation must be auto, date, hour, minute or none")
    horizons = list(horizons)
    if not horizons or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
        raise ValueError("horizons must be nonempty positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be distinct")
    repeat_seeds = list(seeds) if seeds is not None else [seed]
    if (not repeat_seeds or len(set(repeat_seeds)) != len(repeat_seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in repeat_seeds)):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    baseline = use_baseline or ckpt_path is None
    if use_baseline and ckpt_path:
        raise ValueError("choose either a trained checkpoint or the original baseline")
    if seeds is not None and not baseline:
        raise ValueError("multiple seeds apply only to the stochastic original baseline")
    device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with _seeded(repeat_seeds[0]):
        predictor = (_load_baseline_predictor(cfg, device, tokenizer_path) if baseline else
                     _load_trained_predictor(ckpt_path, device, tokenizer_path))
    manifest = ({"freq": cfg.freq, "market": cfg.market, "lookback_window": cfg.lookback_window,
                 "return_horizon": cfg.return_horizon, "feature_cols": MAIN_COLUMNS}
                if baseline else predictor.manifest)
    lookback, horizon = int(manifest["lookback_window"]), max(horizons)
    if horizon > manifest["return_horizon"]:
        raise ValueError(f"requested horizon {horizon} exceeds model horizon {manifest['return_horizon']}")
    if lookback < 2 or lookback > getattr(predictor, "max_context", cfg.max_context):
        raise ValueError("lookback must be at least two and within the model max_context")
    freq = manifest["freq"]
    delta = bar_delta(freq)
    root = Path(cfg.dataset_path)
    meta = _load_dataset_meta(root)
    _validate_dataset_meta(meta, manifest, require_provenance=not baseline)
    with (root / "test_data.pkl").open("rb") as fh:
        raw_main = pickle.load(fh)
    if not isinstance(raw_main, dict):
        raise ValueError("test_data.pkl must map symbols to data frames")
    raw_exog = {}
    use_exog = not baseline and manifest.get("use_exog", True)
    if use_exog:
        with (root / "exog_test.pkl").open("rb") as fh:
            raw_exog = pickle.load(fh)
    symbols = sorted(raw_main)[:max_symbols]
    frames, candidates = {}, {}
    for symbol in symbols:
        main = validate_main_frame(raw_main[symbol])
        if use_exog:
            if symbol not in raw_exog:
                raise ValueError(f"missing exogenous data for {symbol}")
            exog = validate_exog_frame(main, raw_exog[symbol], manifest["exog_cols"])
        else:
            exog = None
        starts = contiguous_starts(main.index, lookback + horizon, freq)
        # Anchor against a shared UTC grid, not offsets relative to each listing date.
        anchors = main.index[starts + lookback - 1]
        starts = starts[((anchors - pd.Timestamp("1970-01-01")) // delta) % stride == 0]
        candidates[symbol] = starts
        frames[symbol] = (main, exog)
    allowed = None
    if per_symbol_limit:
        dates = sorted({frames[s][0].index[int(start) + lookback - 1]
                        for s in symbols for start in candidates[s]})
        selected = np.linspace(0, len(dates) - 1, min(per_symbol_limit, len(dates)), dtype=int)
        allowed = {dates[i] for i in selected}

    def evaluate(run_seed: int) -> dict:
        records, main_buffer, exog_buffer, meta_buffer = [], [], [], []

        def flush():
            if not main_buffer:
                return
            if baseline:
                scores = _baseline_scores(predictor, main_buffer, horizon, freq)
            else:
                quantiles = np.asarray(predictor.predict_batch(main_buffer, exog_buffer))
                expected = (len(main_buffer), manifest["return_horizon"], manifest["n_quantiles"])
                if quantiles.shape != expected or not np.isfinite(quantiles).all():
                    raise ValueError("trained predictor returned invalid quantile shape or values")
                mid = manifest["n_quantiles"] // 2
                scores = quantiles[:, :, mid]
            for info, values in zip(meta_buffer, scores):
                records.append({**info, **{f"score_h{h}": float(values[h - 1]) for h in horizons}})
            main_buffer.clear()
            exog_buffer.clear()
            meta_buffer.clear()

        with _seeded(run_seed):
            for symbol in symbols:
                main, exog = frames[symbol]
                for start in candidates[symbol]:
                    anchor = int(start) + lookback - 1
                    date = main.index[anchor]
                    if allowed is not None and date not in allowed:
                        continue
                    # Validate only this target span; full series validation already ran once.
                    targets = log_return_targets(main.close.iloc[anchor:anchor + horizon + 1], 0, horizon)
                    main_buffer.append(main.iloc[start:anchor + 1])
                    exog_buffer.append(exog.iloc[start:anchor + 1] if exog is not None else None)
                    meta_buffer.append({"symbol": symbol, "date": date,
                                        **{f"ret_h{h}": float(targets[h - 1]) for h in horizons}})
                    if len(main_buffer) >= batch_size:
                        flush()
            flush()
        report = summarize_records(records, horizons, aggregation)
        report.update(seed=run_seed, model={
            "mode": "original_kronos" if baseline else "kairos_log_return_v2",
            "source": cfg.pretrained_predictor_path if baseline else str(ckpt_path),
            "tokenizer_source": (tokenizer_path or cfg.pretrained_tokenizer_path) if baseline else "bundle",
        })
        report["evaluation"] = {
            "freq": freq, "lookback_window": lookback, "horizons": horizons,
            "stride": stride, "per_symbol_limit": per_symbol_limit, "batch_size": batch_size,
            "sampling": "shared_utc_anchor_grid", "dataset_meta_present": bool(meta),
            "seed_scope": "same software, device, batch size and ordered dataset",
        }
        return report

    runs = [evaluate(run_seed) for run_seed in repeat_seeds]
    if seeds is None:
        return runs[0]
    return {"seeds": repeat_seeds, "runs": runs, "seed_summary": _seed_summary(runs, horizons),
            "methodology": runs[0]["methodology"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    model = ap.add_mutually_exclusive_group()
    model.add_argument("--ckpt", help="version 2 log-return bundle directory")
    model.add_argument("--baseline", action="store_true", help="sample actual original-Kronos price forecasts")
    ap.add_argument("--out", default="artifacts/backtest_report.json")
    ap.add_argument("--horizons", default="1,5", help="distinct h in 1..model return_horizon")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-symbols", type=int)
    ap.add_argument("--market", help="baseline market; trained model metadata is authoritative")
    ap.add_argument("--dataset-path")
    ap.add_argument("--preset", help="baseline preset, e.g. crypto-1min")
    ap.add_argument("--aggregation", default="auto", help="average timestamp IC by date/hour/minute/none")
    ap.add_argument("--stride", type=int, default=1, help="sample on a shared UTC grid of N bars")
    ap.add_argument("--per-symbol-limit", type=int, default=0, help="shared maximum N anchor timestamps")
    ap.add_argument("--tokenizer", help="explicit original tokenizer source; trained bundles bind their tokenizer")
    ap.add_argument("--predictor", help="original baseline predictor source")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--seeds", help="comma-separated independent baseline seeds; writes runs and mean/std")
    args = ap.parse_args()
    overrides = preset_for(args.preset) if args.preset else {}
    for arg, name in ((args.dataset_path, "dataset_path"), (args.market, "market"),
                      (args.predictor, "pretrained_predictor_path")):
        if arg is not None:
            overrides[name] = arg
    cfg = TrainConfig(**overrides)
    meta = _load_dataset_meta(cfg.dataset_path)
    for field in ("market", "freq"):
        if field in meta and field not in overrides:
            setattr(cfg, field, meta[field])
    report = run_backtest(
        args.ckpt, cfg, horizons=[int(h) for h in args.horizons.split(",")],
        batch_size=args.batch_size, max_symbols=args.max_symbols, device=args.device,
        use_baseline=args.baseline, aggregation=args.aggregation, stride=args.stride,
        per_symbol_limit=args.per_symbol_limit or None, tokenizer_path=args.tokenizer,
        seed=args.seed, seeds=[int(s) for s in args.seeds.split(",")] if args.seeds else None,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"[save] {output}")
    print(json.dumps(report.get("seed_summary", report.get("by_date_mean")), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
