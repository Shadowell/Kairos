"""Versioned, self-contained predictor bundles and collision-safe run paths."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from uuid import uuid4

from kairos.data.features import exog_cols_for
from kairos.data.contracts import TIME_COLUMNS, bar_delta
from kairos.training.config import TrainConfig


MANIFEST_NAME = "kairos_manifest.json"
CONTRACT_VERSION = 2
_REQUIRED = {
    "contract_version", "target", "feature_cols", "exog_cols", "freq", "market",
    "market_type", "lookback_window", "clip", "return_horizon", "n_quantiles",
    "use_exog", "tokenizer_path", "tokenizer_files", "dataset_hashes", "source_sha",
    "training_config",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_dir(cfg: TrainConfig) -> Path:
    """Reserve a new run directory; even explicit run names never overwrite."""
    run_id = getattr(cfg, "run_id", None) or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be a single safe directory name")
    path = Path(cfg.save_path) / cfg.predictor_save_folder_name / run_id
    path.mkdir(parents=True, exist_ok=False)
    cfg.run_id = run_id
    return path


def _source_version() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        )
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        # Wheels/source archives have no Git identity. Unknown stays explicit.
        return None, None


def hash_training_data(cfg: TrainConfig) -> dict[str, str]:
    """Hash inputs once at run startup, before loaders read their snapshots."""
    names = ["meta.json", "train_data.pkl", "val_data.pkl"]
    if cfg.use_exog:
        names += ["exog_train.pkl", "exog_val.pkl"]
    return {name: _sha256(Path(cfg.dataset_path) / name) for name in names}


def _training_config(cfg: TrainConfig) -> dict:
    # Logging credentials do not belong in published model artifacts.
    return {key: value for key, value in asdict(cfg).items()
            if not key.startswith("comet_") and not re.search(
                r"password|secret|credential|api_key|access_token", key, re.I)}


def save_checkpoint(model, tokenizer, cfg: TrainConfig, checkpoint: str | Path,
                    dataset_hashes: dict[str, str] | None = None) -> dict:
    """Save model and exact in-memory tokenizer snapshot with their contract.

    The caller owns the run directory. Replacing a best checkpoint within that
    newly reserved run is allowed; a legacy checkpoint is never overwritten.
    """
    checkpoint = Path(checkpoint)
    if list(cfg.time_feature_list) != TIME_COLUMNS:
        raise ValueError("time_feature_list must match the fixed tokenizer/model time slots")
    if checkpoint.exists() and any(checkpoint.iterdir()):
        load_manifest(checkpoint)
    root = Path(cfg.dataset_path)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise ValueError("dataset meta.json is required for a version 2 checkpoint")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if (metadata.get("market", cfg.market) != cfg.market
            or bar_delta(metadata.get("freq", cfg.freq)) != bar_delta(cfg.freq)):
        raise ValueError("dataset market/freq does not match training configuration")
    feature_cols = list(cfg.feature_list)
    exog_cols = exog_cols_for(cfg.market)
    if metadata.get("feature_cols", feature_cols) != feature_cols:
        raise ValueError("dataset feature_cols does not match training configuration")
    if metadata.get("exog_cols", exog_cols) != exog_cols or len(exog_cols) != cfg.n_exog:
        raise ValueError("dataset exog_cols does not match training configuration")
    hashes = hash_training_data(cfg) if dataset_hashes is None else dict(dataset_hashes)
    source_sha, source_dirty = _source_version()
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "target": "log_return",
        "feature_cols": feature_cols,
        "exog_cols": exog_cols,
        "freq": cfg.freq,
        "market": cfg.market,
        "market_type": metadata.get("market_type"),
        "lookback_window": cfg.lookback_window,
        "clip": cfg.clip,
        "return_horizon": cfg.return_horizon,
        "n_quantiles": cfg.n_quantiles,
        "use_exog": cfg.use_exog,
        "tokenizer_path": "tokenizer",
        "dataset_hashes": hashes,
        "source_sha": source_sha,
        "source_dirty": source_dirty,
        "training_config": _training_config(cfg),
    }
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint))
    tokenizer_dir = checkpoint / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    manifest["tokenizer_files"] = {
        path.relative_to(tokenizer_dir).as_posix(): _sha256(path)
        for path in sorted(tokenizer_dir.rglob("*")) if path.is_file()
    }
    if not manifest["tokenizer_files"]:
        raise ValueError("tokenizer snapshot is empty")
    (checkpoint / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return load_manifest(checkpoint)


def load_manifest(checkpoint: str | Path) -> dict:
    """Validate a version 2 bundle and its tokenizer before any inference."""
    checkpoint = Path(checkpoint)
    path = checkpoint / MANIFEST_NAME
    if not path.is_file():
        raise ValueError("Missing version 2 Kairos manifest; legacy checkpoints require retraining")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid Kairos manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Kairos manifest must be an object")
    missing = _REQUIRED.difference(manifest)
    if missing:
        raise ValueError(f"Kairos manifest missing required fields: {sorted(missing)}")
    if type(manifest["contract_version"]) is not int or manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError("Unsupported Kairos contract_version; expected 2")
    if manifest["target"] != "log_return":
        raise ValueError("Unsupported Kairos target; expected log_return")
    for key in ("lookback_window", "return_horizon", "n_quantiles"):
        if type(manifest[key]) is not int or manifest[key] <= 0:
            raise ValueError(f"Invalid manifest {key}")
    if manifest["n_quantiles"] < 3 or manifest["n_quantiles"] % 2 == 0:
        raise ValueError("manifest n_quantiles must be odd and at least 3")
    if type(manifest["use_exog"]) is not bool:
        raise ValueError("Invalid manifest use_exog")
    if (type(manifest["clip"]) not in (int, float) or not math.isfinite(manifest["clip"])
            or manifest["clip"] <= 0):
        raise ValueError("Invalid manifest clip")
    for key in ("freq", "market"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"Invalid manifest {key}")
    bar_delta(manifest["freq"])
    if manifest["market_type"] is not None and not isinstance(manifest["market_type"], str):
        raise ValueError("Invalid manifest market_type")
    for key in ("feature_cols", "exog_cols"):
        cols = manifest[key]
        if (not isinstance(cols, list) or not cols or
                any(not isinstance(col, str) or not col for col in cols) or len(set(cols)) != len(cols)):
            raise ValueError(f"Invalid manifest {key}")
    if len(manifest["exog_cols"]) != 32:
        raise ValueError("manifest exog_cols must contain exactly 32 columns")
    if not isinstance(manifest["training_config"], dict):
        raise ValueError("Invalid manifest training_config")
    if manifest["training_config"].get("time_feature_list") != TIME_COLUMNS:
        raise ValueError("manifest training_config time_feature_list does not match fixed time slots")
    if manifest["source_sha"] is not None and (
            not isinstance(manifest["source_sha"], str) or
            not re.fullmatch(r"[0-9a-f]{40}", manifest["source_sha"])):
        raise ValueError("Invalid manifest source_sha")
    if manifest.get("source_dirty") is not None and type(manifest["source_dirty"]) is not bool:
        raise ValueError("Invalid manifest source_dirty")
    if manifest["tokenizer_path"] != "tokenizer":
        raise ValueError("manifest tokenizer_path must be the local tokenizer snapshot")
    tokenizer_dir = checkpoint / "tokenizer"
    if tokenizer_dir.is_symlink():
        raise ValueError("tokenizer snapshot must be local, not a symlink")
    files = manifest["tokenizer_files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Invalid manifest tokenizer_files")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe tokenizer file path")
        file = tokenizer_dir / relative
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or not file.is_file() or file.is_symlink()
                or not file.resolve().is_relative_to(tokenizer_dir.resolve())
                or _sha256(file) != expected):
            raise ValueError(f"Tokenizer checksum/hash mismatch: {name}")
    actual_files = {p.relative_to(tokenizer_dir).as_posix()
                    for p in tokenizer_dir.rglob("*") if p.is_file()}
    if actual_files != set(files):
        raise ValueError("Tokenizer snapshot files differ from manifest")
    hashes = manifest["dataset_hashes"]
    if not isinstance(hashes, dict) or not {"meta.json", "train_data.pkl", "val_data.pkl"} <= set(hashes):
        raise ValueError("Invalid manifest dataset_hashes")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in hashes.values()):
        raise ValueError("Invalid dataset hash")
    return manifest
