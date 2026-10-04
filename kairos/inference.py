"""One history-to-log-return contract for evaluation and HTTP serving."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kairos.data.contracts import (
    TIME_COLUMNS, contiguous_starts, normalize_window, validate_exog_frame, validate_main_frame,
)
from kairos.data.features import EXOG_COLS
from kairos.models import KronosWithExogenous
from kairos.vendor.kronos import KronosTokenizer


class KairosPredictor:
    def __init__(self, model, tokenizer, manifest: dict, device: str = "cpu"):
        if manifest.get("contract_version") != 2 or manifest.get("target") != "log_return":
            raise ValueError("checkpoint must declare contract v2 target=log_return; retrain legacy models")
        if manifest.get("feature_cols") != ["open", "high", "low", "close", "vol", "amt"]:
            raise ValueError("checkpoint main feature schema does not match Kairos")
        if manifest.get("exog_cols") != EXOG_COLS:
            raise ValueError("checkpoint exogenous schema does not match Kairos")
        if manifest.get("training_config", {}).get("time_feature_list") != TIME_COLUMNS:
            raise ValueError("checkpoint time feature order does not match Kairos")
        for name in ("return_horizon", "n_quantiles"):
            if getattr(model, name) != manifest[name]:
                raise ValueError(f"checkpoint model and manifest disagree on {name}")
        if model.n_exog != 32 or not model.use_return_head:
            raise ValueError("checkpoint requires a 32-column exogenous model with a return head")
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer.eval().to(self.device)
        self.max_context = int(manifest["lookback_window"])
        self.quantiles = np.linspace(.1, .9, manifest["n_quantiles"])
        if .5 not in self.quantiles:
            raise ValueError("checkpoint quantile grid must include the median (odd n_quantiles)")

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None,
                        tokenizer_path: str | None = None):
        from kairos.training.artifacts import load_manifest

        root = Path(path).expanduser()
        if not root.is_dir():
            raise ValueError("provide a local complete Kairos checkpoint bundle directory")
        manifest = load_manifest(root)
        bound_tokenizer = (root / manifest["tokenizer_path"]).resolve()
        if tokenizer_path is not None and Path(tokenizer_path).expanduser().resolve() != bound_tokenizer:
            raise ValueError("tokenizer override differs from the checkpoint-bound tokenizer")
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(KronosWithExogenous.from_pretrained(str(root)),
                   KronosTokenizer.from_pretrained(str(bound_tokenizer)), manifest, selected_device)

    @torch.inference_mode()
    def predict_batch(self, main_frames: list[pd.DataFrame],
                      exog_frames: list[pd.DataFrame | None]) -> np.ndarray:
        if not main_frames or len(main_frames) != len(exog_frames):
            raise ValueError("main and exogenous batches must be nonempty and have equal lengths")
        xs, stamps, extras = [], [], []
        for main, exog in zip(main_frames, exog_frames):
            main = validate_main_frame(main)
            if len(main) < self.max_context:
                raise ValueError(f"history needs at least {self.max_context} bars")
            if self.manifest.get("use_exog", True):
                exog = validate_exog_frame(main, exog, self.manifest["exog_cols"])
            else:
                exog = pd.DataFrame(0., index=main.index, columns=self.manifest["exog_cols"])
            main, exog = main.iloc[-self.max_context:], exog.iloc[-self.max_context:]
            if len(contiguous_starts(main.index, self.max_context, self.manifest["freq"])) != 1:
                raise ValueError("history contains a gap or differs from the checkpoint frequency")
            xs.append(normalize_window(main[self.manifest["feature_cols"]].to_numpy(),
                                       self.max_context, self.manifest["clip"]))
            dates = main.index
            stamps.append(np.column_stack([dates.minute, dates.hour, dates.dayofweek, dates.day, dates.month]))
            extras.append(exog.to_numpy(dtype=np.float32))
        x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=self.device)
        stamp = torch.as_tensor(np.stack(stamps), dtype=torch.float32, device=self.device)
        exog = torch.as_tensor(np.stack(extras), dtype=torch.float32, device=self.device)
        s1, s2 = self.tokenizer.encode(x, half=True)
        # No future bar is tokenized or dropped. The last hidden state belongs
        # to exactly the same anchor used to construct training targets.
        _, hidden = self.model.decode_s1(s1, s2, stamp=stamp,
                                        exog=exog if self.manifest.get("use_exog", True) else None)
        values = self.model.return_head(hidden[:, -1:])[:, 0].cpu().numpy()
        expected = (len(xs), self.manifest["return_horizon"], self.manifest["n_quantiles"])
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError("model returned invalid log-return quantiles")
        return values
