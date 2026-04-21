"""Push fine-tuned Kairos checkpoints to the Hugging Face Hub.

Two shapes are supported:

* **Tokenizer only** — pass ``--tokenizer-ckpt`` + ``--repo-tokenizer``;
  ``--predictor-ckpt`` can be omitted.
* **Predictor only** — pass ``--predictor-ckpt`` + ``--repo-predictor``;
  ``--tokenizer-ckpt`` can be omitted.
* **Both** — pass all four arguments.

Typical tokenizer-only invocation (from the Kairos-base-crypto run)::

    kairos-push-hf \
        --tokenizer-ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
        --repo-tokenizer Shadowell/Kairos-base-crypto \
        --market-tag crypto \
        --metrics-file artifacts/tokenizer_eval_summary.md
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from kairos.models import KronosWithExogenous
from kairos.vendor.kronos import Kronos, KronosTokenizer

CARD_TOKENIZER_TMPL = """---
license: mit
tags: [time-series, finance, kronos, kairos{market_tag}]
library_name: pytorch
---

# {repo}

Fine-tuned **Kronos-Tokenizer-base** ({market_human}), produced by
**[Kairos](https://github.com/Shadowell/Kairos)**.

The tokenizer encodes a rolling 6-dim OHLCV+amount window into two streams of
discrete tokens (BSQ — Binary Spherical Quantization) that Kronos predictors
consume downstream. Only the OHLCV inputs are used for training; the 32-dim
exogenous channel required by `KronosWithExogenous` is not needed for the
tokenizer itself.

{metrics_block}

## Usage

```python
from kairos import KronosTokenizer
tok = KronosTokenizer.from_pretrained("{repo}")
# Encode a [B, T, 6] OHLCV tensor into (s1_ids, s2_ids)
s1, s2 = tok.encode(x, half=True)
```

## Training recipe

See `docs/CRYPTO_TOKENIZER_RUN.md` in the upstream repo for the full
reproduction checklist (data collection, preparation, 15-epoch + patience 3
fine-tune, reconstruction evaluation).
"""

CARD_PREDICTOR_TMPL = """---
license: mit
tags: [time-series, finance, kronos, kairos{market_tag}]
library_name: pytorch
pipeline_tag: time-series-forecasting
---

# {repo}

{desc}

## Usage

```python
from kairos import KronosTokenizer, {cls}
tok = KronosTokenizer.from_pretrained("{tok_repo}")
model = {cls}.from_pretrained("{repo}")
```
"""


def _format_market_tag(raw: str) -> tuple[str, str]:
    """Return the YAML-tag fragment and a human-readable label."""
    raw = (raw or "").strip().lower()
    human = {"crypto": "crypto (BTC/USDT + ETH/USDT 1-min)",
             "ashare": "A-share (CSI300 daily)",
             "": "generic"}.get(raw, raw)
    return (f", {raw}" if raw else ""), human


def _push(local_dir: Path, repo_id: str, private: bool, token: str | None, card: str):
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
    (local_dir / "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        commit_message="Upload Kairos checkpoint",
    )
    print(f"[OK] pushed {repo_id}  →  https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-ckpt", default=None,
                    help="Path to a fine-tuned tokenizer best_model dir (omit to skip tokenizer push)")
    ap.add_argument("--predictor-ckpt", default=None,
                    help="Path to a fine-tuned predictor best_model dir (omit to skip predictor push)")
    ap.add_argument("--repo-tokenizer", default=None,
                    help="Target HF repo id for the tokenizer, e.g. Shadowell/Kairos-base-crypto")
    ap.add_argument("--repo-predictor", default=None,
                    help="Target HF repo id for the predictor, e.g. Shadowell/Kairos-small-crypto")
    ap.add_argument("--predictor-class", default="ext",
                    choices=["base", "ext"],
                    help="base = Kronos; ext = KronosWithExogenous")
    ap.add_argument("--market-tag", default="",
                    help="Extra tag appended to the card's `tags:` list (e.g. 'crypto', 'ashare')")
    ap.add_argument("--metrics-file", default=None,
                    help="Optional markdown snippet inlined into the tokenizer card (e.g. eval summary)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.tokenizer_ckpt and not args.predictor_ckpt:
        raise SystemExit("Must pass at least one of --tokenizer-ckpt / --predictor-ckpt")

    market_tag, market_human = _format_market_tag(args.market_tag)

    metrics_block = ""
    if args.metrics_file:
        mf = Path(args.metrics_file)
        if mf.exists():
            metrics_block = mf.read_text(encoding="utf-8").strip()
        else:
            print(f"[warn] --metrics-file {mf} not found, skipping")

    # ---- Tokenizer branch ---------------------------------------------------
    if args.tokenizer_ckpt:
        if not args.repo_tokenizer:
            raise SystemExit("--tokenizer-ckpt provided but --repo-tokenizer missing")
        tok_dir = Path(args.tokenizer_ckpt).resolve()
        if not tok_dir.exists():
            raise SystemExit(f"tokenizer checkpoint {tok_dir} not found")
        print(f"[check] load tokenizer  {tok_dir}")
        tok = KronosTokenizer.from_pretrained(str(tok_dir))
        print(f"        OK d_in={tok.d_in} d_model={tok.d_model} "
              f"s1={tok.s1_bits} s2={tok.s2_bits}")
        card = CARD_TOKENIZER_TMPL.format(
            repo=args.repo_tokenizer,
            market_tag=market_tag,
            market_human=market_human,
            metrics_block=metrics_block or "_No quantitative metrics were supplied at upload time._",
        )
        if args.dry_run:
            print("[dry-run] would push tokenizer → ", args.repo_tokenizer)
            print("[dry-run] card preview (first 20 lines):")
            for line in card.splitlines()[:20]:
                print("  " + line)
        else:
            _push(tok_dir, args.repo_tokenizer, args.private, args.token, card)

    # ---- Predictor branch ---------------------------------------------------
    if args.predictor_ckpt:
        if not args.repo_predictor:
            raise SystemExit("--predictor-ckpt provided but --repo-predictor missing")
        pred_dir = Path(args.predictor_ckpt).resolve()
        if not pred_dir.exists():
            raise SystemExit(f"predictor checkpoint {pred_dir} not found")
        print(f"[check] load predictor {pred_dir}")
        if args.predictor_class == "ext":
            model = KronosWithExogenous.from_pretrained(str(pred_dir))
            desc = "Fine-tuned Kronos with exogenous channel + quantile return head."
        else:
            model = Kronos.from_pretrained(str(pred_dir))
            desc = "Fine-tuned Kronos predictor."
        print(f"        OK n_layers={model.n_layers} d_model={model.d_model}")

        tok_repo = args.repo_tokenizer or "NeoQuasar/Kronos-Tokenizer-base"
        card = CARD_PREDICTOR_TMPL.format(
            repo=args.repo_predictor,
            tok_repo=tok_repo,
            market_tag=market_tag,
            desc=desc,
            cls="KronosWithExogenous" if args.predictor_class == "ext" else "Kronos",
        )
        if args.dry_run:
            print("[dry-run] would push predictor → ", args.repo_predictor)
        else:
            _push(pred_dir, args.repo_predictor, args.private, args.token, card)


if __name__ == "__main__":
    main()
