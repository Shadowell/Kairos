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

{tokenizer_intro}

{tokenizer_notes}

{metrics_block}

## Usage

```python
from kairos import KronosTokenizer
tok = KronosTokenizer.from_pretrained("{repo}")
# Encode a [B, T, 6] OHLCV tensor into (s1_ids, s2_ids)
s1, s2 = tok.encode(x, half=True)
```

{training_block}

{recipe_block}
"""

CARD_PREDICTOR_TMPL = """---
license: mit
tags: [time-series, finance, kronos, kairos{market_tag}]
library_name: pytorch
pipeline_tag: time-series-forecasting
---

# {repo}

{predictor_intro}

{predictor_notes}

{metrics_block}

## Usage

```python
from kairos import KronosTokenizer, {cls}
tok = KronosTokenizer.from_pretrained("{tok_repo}")
model = {cls}.from_pretrained("{repo}")
```

{training_block}

{recipe_block}
"""


def _format_market_tag(raw: str) -> tuple[str, str]:
    """Return the YAML-tag fragment and a human-readable label."""
    raw = (raw or "").strip().lower()
    human = {"crypto": "crypto (BTC/USDT + ETH/USDT 1-min)",
             "ashare": "A-share (CSI300 daily)",
             "": "generic"}.get(raw, raw)
    return (f", {raw}" if raw else ""), human


def _tokenizer_card_parts(raw: str) -> dict[str, str]:
    raw = (raw or "").strip().lower()
    if raw == "crypto":
        return {
            "tokenizer_intro": (
                "Fine-tuned **Kronos-Tokenizer-base** on BTC/USDT + ETH/USDT 1-min "
                "K-lines (2024-01 ~ 2026-04) using "
                "**[Kairos](https://github.com/Shadowell/Kairos)**."
            ),
            "tokenizer_notes": (
                "This tokenizer run reuses the same BTC/ETH spot corpus as "
                "[`Shadowell/Kairos-small-crypto`](https://huggingface.co/Shadowell/Kairos-small-crypto), "
                "but fine-tunes the BSQ tokenizer itself instead of only adapting "
                "the downstream predictor. The model encodes a rolling 6-dim "
                "OHLCV+amount window into two streams of discrete tokens that "
                "Kronos predictors consume downstream. Only the OHLCV inputs are "
                "used for tokenizer training; the 32-dim exogenous channel "
                "required by `KronosWithExogenous` is not needed here."
            ),
            "training_block": """## Training config (preset `crypto-1min`)

- lookback 256 min
- batch 50, OneCycleLR, early-stop patience 3
- full-model fine-tune on `NeoQuasar/Kronos-Tokenizer-base`
- training stopped at epoch 7; best checkpoint = epoch 4
- total wall time: ~18 min 35 s on a single RTX 5090""",
            "recipe_block": (
                "## Training recipe\n\n"
                "Full command log, evaluation commands, pitfalls and the "
                "reproduction checklist are in "
                "[`docs/CRYPTO_TOKENIZER_RUN.md`](https://github.com/Shadowell/Kairos/blob/main/docs/CRYPTO_TOKENIZER_RUN.md)."
            ),
        }

    return {
        "tokenizer_intro": (
            f"Fine-tuned **Kronos-Tokenizer-base** ({_format_market_tag(raw)[1]}), "
            "produced by **[Kairos](https://github.com/Shadowell/Kairos)**."
        ),
        "tokenizer_notes": (
            "The tokenizer encodes a rolling 6-dim OHLCV+amount window into two "
            "streams of discrete tokens (BSQ — Binary Spherical Quantization) "
            "that Kronos predictors consume downstream. Only the OHLCV inputs "
            "are used for training; the 32-dim exogenous channel required by "
            "`KronosWithExogenous` is not needed for the tokenizer itself."
        ),
        "training_block": "## Training config\n\nSee the upstream Kairos repo for the market-specific preset used for this run.",
        "recipe_block": (
            "## Training recipe\n\n"
            "See the corresponding training doc in the upstream "
            "[Kairos](https://github.com/Shadowell/Kairos) repo for the full "
            "reproduction checklist."
        ),
    }


def _predictor_card_parts(raw: str, tok_repo: str) -> dict[str, str]:
    raw = (raw or "").strip().lower()
    if raw == "crypto":
        tokenizer_note = (
            "This run keeps the original tokenizer "
            "[`NeoQuasar/Kronos-Tokenizer-base`](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base), "
            "matching the original `Kairos-small-crypto` training flow."
            if tok_repo == "NeoQuasar/Kronos-Tokenizer-base"
            else
            "This version uses the fine-tuned tokenizer "
            f"[`{tok_repo}`](https://huggingface.co/{tok_repo}) instead of the "
            "original `NeoQuasar/Kronos-Tokenizer-base`."
        )
        return {
            "predictor_intro": (
                "Fine-tuned **Kronos-small** on BTC/USDT + ETH/USDT 1-min K-lines "
                "(2024-01 ~ 2026-04) using "
                "**[Kairos](https://github.com/Shadowell/Kairos)**.\n"
                "Architecture = Kronos + exogenous bypass channel (32-d) + quantile return head."
            ),
            "predictor_notes": (
                f"{tokenizer_note} Training data comes "
                "from the public Binance Vision spot mirror, so the 5 crypto-native "
                "exogenous features (`funding_rate` / `funding_rate_z` / "
                "`oi_change` / `basis` / `btc_dominance`) remain padded to zero; "
                "the other 27 dimensions are real."
            ),
            "training_block": f"""## Training config (preset `crypto-1min`)

- lookback 256 min, predict 30 min
- batch 50, OneCycleLR, early-stop patience 3
- progressive unfreeze: only last transformer block + exog bypass + return head
- tokenizer source = `{tok_repo}`
- 32-d EXOG = 24 common + 8 crypto-market features""",
            "recipe_block": (
                "## Training recipe\n\n"
                "Full command log, backtest commands, pitfalls and the reproduction "
                "checklist are in "
                "[`docs/CRYPTO_BTC_ETH_RUN.md`](https://github.com/Shadowell/Kairos/blob/main/docs/CRYPTO_BTC_ETH_RUN.md)."
            ),
        }

    return {
        "predictor_intro": "Fine-tuned Kronos predictor.",
        "predictor_notes": "",
        "training_block": "## Training config\n\nSee the upstream Kairos repo for the market-specific preset used for this run.",
        "recipe_block": (
            "## Training recipe\n\n"
            "See the corresponding training doc in the upstream "
            "[Kairos](https://github.com/Shadowell/Kairos) repo for the full "
            "reproduction checklist."
        ),
    }


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
        tok_card_parts = _tokenizer_card_parts(args.market_tag)
        card = CARD_TOKENIZER_TMPL.format(
            repo=args.repo_tokenizer,
            market_tag=market_tag,
            metrics_block=metrics_block or "_No quantitative metrics were supplied at upload time._",
            tokenizer_intro=tok_card_parts["tokenizer_intro"],
            tokenizer_notes=tok_card_parts["tokenizer_notes"],
            training_block=tok_card_parts["training_block"],
            recipe_block=tok_card_parts["recipe_block"],
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
        pred_card_parts = _predictor_card_parts(args.market_tag, tok_repo)
        card = CARD_PREDICTOR_TMPL.format(
            repo=args.repo_predictor,
            tok_repo=tok_repo,
            market_tag=market_tag,
            desc=desc,
            cls="KronosWithExogenous" if args.predictor_class == "ext" else "Kronos",
            predictor_intro=pred_card_parts["predictor_intro"],
            predictor_notes=pred_card_parts["predictor_notes"],
            metrics_block=metrics_block or "_No quantitative metrics were supplied at upload time._",
            training_block=pred_card_parts["training_block"],
            recipe_block=pred_card_parts["recipe_block"],
        )
        if args.dry_run:
            print("[dry-run] would push predictor → ", args.repo_predictor)
        else:
            _push(pred_dir, args.repo_predictor, args.private, args.token, card)


if __name__ == "__main__":
    main()
