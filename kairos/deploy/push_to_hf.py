"""Push fine-tuned Kairos checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from kairos.models import KronosWithExogenous
from kairos.vendor.kronos import Kronos, KronosTokenizer

CARD_TOKENIZER_TMPL = """---
license: mit
tags: [time-series, finance, kronos, kairos, a-share]
library_name: pytorch
---

# {repo}

A-share fine-tuned Kronos tokenizer, produced by **[Kairos](https://github.com/your-user/kairos)**.

## Usage

```python
from kairos import KronosTokenizer
tok = KronosTokenizer.from_pretrained("{repo}")
```
"""

CARD_PREDICTOR_TMPL = """---
license: mit
tags: [time-series, finance, kronos, kairos, a-share]
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
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--predictor-ckpt", required=True)
    ap.add_argument("--repo-tokenizer", required=True)
    ap.add_argument("--repo-predictor", required=True)
    ap.add_argument("--predictor-class", default="ext",
                    choices=["base", "ext"],
                    help="base = Kronos; ext = KronosWithExogenous")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok_dir = Path(args.tokenizer_ckpt).resolve()
    pred_dir = Path(args.predictor_ckpt).resolve()
    if not tok_dir.exists() or not pred_dir.exists():
        raise SystemExit("checkpoint 路径不存在")

    print(f"[check] load tokenizer  {tok_dir}")
    tok = KronosTokenizer.from_pretrained(str(tok_dir))
    print(f"        OK d_in={tok.d_in} d_model={tok.d_model} "
          f"s1={tok.s1_bits} s2={tok.s2_bits}")

    print(f"[check] load predictor {pred_dir}")
    if args.predictor_class == "ext":
        model = KronosWithExogenous.from_pretrained(str(pred_dir))
        desc = "Fine-tuned Kronos with exogenous channel + quantile return head."
    else:
        model = Kronos.from_pretrained(str(pred_dir))
        desc = "Fine-tuned Kronos predictor for A-share."
    print(f"        OK n_layers={model.n_layers} d_model={model.d_model}")

    if args.dry_run:
        print("[dry-run] skipped upload"); return

    _push(tok_dir, args.repo_tokenizer, args.private, args.token,
          CARD_TOKENIZER_TMPL.format(repo=args.repo_tokenizer))
    _push(pred_dir, args.repo_predictor, args.private, args.token,
          CARD_PREDICTOR_TMPL.format(
              repo=args.repo_predictor, tok_repo=args.repo_tokenizer,
              desc=desc,
              cls="KronosWithExogenous" if args.predictor_class == "ext" else "Kronos"))


if __name__ == "__main__":
    main()
