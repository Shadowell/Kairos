# Kairos

> Seize **the right moment**.
>
> **Kairos** is a multi-market fine-tuning and deployment toolbox built on top of the [Kronos](https://github.com/shiyu-coder/Kronos) foundation model.
> It focuses on practical workflows for **A-shares** and **crypto**: data collection, feature engineering, predictor fine-tuning, IC backtesting, and Hugging Face deployment.

## What It Does

- Fine-tunes Kronos predictors with a fixed 32-dimensional exogenous feature schema.
- Supports both **A-shares** and **crypto** through a shared `MarketAdapter` interface.
- Runs predictor training, tokenizer evaluation, and IC / Rank-IC / ICIR backtests.
- Packages datasets with `meta.json` so training and backtests can recover market context automatically.
- Pushes trained checkpoints to Hugging Face and serves them through FastAPI.

## Current Status

### What is working

- **Crypto 1-minute, h30** is the only direction that has shown consistent alpha so far.
- **Top100 spot** improved signal stability meaningfully over the original BTC/ETH-only run.
- **Kronos-base** outperformed **Kronos-small** on the same BTC/ETH dataset.
- The **OKX perp multichannel path** is connected, but the first real run is a failed experiment and should be treated as a post-mortem, not a result.

### Headline results

| Run | Universe | Best h30 rank-IC | Best h30 ICIR | Notes |
|---|---|---:|---:|---|
| BTC/ETH 2y spot | 2 symbols | `+0.050` | `+0.325` | First usable crypto signal |
| Top100 1y spot | 100 symbols | `+0.030` | `+0.454` | Lower raw IC, much better stability |
| BTC/ETH 2y spot + `Kronos-base` | 2 symbols | `+0.076` | `+0.484` | Current best public result |

### What is not working yet

- **A-shares daily** remains negative after multiple training variants.
- **h1 / h5** are still weak or unstable.
- **Perpetual-data experiments** need a cleaner rerun with stricter controls.

## Public Models

| Repo | Base model | Data | Purpose |
|---|---|---|---|
| [`Shadowell/Kairos-small-crypto`](https://huggingface.co/Shadowell/Kairos-small-crypto) | [`NeoQuasar/Kronos-small`](https://huggingface.co/NeoQuasar/Kronos-small) | BTC/USDT + ETH/USDT, 1-min | Best public small predictor checkpoint |
| [`Shadowell/Kairos-base-crypto`](https://huggingface.co/Shadowell/Kairos-base-crypto) | [`NeoQuasar/Kronos-base`](https://huggingface.co/NeoQuasar/Kronos-base) | BTC/USDT + ETH/USDT, 1-min | Best public base predictor checkpoint |

Minimal loading example:

```python
from kairos import KronosTokenizer
from kairos.models import KronosWithExogenous

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = KronosWithExogenous.from_pretrained("Shadowell/Kairos-small-crypto")
```

## Core Design

Kairos keeps the Kronos tokenizer input unchanged at **6 price-volume dimensions**:

- `open, high, low, close, volume, amount`

It then adds a fixed **32-dimensional exogenous channel**:

- `24` common factors shared across markets
- `8` market-specific factors contributed by each adapter

The model side is intentionally conservative:

- Reuse the original Kronos tokenizer and most predictor weights.
- Add an exogenous bypass encoder instead of changing tokenizer dimensionality.
- Add a quantile return head for downstream horizon-specific supervision.
- Keep `n_exog=32` fixed across markets so checkpoints stay portable.

## Quick Start

### Install

```bash
git clone https://github.com/Shadowell/Kairos.git
cd Kairos
pip install -e '.[serve,train]'
```

### Collect data

```bash
# A-shares
kairos-collect --universe csi300 --freq daily \
    --start 2018-01-01 --adjust qfq --out ./raw/daily

# Crypto
pip install -e '.[crypto]'
kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2024-01-01 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_btc_eth --workers 1
```

### Package a dataset

```bash
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_btc_eth \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_btc_eth
```

### Train a predictor

```bash
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=./finetune/data/crypto_1min_btc_eth
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
```

### Run a backtest

```bash
python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --out artifacts/backtest_baseline.json

python -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --out artifacts/backtest_finetuned.json
```

For real training, use the remote GPU workflow in [docs/AUTODL_REMOTE_TRAINING_GUIDE.md](docs/AUTODL_REMOTE_TRAINING_GUIDE.md).

## Repository Shape

```text
Kairos/
├── kairos/
│   ├── data/        # collection, features, dataset packaging
│   ├── models/      # KronosWithExogenous
│   ├── training/    # train_predictor, train_tokenizer, backtest_ic
│   ├── deploy/      # Hugging Face push and FastAPI serving
│   └── vendor/      # vendored Kronos source
├── docs/            # guides, run logs, roadmap, post-mortems
├── examples/        # small usage references
├── scripts/         # AutoDL and packaging helpers
├── tests/
├── README.md
└── AGENTS.md
```

## Where To Read Next

| If you want to... | Read this first |
|---|---|
| Find the full doc map | [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) |
| Understand terminology and metrics | [docs/CONCEPTS_AND_GLOSSARY.md](docs/CONCEPTS_AND_GLOSSARY.md) |
| Run training on AutoDL | [docs/AUTODL_REMOTE_TRAINING_GUIDE.md](docs/AUTODL_REMOTE_TRAINING_GUIDE.md) |
| Work on crypto data pipelines | [docs/CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](docs/CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) |
| Call the `kairos-serve` HTTP API | [docs/SERVE_HTTP_API.md](docs/SERVE_HTTP_API.md) |
| Study the best successful crypto run | [docs/CRYPTO_TOP100_1Y_SPOT_RUN.md](docs/CRYPTO_TOP100_1Y_SPOT_RUN.md) |
| Understand the current roadmap | [docs/PROJECT_ROADMAP_AND_NEXT_STEPS.md](docs/PROJECT_ROADMAP_AND_NEXT_STEPS.md) |
| Follow repository agent rules | [AGENTS.md](AGENTS.md) |

## License

MIT. See [LICENSE](LICENSE).

This repository also vendors the original Kronos model code under [`kairos/vendor/kronos/`](kairos/vendor/kronos/), which is also MIT-licensed.
