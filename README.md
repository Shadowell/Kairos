# Kairos

Kairos is a crypto fine-tuning and evaluation toolbox for the
[Kronos](https://github.com/shiyu-coder/Kronos) time-series foundation model. It
focuses on two instrument families: **spot** and **USDT-margined perpetual
swaps**.

## What It Does

- Collects OKX-compatible crypto OHLCV data for spot and perpetual swaps.
- Adds a fixed 32-dimensional exogenous channel: 24 common OHLCV-derived factors
  plus 8 crypto factors.
- Trains `KronosWithExogenous` predictor checkpoints and evaluates them with IC /
  Rank-IC / ICIR backtests.
- Pushes trained checkpoints to Hugging Face and serves prediction from caller
  supplied OHLCV bars through FastAPI.

## Current Status

The public predictor runs below are the main usable references. Report deltas
against the baseline model, not absolute IC alone.

| Run | Universe | h30 Rank-IC | h30 ICIR | Notes |
| --- | --- | ---: | ---: | --- |
| BTC/ETH 2y spot | 2 symbols | `+0.050` | `+0.325` | First usable crypto signal |
| Top100 1y spot | 100 symbols | `+0.030` | `+0.454` | Better stability from broader universe |
| BTC/ETH 2y spot + `Kronos-base` | 2 symbols | `+0.076` | `+0.484` | Current best public result |

OKX perpetual-swap multichannel work is connected but still experimental. The
first Top10 30-day run is documented as a post-mortem, not a production result.

## Public Models

| Repo | Base model | Data | Purpose |
| --- | --- | --- | --- |
| [`Shadowell/Kairos-small-crypto`](https://huggingface.co/Shadowell/Kairos-small-crypto) | [`NeoQuasar/Kronos-small`](https://huggingface.co/NeoQuasar/Kronos-small) | BTC/USDT + ETH/USDT, 1-min | Public small predictor checkpoint |
| [`Shadowell/Kairos-base-crypto`](https://huggingface.co/Shadowell/Kairos-base-crypto) | [`NeoQuasar/Kronos-base`](https://huggingface.co/NeoQuasar/Kronos-base) | BTC/USDT + ETH/USDT, 1-min | Public base predictor checkpoint |

```python
from kairos.models import KronosWithExogenous

model = KronosWithExogenous.from_pretrained("Shadowell/Kairos-base-crypto")
```

## Feature Schema

The exogenous vector is fixed at 32 dimensions:

- Common OHLCV block, 24 dims: returns, volatility, volume/amount, range, VWAP,
  and padding factors from `kairos.data.common_features`.
- Crypto block, 8 dims: `market_ret_1`, `market_vol_20`, `hour_sin`,
  `hour_cos`, `funding_rate`, `funding_rate_z`, `oi_change`, `basis`.

Spot datasets use the four universal crypto factors and keep swap-only columns
at zero. Swap datasets can additionally use funding, OI, and spot-vs-swap basis
sidecars from OKX.

## Quick Start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,serve]'
```

### Collect OKX Spot

```bash
kairos-collect --market-type spot \
  --universe "BTC/USDT,ETH/USDT" \
  --freq 1min --start 2026-04-01 --end 2026-04-30 \
  --out ./raw/crypto/okx_spot_btc_eth_1min --workers 1
```

### Collect OKX Perpetual Swaps

```bash
kairos-collect --market-type swap \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" \
  --freq 1min --start 2026-04-01 --end 2026-04-30 \
  --out ./raw/crypto/okx_swap_btc_eth_1min --workers 1 \
  --crypto-extras funding,open_interest,spot,reference
```

### Package A Dataset

```bash
kairos-prepare --market crypto --market-type swap \
  --raw ./raw/crypto/okx_swap_btc_eth_1min \
  --train 2026-04-01:2026-04-20 \
  --val 2026-04-21:2026-04-25 \
  --test 2026-04-26:2026-04-30 \
  --out ./finetune/data/crypto_swap_btc_eth_1min
```

### Train And Backtest

```bash
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=./finetune/data/crypto_swap_btc_eth_1min
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
  --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_baseline.json

python -m kairos.training.backtest_ic --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
  --preset crypto-1min --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_finetuned.json
```

## Serving

`kairos-serve` does not fetch exchange data. The `/predict` endpoint accepts a
JSON body with `symbol`, `market_type`, `freq`, and a `bars` array containing
`datetime`, `open`, `high`, `low`, `close`, `volume`, and optional `amount`.

## Repository Shape

```text
kairos/
  data/       # OKX-compatible collection, extras, feature packaging
  models/     # KronosWithExogenous and quantile return head
  training/   # predictor/tokenizer training and IC backtests
  deploy/     # Hugging Face push and FastAPI serving
  vendor/     # Kronos source snapshot
docs/         # guides, run logs, plans, and interpretation notes
examples/     # quickstart scripts and universe snapshots
tests/        # pytest smoke and feature tests
```

## Where To Read Next

| Goal | Document |
| --- | --- |
| Navigate the docs | [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) |
| Understand OKX spot/perp exogenous factors | [docs/CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md](docs/CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md) |
| Interpret IC backtests | [docs/BACKTEST_IC_INTERPRETATION_GUIDE.md](docs/BACKTEST_IC_INTERPRETATION_GUIDE.md) |
| Run remote training | [docs/AUTODL_REMOTE_TRAINING_GUIDE.md](docs/AUTODL_REMOTE_TRAINING_GUIDE.md) |
| Study the best successful run | [docs/CRYPTO_TOP100_1Y_SPOT_RUN.md](docs/CRYPTO_TOP100_1Y_SPOT_RUN.md) |

## License

MIT.
