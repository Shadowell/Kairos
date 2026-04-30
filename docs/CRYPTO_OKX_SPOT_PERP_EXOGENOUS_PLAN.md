# OKX Spot And Perpetual Exogenous Factor Training Plan

Last updated: 2026-04-30.

## 1. Scope

Kairos is now scoped to crypto only:

- **Spot**: OKX `SPOT` instruments such as `BTC-USDT`, represented in ccxt form as `BTC/USDT`.
- **Perpetual swaps**: OKX linear USDT-margined `SWAP` instruments such as `BTC-USDT-SWAP`, represented in ccxt form as `BTC/USDT:USDT`.
- Out of scope: equities, options, inverse/coin-margined contracts, execution/private-account data, and non-OKX exchange-specific signals.

The training target remains predictor fine-tuning on Kronos, evaluated primarily with h30 IC / Rank-IC / ICIR for 1-minute bars.

## 2. Connectivity Check

Local macOS connectivity to `www.okx.com` timed out. The fallback server
`root@47.79.36.92` could reach OKX public endpoints when requests used an
HTTP/1.1 browser-like `User-Agent`. Plain Python urllib without those headers
received HTTP 403.

The following public endpoints were verified from the fallback server:

| Endpoint | Spot | Perp | Available fields |
| --- | --- | --- | --- |
| `/api/v5/public/instruments` | Yes | Yes | instrument metadata, tick size, lot size, contract metadata for swaps |
| `/api/v5/market/ticker` | Yes | Yes | last, bid/ask price and size, 24h high/low/open, 24h volume |
| `/api/v5/market/history-candles` | Yes | Yes | timestamp, open, high, low, close, volume, quote volume, confirm flag |
| `/api/v5/market/books` | Yes | Yes | top-of-book and depth snapshots |
| `/api/v5/public/funding-rate` | No | Yes | current/next funding, premium, impact value |
| `/api/v5/public/funding-rate-history` | No | Yes | historical funding settlement rates |
| `/api/v5/public/open-interest` | No | Yes | current open interest, OI currency, OI USD |
| `/api/v5/market/mark-price-candles` | No | Yes | mark-price OHLC |
| `/api/v5/market/index-candles` | Reference | Yes | index-price OHLC |
| `/api/v5/public/price-limit` | No | Yes | contract buy/sell price limits |
| `/api/v5/rubik/stat/contracts/open-interest-volume` | No | Yes | aggregate OI and volume history by currency |
| `/api/v5/rubik/stat/contracts/long-short-account-ratio` | No | Yes | aggregate account long/short ratio by currency |

Implementation consequence: the OKX ccxt adapter now sets browser-like headers.
Training data collection should still run on a network path that can reach OKX.

## 3. Factor Availability Matrix

| Factor family | Feature | Spot | Perp | Source | Core training use |
| --- | --- | --- | --- | --- | --- |
| Common OHLCV | returns, volatility, volume, amount, range, VWAP | Yes | Yes | history candles | Yes |
| Universal crypto | `market_ret_1` | Yes | Yes | BTC/USDT reference close, fallback own close | Yes |
| Universal crypto | `market_vol_20` | Yes | Yes | BTC/USDT reference close, fallback own close | Yes |
| Universal crypto | `hour_sin`, `hour_cos` | Yes | Yes | timestamp | Yes |
| Spot-specific | spot bid/ask spread | Yes | No | ticker/books snapshot | Not in core offline training yet |
| Spot-specific | spot depth imbalance | Yes | No | books snapshot | Not in core offline training yet |
| Perp-specific | `funding_rate` | No | Yes | funding-rate-history | Yes |
| Perp-specific | `funding_rate_z` | No | Yes | funding-rate-history | Yes |
| Perp-specific | `oi_change` | No | Yes | open-interest / Rubik OI | Yes when historical OI exists |
| Perp-specific | `basis` | No | Yes | swap close vs spot close, or mark/index | Yes |
| Perp-specific | premium / mark-index spread | No | Yes | funding-rate current, mark/index candles | Candidate v2 |
| Perp-specific | long/short account ratio | No | Yes | Rubik stats | Candidate v2 |
| Perp-specific | price-limit distance | No | Yes | price-limit endpoint | Candidate v2 |

## 4. Selected 32-Dim Training Schema

Kairos keeps a fixed exogenous shape to avoid changing model architecture:

- 24 common factors from `kairos.data.common_features`.
- 8 crypto factors from `CryptoAdapter.MARKET_EXOG_COLS`.

Current crypto factor block:

| Slot | Feature | Type | Spot fill policy | Perp fill policy |
| ---: | --- | --- | --- | --- |
| 1 | `market_ret_1` | Universal | BTC/USDT reference close or own close | BTC/USDT reference close or own close |
| 2 | `market_vol_20` | Universal | BTC/USDT reference close or own close | BTC/USDT reference close or own close |
| 3 | `hour_sin` | Universal | computed from timestamp | computed from timestamp |
| 4 | `hour_cos` | Universal | computed from timestamp | computed from timestamp |
| 5 | `funding_rate` | Perp-specific | `0.0` | forward-filled funding history |
| 6 | `funding_rate_z` | Perp-specific | `0.0` | rolling z-score of funding |
| 7 | `oi_change` | Perp-specific | `0.0` | log-change of OI snapshots |
| 8 | `basis` | Perp-specific | `0.0` | swap close / spot close - 1 |

Why this schema:

- It is portable across spot and swaps.
- It uses only information available at or before bar time `t`.
- It avoids order-book factors in offline training until we have a reliable
  historical snapshot collector.
- It avoids spot-only features in the shared 32-dim schema because they would
  reduce comparability between spot and swap models.

## 5. Data Collection Plan

### Spot dataset

Command shape:

```bash
kairos-collect --market-type spot \
  --universe "BTC/USDT,ETH/USDT" \
  --freq 1min --start <START> --end <END> \
  --out ./raw/crypto/okx_spot_<run_name> --workers 1 \
  --crypto-extras reference
```

Use this for:

- BTC/ETH controlled spot runs.
- TopN spot universe runs.
- Baseline comparison against older Binance Vision spot experiments.

### Perpetual-swap dataset

Command shape:

```bash
kairos-collect --market-type swap \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" \
  --freq 1min --start <START> --end <END> \
  --out ./raw/crypto/okx_swap_<run_name> --workers 1 \
  --crypto-extras funding,open_interest,spot,reference
```

Use this for:

- Swap-only predictor runs.
- Spot-vs-swap basis experiments.
- Ablation runs that compare OHLCV-only vs multichannel sidecars.

Known OKX retention constraints:

- Funding history is useful but not infinite; older windows may return empty.
- Public OI endpoints are more constrained than candles. For long historical
  training windows, expect partial OI coverage unless we accumulate snapshots or
  use a paid historical data source.
- Order-book factors are online/snapshot data, not reliable historical factors
  unless we explicitly collect and store depth snapshots.

## 6. Packaging And Training Plan

Package spot and swap separately. Do not mix them in the same first official
run; keeping them separate makes attribution clearer.

```bash
kairos-prepare --market crypto --market-type spot \
  --raw ./raw/crypto/okx_spot_<run_name> \
  --train <TRAIN_RANGE> --val <VAL_RANGE> --test <TEST_RANGE> \
  --split-mode interleave --val-ratio 0.15 --block-days 20 \
  --out ./finetune/data/crypto_spot_<run_name>

kairos-prepare --market crypto --market-type swap \
  --raw ./raw/crypto/okx_swap_<run_name> \
  --train <TRAIN_RANGE> --val <VAL_RANGE> --test <TEST_RANGE> \
  --split-mode interleave --val-ratio 0.15 --block-days 20 \
  --out ./finetune/data/crypto_swap_<run_name>
```

Training command:

```bash
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=./finetune/data/<dataset_name>
unset KAIROS_N_TRAIN_ITER
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
```

Mandatory evaluation:

```bash
python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
  --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_baseline_<run_name>.json

python -m kairos.training.backtest_ic \
  --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
  --preset crypto-1min --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_finetuned_<run_name>.json
```

## 7. Experiment Sequence

1. **Spot BTC/ETH smoke**: verify OKX spot collection, packaging, local smoke
   training, and h30 backtest plumbing.
2. **Swap BTC/ETH multichannel smoke**: verify funding, OI, spot basis, and
   reference sidecars are non-empty where OKX provides data.
3. **Spot TopN official run**: reproduce the broader-universe signal with OKX
   as the source.
4. **Swap TopN official run**: train with funding/OI/basis sidecars and compare
   against OHLCV-only baseline.
5. **Ablation**: run swap with sidecars disabled, then with funding only, then
   funding+OI+basis to isolate which exogenous family contributes signal.

## 8. Acceptance Criteria

A run is reportable only if all items are satisfied:

- `meta.json` records `market=crypto`, `market_type`, `exog_cols`, ranges, and
  discovered extras channels.
- Feature packaging confirms the expected 32 columns.
- Sidecar coverage is reported: funding rows, OI rows, spot basis rows,
  reference rows.
- Baseline and finetuned h30 IC reports are both produced.
- The README/model card reports delta vs baseline, not just absolute IC.
- For test windows shorter than 15 days, use `--aggregation none` and mark the
  result as smoke/noise, not as an official ICIR.

## 9. Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| OKX public access blocked locally | Collection fails | Run collection on a server with OKX access; keep browser-like headers in adapter |
| OI history is sparse or short-retention | `oi_change` mostly zero | Report coverage and run ablation without OI |
| Funding history missing for older ranges | funding columns partly zero | Choose recent windows or document coverage |
| Order-book factors are snapshot-only | backfilled training leakage/inconsistency risk | Exclude from core schema until a depth snapshot collector exists |
| Spot and swap mixed too early | Attribution becomes unclear | Train spot and swap separately first |
