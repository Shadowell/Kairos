# Crypto data source and exchange access guide

> This document explains how Kairos' crypto data layer works: which exchange is connected by default, how to collect data in different network environments, and how to extend to new exchange backends.

Kairos supports crypto markets out of the box through a pluggable exchange
backend. By default it connects to **OKX USDT-margined perpetual swaps**; the
abstraction (`kairos.data.markets.crypto_exchanges.CryptoExchange`) is
deliberately thin so that Binance, Bybit, Hyperliquid and friends can be
added by dropping in one file.

This guide is a companion to [`AUTODL_REMOTE_TRAINING_GUIDE.md`](AUTODL_REMOTE_TRAINING_GUIDE.md) and
[`CONCEPTS_AND_GLOSSARY.md`](CONCEPTS_AND_GLOSSARY.md); keep them handy.

---

## 1. Why crypto?

The full rationale lives in the project write-ups, but the short version:

- **Free high-frequency data.** OKX public candles require no credentials.
- **Enormous sample density.** `BTC/USDT:USDT` 1min ≈ 525k bars per year.
- **True T+0 and 24/7 trading.** No trading halts, no T+1 lock-up, no
  lunch break, no circuit breakers — the same data shape the Kronos model
  was pre-trained on.
- **Strong price-action alpha.** Funding rate, open interest, and tick
  direction are widely documented signals that a transformer can pick up.

A-shares remain a first-class citizen (`--market ashare`); crypto is just
another adapter on the same rails.

---

## 2. Install the crypto extras

```bash
# From the repo root, inside your venv:
pip install -e '.[crypto]'
```

`kairos-kronos[crypto]` pulls in:

| Package | Purpose |
|---|---|
| `ccxt>=4.3` | Cross-venue exchange client used by OKX / Binance / ... |
| `python-dotenv>=1.0` | Optional, loads `.env` secrets for private endpoints |

No extras are needed if you only use A-share data.

---

## 3. Authentication (optional)

Public market-data endpoints (candles, funding, open interest) work without
any credentials. You only need an API key if you later want to:

- Pull private account data (positions, balance history, fills).
- Place orders from `kairos.deploy.serve`.

Keys are loaded from environment variables. Copy the template:

```bash
cp .env.example .env
$EDITOR .env            # fill in OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE
```

Kairos itself never reads `.env` automatically — point your shell at it:

```bash
set -a; source .env; set +a
```

Or in a Python entrypoint:

```python
from dotenv import load_dotenv
load_dotenv()
```

**Security rules** (also enforced in `.gitignore`):

- Never commit `.env`, `secrets/`, `*.pem`, `*.key`.
- Rotate API keys if you paste them into a shell history.
- OKX keys should be scoped **read-only** until you actually wire up trading.

---

## 4. Collecting OHLCV

### 4.1 Basic invocation

```bash
kairos-collect \
  --market crypto \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" \
  --freq 1min \
  --start 2023-01-01 \
  --end   2025-01-01 \
  --out   ./raw/crypto/1min \
  --workers 1
```

Notes:

- ccxt symbols are used natively: `BTC/USDT:USDT` means "BTC / USDT perpetual
  margined in USDT". For spot replace with `BTC/USDT` (and set `--exchange`
  options accordingly — spot is not the default market type).
- The output file for `BTC/USDT:USDT` is written to
  `./raw/crypto/1min/BTC_USDT-USDT.parquet` (`/` → `_`, `:` → `-`).
- Set `--workers 1` until you've verified your proxy is stable; ccxt
  throttles per-exchange-instance, so parallelism across symbols is
  generally fine but can still trip rate limits on congested networks.

### 4.2 Named universes

```bash
# Top 10 USDT-margined perps by 24h quote volume (resolved live)
kairos-collect --market crypto --universe top10 --freq 1h --out ./raw/crypto/1h
```

- `topN` ranks all **active, linear, USDT-quoted swap** markets by the
  venue's reported 24h quote volume, so the constituent list is the "live"
  top-N at the time you run the command. If you want a frozen universe for
  reproducible research, pass the symbols explicitly.

### 4.3 Proxies

China-based runners commonly need a proxy to reach OKX. Two options, in
priority order:

1. **Explicit CLI flag:** `--proxy http://127.0.0.1:7890`
2. **Environment variables:** `HTTPS_PROXY` / `HTTP_PROXY` (standard
   curl-style URLs). Kairos picks them up automatically.

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
kairos-collect --market crypto --universe top10 --freq 1min --start 2024-01-01 --out ./raw/crypto/1min
```

If OKX returns `403 Forbidden` repeatedly: verify the proxy, then try
`--exchange` with a different venue (once more venues are wired up).

### 4.4 Frequencies

The OKX backend supports:

| Kairos freq | OKX timeframe | ~bars / year / symbol |
|---|---|---|
| `1min` | `1m` | ~525,600 |
| `3min` | `3m` | ~175,200 |
| `5min` | `5m` | ~105,120 |
| `15min` | `15m` | ~35,040 |
| `30min` | `30m` | ~17,520 |
| `60min` / `1h` | `1h` | ~8,760 |
| `2h` / `4h` | `2h` / `4h` | ~4,380 / ~2,190 |
| `1d` / `daily` | `1d` | ~365 |

Pagination is handled internally (OKX caps to 300 bars per request).

---

## 4.5 Office-network fallback: `binance_vision`

Many China-based corporate networks block `api.binance.com`,
`fapi.binance.com`, **and** `okx.com`, but leave the Binance public data
mirror `data-api.binance.vision` reachable. Kairos ships a thin backend
that talks to that mirror so you can validate the entire
collect → prepare → train pipeline without leaving the office.

```bash
kairos-collect --market crypto --exchange binance_vision \
  --universe "BTC/USDT,ETH/USDT" --freq 1min \
  --start 2024-01-01 --end 2024-02-01 \
  --out ./raw/crypto/binance_spot_1min --workers 1
```

Key differences vs the OKX default:

| Capability | `okx` (default) | `binance_vision` |
|---|---|---|
| Perpetual OHLCV | ✅ | ❌ (spot mirror only) |
| Funding rate history | ✅ | ❌ (`NotImplementedError`) |
| Open-interest history | ✅ | ❌ |
| Basis (perp vs spot) | ✅ | ❌ |
| Symbol format | `BTC/USDT:USDT` | `BTC/USDT` (perp form auto-degraded with warning) |
| Credentials | optional | never accepted |
| Blocked in Chinese office networks | often yes | usually **no** |

Use it for:
- Smoke-testing code changes end-to-end without waiting for a VPN.
- Running a "price-action only" baseline so you can attribute IC
  improvements later to the funding/OI/basis features when you add them.

Do **not** use it for the canonical Phase 2 run — crypto's edge over
A-shares comes largely from perp-derived features, which this backend
cannot serve.

---

## 5. Extending to other exchanges

Adding Binance / Bybit is roughly three steps:

1. Create `kairos/data/markets/crypto_exchanges/binance.py`.
2. Subclass `CryptoExchange`, implement `list_markets` and `fetch_ohlcv`
   (ccxt makes this ~60 lines), then call `register_exchange("binance", ...)`.
3. Wire it into `crypto_exchanges/__init__.py`'s `_safe_import` tuple.

Then users can opt in with `--exchange binance` or
`KAIROS_CRYPTO_EXCHANGE=binance`. No change to `CryptoAdapter`, no change to
anything downstream of the adapter boundary.

See `crypto_exchanges/okx.py` for the reference implementation.

---

## 6. Known pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: ccxt is required` | Installed without `[crypto]` extra | `pip install -e '.[crypto]'` |
| `403 Forbidden` / timeouts | Network blocks OKX (common in corporate / office networks) | Set a proxy or move to a home / cloud runner |
| `unknown crypto universe` | Passed an A-share style universe name | Use `topN`, a single symbol, or a comma-separated list |
| `freq '1min' not supported for crypto` | Typo / unsupported freq | Run `kairos-collect --help` and pick from §4.4 |
| Fewer bars than expected | OKX returns empty pages during upstream outages; we stop after 3 in a row | Re-run with `--daily-append` to top up missing windows |

---

## 7. Feature schema (Phase 2)

The exogenous vector is always 32-wide. It is built as
`COMMON_EXOG_COLS (24) + MarketAdapter.MARKET_EXOG_COLS (8)`:

| Slot | Source | Column | Description |
|---|---|---|---|
| 0-23 | `kairos.data.common_features.COMMON_EXOG_COLS` | `log_ret_*`, `rsi_14`, `macd_hist`, `atr_14`, `parkinson_20`, `ma{5,20,60}_dev`, `boll_z`, `obv_z`, `mfi_14`, `amount_z`, `vwap_dev`, `amplitude`, `upper/lower_shadow`, `body_ratio`, `pad_common_{0,1}` | Market-agnostic technicals, reused across every adapter. |
| 24 | Crypto | `funding_rate` | Filled from `OkxExchange.fetch_funding_rate_history` when provided via `FeatureContext.extras["funding_rate"]`. |
| 25 | Crypto | `funding_rate_z` | 60-bar rolling z-score of `funding_rate`. |
| 26 | Crypto | `oi_change` | Log-delta of open interest (`extras["open_interest"]`). |
| 27 | Crypto | `basis` | `perp_close / spot_close - 1`, expects `extras["spot_close"]`. |
| 28 | Crypto | `btc_dominance` | BTC market cap share (`extras["btc_dominance"]`), 0..1. |
| 29 | Crypto | `hour_sin` | `sin(2π · hour_of_day / 24)` — always populated. |
| 30 | Crypto | `hour_cos` | `cos(2π · hour_of_day / 24)` — always populated. |
| 31 | Crypto | `pad_crypto_0` | Reserved for future cross-sectional factor. |

**Initial crypto run** uses only slots 0-23, 29, 30 (price-action +
intraday cycle). Plug in the three external series via
`FeatureContext.extras` once you have collected them:

```python
from kairos.data.markets.crypto_exchanges.okx import OkxExchange, to_unix_ms

ex = OkxExchange()
start_ms = to_unix_ms("2024-01-01")
end_ms = to_unix_ms("2024-04-01")

funding = ex.fetch_funding_rate_history("BTC/USDT:USDT", start_ms, end_ms)
oi = ex.fetch_open_interest_history("BTC/USDT:USDT", "1h", start_ms, end_ms)
spot = ex.fetch_spot_ohlcv("BTC/USDT", "1min", start_ms, end_ms)["close"]

# Pipe them into build_features for enrichment (usually done inside
# prepare_dataset; this is the raw wiring if you want a notebook flow):
df_feat = build_features(
    df_perp_ohlcv,
    market="crypto",
    extras={
        "funding_rate": funding["funding_rate"],
        "open_interest": oi["open_interest"],
        "spot_close": spot,
    },
)
```

The three `fetch_*_history` methods are already implemented and ready to
call the moment your runner has network access to OKX.

---

## 8. Training a crypto model

Once you have a prepared dataset under `./finetune/data/crypto_1min`:

```bash
# preset from kairos.training.config.preset_for
python - <<'PY'
from kairos.training.config import TrainConfig, preset_for
cfg = TrainConfig(
    **preset_for("crypto-1min"),
    dataset_path="./finetune/data/crypto_1min",
    save_path="./artifacts/checkpoints_crypto",
)
print(cfg)
PY
```

`preset_for("crypto-1min")` flips the knobs that differ from the A-share
daily baseline:

| Knob | Default (A-share) | Crypto-1min |
|---|---|---|
| `market` | `ashare` | `crypto` |
| `freq` | `daily` | `1min` |
| `lookback_window` | 90 | 256 |
| `predict_window` | 10 | 32 |
| `return_horizon` | 5 | 30 |
| `ce_weight` | 0.5 | 0.7 |
| `quantile_weight` | 2.0 | 1.5 |

Backtest on the same bundle (market / freq auto-read from `meta.json`):

```bash
python -m kairos.training.backtest_ic \
  --ckpt artifacts/checkpoints_crypto/predictor/checkpoints/best_model \
  --dataset-path ./finetune/data/crypto_1min \
  --horizons 1,5,30 \
  --aggregation date \
  --out artifacts/backtest_crypto.json
```

`--aggregation` controls the cross-section: with only a handful of
symbols, bucket by `date` (one cross-section per day); with a broader
universe you can bump to `hour` or `minute`.

### 8.1 Running on AutoDL end-to-end

The repo ships two scripts that wrap the full flow:

```bash
# On the Mac: pack code via git-archive + dataset tar.gz, scp to instance
scripts/package_and_upload.sh <PORT> <HOST> data/crypto/bv_1min_2y

# On the AutoDL box (after ssh):
cd /root/autodl-tmp
mkdir -p Kairos && tar xzf kairos_code.tar.gz -C Kairos
cd Kairos
bash scripts/autodl_bootstrap.sh /root/autodl-tmp/bv_1min_2y.tar.gz
```

`autodl_bootstrap.sh` is idempotent:

1. clears AutoDL's academic-turbo proxy env (otherwise it hijacks
   `hf-mirror.com`),
2. creates `.venv`, runs `pip install -e '.[train]'`, pins `numpy<2`,
3. writes `HF_ENDPOINT` + `HF_HOME` into `~/.bashrc`,
4. extracts the dataset tarball and verifies `meta.json`,
5. runs a short `KAIROS_SMOKE=1` + `KAIROS_PRESET=crypto-1min` training
   end-to-end. If smoke passes, it prints the exact `tmux` command to
   kick off the real training.

Typical real-run cost on RTX 5090 32GB, BTC+ETH 1min, 2 years of data,
`crypto-1min` preset, 15 epochs: **~2–3h, ¥6–10**.

---

## 9. Roadmap

- **Phase 2 (now):** implement funding / OI / basis ingestion end-to-end,
  plumb into `prepare_dataset` via `FeatureContext`, run first
  `crypto-1min` fine-tune, compare IC vs a funding-rate-only baseline.
- **Phase 3 (if IC > 0):** 15-second bars on a curated universe, funding
  arbitrage, optional on-chain features.

The adapter architecture means swapping the data source for any of the
above is local to `kairos/data/markets/` — the training and backtest code
stays put.
