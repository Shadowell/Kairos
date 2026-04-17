# Crypto Market Guide

Kairos supports crypto markets out of the box through a pluggable exchange
backend. By default it connects to **OKX USDT-margined perpetual swaps**; the
abstraction (`kairos.data.markets.crypto_exchanges.CryptoExchange`) is
deliberately thin so that Binance, Bybit, Hyperliquid and friends can be
added by dropping in one file.

This guide is a companion to [`AUTODL_GUIDE.md`](AUTODL_GUIDE.md) and
[`GLOSSARY.md`](GLOSSARY.md); keep them handy.

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

## 7. Roadmap

- **Phase 2 factors (in progress):** funding rate, 8h funding average,
  open interest delta, basis (perp vs spot), BTC dominance, per-symbol
  beta-to-BTC.
- **Phase 2 training:** Kronos + crypto exog → predict T+30min / T+4h
  quantile returns, compared against a funding-rate-only baseline.
- **Phase 3 (if IC > 0):** 15-second bars on a curated universe, funding
  arbitrage, optional on-chain features.

The adapter architecture means swapping the data source for any of the
above is local to `kairos/data/markets/` — the training and backtest code
stays put.
