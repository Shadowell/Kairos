# Crypto OKX perpetual multi-channel transformation plan

> **Status**: Draft/To be reviewed
> **Purpose**: Let `--market crypto` truly utilize funding / OI / basis and completely unlock the 5 "data-driven" bits of `MARKET_EXOG_COLS`.
> **No damage**: A-shares main path, binance_vision spot collection, existing spot Top100 run.

---

## 0. Motivation – Why the change is necessary

CRYPTO_TOP100_1Y_SPOT_RUN.md §11's h30 result is bright (ICIR 0.454), but h1/h5 are negative.
After investigation, it was found that the root cause = **EXOG's 5 data driver bits (`funding_rate`, `funding_rate_z`, `oi_change`, `basis`, `btc_dominance`) are always 0** in any existing run.

reason:

1. `CryptoAdapter.market_features` is designed to eat `context.extras["funding_rate"]` and so on, but **no code is inserted into `extras`**.
2. The crypto path of `kairos-collect` only stores OHLCV parquet and does not collect funding / OI / spot.
3. `prepare_dataset.build_features(...)` does not pass `extras`, `extras={}` → adapter `_align_series` returns all NaN → fillna(0) → is 0.

Therefore: even if we switch to OKX perpetual, as long as we do not change the code, the 5 columns of EXOG's "perpetual exclusive" are still 0, and the "perpetual" and "spot" seen by the model are almost the same.

---

## 1. Data placement specifications

### 1.1 Directory layout

```
raw/crypto/perp_1min_top100/          # Home directory, save perp OHLCV parquet
  BTC_USDT-USDT.parquet               # Consistent with the existing `sanitize_symbol` rule
  ETH_USDT-USDT.parquet
  ...
  _extras/                            # Auxiliary factor directory (double underscore prefix avoids conflict with symbol names)
    funding/
      BTC_USDT-USDT.parquet           # Single column funding_rate, 8h one
      ...
    open_interest/
      BTC_USDT-USDT.parquet           # Single column open_interest, 5min one
      ...
    spot/
      BTC_USDT-USDT.parquet           # spot close 1min, used to calculate basis
      ...
    btc_dominance.parquet             # Shared by the whole market, with 1min or 1h granularity
```

### 1.2 parquet schema (convention shared by all auxiliary parquet)

|List|type|illustrate|
|---|---|---|
| `datetime` | `datetime64[ns]` (naive, UTC) |Synchronized with the main parquet for easy `_align_series` reindex|
| `<payload>` | float64 |Single column payload: `funding_rate` / `open_interest` / `spot_close` / `btc_dominance`|

- An auxiliary parquet only stores one column of payload + datetime, without adding anything else, so it is convenient to directly `df.set_index("datetime")["<col>"]`.
- `datetime` Must be deduplicated and sorted in ascending order.

### 1.3 Why not merge directly into main parquet

- Different frequencies: K-line 1min, funding 8h, OI 5min - merging into the main parquet requires ffill/reindex, and the original frequency information will be lost.
- When collecting again, rerunning a certain channel will not involve the main K-line.
- Keep decoupled from the existing `fetch_one → to_parquet` and do not change the old dataframe schema.

---

## 2. Changes to the collection layer (Phase 2)

### 2.1 API extension

Add CLI flag in `kairos-collect` (closed by default, does not affect spot run):

```
--crypto-extras {none,funding,oi,spot,basis,all}[,...]
    default: none (backwards-compatible)
    values:
      funding   → collect OKX funding-rate-history (8h frequency)
      oi        → collect OKX open-interest-history (5m frequency)
spot → adopt the 1min OHLCV corresponding to the spot symbol (used to calculate basis)
basis → only use spot, then calculate basis when packaging (= alias of spot, keep two names)
      all       → funding + oi + spot
```

### 2.2 Division of labor

|module|Responsibilities|
|---|---|
| `kairos/data/collect.py::main` |Parse `--crypto-extras` and send it to adapter|
| `kairos/data/markets/crypto.py::CryptoAdapter.fetch_extras(symbol, …)` |New method: call the OKX corresponding endpoint according to the request set and return `{"funding": df, "oi": df, "spot": df}`|
|`kairos/data/crypto_extras.py` (new file)|Drop util: `save_extras(out_dir, symbol, extras_dict)` + `load_extras(out_dir, symbol, kinds)`|
| `kairos/data/collect.py::fetch_one` |After collecting OHLCV, if `--crypto-extras != none`, then serially adjust `fetch_extras` and write `_extras/<kind>/`|

### 2.3 Why serial and not concurrent

- OKX's current limit is **endpoint-level IP current limit** (`public/*` 20 req/2s, `rubik/stat/*` 5 req/2s). Each of the 4 types of extras in the same symbol is run once. The total number of requests is small (funding 10-50 times + OI ~30 times + spot and main K-line peering), and 4 worker concurrency will basically not be triggered.
- In extreme cases, it is enough to add a global `ccxt enableRateLimit=True`, and there is no need to write the current limit yourself.

### 2.4 Resume download from breakpoint

Reuse existing `daily_append` logic:

- Each extras parquet is `_load_existing` individually. If it already exists, only the new time period will be incrementally captured.
- Failed extras do not affect the main OHLCV save (main acquisition success + extras failure = warn, the task is still considered ok).

---

## 3. Packaging layer changes (Phase 3)

### 3.1 Changes

`kairos/data/prepare_dataset.py::per_symbol_split` (L120) Current:

```python
df = build_features(df, index_df, market=market, symbol=path.stem)
```

Change to:

```python
extras = None
if market == "crypto":
    extras = crypto_extras.load_for_symbol(
        raw_dir=path.parent,        # .../perp_1min_top100/
        symbol_stem=path.stem,      # "BTC_USDT-USDT"
        freq=freq,
    )
df = build_features(df, index_df, market=market, symbol=path.stem, extras=extras)
```

### 3.2 `load_for_symbol` Behavior

- Check whether `path.parent / "_extras" / <kind> / <stem>.parquet` exists, read out if it exists, and return the Series with `datetime` as the index.
- The missing kind will not be entered directly into the `extras` dict, and the adapter side `_align_series` will return NaN → fillna(0). **Backward Compatibility**: If the user does not choose extras, the packaging behavior is exactly the same as today.
- The returned dict key is consistent with the key read by `CryptoAdapter.market_features`:
  - `"funding_rate"` → `funding_rate` column of funding parquet
  - `"open_interest"` → `open_interest` column of OI parquet
  - `"spot_close"` → spot parquet’s `close` column
  - `"btc_dominance"` → `_extras/btc_dominance.parquet` (market-wide shared file)

### 3.3 meta.json extension

`meta.json` written after packaging is completed adds a field:

```json
{
  "market": "crypto",
  "freq": "1min",
  "exog_channels_available": ["funding_rate", "oi_change", "basis"],
  ...
}
```

Downstream backtest or document generation can read this directly without re-exploring the directory.

---

## 4. Things not to do

1. **Not moving `n_exog=32`** of KronosWithExogenous. Architectural invariants.
2. **Does not do spot VWAP/spot volume**. basis only use spot close.
3. **Does not make binance_futures adapter**. Leave it to follow.
4. **Do not delete or rename** The existing parquet directory (`binance_vision/*` spot will continue to be available).

---

## 5. Time budget

| Phase |Work|estimate time|
|---|---|---|
| 2 |Change collect.py + crypto adapter + crypto_extras.py and write| 2-3 h |
| 3 |Change prepare_dataset.py + crypto_extras.py to read| 1 h |
| 4 |Local Top3 × 1 day smoke (verify that the extras values ​​are different and the build_features 32 columns are non-zero)| 30 min |
| 5 | AutoDL Top5 × 1 day mini end-to-end | 2 h |
| 6 |Top100 × 1 year full collection + training + backtest|10-12 h (big head in acquisition and backtest stride=10)|
| 7 |Write docs/CRYPTO_PERP_RUN.md + comparison table| 1 h |
|**total**| | **~18 h** |

---

## 6. Failure scenarios & fallback

|scene| fallback |
|---|---|
|OKX API symbol funding/OI 404 (non-perpetual or removed from the shelves)|This kind does not write parquet, pack the side `_align_series → NaN → 0`, and continue training.|
|mihomo disconnected/node blocked|The ccxt retry of `fetch_extras` will report an error, `fetch_one` will remember `fail` but the entire batch will not be interrupted.|
|spot symbol does not exist (some alt coins only have perp and do not correspond to spot)|Skip spot, basis=0|
|**OKX funding-rate-history is retained for ~90 days** (Phase-5 measured dichotomy: 90d ok, 120d empty table)|The oldest ~9 months funding in the 1 year run is all empty; accept `_align_series` and fill in 0, or rewrite adapter and go to crontab to build history from `fetchFundingRate` (current real-time value)|
|**OKX contracts/open-interest-history only returns the last ~8 hours** (actual measurement: default 100 items 5m; `after`/`before`/`begin`/`end` parameters do not provide old data)|Formal training must give up historical OI; either subscribe in real time and build your own parquet accumulation, or accept the `oi_change` column to be 0 in the long term (the same as the BTC/ETH spot run). Phase 5 mini cannot be obtained because the window is > 8h → This is **expected**, not a bug. If mini wants real OI, just receive the window in the last 5 hours|

---

## 7. Verification Checklist (for Phase 4/5)

- [ ] After mining BTC-USDT-SWAP for one day: main parquet 1440 lines, funding parquet 3 lines (8h), OI parquet 288 lines (5m), spot parquet 1440 lines
- [ ] `build_features(perp_df, market="crypto", extras={funding, oi, spot})` Output 32 columns where `funding_rate` non-zero scale > 0
- [ ] `prepare_dataset` Outputs the mean/variance pair of `exog_train.pkl` 5 perpetual exclusive columns **non-zero** (z-score should be distributed)
- [ ] `train_predictor` smoke (KAIROS_SMOKE=1) runs through
- [ ] `backtest_ic` smoke runs through, and the result file is placed on disk
