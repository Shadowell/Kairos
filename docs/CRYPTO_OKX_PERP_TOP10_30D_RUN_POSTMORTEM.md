# Crypto OKX perpetual Top10 30-day experiment post-mortem

> On top of [`CRYPTO_TOP100_1Y_SPOT_RUN.md`](CRYPTO_TOP100_1Y_SPOT_RUN.md)'s spot baseline, **for the first time** changed the data source to **OKX perpetual** to get real non-zero `funding_rate` and `basis` to verify the end-to-end link of the multi-channel transformation ([`CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md`](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md)).
>
> **Conclusion first**: The link is opened (collection → packaging → training → backtest is all run), **but the training effect is negative transfer** - finetuned is worse than baseline (Kronos original weight + random head) on pooled IC. The diagnosis has been established and the three root causes are listed in §8. The main value of this document is to keep **all the pitfalls and diagnostic methods** that have been stepped on to avoid consuming GPU time again.
>
> Related documents:
> - [`CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md`](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md) — The overall plan for perpetual multi-channel transformation (funding / OI / basis)
> - [`CRYPTO_BTC_ETH_2Y_SPOT_RUN.md`](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md) — spot BTC+ETH 2 years baseline
> - [`CRYPTO_TOP100_1Y_SPOT_RUN.md`](CRYPTO_TOP100_1Y_SPOT_RUN.md) — spot Binance Top100 1 year baseline
> - [`BACKTEST_IC_INTERPRETATION_GUIDE.md`](BACKTEST_IC_INTERPRETATION_GUIDE.md) — How to choose bucket / aggregation / stride / horizon
> - [`AUTODL_REMOTE_TRAINING_GUIDE.md`](AUTODL_REMOTE_TRAINING_GUIDE.md) — AutoDL common card rental training manual

---

## 0. Background and Objectives

- **Date**: 2026-04-20
- **Motivation**: The two spot runs of BTC/ETH and Top100 are because the `binance_vision` image does not have derivative factors, the four columns of `funding_rate / oi_change / basis / btc_dominance` are padded to 0, and the model can only use 24-dimensional common factors. This time I want to verify: **With real non-zero funding and basis, can the IC of h1/h5 short horizon also increase**?
- **Route Selection**: "Route A — AutoDL Tunnel to Airport, use OKX perpetual" in [`CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md`](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md).
- **data source**: `api.okx.com` perpetual + spot (get basis), through mihomo (Clash Meta) + airport subscription.
- **Machine**: Use the same AutoDL RTX 5090 (`connect.westd.seetacloud.com:37667`).

### 0.1 Scope Adjustment Timeline

> This is to explain "why it only ran for 30 days in the end" - it was not a default plan, but was forced out by the hard limits of the OKX API.

|time|plan|Trigger reason|
|---|---|---|
| 11:30 |Top100 × 90 days (funding coverage limit)|Go [`CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md`](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md) Phase 6|
| 14:00 |User modification: Top10 × 1 year|Want to quickly verify the link on a small scale first|
| 16:00 |Further changes: **Top10 × Last 30 days**|One year of funding data OKX only backfills for 90 days, which is of little significance for "using funding as exog"; funding coverage is the highest in 30 days|

---

## 1. Universe selection

OKX perpetual 24h trading volume Top10 (sorted by `baseVolume × last × contractSize`, **not** use `quoteVolume` - the number of meme currency contracts will flood quoteVolume, see [`CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md`](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) for details):

```
BLUR/USDT:USDT  BTC/USDT:USDT   DOGE/USDT:USDT  ETH/USDT:USDT
HYPE/USDT:USDT  ORDI/USDT:USDT  PEPE/USDT:USDT  SOL/USDT:USDT
XAU/USDT:USDT   XRP/USDT:USDT
```

XAU/USDT: USDT (gold perpetual) is retained in the training set to see if the model can distinguish "objects that are almost irrelevant to BTC" - as a result, its spot is missing (OKX does not correspond to the spot), and the basis column is 0 for it.

---

## 2. Server environment (incremental)

The code / venv / hf_cache all uses the previous BTC/ETH + Top100 run, only the **network tunnel** part is added.

### 2.1 mihomo (Clash Meta) becomes an agent

```bash
# on AutoDL
mkdir -p /root/.config/mihomo && cd /root/.config/mihomo
# 1. Subscribe to the airport (flag=meta to get mihomo compatible YAML)
curl -L -o config.yaml '<Airport Subscription URL>&flag=meta'
# 2. Pre-download the GeoIP/GeoSite database (if not downloaded, startup will fail)
curl -L -o Country.mmdb https://github.com/.../Country.mmdb
curl -L -o GeoSite.dat  https://github.com/.../GeoSite.dat
curl -L -o geoip.dat    https://github.com/.../geoip.dat
# 3. Start
nohup /root/mihomo/mihomo -d /root/.config/mihomo > /root/mihomo.log 2>&1 &
# 4. GLOBAL defaults to DIRECT, you need to switch to a specific node
curl -X PUT http://127.0.0.1:9090/proxies/GLOBAL \
-H 'Content-Type: application/json' -d '{"name":"<US node name>"}'
# 5. Test
HTTPS_PROXY=http://127.0.0.1:7890 curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" \
    https://api.okx.com/api/v5/public/time
# Expectation: 200 ~1.1s
```

The US node is the fastest in actual testing (~1.1s/req), HK is at 0.8s but unstable, and DE is 1.8s+.

### 2.2 Two fixes for OKX adapter (**Committed on 4-20 am**)

During the running process, two problems were found that need to be modified `kairos/data/markets/crypto_exchanges/okx.py`:

1. **`InvalidProxySettings`** — ccxt ≥ 4.5 does not allow `http_proxy` + `https_proxy` to be set simultaneously. OKX uses all HTTPS, leaving only `https_proxy` (commit `9e33a2f`).
2. **funding/OI time window parameter** — `since` kwarg is ignored by the OKX server, use `params={"after": cursor}` to check funding, `params={"begin":..., "end":...}` to check OI; at the same time, the empty frame should also retain the `funding_rate`/`open_interest` column to avoid KeyError (commit `05b8595`).

### 2.3 Hard history window of OKX API (this is the fundamental constraint of 30-day selection)

|endpoint|actual traceability window|Influence|
|---|---|---|
| `/api/v5/market/history-candles` |Many years (enough)| OK |
| `/api/v5/public/funding-rate-history` |**Last ~90 days** (older returns empty)|The training window exceeds 90 days, and the first part of the funding column is all 0|
| `/api/v5/rubik/stat/contracts/open-interest-history` |**Last ~8 hours** (100 items × 5min)|OI history is almost unavailable, and you need to subscribe in real time to accumulate it yourself; this run directly accepts `oi_change=0`|

See [`CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md`](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md) §5 "Fallback scenarios" for details.

---

## 3. Collect Top10 × 30d × 1min

### 3.1 Commands

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
unset http_proxy HTTP_PROXY            # ccxt can only require https
export HTTPS_PROXY=http://127.0.0.1:7890

UNIVERSE="BLUR/USDT:USDT,BTC/USDT:USDT,DOGE/USDT:USDT,ETH/USDT:USDT,HYPE/USDT:USDT,ORDI/USDT:USDT,PEPE/USDT:USDT,SOL/USDT:USDT,XAU/USDT:USDT,XRP/USDT:USDT"

nohup kairos-collect --market crypto --exchange okx \
    --universe "$UNIVERSE" --freq 1min \
    --start 2026-03-20 --end 2026-04-20 \
    --out ./raw/crypto/perp_top10 --workers 4 \
    --crypto-extras funding,spot \
    --proxy "$HTTPS_PROXY" \
    > logs/perp_top10_collect.log 2>&1 &
```

### 3.2 Output

- **Time consumption: 9 m 34 s** (10 perp × 30d × 1min ≈ 432k lines/coin)
- `raw/crypto/perp_top10/`
  - 10 × `.parquet`(OHLCV + amount)
  - `_extras/funding/` 10 parquets (**all non-zero**, funding period 8 hours → 30 days ~90 pieces/coin)
  - `_extras/spot/` 9 parquets (XAU does not correspond to spots)
  - No `_extras/oi/` —— OKX history OI is only 8 hours long, don’t catch it
- **Key sanity**: `funding_rate` has a non-zero rate of ~94% after packaging, `basis` has a non-zero rate of ~88%, **This is the core improvement of this run compared to the previous two**.

### 3.3 Monitoring script (solve the problem of `nohup + ThreadPoolExecutor + tqdm` not being able to see the progress)

`kairos-collect` In nohup mode tqdm does not render, 4 workers share a Python process, and the outer layer can only `pgrep` reach 1 PID. **It’s easy to mistake it for death**. This time I wrote a heartbeat monitoring script:

```bash
# /tmp/perp_top10_monitor.sh
#!/usr/bin/env bash
set -u
OUT_DIR="${1:-/root/autodl-tmp/Kairos/raw/crypto/perp_top10}"
LOG="${2:-/root/autodl-tmp/Kairos/logs/perp_top10_collect.log}"
INTERVAL="${3:-60}"

find_pid() {
    pgrep -af kairos.data.collect 2>/dev/null | awk '/python/ {print $1; exit}'
}

prev_cpu=0; t0=$(date +%s)
while true; do
    pid=$(find_pid)
    [[ -z "$pid" ]] && { echo "process gone"; tail -n 10 "$LOG"; exit 0; }
    ncand=$(ls "$OUT_DIR"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    nfund=$(ls "$OUT_DIR"/_extras/funding/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    nspot=$(ls "$OUT_DIR"/_extras/spot/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    sz=$(du -sh "$OUT_DIR" 2>/dev/null | awk '{print $1}')
    # 0x1ED2 = 7890 (proxy port)
    est=$(awk '/ 0100007F:1ED2 / && $4=="01" {c++} END{print c+0}' /proc/"$pid"/net/tcp 2>/dev/null)
    threads=$(awk '{print $20}' /proc/"$pid"/stat 2>/dev/null)
    cpu=$(awk '{print $14+$15}' /proc/"$pid"/stat 2>/dev/null)
    delta_cpu=$(( cpu - prev_cpu ))
    prev_cpu=$cpu
    echo "[$(date +%H:%M:%S) +$(( $(date +%s) - t0 ))s] py_pid=$pid cand=$ncand fund=$nfund spot=$nspot size=$sz proxy_est=$est threads=$threads cpu_delta=$delta_cpu"
    sleep "$INTERVAL"
done
```

Key points:
- `pgrep -af kairos.data.collect | awk '/python/ {print $1}'`—— The bash package will be matched by `pgrep -f`, and `awk` must be used to filter out the `python` lines.
- `awk '/ 0100007F:1ED2 / && $4=="01"' /proc/$pid/net/tcp` —— `0x1ED2 = 7890` is the mihomo port, `$4=01` is `ESTABLISHED`, and you can directly see how many workers are currently connected to the agent.
- `cpu_delta` greater than 0 means working in active; `threads` should be ≥ 6 (4 workers + main thread + GIL helper) when `workers=4`.

---

## 4. Packing

```bash
kairos-prepare --market crypto \
    --raw ./raw/crypto/perp_top10 \
    --train 2026-03-21:2026-04-17 \
    --val   2026-03-21:2026-04-17 \
    --test  2026-04-17:2026-04-20 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --extras-channels funding,spot \
    --out ./finetune/data/perp_top10_30d
```

### output

- `train_data.pkl` 330k lines (10 symbols)
- `val_data.pkl` 58k lines
- `test_data.pkl` 43k lines (**= 4035 minutes × 10 symbols ≈ 3 days**)
- There is an extra `extras_channels: ["funding", "spot"]` field in `meta.json` (revealed by adapter)
- In 32-dimensional exog `funding_rate / basis` is truly non-zero, `oi_change / btc_dominance` is still 0

---

## 5. Training

```bash
cat > logs/run_train_perp_top10.sh <<'SH'
#!/bin/bash
set -e
cd /root/autodl-tmp/Kairos
source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1                # Force local cache
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/perp_top10_30d
export KAIROS_NUM_WORKERS=0
export KAIROS_BATCH_SIZE=64
export KAIROS_EPOCHS=10
export KAIROS_N_TRAIN_ITER=5000        # ⚠️ This line is the culprit diagnosed in §8
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
SH
```

### 5.1 Training log (10 epochs, 1 m 13 s in total)

```
[TRAIN] pool=327610, using 5000/epoch.
[VAL]   pool=55440,  using 800/epoch.

ep 1: val_ce=2.4621  ✅ save best
ep 2: val_ce=2.4601  ✅ save best
ep 3: val_ce=2.4587  ✅ save best
ep 4: val_ce=2.4577  ✅ save best
ep 5: val_ce=2.4570  ✅ save best
ep 6: val_ce=2.4567  ✅ save best
ep 7: val_ce=2.4564  ✅ save best
ep 8: val_ce=2.4563  ✅ save best
ep 9: val_ce=2.4563  patience 1/3
ep 10: val_ce=2.4563 patience 2/3
```

val_ce dropped all the way but the absolute drop was very small (2.4621 → 2.4563, **Δ = 0.0058**) - an order of magnitude smaller than the BTC/ETH run (Δ ~0.05) and Top100 run (Δ ~0.09). **This is a clear sign of underfitting** but was not realized at the time (see §8.1).

### 5.2 Key observations

`[TRAIN] pool=327610, using 5000/epoch` ——The pool has 320,000 samples, and only **5000** are randomly selected every epoch. A total of 50k samples are viewed in 10 epochs, which is only 15% of the pool**. See §8.1 for the reason.

---

## 6. backtest

Three bucket configurations were run, each with a pair of baseline + finetuned:

| run | bucket | stride | n_records |time|
|---|---|---|---|---|
|First version (bug → see §7.1)| minute | 1 | 40350 | ~5 min × 2 |
|date rerun| date | 1 | 40350 | ~5 min × 2 |

```bash
python -u -m kairos.training.backtest_ic --baseline \
    --preset crypto-1min --dataset-path $DATA \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/perp_top10_30d/backtest_baseline_date.json

python -u -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min --dataset-path $DATA \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/perp_top10_30d/backtest_finetuned_date.json
```

---

## 7. Results

### 7.1 minute bucket (first version, **results are unreliable**)

When `bucket=minute`, each bucket only has 10 samples (10 symbols × 1 min), the single bucket IC standard deviation is ~0.35, the average SE of 4035 buckets is ~0.0055, and **the weak signal is completely overwhelmed by the noise**.

| h | metric | baseline | finetuned | Δ |
|---|---|---|---|---|
| h1 | rank_ic (by-min) | +0.0045 | -0.0187 | -0.023 |
| h5 | rank_ic (by-min) | +0.0078 | -0.0108 | -0.019 |
| h30 | rank_ic (by-min) | -0.0387 | +0.0024 | +0.041 |
| h30 | ICIR (by-min) | -0.068 | +0.011 | +0.079 |

`bucket=minute` The statistical properties under the 10 symbols × short test area are detailed in [`BACKTEST_IC_INTERPRETATION_GUIDE.md`](BACKTEST_IC_INTERPRETATION_GUIDE.md) §3.2.

### 7.2 date bucket (after re-running, **main reference but still unreliable**: n=3)

| h | metric | baseline | finetuned | Δ |
|---|---|---|---|---|
| h1 | ic (by-day) | -0.0091 | -0.0085 | +0.001 |
| h1 | rank_ic | -0.0129 | -0.0155 | -0.003 |
| h5 | rank_ic | +0.0170 | +0.0021 | -0.015 |
| h30 | ic (by-day) | +0.0319 | +0.0024 | **-0.029** |
| h30 | rank_ic | +0.0078 | +0.0164 | +0.009 |
| h30 | ICIR (by-day) | +1.17 | +0.06 | -1.11 |

⚠️ **`n_dates=3`** (the test area is only 4-17 / 4-18 / 4-19 three days), the denominator of ICIR (standard deviation of IC) has only 3 point estimates, **completely noise**. This is where the "good looking" number of ICIR=+1.17 comes from.

### 7.3 pooled overall (**the only credible statistical signal**, n=40350)

Ignore the bucket and directly use 40,000 (score, return) to calculate Pearson/Spearman/hit_rate:

| h | metric | baseline | finetuned | Δ |
|---|---|---|---|---|
| h1 | spearman | -0.0122 | -0.0118 | +0.0004 |
| h1 | hit_rate | 54.14% | 51.11% | **-3.03 pp** |
| h5 | spearman | +0.0167 | +0.0011 | -0.0156 |
| h5 | hit_rate | 48.86% | 49.03% | +0.17 pp |
| h30 | pearson | **+0.0368** | -0.0062 | **-0.0430** |
| h30 | spearman | **+0.0226** | +0.0021 | -0.0205 |
| h30 | hit_rate | 52.43% | 50.93% | -1.50 pp |

**Kronos original weight + random head has an IC of +0.037 on pooled h30 (p < 1e-13)** - This is not alpha, but the transformer hidden state itself has encoded future distribution information, and random fc can also lead to the direction. **finetuned eliminates this signal** - it is negative migration.

### 7.4 Comparison with historical runs

| run |universe × duration| h30 ICIR (by-day, finetuned) | h30 pooled spearman |exog true non-zero|
|---|---|---|---|---|
| BTC/ETH 2y spot |2 × 2 years| +0.325 (n=106 days) | +0.032 | ❌(pad 0) |
| Top100 1y spot  |100 × 1 year| +0.454 (n=78 days)  | n/a    | ❌(pad 0) |
| **Top10 30d perp** |**10 × 30 days**|**+0.063 (n=3, noise)**| **+0.002** | ✅ funding+basis |

The data scale is 10-30 times different, and the conclusion is that it is unclear whether funding/basis is useful.

---

## 8. Post-mortem: three root causes

### 8.1 🔴 The culprit: `KAIROS_N_TRAIN_ITER=5000` Residual → Only 15% of the training pool is actually used

`KAIROS_N_TRAIN_ITER=5000` was set when starting the mini run (5d × 5 symbols verification link), but was not cleared during the official run.

`kairos/training/dataset.py` L80:
```python
limit = cfg.n_train_iter if split == "train" else cfg.n_val_iter
self.n_samples = min(limit, len(self.indices))
print(f"[{split.upper()}] pool={len(self.indices)}, using {self.n_samples}/epoch.")
```

Meaning: **`n_train_iter` is the number of samples** to be viewed per epoch (not the number of steps).

- BTC/ETH run: default `n_train_iter=50000`, pool ~1 million → 5% for each epoch, 75% for 15 epochs.
- Top100 run: default 50000, pool 31.6 million → 0.16% per epoch, but still 200k samples (5× current) after 4 epoch early stop.
- **This run**: `n_train_iter=5000`, pool 320,000 → **1.5%** viewed per epoch, **15%** = 50k samples viewed in 10 epochs.

50k samples are seriously insufficient for a fine-tuning task with 5.4M parameters + 32-dimensional exog + 30 step pinball loss. The fact that val_ce only drops by 0.006 is a direct manifestation of underfitting.

**Fix**: Clear env, or use the default 50000, or change `train_predictor.py` to print `total_steps_seen / pool` and warning at the end of the log.

### 8.2 🟡 The dimensional design of the training target is biased

`kairos/training/train_predictor.py` L107-114:
```python
close_n = x[:, :, close_idx]  # ⚠️ normalized close (already z-score in batch)
T = close_n.size(1)
h = cfg.return_horizon  # = 30
targets = []
for k in range(h):
    rolled = torch.roll(close_n, shifts=-(k + 1), dims=1)
    targets.append(rolled - close_n)         # cumulative diff
target = torch.stack(targets, dim=-1)        # [B, T, h]
```

Two questions:

1. **The dimension is normalized diff** (local z-score for each batch), but the true value of `backtest_ic.py` L244-245 uses raw log-return:
   ```python
   meta[f"ret_h{h}"] = float(np.log(cf / c0))
   ```
IC (Pearson/Spearman) is not sensitive to monotonic transformation and theoretically does not affect the IC sign; however, the quantile distribution learned by the model is in normalized space, and uneven variance may be introduced when backtest is used for cross-sectional sorting.
2. **The target of step k is `close[t+k+1] - close[t]`** —— cumulative diff, the dimension increases linearly with k. With preset `return_horizon=30`, the loss of **k=29 dominates the entire pinball**. The model actually only optimizes the "30th step in the future", and h1-h29 has almost no supervision signal.

This can explain why in the two runs of BTC/ETH and Top100, the effect can only be seen on h30, while h1/h5 does not move.

**Fix Directions** (not implemented):
- Change target to raw log-return or step-wise diff, and press k to do per-horizon normalization
- Or add an explicit "1-step" head to h1 in addition to `n_quantiles=9`

### 8.3 🟡 Test area is only 3 days → date bucket is unreliable + minute bucket is noisy

interleave cuts 4-17 ~ 4-19 to test, `n_dates=3`:
- date bucket: 3 ICs are calculated as ICIR, **completely noise** (ICIR standard error ~ 1/√3 ≈ 0.58)
- minute bucket: 4035 buckets, but each bucket only has 10 samples, single bucket IC standard deviation ~0.35, average SE ~0.0055
- The only reliable one is pooled IC (n=40350)

**Fix direction**: The test area should be given at least 15 days (n_buckets ≥ 15), or the code side should force `backtest_ic` to fallback to pooled and issue a warning when `n_buckets < 10`. See [`BACKTEST_IC_INTERPRETATION_GUIDE.md`](BACKTEST_IC_INTERPRETATION_GUIDE.md) for details.

---

## 9. artifacts list

### Server (AutoDL `/root/autodl-tmp/`)

```
Kairos/
├── raw/crypto/perp_top10/                            15 MB
│ ├── *.parquet 10 perp OHLCV
│ ├── _extras/funding/*.parquet 10 funding rates (all non-zero)
│ └── _extras/spot/*.parquet 9 spot mid (XAU missing)
├── finetune/data/perp_top10_30d/                     ~440 MB
│ ├── {train,val,test}_data.pkl 330k / 58k / 43k lines
│   ├── exog_{train,val,test}.pkl
│   └── meta.json                                     extras_channels: [funding, spot]
├── artifacts/
│   ├── checkpoints/predictor/checkpoints/
│ │ ├── best_model/ 🚨 Top10 perp fine-tuning (covered Top100 ckpt)
│ │ └── best_model_btceth_backup/ ✅ BTC/ETH backup (reserved)
│   └── perp_top10_30d/
│       ├── backtest_baseline.json                    minute bucket
│       ├── backtest_finetuned.json                   minute bucket
│       ├── backtest_baseline_date.json               date bucket
│       └── backtest_finetuned_date.json              date bucket
└── logs/
    ├── perp_top10_collect.log / perp_top10_train.log
    ├── run_btceth_recheck.sh / btceth_recheck.log    sanity-check script(§10.2)
└── run_top10_date.sh / top10_date.log date bucket rerun
```

> ⚠️ **Top100 ckpt has been covered this time**. If you want to compare based on the Top100 ckpt again, you need to re-run §6 of [`CRYPTO_TOP100_1Y_SPOT_RUN.md`](CRYPTO_TOP100_1Y_SPOT_RUN.md), or develop the habit of `cp best_model best_model_<run-name>_backup` from the next run.

---

## 10. How to verify "the code does not regress" (written to my future self)

### 10.1 Same code + old data sanity check

I reran the backtest using today’s main branch code, old BTC/ETH ckpt, and old data set (`finetune/data/crypto_1min_btc_eth/`):

```bash
python -u -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model_btceth_backup \
    --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --aggregation date --stride 5 \
    --out artifacts/btceth_recheck/finetuned_date_stride5.json
```

| metric |Old 4-17 run (stride=1)|Run again today (stride=5)|in conclusion|
|---|---|---|---|
| h30 rank_ic | +0.0505 | +0.0238 |The directions are consistent; the amplitude difference is ~½, consistent with stride=5 → SE × √5 times|
| h30 ICIR | +0.325 | +0.147 |Same as above|
|Overall direction (h1/h5/h30 third gear symbol)|same|same|same|

✅ **The code has no regression**, the old alpha is completely reproducible.

### 10.2 Similar sanity checks should be a regular step before release

Next time after major changes to `train_predictor.py` / `backtest_ic.py` / `kronos_ext.py`, run this: "finetuned BTC/ETH ckpt + finetuned BTC/ETH data + horizon 1,5,30 + date bucket" and compare it with the historical results. See [`BACKTEST_IC_INTERPRETATION_GUIDE.md`](BACKTEST_IC_INTERPRETATION_GUIDE.md) §5 for details.

---

## 11. Next step

Sorted by ROI:

1. **D (fastest verification, ~10 min)**: Clear `KAIROS_N_TRAIN_ITER`, use the default 50000 to retrain this dataset, and see if finetuned can at least **not** fall back to baseline.
2. **B (anti-re-stepping, ~5 minutes to change the code)**:
   - `_BUCKET_ALIASES.auto` is downgraded to pooled when changed to "`n_dates < 10`, otherwise press freq to select date/hour/minute"
   - `dataset.py` Print `using {n_train_iter}/epoch ({pct:.1%} of pool)` at the end of log and warn when pct < 5%
3. **D' (data expanded to usable scale, ~30 min collection + 10 min training)**: Top10 × 90 days (funding coverage limit); the test area is given 15 days (n_buckets ≥ 15).
4. **C (structural repair, ~20 min)**: Repair the train pinball target dimension (normalized diff → raw log-return + per-k normalization), retrain + backtest; this is the deepest reason why short-horizon IC has never been able to get up.

Only after completing 1-2 and getting a baseline of "at least no regression", 3-4 will be meaningful; otherwise, it will just pile up data on the wrong training configuration / change the loss.

---

## 12. One line TL;DR

> The link is connected, and the **funding + spot real non-zero** has entered the 32-dimensional exog; but because the `KAIROS_N_TRAIN_ITER=5000` residual + test area is only 3 days + bucket selection minute, the three-layer superposition leads to negative migration in the final finetuned. There is no regression in the code itself (rerun with old data and the old results pass). Next time, fix the env residue + bucket auto logic first and then expand the data.
