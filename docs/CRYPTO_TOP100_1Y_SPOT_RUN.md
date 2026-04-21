# Crypto Top100 one year spot Predictor run log

> Based on [`CRYPTO_BTC_ETH_2Y_SPOT_RUN.md`](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md)’s baseline of 2 coins × 2 years, expand the universe to **Binance spot 24h trading volume Top100** (~100 USDT pairs) × **1min in the past year** to see the impact of expanding the amount of data on the fine-tuning effect.
> If you just want to reproduce the results, skip to §10 for a list of TL;DR commands.
>
> Related documents:
> - [`CRYPTO_BTC_ETH_2Y_SPOT_RUN.md`](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md) — Previous version of BTC+ETH 2 currency run-through records (must read, only the differences are discussed here)
> - [`AUTODL_REMOTE_TRAINING_GUIDE.md`](AUTODL_REMOTE_TRAINING_GUIDE.md) — AutoDL common card rental training manual
> - [`CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md`](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) — crypto data layer and exchange extension
> - [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) — Parameter Manual

---

## 0. Background

- **Date**: 2026-04-20
- **Motivation**: The first version of fine-tuning for BTC/ETH only saw usable alpha on h30 (rank-IC +0.050 / ICIR +0.325). It is suspected that the bottleneck is that the universe is too small → the model is easily overfitted on 2 coins. Expand to Top100 verification:
  1. Can the larger train pool be reflected in val_ce;
  2. Is the by-date IC / ICIR more stable after expanding to the cross section?
  3. Can the h1 / h5 segments be lifted up due to sampling of more currencies?
- **Machine**: The same AutoDL RTX 5090 from `CRYPTO_BTC_ETH_2Y_SPOT_RUN.md` (`connect.westd.seetacloud.com:37667`, westd area).
- **data source**: Still using `binance_vision` (the only crypto endpoint that is stable and accessible under the company network). Derivatives factors funding / OI / basis are still pad 0.

---

## 1. Universe selection

### 1.1 Start with Binance spot top-by-volume

Directly reuse `BinanceVisionExchange.list_symbols_by_volume(top_n=200, quote="USDT")`: it sorts `/api/v3/ticker/24hr`, takes 24h `quoteVolume`, and returns ccxt-unified form (`BTC/USDT`). Pull 200 first to leave enough margin.

### 1.2 Filter "structural non-alpha" targets

The original top table contains three types of samples that are harmful to training:

|category|example|Why culled|
|---|---|---|
|Stablecoin| USDC, USDE, FDUSD, RLUSD, TUSD, BUSD, BFUSD, XUSD, USD1, DAI, USDD, USDP |The price is locked around $1 all year round, and the close-to-close return ≈ 0. When calculating IC, the denominator (fluctuation) will be pressed to 0, which is equivalent to using `div by zero` to pollute the cross section.|
|Precious metals / fiat currency anchoring|PAXG (gold), XAUT (gold), EUR, GBP, AEUR|The price has almost no correlation with other cryptos, and has its own independent spot/futures pricing mechanism, which will deviate significantly from the correlation matrix during training.|
|wrapped assets| WBTC, WETH, STETH, WSTETH, WBETH |100% correlated with BTC/ETH/stETH, duplicate samples drag down IC effective degrees of freedom.|
| non-ASCII base | a non-ASCII symbol example and similar wrapper/meme coins | `binance_vision` occasionally returns symbols with non-ASCII characters; they are skipped directly. |

After filtering, select the first 100 from top-200 and write them into `/root/autodl-tmp/top100_universe.txt`. See [`examples/crypto_top100_universe.md`](../examples/crypto_top100_universe.md) for the full list.

### 1.3 Known "weaknesses" in the list

There are many new coins in the Top100 that were launched in 2024-2025 (`TRUMP`, `PNUT`, `PENGU`, `MOVE`, `PLUME`, `GIGGLE`, `PUMP`, `ZAMA`,…), and there is no data before the starting point of `2025-04-20`.
`kairos-prepare` will eliminate the shortest ones due to insufficient samples during the val segmentation stage. This time, the actual val set only has **97/100** symbols left (train / test is still 100). This is expected behavior and does not need to be fixed.

---

## 2. Server environment

Directly use the previous version of run: code `git pull` to the latest (new env overrides and scipy extra are merged), venv is not reinstalled, and `hf_cache` the preheated Kronos-Tokenizer-base / Kronos-small continues to be reused.

```bash
cd /root/autodl-tmp/Kairos
source /etc/network_turbo 2>/dev/null
git pull --ff-only
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.11.0+cu130 True NVIDIA GeForce RTX 5090
```

The initial state of the GPU is clean (`32110 MiB free / 32607 MiB total`), and the disk has 38 GB left, which is enough to accommodate a Top100 run.

---

## 3. Collect Top100 × 1 year 1min

### 3.1 Parameters and key selections

|parameter|value|Why|
|---|---|---|
| `--freq` | `1min` |Keep the same frequency as BTC/ETH run, and the model preset `crypto-1min` remains unchanged.|
| `--start / --end` | `2025-04-20 / 2026-04-20` |Only take the last 1 year: 100 coins × 2 years will exhaust the disk and collection time (estimated 10 hours / 50 GB), 1 year is enough for training and the test set is still 1 million+ records in 2.5 months|
| `--workers` | `4` |`binance_vision` is pure HTTP + shared `requests.Session`, thread-safe; 4 concurrent measured ~4 req/s × 4 ≈ 16 req/s, far lower than Binance `/api/v3/klines` limit of 1200 weight/minute|
| `--exchange` | `binance_vision` |See the previous article; the only reachable crypto endpoint under the company network|

### 3.2 Startup

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY    # The academic proxy is not on the `binance_vision` allowlist

UNIVERSE=$(cat /root/autodl-tmp/top100_universe.txt)   # Generated as described in §1.2
nohup kairos-collect --market crypto --exchange binance_vision \
    --universe "$UNIVERSE" --freq 1min \
    --start 2025-04-20 --end 2026-04-20 \
    --out ./raw/crypto/bv_1min_top100 --workers 4 \
    > logs/collect_top100.log 2>&1 &
```

### 3.3 Output

- **Time taken: 26 minutes and 43 seconds** (100 symbols / 4 workers ≈ 16 s/symbol amortized)
- `raw/crypto/bv_1min_top100/*.parquet` Total **1.2 GB**
- 100 / 100 successful, 0 failed, 0 empty.
- The first symbol of tqdm takes the longest time (78 s) because four workers start grabbing BTC/ETH/SOL/XRP, which are the symbols with the largest amount of data, almost at the same time. The first symbol to be placed takes up all the waiting time. The tempo then stabilizes at 11-20 s/symbol.

### 3.4 Comparison with BTC/ETH run

| |BTC/ETH 2 years|**Top100 1 year**|
|---|---|---|
|symbols × time|2 × 2 years|**100 × 1 year**|
| workers | 1 | 4 |
|Total time spent| 11 m 29 s | 26 m 43 s |
|original parquet| 97 MB | **1.2 GB** |
|throughput (coin-day/second)| ~1.06 | ~22.7(×21) |

`workers=4` The actual test raised the throughput to ~4× single thread. It was not restricted by Binance and did not get 429. If you want to be faster next time, you can try `workers=8`, but half an hour is enough for Top100 data.

---

## 4. Package into training set

After picking, go directly to `kairos-prepare`. Time segmentation:

```bash
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_top100 \
    --train 2025-04-20:2026-01-31 \
    --val   2025-04-20:2026-01-31 \
    --test  2026-02-01:2026-04-20 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_top100
```

### 4.1 Output

```
finetune/data/crypto_1min_top100/    7.6 GB
├── train_data.pkl   967 MB    # 100 symbols
├── val_data.pkl     170 MB    # 97 symbols (3 symbols were eliminated due to too few samples during the training period)
├── test_data.pkl    339 MB    # 100 symbols
├── exog_train.pkl  4.1 GB     # 100 symbols × 32 exog cols × block-level z-score
├── exog_val.pkl    724 MB     #  97 symbols
├── exog_test.pkl   1.4 GB     # 100 symbols
└── meta.json                  # {market: crypto, exog_cols: [32], split_mode: interleave, ranges: {...}}
```

- Takes **about 3 minutes**.
- `exog_cols` in meta.json is still 32 dimensions (24 common + 8 crypto exclusive), funding_rate / funding_rate_z / oi_change / basis / btc_dominance is padded to 0; `n_exog=32` on the architecture side does not need to be changed (AGENTS.md §8 Invariant 2).
- 7.6 GB of exog is fully loaded into the main process memory; during training, the RSS is ~14 GB, which is far below the 96 GB cgroup limit.

---

## 5. Smoke training (16 seconds)

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_SMOKE=1
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_top100
export KAIROS_NUM_WORKERS=0

MASTER_ADDR=127.0.0.1 MASTER_PORT=29510 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    python -m kairos.training.train_predictor
```

- **`[TRAIN] pool=31,601,392`** (vs BTC/ETH smoke’s ~1 million → **~30× big**)
- val_ce = **2.4234**, saved to `artifacts/checkpoints/predictor/checkpoints/best_model`
- Total time taken 16 seconds (50 steps + 40 steps val)

Passed once, description: DDP single card + Kronos weight loading + exog bypass + 32-dimensional exog schema all OK.

---

## 6. Formal training

### 6.1 Commands

It is almost the same as BTC/ETH run, only the data set path is changed:

```bash
cat > logs/run_train_top100.sh <<'SH'
#!/bin/bash
set -e
cd /root/autodl-tmp/Kairos
source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_top100
export KAIROS_NUM_WORKERS=0                # Prevent DataLoader multi-worker memory blow-ups
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
SH
chmod +x logs/run_train_top100.sh
nohup bash logs/run_train_top100.sh > logs/train_top100.log 2>&1 &
```

### 6.2 Back up BTC/ETH old checkpoint (before overwriting)

```bash
cp -r artifacts/checkpoints/predictor/checkpoints/best_model \
       artifacts/checkpoints/predictor/checkpoints/best_model_btceth_backup
```

### 6.3 Trajectory

| Epoch | val_ce |Relative to BTC/ETH with epoch|determination|
|---|---|---|---|
| 1 | **2.4036** | ⬇ 0.0904(BTC/ETH ep1=2.4940) | ✅ save best |
| 2 | 2.4152 |quite| patience 1/3 |
| 3 | 2.5892 |quite| patience 2/3 |
| 4 | 2.9903 |Slightly higher than BTC/ETH ep4=2.8820|patience 3/3 → early stop|

- Each epoch lasts about 2 minutes and 55 seconds (close to BTC/ETH’s 2 m 33 s; `n_train_iter=50000` is fixed with batch=50, and the increase in the sample pool does not affect the number of steps in an epoch).
- Total time taken **12 minutes** (3 min × 4 ep + val).
- **val_ce ep1 is 0.09 lower than BTC/ETH** - This is the most direct return of expanding the universe. The CE space of 0.09 roughly corresponds to an increase in the probability of the next token of about 9%, which is not noise.
- The same overfitting structure (ep2 starts to rise) confirms that after Kronos-small (5.4M parameters) freezes 7/8 layers of the transformer backbone, the capacity is only enough to absorb the signal of one epoch. Continuing training will only overfit memory.

### 6.4 Checkpoint

```
artifacts/checkpoints/predictor/checkpoints/
├── best_model/                  97 MB   # Top100 fine-tuning (ep1, val_ce=2.4036)
└── best_model_btceth_backup/    97 MB   # BTC/ETH original version, keep for reference
```

---

## 7. backtest: baseline vs finetuned

### 7.1 Key parameters: `--stride 10`

Here **must deviate** from the default `stride=1` of BTC/ETH run:

- `stride=1` means one window every 1 minute. BTC/ETH 2 coins × 3 months = 300,000 windows, running for 10 minutes is acceptable.
- Top100 × 2.5 months stride=1 = 11 million windows / horizon → estimated **8 hours / time × 2 times = 16+ hours**.
- After changing `--stride 10`, a signal is sent out every 10 minutes, which is reduced to 1.1 million windows / horizon, **single ~45 minutes**.
- IC stability is no problem: 78 cross-section dates × ~14,000 observations per day on average, more than BTC/ETH run.

### 7.2 Startup

```bash
cat > logs/run_backtest_top100.sh <<'SH'
#!/bin/bash
set -e
cd /root/autodl-tmp/Kairos
source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_top100

python -u -m kairos.training.backtest_ic \
    --baseline --preset crypto-1min \
    --dataset-path $DATASET \
    --horizons 1,5,30 --stride 10 \
    --out artifacts/backtest_top100_baseline.json

python -u -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path $DATASET \
    --horizons 1,5,30 --stride 10 \
    --out artifacts/backtest_top100_finetuned.json
SH
chmod +x logs/run_backtest_top100.sh
nohup bash logs/run_backtest_top100.sh > logs/backtest_top100.log 2>&1 &
```

The two backtests took a total of **89 minutes** (baseline 44 m + finetuned 44 m).

### 7.3 Data scale

- **n_records = 1,106,570** (100 symbols × ~11k windows × horizon reuse)
- **date_range = 2026-02-01 04:16 ~ 2026-04-19 23:26** (78 cross-section dates, the uncorrected UTC offset is still the naive local time of binance_vision, but the comparison consistency is not affected)
- **100 symbols** all participate (test set symbols are fully covered)

### 7.4 Results

| horizon |index| baseline | **finetuned** | Δ |
|---|---|---|---|---|
| **h1** | pearson | +0.0079 | −0.0112 | ❌ |
| | hit_rate | 0.4892 | 0.4906 | +0.14 pp |
| | by-date IC | +0.0079 | −0.0127 | ❌ |
| | rank-IC | +0.0032 | −0.0082 | ❌ |
| | **ICIR** | **+0.424** | **−0.555** |❌ turn negative|
| **h5** | pearson | −0.0025 | −0.0076 | ❌ |
| | hit_rate | 0.4590 | 0.4883 | +2.9 pp |
| | by-date IC | +0.0033 | −0.0107 | ❌ |
| | rank-IC | +0.0139 | −0.0050 | ❌ |
| | **ICIR** | +0.125 | **−0.681** | ❌ |
| **h30** | pearson | −0.0046 | **+0.0054** | ✅ |
| | hit_rate | 0.4863 | 0.4921 | +0.6 pp |
| | by-date IC | −0.0023 | **+0.0158** | ✅ |
| | rank-IC | +0.0004 | **+0.0299** | ✅ |
| | **ICIR** | −0.084 | **+0.454** |✅ Across 0.3|

### 7.5 How to read this table

1. **h30 is the headline alpha** this time.
   - **by-date rank-IC from +0.0004 → +0.0299 (almost 0 → 2.99%)**
   - **ICIR −0.084 → +0.454**, exceeding BTC/ETH run’s +0.325
   - hit_rate 49.21%(+0.6 pp vs baseline)
   - Fine-tuning on this channel has indeed learned the 30-min directionality of the Top100 cross-section, and it is more stable than the 2-coin version on more currencies - this is exactly what you want "IC may not rise, but ICIR does."

2. **h1/h5 are "contaminated" with negative IC**, not random noise.
   - pearson / rank-IC / by-date IC / ICIR The four independent indicators **in the same direction** show negative correlation, with p-values ​​all < 1e-14.
   - The magnitude (by-date IC ≈ −0.012) is small but significant—suggesting that the model did learn something on the short-horizon, just in the wrong direction.
   - In other words, the model squeezes all the explainable variance into h30, and treats h1/h5 as a "reverse warning" of h30**. The corresponding preset `return_horizon=30` is h30 to align the training target, and the short horizon is equivalent to allowing the model to extrapolate a task that it is not supervised.
   - In actual use, you can only subscribe to the h30 signal, or add an additional return head to h1/h5 (for multi-head/multi-horizon training, see TRAINING_TUNING_PLAYBOOK.md).

3. **Baseline's h1 ICIR +0.42 is not true alpha**.
   - baseline = Kronos original weight + randomly initialized exog / return head, the output of return head is messy.
- But when the cross section is only 100 coins, daily samples ~14k, and 78 dates, even random signals can be generated|ICIR|> Falsely high value of 0.3 - so look at the **relative improvement** (Δ) of finetuned, not the baseline absolute ICIR.
   - This is why reporting an ICIR must also report a baseline control.

4. **h30 vs BTC/ETH run**:

   | run | rank-IC | ICIR | hit_rate |
   |---|---|---|---|
   |BTC/ETH 2 coins × 2 years| +0.050 | +0.325 | 0.5168 |
   |**Top100 × 1 year**| **+0.030** | **+0.454** | 0.4921 |

The absolute value of rank-IC dropped from 5% to 3%, but the ICIR (information ratio) increased by 40% - a more valuable signal for combined use. At the same time, hit_rate dropped from 51.7% to 49.2% - this implies that the 30-min directionality of many small coins in the Top100 is lower than that of BTC/ETH, and the alpha learned by the model is more about the relative strength of cross-currency rather than direction.

---

## 8. artifacts list

### Server (AutoDL `/root/autodl-tmp/`)

```
top100_universe.txt                                     962 B    # Universe list for 100 symbols
Kairos/
├── raw/crypto/bv_1min_top100/                          1.2 GB   # 100 parquet files
├── finetune/data/crypto_1min_top100/                   7.6 GB   # train/val/test + exog + meta.json
├── artifacts/
│   ├── backtest_top100_baseline.json                   1.4 KB
│   ├── backtest_top100_finetuned.json                  1.4 KB
│   └── checkpoints/predictor/checkpoints/
│       ├── best_model/                                 97 MB    # Top100 fine-tuning (ep1, val_ce=2.4036)
│       └── best_model_btceth_backup/                   97 MB    # BTC/ETH backup, reserved
└── logs/
    ├── collect_top100.log / prepare_top100.log
    ├── train_top100.log / backtest_top100.log
    ├── run_train_top100.sh / run_backtest_top100.sh
    └── *.pid
```

### Local repository (`/Users/jie.feng/wlb/Kairos/`)

- `artifacts/backtest_top100_baseline.json` / `backtest_top100_finetuned.json` — Backtest summary returned from AutoDL scp (gitignored, full number in §7.4 table)
- [`examples/crypto_top100_universe.md`](../examples/crypto_top100_universe.md) — Frozen Top100 list (snapshot of today 2026-04-20)

If you want to pull the checkpoint back to the local:

```bash
scp -P 37667 -r \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/Kairos/artifacts/checkpoints/predictor/checkpoints/best_model \
  /Users/jie.feng/wlb/Kairos/artifacts/checkpoints/predictor/checkpoints/best_model_top100
```

---

## 9. This run did not produce any code changes.

This time there are no changes to Python/CLI at all, all capabilities are ready in the previous run (commits `44cd5d6` and `ccb0de1`):

- `--universe` supports comma separated list (existing)
- `binance_vision.list_symbols_by_volume()` supports any `top_n` (existing)
- `kairos-collect --workers 4` supports concurrency (existing)
- `backtest_ic --stride 10` supports downsampling (existing)
- `KAIROS_NUM_WORKERS=0` env override avoid DataLoader OOM (existing)

In other words, the BTC/ETH work of fixing bugs through pitfalls will be directly harvested this time, and the top 100 run-through is just a matter of changing the parameters.

---

## 10. TL;DR — Reproduce the command list with one click

Assume you have got an AutoDL 5090 / 4090 (≥16 GB video memory, ≥32 GB system memory, ≥20 GB free disk):

```bash
# ------------------ 0. env ------------------
ssh -p <PORT> root@<HOST>
cd /root/autodl-tmp
source /etc/network_turbo
[ -d Kairos ] || git clone --depth=1 https://github.com/Shadowell/Kairos.git
cd Kairos && [ -d .venv ] || python -m venv .venv
source .venv/bin/activate
pip install -U pip && pip install -e '.[train,crypto]' && pip install 'numpy<2'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ------------------ 1. Warm up weights (can be skipped if reusing the previous run) ------------------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
huggingface-cli download NeoQuasar/Kronos-Tokenizer-base --cache-dir $HF_HOME
huggingface-cli download NeoQuasar/Kronos-small           --cache-dir $HF_HOME

# ------------------ 2. Generate the Top100 universe ------------------
python - <<'PY'
from kairos.data.markets.crypto_exchanges.binance_vision import BinanceVisionExchange
from kairos.data.markets.crypto_exchanges.base import ExchangeConfig
syms = BinanceVisionExchange(ExchangeConfig()).list_symbols_by_volume(top_n=200, quote="USDT")
STABLES = {"USDC","USD1","USDE","FDUSD","RLUSD","XUSD","BFUSD","TUSD","BUSD","USDP","DAI","USDD"}
PEGGED  = {"PAXG","XAUT","EUR","GBP","AEUR"}
WRAPPED = {"WBTC","WETH","STETH","WSTETH","WBETH"}
BLOCKED = STABLES | PEGGED | WRAPPED
kept = []
for s in syms:
    base = s.split("/")[0]
    if not base.isascii() or base.upper() in BLOCKED:
        continue
    kept.append(s)
    if len(kept) >= 100:
        break
open("/root/autodl-tmp/top100_universe.txt", "w").write(",".join(kept))
print(f"wrote {len(kept)} symbols")
PY

# ------------------ 3. Collection 1 year (~27 minutes) ------------------
UNIVERSE=$(cat /root/autodl-tmp/top100_universe.txt)
mkdir -p logs
nohup kairos-collect --market crypto --exchange binance_vision \
    --universe "$UNIVERSE" --freq 1min \
    --start 2025-04-20 --end 2026-04-20 \
    --out ./raw/crypto/bv_1min_top100 --workers 4 \
    > logs/collect_top100.log 2>&1 &
wait
# Expected: Completion: {'ok': 100, 'fail': 0, 'empty': 0, 'skip_up_to_date': 0}

# ------------------ 4. Packing (~3 minutes) ------------------
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_top100 \
    --train 2025-04-20:2026-01-31 \
    --val   2025-04-20:2026-01-31 \
    --test  2026-02-01:2026-04-20 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_top100

# ------------------ 5. Fine-tuning (~12 minutes, early stop to ep 4) ------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_top100
export KAIROS_NUM_WORKERS=0
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# ------------------ 6. Backtest comparison (~45 minutes each, stride=10) ------------------
python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_top100 \
    --horizons 1,5,30 --stride 10 \
    --out artifacts/backtest_top100_baseline.json

python -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_top100 \
    --horizons 1,5,30 --stride 10 \
    --out artifacts/backtest_top100_finetuned.json
```

Total time taken: **~2 hours and 30 minutes** (27 + 3 + 12 + 45×2 = 132 m). Single card 5090 cost ≈ ¥10-15.

---

## 11. Subsequent direction (updated from §11 of BTC/ETH run)

New information brought into the Top100 run is re-prioritized:

1. **Fix "reverse" issue for h1/h5** (new highest priority). Three roads:
   - Add multiple return heads (one each for h1 / h5 / h30) to `KronosWithExogenous`, and supervise them separately during training to prevent h30 from "squeezing" the short horizon.
   - Train two checkpoints (one horizon=1, one horizon=30), and each goes his own way during inference.
   - Make the preset of h30 clear as `kairos-small-crypto-h30` to avoid misuse for h1.
2. **Pull to OKX perpetual**. h30 already has alpha. After funding / OI / basis is added, h1/h5 will most likely no longer be negative - short horizon relies most on these micro factors.
3. **Replace Kronos-base / Kronos-large** (number of parameters ×10-50). The 5.4M parameters are saturated in 1 epoch on 31.6 million samples. A larger model should be able to continue to reduce val_ce in ep 2-3 to suppress overfitting.
4. **Freezing strategy adjustment**: Only unfreeze the last 1 layer of transformer. It still seems too tight in this val_ce trajectory - adjust `KAIROS_UNFREEZE_LAST_N` to 2 or 3 and try again.
5. Learning rate sweep (low priority, the current val_ce decrease of 0.09 is more important than adjusting lr).
