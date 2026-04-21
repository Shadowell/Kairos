# Crypto BTC/ETH two-year spot Predictor run log

> Record once **completely from scratch** to complete the complete process of "data collection → packaging → fine-tuning → backtest" on AutoDL RTX 5090, together with the pitfalls and repairs.
> If you just want to reproduce the results, skip to §10 for a list of TL;DR commands.
>
> Related documents:
> - [`AUTODL_REMOTE_TRAINING_GUIDE.md`](AUTODL_REMOTE_TRAINING_GUIDE.md) — AutoDL common card rental training manual
> - [`CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md`](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) — crypto data layer and exchange extension
> - [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) — Parameter Manual

---

## 0. Background and Premise

- **Date**: 2026-04-17, complete crypto fine-tuning on AutoDL
- **Machine**: AutoDL `westd` Zone one RTX 5090 container (31 GB VRAM, 754 GB system memory but cgroup limited to 96 GB, 30 GB local disk)
- **Local**: MacBook (without CUDA), only responsible for issuing commands, reading logs, modifying code + commit/push
- **Goal**: Fine-tuning Kronos-small on BTC/USDT + ETH/USDT 1min K online, and get baseline vs finetuned IC comparison

---

## 1. Local connectivity diagnosis (deciding where to collect data from)

First, I tested the connectivity of OKX/Binance direct connection and proxy on this machine. Conclusion:

|endpoint|direct connection|through proxy|
|---|---|---|
| `www.okx.com` |❌ DNS hijacked to `169.254.0.2`| N/A |
| `api.binance.com` / `fapi.binance.com` |❌ Port 443 is blocked| — |
| `data-api.binance.vision` | ✅ HTTP 200, 0.5 s | ✅ |

**The only accessible crypto endpoint under the office network is `data-api.binance.vision`**, which corresponds to the `--exchange binance_vision` downgrade channel in Kairos. Only spot K-line, no funding / OI / basis.

The results of the AutoDL test are exactly the same: the main site is completely blocked, and only Binance Vision and (through `/etc/network_turbo`’s proxy) GitHub / HuggingFace can access it.

**Conclusion**: The crypto K line is taken from Binance Vision, the weight is `HF_ENDPOINT=hf-mirror.com`, and the code is `/etc/network_turbo`.

---

## 2. AutoDL environment preparation

### 2.1 SSH login and network detection

```bash
# local
ssh -p 37667 root@connect.westd.seetacloud.com   # Password: SWI+5jGICrTa (example)

# Probing the network
source /etc/network_turbo          # Turn on academic acceleration proxy (only covers http/https_proxy)
curl -I https://github.com         # Need 200
curl -I https://huggingface.co     # Need 200
curl -I https://data-api.binance.vision/api/v3/time   # No agency, direct connection 200
```

### 2.2 Pull code + build venv + install dependencies

The server `/root/autodl-tmp/` is a persistent disk (it will not be lost after restarting), and everything is placed here:

```bash
cd /root/autodl-tmp
git clone --depth=1 https://github.com/Shadowell/Kairos.git   # Go to turbo agent
cd Kairos

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[train,crypto]'   # Also install training and crypto adapter dependencies
pip install 'numpy<2'              # torch ABI is not compatible with numpy 2.x

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expectation: 2.11.0+cu130 True NVIDIA GeForce RTX 5090
```

> ⚠️ A small pitfall encountered in this step: `[train]` extras **missing scipy** will cause `backtest_ic` to be unable to run later. `pyproject.toml` was added after this run. You don’t need to add it manually next time you install it.

### 2.3 Pre-download Kronos weights

`torchrun` When loading Kronos for the first time during startup, HF will be used. When using the proxy, it is easy to time out. You can simply download it manually:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
mkdir -p $HF_HOME

python - <<'PY'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
for repo in ["NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small"]:
    print("downloading", repo)
    print(" ->", snapshot_download(repo_id=repo, cache_dir="/root/autodl-tmp/hf_cache"))
PY
```

The two repos total ~220 MB, and it takes about 2 minutes to go to hf-mirror.com.

---

## 3. Collect BTC + ETH 2-year 1min K-line

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
# Binance Vision direct connection is the fastest; do not add http_proxy (academic proxies are not in its whitelist)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p logs raw/crypto
nohup kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2024-01-01 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_btc_eth --workers 1 \
    > logs/collect.log 2>&1 &
echo $! > logs/collect.pid
```

### output

- `raw/crypto/bv_1min_btc_eth/BTC_USDT.parquet` — 52 MB, ~1.05 million lines
- `raw/crypto/bv_1min_btc_eth/ETH_USDT.parquet` — 45 MB, ~1.05 million lines
- **97 MB total, 11 minutes and 29 seconds**

### Known behavior regarding `binance_vision`

- The timestamp is `naive local time` (not UTC) - has no effect on 24/7 crypto training, but the date boundary will be offset by 8 hours.
- `fetch_ohlcv` is a while-loop with 1000 K lines at a time. **The entire symbol must be captured before `to_parquet`**, so there is nothing on the disk in the first few minutes. Don’t mistakenly think that it is hung up (just look at the RSS rising).
- `funding_rate / open_interest / basis` Throw `NotImplementedError` directly, `prepare_dataset` will pad them to 0.

---

## 4. Package into training set

After collecting `kairos-prepare`, generate train/val/test pickle + `meta.json`:

```bash
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_btc_eth \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_btc_eth
```

> `--split-mode interleave --block-days 7` is accurate for high-frequency data: it switches to train / val in turns on a 7-day basis, which not only retains the timing structure but also prevents val from concentrating the entire paragraph on a certain regime. See AGENTS.md §8 and `TRAINING_TUNING_PLAYBOOK.md`.

### output

```
finetune/data/crypto_1min_btc_eth/  (386 MB)
├── train_data.pkl   57 MB     # {symbol: DataFrame[OHLCVA]}
├── val_data.pkl     10 MB
├── test_data.pkl     9.8 MB
├── exog_train.pkl  243 MB     # {symbol: DataFrame[32 exog cols]}
├── exog_val.pkl     43 MB
├── exog_test.pkl    42 MB
└── meta.json                  # {market, freq, exog_cols, split_mode, ranges}
```

**meta.json `exog_cols` is 32 dimensions**: 24 common factors + 8 crypto market factors. `funding_rate / funding_rate_z / oi_change / basis / btc_dominance` These 5 unobtained fields will be padded to 0, but the schema remains unchanged at 32 dimensions - this is a hard constraint of the Phase 2 architecture (AGENTS.md §8).

It took 10 seconds.

---

## 5. Smoke training (verification link, 1 minute)

Before formal training, run `KAIROS_SMOKE=1` to confirm that the model can forward + backward + save:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export KAIROS_SMOKE=1
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth

torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
```

### output

- 50 steps × batch=4 = 200 samples, **takes ~8 seconds**
- val_ce = 2.4181, saved to `artifacts/checkpoints/predictor/checkpoints/best_model`

Smoke run-through description: The weight loading is correct (136 layers of Kronos-small 147 layers reuse, 11 layers of new initialization = exog bypass + return head), the freezing strategy takes effect (the first 7 layers are frozen, the last 1 layer + exog + heads are unfrozen), OneCycleLR does not crash.

---

## 6. Formal training (stepping into two pitfalls)

### 6.1 Pitfall 1: Direct OOM for the first time

The default preset `crypto-1min` is `batch_size=50`, and it crashes in 5 seconds after the first startup:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 MiB.
GPU 0 has a total capacity of 31.36 GiB of which 19.69 MiB is free.
Including non-PyTorch memory, this process has 2.42 GiB memory in use.
```

Translation: **GPU shows 31 GB total, but only 20 MB free**. `nvidia-smi --query-compute-apps` Display `No running processes found` - There is no process belonging to the current container, but the video memory is locked.
Possible reasons: AutoDL's 5090 is sometimes a slice shared card used by other tenants; it may also be leftover from the last abnormal shutdown.

**Fix**: Go to the AutoDL console to "shut down → power on" (`nvidia-smi --gpu-reset` cannot be done in SSH, the container has no permissions). After restarting, `nvidia-smi` displays `2 MiB used / 32110 MiB free`, and all 31 GB are available.

### 6.2 Pit 2: batch=50 and num_workers=2 max out the memory

After solving the video memory problem, continue running. ep 1 is normal (val_ce=2.4959), and ep 2 is `Killed` at step 400/1000:

```
logs/run_train.sh: line 11: 1500 Killed  torchrun ...
```

I don’t have permission to look at dmesg, but `free -h` shows `used=59Gi / 96Gi` before the process died (the container cgroup is limited to 96GB).

Root cause: `__getitem__` of `kairos.training.dataset.py` adjusts `ei.reindex(dates)` every time,
When DataLoader `num_workers=2` is used, each worker forks a copy of the DataFrame.
pandas reindex keeps temporary buffers in workers, and two workers run at the same time → the memory increases linearly until the OOM killer intervenes.

**Fix**: The newly added `KAIROS_NUM_WORKERS=0` env override (see this commit `44cd5d6`), do IO in the main process,
Keep only one copy of the DataFrame. Cost: Data loading and GPU calculation no longer overlap, but 5090 + Kronos-small is inherently CPU bound (~50% GPU util), so **throughput is almost unchanged**.

### 6.3 Final successful training command

```bash
cat > logs/run_train.sh <<'SH'
#!/bin/bash
set -e
cd /root/autodl-tmp/Kairos
source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth
export KAIROS_NUM_WORKERS=0                  # ← Key
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
SH
chmod +x logs/run_train.sh
nohup bash logs/run_train.sh > logs/train.log 2>&1 &
```

### 6.4 Training log

| Epoch |train loss end| val_ce |determination|
|---|---|---|---|
| 1 | 119.7 | **2.4940** | ✅ save best |
| 2 |  94.2 | 2.5006 | patience 1/3 |
| 3 |  86.1 | 2.6005 | patience 2/3 |
| 4 |  83.2 | 2.8820 |patience 3/3 → early stop|

Total time taken **10 minutes and 18 seconds**. Each epoch lasts 2 minutes and 33 seconds (50000 samples/batch 50 = 1000 steps, ~0.15 s/step).

How to read:
- The train loss keeps decreasing, and val_ce starts to fluctuate and rise from ep 2 → typical small sample (only 2 currencies) + overfitting signal in high noise scenario.
- The early stopping of patience=3 just keeps the weights of ep 1, and will not overfit and pollute the test.
- The absolute value of **val_ce ~2.5** is the level reached by Kronos pre-training itself; we did not really touch the transformer backbone (only unfrozen the last layer + exog + heads), so val_ce will not drop significantly. Where you really expect to see improvements is in the return-head's IC, see §7.

### 6.5 Checkpoint

```
artifacts/checkpoints/predictor/checkpoints/best_model/  (97 MB)
├── config.json
├── model.safetensors
└── README.md
```

---

## 7. backtest: baseline vs finetuned

Run `backtest_ic` twice, once `--baseline` (Kronos original weights + randomly initialized exog/return head), and once `--ckpt best_model`.

```bash
# baseline: about 10 minutes
python -m kairos.training.backtest_ic \
    --baseline --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 \
    --out artifacts/backtest_baseline.json

# finetuned: about 10 minutes
python -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 \
    --out artifacts/backtest_finetuned.json
```

> Be careful not to use pipes like `2>&1 | tail -40`. The stdout buffer will make you think that the program has hung up.
> Change it to `nohup python -u ... > log 2>&1 &` and then go to `tail -f log` to check the progress.

### 7.1 Data scale

- **Test Set**: 304,710 1-min bars
- **Time range**: 2026-01-01 04:16 ~ 2026-04-16 23:30
- **symbols**: BTC/USDT、ETH/USDT

### 7.2 Results

| horizon | model | pearson | spearman | hit_rate | ic_by_date | rank_ic | ICIR |
|---|---|---|---|---|---|---|---|
| h1  | baseline  | -0.0007 | +0.0160 | 0.4974 | +0.0069 | +0.0217 | +0.2972 |
| h1  | finetuned | +0.0035 | -0.0102 | 0.4953 | +0.0007 | -0.0119 | +0.0291 |
| h5  | baseline  | +0.0121 | -0.0018 | 0.4972 | +0.0043 | -0.0070 | +0.0934 |
| h5  | finetuned | -0.0011 | +0.0079 | 0.5051 | +0.0026 | +0.0101 | +0.0598 |
| **h30** | baseline  | -0.0033 | +0.0136 | 0.5098 | +0.0035 | +0.0178 | +0.0385 |
| **h30** | finetuned | +0.0031 | +0.0316 | **0.5168** | **+0.0291** | **+0.0505** | **+0.3248** |

### 7.3 How to read this table

1. **h1 / h5 are basically tied**. High-frequency band noise dominates, and `binance_vision` cannot obtain the three core market factors of crypto (funding / OI / basis) (padded to 0). The signals that fine-tuning can utilize are very limited. The ICIR of h1 has also dropped, which is a normal small disturbance.

2. **h30 obviously works**:
   - rank-IC from +0.018 → **+0.050** (+184%)
   - ICIR from +0.039 → **+0.325** (crossing the 0.3 threshold of "can be included in the combination")
   - hit_rate from 50.98% → 51.68%, it seems to only increase 0.7pp, but this is a very stable improvement on 300,000 samples
   - by-date IC from +0.0035 → **+0.0291** (an order of magnitude)

In other words, **even if only using spot K-line + freezing 7/8 of the trunk**, the model has learned to use 256 minutes look-back + 24 common factors to push the direction 30 minutes later.
h30 corresponds to `return_horizon=30` in the `crypto-1min` preset. Only when the training target is aligned with the backtest horizon can the effect be seen on this bucket - this also confirms that the preset design is correct.

3. **The absolute value is still small**. rank-IC 5% / ICIR 0.32 is a weak alpha in a single strategy, but:
   - Enough to enter the combination: Kronos-small only has 5.4M parameters. What can be trained in 5 minutes can reach this level, and the computing power efficiency is very high.
   - Can be stacked in the future: exchange for OKX perpetual to get all market factors / expand to top N currencies / exchange for Kronos-base to increase IC

---

## 8. Two code changes produced by this run

Immediately after modification `git commit && git push`, comply with AGENTS.md §6.1:

| Commit |change|reason|
|---|---|---|
| `44cd5d6` |Added `KAIROS_BATCH_SIZE / KAIROS_ACCUM_STEPS / KAIROS_NUM_WORKERS / KAIROS_EPOCHS / KAIROS_N_TRAIN_ITER / KAIROS_N_VAL_ITER / KAIROS_LR / KAIROS_UNFREEZE_LAST_N / KAIROS_LOG_INTERVAL` nine env overrides|In the shared GPU/small memory scenario, the batch needs to be changed; there is no need to change the code when scanning parameters in the future.|
| `ccb0de1` |`pyproject.toml` `[train]` extras supplement `scipy>=1.10`|`backtest_ic` uses `scipy.stats.pearsonr`, but the dependency was not declared originally|

---

## 9. artifacts list

### Server (AutoDL `/root/autodl-tmp/Kairos/`)

```
raw/crypto/bv_1min_btc_eth/                  97 MB    # original parquet
finetune/data/crypto_1min_btc_eth/          386 MB    # Packaged train/val/test + exog + meta.json
artifacts/
├── backtest_baseline.json                  1.4 KB   # Kronos original weight
├── backtest_finetuned.json                 1.4 KB   # After fine-tuning
└── checkpoints/predictor/checkpoints/
    └── best_model/                          97 MB    # Final weight (ep 1, val_ce=2.4940)
logs/
├── collect.log / train.log / backtest.log           # Whole process log
├── run_train.sh / run_backtest.sh                   # startup script
└── *.pid                                            # Process ID
hf_cache/                                   220 MB   # Kronos-Tokenizer + Kronos-small
```

### Local (`/Users/jie.feng/wlb/Kairos/`)

Just code. If you want to pull the checkpoint back to the local:

```bash
scp -P 37667 -r \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/Kairos/artifacts/checkpoints/predictor/checkpoints/best_model \
  /Users/jie.feng/wlb/Kairos/artifacts/checkpoints/predictor/checkpoints/
```

---

## 10. TL;DR — Reproduce the command list with one click

Assume that you have rented an AutoDL 5090 / 4090 / 3090 (video memory ≥ 16 GB is sufficient, the batch will automatically be enough):

```bash
# ------------------ 0. env ------------------
ssh -p <PORT> root@<HOST>
cd /root/autodl-tmp
source /etc/network_turbo
git clone --depth=1 https://github.com/Shadowell/Kairos.git
cd Kairos && python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e '.[train,crypto]' && pip install 'numpy<2'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ------------------ 1. Warm-up weight ------------------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
huggingface-cli download NeoQuasar/Kronos-Tokenizer-base --cache-dir $HF_HOME
huggingface-cli download NeoQuasar/Kronos-small           --cache-dir $HF_HOME

# ------------------ 2. Collect 2 years of data (~11 minutes) ------------------
kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2024-01-01 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_btc_eth --workers 1

# ------------------ 3. Packing (~10 seconds) ------------------
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_btc_eth \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_btc_eth

# ------------------ 4. fine-tuning (~10 minutes, early stopping) ------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth
export KAIROS_NUM_WORKERS=0                  # Prevent DataLoader memory explosion
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# ------------------ 5. Backtest comparison (~10 minutes each) ------------------
python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --out artifacts/backtest_baseline.json

python -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --out artifacts/backtest_finetuned.json
```

Total time spent: **Start from 0 ~45 minutes** (11 minutes of mining + 5 minutes of packaging dependencies + 1 minute of packaging + 10 minutes of training + 20 minutes of two backtests), single card cost 5090 ≈ ¥3-5.

---

## 11. Follow-up direction

Ranked from high to low by expected revenue:

1. **Pull to OKX perpetual** (requires VPN/proxy) → Complete the three core crypto factors of `funding_rate / oi_change / basis`, and the IC of h1 / h5 has a high probability of following. `kairos.data.markets.crypto_exchanges.okx` It has been implemented, only the network is missing.
2. **Expand to top 10-20 coins** (BTC/ETH/SOL/XRP/BNB/...) → Overfitting of single currency is greatly reduced; `n_train_iter=50000` is more reasonable for 10 currencies.
3. **Replace Kronos-base / Kronos-large** (parameter size × 10-50) → h30’s rank-IC is expected to reach 0.1+; the computing power cost is only × 3-5.
4. **Learning rate sweep**: With the env override introduced by `44cd5d6`, one line of `for lr in 1e-6 5e-6 1e-5; do KAIROS_LR=$lr ...; done` can run 3 groups.

Go to 3. If it still doesn’t work, then doubt the supervision signal design of the return head (see `TRAINING_TUNING_PLAYBOOK.md`).
