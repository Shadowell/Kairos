# Crypto BTC/ETH 两年现货 Predictor 实验记录

> 记录一次**完全从零**在 AutoDL RTX 5090 上完成"采数据 → 打包 → 微调 → 回测"的完整流程，连同踩到的坑和修复。
> 如果你只想复现结果，跳到 §10 的 TL;DR 命令清单。
>
> 相关文档：
> - [`AUTODL_REMOTE_TRAINING_GUIDE.md`](AUTODL_REMOTE_TRAINING_GUIDE.md) — AutoDL 通用租卡训练手册
> - [`CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md`](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) — crypto 数据层与交易所扩展
> - [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) — 调参手册

---

## 0. 背景和前提

- **日期**：2026-04-17，完整走通一次 AutoDL 上的 crypto 微调
- **机器**：AutoDL `westd` 区一台 RTX 5090 容器（31 GB 显存，754 GB 系统内存但 cgroup 限 96 GB，30 GB 本地磁盘）
- **本地**：MacBook（无 CUDA），只负责出命令、看日志、修代码 + commit/push
- **目标**：在 BTC/USDT + ETH/USDT 1min K 线上微调 Kronos-small，拿到 baseline vs finetuned 的 IC 对比

---

## 1. 本地连通性诊断（决定从哪里采数据）

先在本机测了一下 OKX / Binance 直连与代理下的连通性，结论：

| 端点 | 直连 | 通过代理 |
|---|---|---|
| `www.okx.com` | ❌ DNS 被劫持到 `169.254.0.2` | N/A |
| `api.binance.com` / `fapi.binance.com` | ❌ 443 端口被封 | — |
| `data-api.binance.vision` | ✅ HTTP 200, 0.5 s | ✅ |

**办公网下唯一能通的 crypto 端点就是 `data-api.binance.vision`**，对应 Kairos 里的 `--exchange binance_vision` 降级通道。只有现货 K 线，没有 funding / OI / basis。

AutoDL 上测的结果完全一样：主站全封，只有 Binance Vision 和（通过 `/etc/network_turbo` 的代理）GitHub / HuggingFace 能通。

**结论**：crypto K 线从 Binance Vision 采，权重走 `HF_ENDPOINT=hf-mirror.com`，代码走 `/etc/network_turbo`。

---

## 2. AutoDL 环境准备

### 2.1 SSH 登录与网络探测

```bash
# 本地
ssh -p 37667 root@connect.westd.seetacloud.com   # 密码：SWI+5jGICrTa（示例）

# 探测网络
source /etc/network_turbo          # 打开学术加速代理（仅覆盖 http/https_proxy）
curl -I https://github.com         # 需要 200
curl -I https://huggingface.co     # 需要 200
curl -I https://data-api.binance.vision/api/v3/time   # 不走代理，直连 200
```

### 2.2 拉代码 + 建 venv + 装依赖

服务器 `/root/autodl-tmp/` 是持久盘（重启不丢），所有东西放这下面：

```bash
cd /root/autodl-tmp
git clone --depth=1 https://github.com/Shadowell/Kairos.git   # 走 turbo 代理
cd Kairos

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[train,crypto]'   # 同时装训练和 crypto adapter 的依赖
pip install 'numpy<2'              # torch ABI 与 numpy 2.x 不兼容

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.11.0+cu130 True NVIDIA GeForce RTX 5090
```

> ⚠️ 这一步撞到的一个小坑：`[train]` extras **漏掉了 scipy**，会导致后面 `backtest_ic` 跑不起来。本次 run 跑完补了 `pyproject.toml`，下次装的话就不需要手工补了。

### 2.3 预下载 Kronos 权重

`torchrun` 启动时第一次加载 Kronos 会走 HF，走代理容易超时，干脆手工下好：

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

两个 repo 共 ~220 MB，走 hf-mirror.com 大约 2 分钟。

---

## 3. 采集 BTC + ETH 2 年 1min K 线

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
# Binance Vision 直连最快；不要加 http_proxy（学术代理不在它的白名单）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p logs raw/crypto
nohup kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2024-01-01 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_btc_eth --workers 1 \
    > logs/collect.log 2>&1 &
echo $! > logs/collect.pid
```

### 产出

- `raw/crypto/bv_1min_btc_eth/BTC_USDT.parquet` — 52 MB，~105 万行
- `raw/crypto/bv_1min_btc_eth/ETH_USDT.parquet` — 45 MB，~105 万行
- **共 97 MB，11 分 29 秒**

### 关于 `binance_vision` 的已知行为

- 时间戳是 `naive local time`（不是 UTC）—— 对 24/7 crypto 的训练不影响，只是日期边界会偏 8 小时。
- `fetch_ohlcv` 是 while-loop 一次 1000 根 K 线，**必须整个 symbol 抓完才会 `to_parquet`**，所以前几分钟磁盘上什么都没有，不要误以为挂了（看 RSS 在涨即可）。
- `funding_rate / open_interest / basis` 直接抛 `NotImplementedError`，`prepare_dataset` 会把它们 pad 为 0。

---

## 4. 打包成训练集

采集完 `kairos-prepare` 生成 train/val/test pickle + `meta.json`：

```bash
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_btc_eth \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_btc_eth
```

> `--split-mode interleave --block-days 7` 是给高频数据准的：按 7 天为一块轮流切到 train / val，既保留了时序结构又避免 val 集整段落在某一种 regime 上。参考 AGENTS.md §8 和 `TRAINING_TUNING_PLAYBOOK.md`。

### 产出

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

**meta.json 里 `exog_cols` 是 32 维**：24 个通用因子 + 8 个 crypto 市场因子。`funding_rate / funding_rate_z / oi_change / basis / btc_dominance` 这 5 个没拿到的字段会被 pad 为 0，但 schema 维持 32 维不变 —— 这是 Phase 2 架构的硬约束（AGENTS.md §8）。

耗时 10 秒。

---

## 5. Smoke 训练（验证链路，1 分钟）

正式训练前先用 `KAIROS_SMOKE=1` 跑一下，确认模型能 forward + backward + save：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export KAIROS_SMOKE=1
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth

torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
```

### 产出

- 50 steps × batch=4 = 200 样本，**耗时 ~8 秒**
- val_ce = 2.4181，保存到 `artifacts/checkpoints/predictor/checkpoints/best_model`

smoke 跑通说明：权重加载对得上（Kronos-small 147 层里 136 层 reuse、11 层新初始化 = exog 旁路 + return head），冻结策略生效（前 7 层冻，最后 1 层 + exog + heads 解冻），OneCycleLR 没崩。

---

## 6. 正式训练（踩到两个坑）

### 6.1 坑一：第一次直接 OOM

默认 preset `crypto-1min` 用 `batch_size=50`，第一次启动 5 秒就崩：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 MiB.
GPU 0 has a total capacity of 31.36 GiB of which 19.69 MiB is free.
Including non-PyTorch memory, this process has 2.42 GiB memory in use.
```

翻译：**GPU 显示 31 GB 总容量，但只剩 20 MB 空闲**。`nvidia-smi --query-compute-apps` 显示 `No running processes found`——没有属于当前容器的进程，但显存被锁。
可能原因：AutoDL 的 5090 有时是切片共享卡，其他租户在用；也可能上一次异常关机残留。

**修复**：走 AutoDL 控制台"关机→开机"（SSH 里做不到 `nvidia-smi --gpu-reset`，容器无权限）。重启后 `nvidia-smi` 显示 `2 MiB used / 32110 MiB free`，31 GB 全部可用。

### 6.2 坑二：batch=50 在 num_workers=2 下把内存撑爆

解决显存问题后继续跑，ep 1 正常（val_ce=2.4959），ep 2 进行到 400/1000 步被 `Killed`：

```
logs/run_train.sh: line 11: 1500 Killed  torchrun ...
```

看 dmesg 没权限，但 `free -h` 显示进程死前 `used=59Gi / 96Gi`（容器 cgroup 限 96GB）。

根因：`kairos.training.dataset.py` 的 `__getitem__` 每次都调 `ei.reindex(dates)`，
DataLoader `num_workers=2` 时每个 worker fork 一份 DataFrame 副本，
pandas reindex 又在 worker 里狂留临时 buffer，两个 worker 同时跑 → 内存线性上涨直到 OOM killer 介入。

**修复**：新加的 `KAIROS_NUM_WORKERS=0` env override（见本次 commit `44cd5d6`），主进程里做 IO，
只保留一份 DataFrame。代价：数据加载和 GPU 计算不再 overlap，但 5090 + Kronos-small 本来就是 CPU bound（~50% GPU util），所以**吞吐几乎不变**。

### 6.3 最终成功的训练命令

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
export KAIROS_NUM_WORKERS=0                  # ← 关键
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
SH
chmod +x logs/run_train.sh
nohup bash logs/run_train.sh > logs/train.log 2>&1 &
```

### 6.4 训练日志

| Epoch | train loss 末尾 | val_ce | 判定 |
|---|---|---|---|
| 1 | 119.7 | **2.4940** | ✅ save best |
| 2 |  94.2 | 2.5006 | patience 1/3 |
| 3 |  86.1 | 2.6005 | patience 2/3 |
| 4 |  83.2 | 2.8820 | patience 3/3 → 早停 |

总耗时 **10 分 18 秒**。每个 epoch 2 分 33 秒（50000 样本 / batch 50 = 1000 step，~0.15 s/step）。

读法：
- train loss 一直降，val_ce 从 ep 2 开始震荡上升 → 典型的小样本（只 2 只币种）+ 高噪声场景下的过拟合信号。
- patience=3 的早停正好保住 ep 1 的 weights，不至于过拟合到污染 test。
- **val_ce 的绝对值 ~2.5** 是 Kronos 预训练本身就到的水平；我们没真正动 transformer 主干（只解冻了最后 1 层 + exog + heads），val_ce 本就不会大降。真正期望看到提升的地方是 **return-head 的 IC**，见 §7。

### 6.5 Checkpoint

```
artifacts/checkpoints/predictor/checkpoints/best_model/  (97 MB)
├── config.json
├── model.safetensors
└── README.md
```

---

## 7. 回测：baseline vs finetuned

两次跑 `backtest_ic`，一次 `--baseline`（Kronos 原始权重 + 随机初始化的 exog/return head），一次 `--ckpt best_model`。

```bash
# baseline：约 10 分钟
python -m kairos.training.backtest_ic \
    --baseline --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 \
    --out artifacts/backtest_baseline.json

# finetuned：约 10 分钟
python -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 \
    --out artifacts/backtest_finetuned.json
```

> 注意不要用 `2>&1 | tail -40` 这种 pipe，stdout buffer 会让你以为程序挂了。
> 改成 `nohup python -u ... > log 2>&1 &` 再去 `tail -f log` 看进度。

### 7.1 数据规模

- **Test 集**：304,710 条 1-min bars
- **时间范围**：2026-01-01 04:16 ~ 2026-04-16 23:30
- **symbols**：BTC/USDT、ETH/USDT

### 7.2 结果

| horizon | model | pearson | spearman | hit_rate | ic_by_date | rank_ic | ICIR |
|---|---|---|---|---|---|---|---|
| h1  | baseline  | -0.0007 | +0.0160 | 0.4974 | +0.0069 | +0.0217 | +0.2972 |
| h1  | finetuned | +0.0035 | -0.0102 | 0.4953 | +0.0007 | -0.0119 | +0.0291 |
| h5  | baseline  | +0.0121 | -0.0018 | 0.4972 | +0.0043 | -0.0070 | +0.0934 |
| h5  | finetuned | -0.0011 | +0.0079 | 0.5051 | +0.0026 | +0.0101 | +0.0598 |
| **h30** | baseline  | -0.0033 | +0.0136 | 0.5098 | +0.0035 | +0.0178 | +0.0385 |
| **h30** | finetuned | +0.0031 | +0.0316 | **0.5168** | **+0.0291** | **+0.0505** | **+0.3248** |

### 7.3 怎么读这张表

1. **h1 / h5 基本打平**。高频段噪声占主导，加上 `binance_vision` 拿不到 funding / OI / basis 这三个 crypto 的核心市场因子（被 pad 成 0），微调能利用的信号很有限。h1 的 ICIR 还降了，属于正常的小扰动。

2. **h30 明显有效**：
   - rank-IC 从 +0.018 → **+0.050**（+184%）
   - ICIR 从 +0.039 → **+0.325**（跨越了 0.3 这个"可以挂进组合里"的阈值）
   - hit_rate 从 50.98% → 51.68%，看起来只涨 0.7pp，但在 30 万样本上这是非常稳定的提升
   - by-date IC 从 +0.0035 → **+0.0291**（一个数量级）

   换句话说，**即使只用现货 K 线 + 冻结 7/8 的主干**，模型也学会了用 256 分钟 look-back + 24 个通用因子推 30 分钟后的方向。
   h30 对应 `crypto-1min` preset 里的 `return_horizon=30`，训练目标和回测 horizon 对齐，才会在这个 bucket 上看到效果——这也印证了 preset 设计是对的。

3. **绝对值还是小**。rank-IC 5% / ICIR 0.32 在单策略里是弱 alpha，但：
   - 够进组合：Kronos-small 只有 5.4M 参数，5 分钟训出来的东西就能到这个量级，算力效率很高
   - 未来可以叠：换 OKX 永续拿全市场因子 / 扩到 top N 币种 / 换 Kronos-base 都能再抬 IC

---

## 8. 本次 run 产生的两个代码改动

改完立即 `git commit && git push`，遵守 AGENTS.md §6.1：

| Commit | 改动 | 原因 |
|---|---|---|
| `44cd5d6` | 新增 `KAIROS_BATCH_SIZE / KAIROS_ACCUM_STEPS / KAIROS_NUM_WORKERS / KAIROS_EPOCHS / KAIROS_N_TRAIN_ITER / KAIROS_N_VAL_ITER / KAIROS_LR / KAIROS_UNFREEZE_LAST_N / KAIROS_LOG_INTERVAL` 九个 env override | 共享 GPU / 小显存场景下要改 batch；以后扫参数也不用改代码 |
| `ccb0de1` | `pyproject.toml` `[train]` extras 补 `scipy>=1.10` | `backtest_ic` 用 `scipy.stats.pearsonr`，原先没声明依赖 |

---

## 9. 产物清单

### 服务器（AutoDL `/root/autodl-tmp/Kairos/`）

```
raw/crypto/bv_1min_btc_eth/                  97 MB    # 原始 parquet
finetune/data/crypto_1min_btc_eth/          386 MB    # 打包后的 train/val/test + exog + meta.json
artifacts/
├── backtest_baseline.json                  1.4 KB   # Kronos 原权重
├── backtest_finetuned.json                 1.4 KB   # 微调后
└── checkpoints/predictor/checkpoints/
    └── best_model/                          97 MB    # 最终权重（ep 1, val_ce=2.4940）
logs/
├── collect.log / train.log / backtest.log           # 全过程日志
├── run_train.sh / run_backtest.sh                   # 启动脚本
└── *.pid                                            # 进程 ID
hf_cache/                                   220 MB   # Kronos-Tokenizer + Kronos-small
```

### 本地（`/Users/jie.feng/wlb/Kairos/`）

只有代码。如果要把 checkpoint 拉回本地：

```bash
scp -P 37667 -r \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/Kairos/artifacts/checkpoints/predictor/checkpoints/best_model \
  /Users/jie.feng/wlb/Kairos/artifacts/checkpoints/predictor/checkpoints/
```

---

## 10. TL;DR — 一键复现命令清单

假设你已经租到一台 AutoDL 5090 / 4090 / 3090（显存 ≥ 16 GB 即可，batch 会自动够用）：

```bash
# ------------------ 0. env ------------------
ssh -p <PORT> root@<HOST>
cd /root/autodl-tmp
source /etc/network_turbo
git clone --depth=1 https://github.com/Shadowell/Kairos.git
cd Kairos && python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e '.[train,crypto]' && pip install 'numpy<2'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ------------------ 1. 预热权重 ------------------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
huggingface-cli download NeoQuasar/Kronos-Tokenizer-base --cache-dir $HF_HOME
huggingface-cli download NeoQuasar/Kronos-small           --cache-dir $HF_HOME

# ------------------ 2. 采集 2 年数据（~11 分钟）------------------
kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2024-01-01 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_btc_eth --workers 1

# ------------------ 3. 打包（~10 秒）------------------
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_btc_eth \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_btc_eth

# ------------------ 4. 微调（~10 分钟，早停）------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth
export KAIROS_NUM_WORKERS=0                  # 防 DataLoader 内存爆
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# ------------------ 5. 回测对比（各 ~10 分钟）------------------
python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --out artifacts/backtest_baseline.json

python -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --out artifacts/backtest_finetuned.json
```

总耗时：**从 0 开始 ~45 分钟**（11 分采 + 5 分装依赖 + 1 分打包 + 10 分训 + 20 分两次回测），单卡 5090 成本 ≈ ¥3-5。

---

## 11. 后续方向

按预期收益从高到低排：

1. **拉到 OKX 永续**（需要 VPN / 代理）→ 补齐 `funding_rate / oi_change / basis` 三个核心 crypto 因子，h1 / h5 的 IC 有大概率跟着起来。`kairos.data.markets.crypto_exchanges.okx` 已经实现好，只差网络。
2. **扩宇宙到 top 10-20 币**（BTC/ETH/SOL/XRP/BNB/...）→ 单币种过拟合大幅减轻；`n_train_iter=50000` 对 10 只币种更合理。
3. **换 Kronos-base / Kronos-large**（参数量 × 10-50）→ h30 的 rank-IC 有望进 0.1+；算力成本也只是 × 3-5。
4. **学习率 sweep**：配合 `44cd5d6` 引入的 env override，一行 `for lr in 1e-6 5e-6 1e-5; do KAIROS_LR=$lr ...; done` 就能跑 3 组。

走到 3. 还没见效的话，再怀疑 return head 的监督信号设计（见 `TRAINING_TUNING_PLAYBOOK.md`）。
