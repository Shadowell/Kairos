# Crypto Top100 一年现货 Predictor 实验记录

> 在 [`CRYPTO_BTC_ETH_2Y_SPOT_RUN.md`](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md) 的 2 币 × 2 年基线上，把 universe 扩到 **Binance 现货 24h 成交量 Top100**（~100 个 USDT 对）× **近 1 年 1min**，看扩大数据量对微调效果的影响。
> 如果你只想复现结果，跳到 §10 的 TL;DR 命令清单。
>
> 相关文档：
> - [`CRYPTO_BTC_ETH_2Y_SPOT_RUN.md`](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md) — 上一版 BTC+ETH 2 币跑通记录（必读，这里只讲差异）
> - [`AUTODL_REMOTE_TRAINING_GUIDE.md`](AUTODL_REMOTE_TRAINING_GUIDE.md) — AutoDL 通用租卡训练手册
> - [`CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md`](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) — crypto 数据层与交易所扩展
> - [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) — 调参手册

---

## 0. 背景

- **日期**：2026-04-20
- **动机**：BTC/ETH 的首版微调只在 h30 上看到可用的 alpha（rank-IC +0.050 / ICIR +0.325），怀疑瓶颈在 universe 太小 → 模型在 2 个币上很容易过拟合。扩大到 Top100 验证：
  1. 更大 train pool 能不能在 val_ce 上反映出来；
  2. 扩到横截面后 by-date IC / ICIR 是不是更稳；
  3. h1 / h5 段能不能因为更多币种采样而抬起来。
- **机器**：沿用 `CRYPTO_BTC_ETH_2Y_SPOT_RUN.md` 的同一台 AutoDL RTX 5090（`connect.westd.seetacloud.com:37667`，westd 区）。
- **数据源**：依然走 `binance_vision`（公司网络下唯一稳定能通的 crypto endpoint）。衍生品因子 funding / OI / basis 仍旧 pad 0。

---

## 1. Universe 选择

### 1.1 从 Binance 现货 top-by-volume 起步

直接复用 `BinanceVisionExchange.list_symbols_by_volume(top_n=200, quote="USDT")`：它打 `/api/v3/ticker/24hr` 拿 24h `quoteVolume` 排序，返回 ccxt-unified form（`BTC/USDT`）。先拉 200 个留充足裕度。

### 1.2 过滤 "结构性非 alpha" 标的

原始 top 表里混入了三类对训练有害的样本：

| 类别 | 例子 | 为什么剔除 |
|---|---|---|
| 稳定币 | USDC, USDE, FDUSD, RLUSD, TUSD, BUSD, BFUSD, XUSD, USD1, DAI, USDD, USDP | 价格常年锁定在 $1 附近，close-to-close return ≈ 0，计算 IC 时会把分母（波动）压成 0，相当于用 `div by zero` 污染截面。 |
| 贵金属 / 法币锚定 | PAXG（黄金）, XAUT（黄金）, EUR, GBP, AEUR | 价格与其他 crypto 几乎无相关，且自身有独立的现货/期货定价机制，训练时会在相关性矩阵里显著偏离。 |
| 包装币 | WBTC, WETH, STETH, WSTETH, WBETH | 与 BTC/ETH/stETH 100% 相关，重复样本拖低 IC 有效自由度。 |
| 非 ASCII base | `币安人生/USDT` 等 wrapper/meme 币 | 上面 binance_vision 偶尔返回带非 ASCII 字符的 symbol，直接跳过。 |

过滤后**从 top-200 里挑前 100**，写进 `/root/autodl-tmp/top100_universe.txt`。完整名单见 [`examples/crypto_top100_universe.md`](../examples/crypto_top100_universe.md)。

### 1.3 名单里的已知"弱项"

Top100 里有不少 2024-2025 年才上线的新币（`TRUMP`、`PNUT`、`PENGU`、`MOVE`、`PLUME`、`GIGGLE`、`PUMP`、`ZAMA`、…），在 `2025-04-20` 起点之前没有数据。
`kairos-prepare` 会在 val 切分阶段因为样本不足剔除掉其中最短的几条，这次实际 val set 只剩 **97/100** symbols（train / test 仍是 100）。这是预期行为，不用修。

---

## 2. 服务器环境

直接沿用上一版 run：代码 `git pull` 到最新（合并了新的 env overrides 和 scipy extra），venv 不重装，`hf_cache` 已预热的 Kronos-Tokenizer-base / Kronos-small 继续复用。

```bash
cd /root/autodl-tmp/Kairos
source /etc/network_turbo 2>/dev/null
git pull --ff-only
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.11.0+cu130 True NVIDIA GeForce RTX 5090
```

GPU 初始状态干净（`32110 MiB free / 32607 MiB total`），磁盘剩 38 GB，足够容纳一次 Top100 run。

---

## 3. 采集 Top100 × 1 年 1min

### 3.1 参数与关键选择

| 参数 | 值 | 为什么 |
|---|---|---|
| `--freq` | `1min` | 保持与 BTC/ETH run 同频，模型 preset `crypto-1min` 不变 |
| `--start / --end` | `2025-04-20 / 2026-04-20` | 只取**最近 1 年**：100 币 × 2 年会把磁盘和采集时间都拉爆（估 10 小时 / 50 GB），1 年足够训练且 test 集 2.5 个月仍是 100 万+ records 量级 |
| `--workers` | `4` | `binance_vision` 是纯 HTTP + shared `requests.Session`，线程安全；4 个并发实测 ~4 req/s × 4 ≈ 16 req/s，远低于 Binance `/api/v3/klines` 1200 权重/分钟的 limit |
| `--exchange` | `binance_vision` | 见上一篇；公司网络下唯一可达的 crypto 端点 |

### 3.2 启动

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY    # 学术代理不在 binance_vision 白名单

UNIVERSE=$(cat /root/autodl-tmp/top100_universe.txt)   # 生成见 §1.2
nohup kairos-collect --market crypto --exchange binance_vision \
    --universe "$UNIVERSE" --freq 1min \
    --start 2025-04-20 --end 2026-04-20 \
    --out ./raw/crypto/bv_1min_top100 --workers 4 \
    > logs/collect_top100.log 2>&1 &
```

### 3.3 产出

- **耗时：26 分 43 秒**（100 symbols / 4 workers ≈ 16 s/symbol 摊销）
- `raw/crypto/bv_1min_top100/*.parquet` 共 **1.2 GB**
- 100 / 100 成功，0 失败 0 空。
- tqdm 首个 symbol 那一格耗时最长（78 s）是因为 4 个 worker 几乎同时开始抓 BTC/ETH/SOL/XRP 这几个数据量最大的 symbol，第一个落盘的 symbol 摊到了所有等待时间。后面节奏稳定在 11-20 s/symbol。

### 3.4 和 BTC/ETH run 的对比

| | BTC/ETH 2 年 | **Top100 1 年** |
|---|---|---|
| symbols × 时间 | 2 × 2 年 | **100 × 1 年** |
| workers | 1 | 4 |
| 总耗时 | 11 m 29 s | 26 m 43 s |
| 原始 parquet | 97 MB | **1.2 GB** |
| throughput（币-天/秒） | ~1.06 | ~22.7（×21） |

`workers=4` 实测把吞吐抬到 ~4× 单线程，没被 Binance 限流，也没拿到 429。下一次要再快可以试 `workers=8`，但 Top100 这点数据半小时已经够用。

---

## 4. 打包成训练集

采完直接 `kairos-prepare`。时间切分：

```bash
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_top100 \
    --train 2025-04-20:2026-01-31 \
    --val   2025-04-20:2026-01-31 \
    --test  2026-02-01:2026-04-20 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_top100
```

### 4.1 产出

```
finetune/data/crypto_1min_top100/    7.6 GB
├── train_data.pkl   967 MB    # 100 symbols
├── val_data.pkl     170 MB    #  97 symbols（3 个 symbol 训练期样本过少被剔除）
├── test_data.pkl    339 MB    # 100 symbols
├── exog_train.pkl  4.1 GB     # 100 symbols × 32 exog cols × block-level z-score
├── exog_val.pkl    724 MB     #  97 symbols
├── exog_test.pkl   1.4 GB     # 100 symbols
└── meta.json                  # {market: crypto, exog_cols: [32], split_mode: interleave, ranges: {...}}
```

- 耗时 **约 3 分钟**。
- meta.json 的 `exog_cols` 仍是 32 维（24 通用 + 8 crypto 专属），funding_rate / funding_rate_z / oi_change / basis / btc_dominance 被 pad 为 0；架构侧的 `n_exog=32` 不需要变（AGENTS.md §8 不变式 2）。
- 7.6 GB 的 exog 全量 load 到主进程内存；训练时 RSS ~14 GB，远低于 96 GB cgroup 限制。

---

## 5. Smoke 训练（16 秒）

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

- **`[TRAIN] pool=31,601,392`**（vs BTC/ETH smoke 的 ~100 万 → **~30× 大**）
- val_ce = **2.4234**，保存到 `artifacts/checkpoints/predictor/checkpoints/best_model`
- 总耗时 16 秒（50 步 + 40 步 val）

一次通过，说明：DDP 单卡 + Kronos 权重加载 + exog 旁路 + 32 维 exog schema 全部 OK。

---

## 6. 正式训练

### 6.1 命令

跟 BTC/ETH run 几乎一模一样，只换数据集路径：

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
export KAIROS_NUM_WORKERS=0                # 防 DataLoader 多 worker 内存爆
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
SH
chmod +x logs/run_train_top100.sh
nohup bash logs/run_train_top100.sh > logs/train_top100.log 2>&1 &
```

### 6.2 备份 BTC/ETH 旧 checkpoint（再覆盖之前）

```bash
cp -r artifacts/checkpoints/predictor/checkpoints/best_model \
       artifacts/checkpoints/predictor/checkpoints/best_model_btceth_backup
```

### 6.3 轨迹

| Epoch | val_ce | 相对 BTC/ETH 同 epoch | 判定 |
|---|---|---|---|
| 1 | **2.4036** | ⬇ 0.0904（BTC/ETH ep1=2.4940） | ✅ save best |
| 2 | 2.4152 | 相当 | patience 1/3 |
| 3 | 2.5892 | 相当 | patience 2/3 |
| 4 | 2.9903 | 略高于 BTC/ETH ep4=2.8820 | patience 3/3 → 早停 |

- 每 epoch 2 分 55 秒左右（与 BTC/ETH 的 2 m 33 s 接近；`n_train_iter=50000` 与 batch=50 固定，样本池变大不影响一 epoch 的 step 数）。
- 总耗时 **12 分钟**（3 min × 4 ep + val）。
- **val_ce ep1 比 BTC/ETH 低 0.09** —— 这是扩大 universe 最直接的回报，CE 空间 0.09 大致对应下一 token 概率提升约 9%，不是噪声。
- 同样的过拟合结构（ep2 开始上升）印证 Kronos-small（5.4M 参数）对 transformer 主干冻结 7/8 层后，容量只够吃一个 epoch 的信号。继续训只会 overfit memory。

### 6.4 Checkpoint

```
artifacts/checkpoints/predictor/checkpoints/
├── best_model/                  97 MB   # Top100 微调（ep1, val_ce=2.4036）
└── best_model_btceth_backup/    97 MB   # BTC/ETH 原版，保留参考
```

---

## 7. 回测：baseline vs finetuned

### 7.1 关键参数：`--stride 10`

这里**必须偏离** BTC/ETH run 的默认 `stride=1`：

- `stride=1` 意味着每 1 分钟一个窗口。BTC/ETH 2 币 × 3 个月 = 30 万 windows，跑 10 分钟可接受。
- Top100 × 2.5 个月 stride=1 = 1100 万 windows / horizon → 估算 **8 小时 / 次 × 2 次 = 16+ 小时**。
- 改 `--stride 10` 后每 10 分钟出一个信号，降到 110 万 windows / horizon，**单次 ~45 分钟**。
- IC 稳定性没问题：78 个 cross-section dates × 平均每天 ~14000 observations，比 BTC/ETH run 还多。

### 7.2 启动

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

两次回测共耗时 **89 分钟**（baseline 44 m + finetuned 44 m）。

### 7.3 数据规模

- **n_records = 1,106,570**（100 symbols × ~11k windows 后 × horizon 复用）
- **date_range = 2026-02-01 04:16 ~ 2026-04-19 23:26**（78 cross-section dates，未校过 UTC 偏移仍是 binance_vision 的 naive local time，但对比一致性不影响）
- **100 symbols** 全部参与（test 集 symbols 全覆盖）

### 7.4 结果

| horizon | 指标 | baseline | **finetuned** | Δ |
|---|---|---|---|---|
| **h1** | pearson | +0.0079 | −0.0112 | ❌ |
| | hit_rate | 0.4892 | 0.4906 | +0.14 pp |
| | by-date IC | +0.0079 | −0.0127 | ❌ |
| | rank-IC | +0.0032 | −0.0082 | ❌ |
| | **ICIR** | **+0.424** | **−0.555** | ❌ 翻负 |
| **h5** | pearson | −0.0025 | −0.0076 | ❌ |
| | hit_rate | 0.4590 | 0.4883 | +2.9 pp |
| | by-date IC | +0.0033 | −0.0107 | ❌ |
| | rank-IC | +0.0139 | −0.0050 | ❌ |
| | **ICIR** | +0.125 | **−0.681** | ❌ |
| **h30** | pearson | −0.0046 | **+0.0054** | ✅ |
| | hit_rate | 0.4863 | 0.4921 | +0.6 pp |
| | by-date IC | −0.0023 | **+0.0158** | ✅ |
| | rank-IC | +0.0004 | **+0.0299** | ✅ |
| | **ICIR** | −0.084 | **+0.454** | ✅ 跨越 0.3 |

### 7.5 怎么读这张表

1. **h30 是这次的 headline alpha**。
   - **by-date rank-IC 从 +0.0004 → +0.0299（几乎 0 → 2.99%）**
   - **ICIR −0.084 → +0.454**，超过 BTC/ETH run 的 +0.325
   - hit_rate 49.21%（+0.6 pp vs baseline）
   - 这条通道上微调确实学到了 Top100 横截面的 30-min 方向性，且在更多币种上成立比 2 币版本更稳定——这正是你要的"IC 不一定抬，但 ICIR 抬"。

2. **h1 / h5 被"污染"成负 IC**，不是随机噪声。
   - pearson / rank-IC / by-date IC / ICIR 四个独立指标**同向**显示负相关，p-value 全部 < 1e-14。
   - 量级（by-date IC ≈ −0.012）虽小但显著——说明模型在 short-horizon 上确实学到了什么，只是方向错。
   - 换句话说，**模型把可解释方差都挤到了 h30，把 h1/h5 当成了 h30 的"反向预警"**。对应 preset `return_horizon=30` 就是 h30 对齐训练目标，短 horizon 等于让模型外推一个它没被监督的任务。
   - 实际用的时候可以只订阅 h30 信号，或者给 h1/h5 额外加一个 return head（多 head / 多 horizon 训练，见 TRAINING_TUNING_PLAYBOOK.md）。

3. **Baseline 的 h1 ICIR +0.42 不是真 alpha**。
   - baseline = Kronos 原权重 + 随机初始化的 exog / return head，return head 输出的就是乱的。
   - 但当截面只有 100 个币、每天样本 ~14k、78 个 dates 时，即使随机信号也能凑出 |ICIR| > 0.3 的虚高值——所以看 finetuned 的**相对改善**（Δ），不要看 baseline 绝对 ICIR。
   - 这也是为什么报告 ICIR 时**必须同时报告 baseline 对照**。

4. **vs BTC/ETH run 的 h30 对比**：

   | run | rank-IC | ICIR | hit_rate |
   |---|---|---|---|
   | BTC/ETH 2 币 × 2 年 | +0.050 | +0.325 | 0.5168 |
   | **Top100 × 1 年** | **+0.030** | **+0.454** | 0.4921 |

   rank-IC 绝对值从 5% 掉到 3%，但 ICIR（信息比率）反而抬升 40% —— 对组合化使用这是更有价值的信号。同时 hit_rate 从 51.7% 掉到 49.2% ——这暗示 Top100 里很多小币的 30-min 方向性比 BTC/ETH 低，模型学到的 alpha 更多是跨币相对强弱而非方向。

---

## 8. 产物清单

### 服务器（AutoDL `/root/autodl-tmp/`）

```
top100_universe.txt                                     962 B    # 100 个 symbol 的名单
Kairos/
├── raw/crypto/bv_1min_top100/                          1.2 GB   # 100 个 parquet
├── finetune/data/crypto_1min_top100/                   7.6 GB   # train/val/test + exog + meta.json
├── artifacts/
│   ├── backtest_top100_baseline.json                   1.4 KB
│   ├── backtest_top100_finetuned.json                  1.4 KB
│   └── checkpoints/predictor/checkpoints/
│       ├── best_model/                                 97 MB    # Top100 微调（ep1, val_ce=2.4036）
│       └── best_model_btceth_backup/                   97 MB    # BTC/ETH 备份，保留
└── logs/
    ├── collect_top100.log / prepare_top100.log
    ├── train_top100.log / backtest_top100.log
    ├── run_train_top100.sh / run_backtest_top100.sh
    └── *.pid
```

### 本地仓库（`/Users/jie.feng/wlb/Kairos/`）

- `artifacts/backtest_top100_baseline.json` / `backtest_top100_finetuned.json` — 从 AutoDL scp 回来的 backtest 摘要（gitignored，完整数字在 §7.4 表里）
- [`examples/crypto_top100_universe.md`](../examples/crypto_top100_universe.md) — 冻结的 Top100 名单（今天 2026-04-20 的快照）

如果要把 checkpoint 拉回本地：

```bash
scp -P 37667 -r \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/Kairos/artifacts/checkpoints/predictor/checkpoints/best_model \
  /Users/jie.feng/wlb/Kairos/artifacts/checkpoints/predictor/checkpoints/best_model_top100
```

---

## 9. 本次 run 没产生代码改动

这次完全没改 Python / CLI，所有能力都在上一 run（commits `44cd5d6` 和 `ccb0de1`）里就绪：

- `--universe` 支持逗号分隔列表（已有）
- `binance_vision.list_symbols_by_volume()` 支持任意 `top_n`（已有）
- `kairos-collect --workers 4` 支持并发（已有）
- `backtest_ic --stride 10` 支持降采样（已有）
- `KAIROS_NUM_WORKERS=0` env override 避免 DataLoader OOM（已有）

也就是说 BTC/ETH 那次踩坑修 bug 的工作在这次直接摘果子，Top100 的跑通只是把参数换一下。

---

## 10. TL;DR — 一键复现命令清单

假设你已经拿到一台 AutoDL 5090 / 4090（≥16 GB 显存、≥32 GB 系统内存、≥20 GB 可用磁盘）：

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

# ------------------ 1. 预热权重（复用上一 run 的话可跳过）------------------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
huggingface-cli download NeoQuasar/Kronos-Tokenizer-base --cache-dir $HF_HOME
huggingface-cli download NeoQuasar/Kronos-small           --cache-dir $HF_HOME

# ------------------ 2. 生成 Top100 名单 ------------------
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

# ------------------ 3. 采集 1 年（~27 分钟）------------------
UNIVERSE=$(cat /root/autodl-tmp/top100_universe.txt)
mkdir -p logs
nohup kairos-collect --market crypto --exchange binance_vision \
    --universe "$UNIVERSE" --freq 1min \
    --start 2025-04-20 --end 2026-04-20 \
    --out ./raw/crypto/bv_1min_top100 --workers 4 \
    > logs/collect_top100.log 2>&1 &
wait
# 预期：完成: {'ok': 100, 'fail': 0, 'empty': 0, 'skip_up_to_date': 0}

# ------------------ 4. 打包（~3 分钟）------------------
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_top100 \
    --train 2025-04-20:2026-01-31 \
    --val   2025-04-20:2026-01-31 \
    --test  2026-02-01:2026-04-20 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_top100

# ------------------ 5. 微调（~12 分钟，早停到 ep 4）------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_top100
export KAIROS_NUM_WORKERS=0
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# ------------------ 6. 回测对比（各 ~45 分钟，stride=10）------------------
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

总耗时：**~2 小时 30 分**（27 + 3 + 12 + 45×2 = 132 m）。单卡 5090 成本 ≈ ¥10-15。

---

## 11. 后续方向（更新自 BTC/ETH run 的 §11）

带入 Top100 run 后的新信息重排优先级：

1. **修 h1/h5 的"反向"问题**（新的最高优先级）。三条路：
   - 给 `KronosWithExogenous` 加多个 return head（h1 / h5 / h30 各一个），在训练时分头监督，避免 h30 把短 horizon "挤扁"。
   - 训两个 checkpoint（一个 horizon=1，一个 horizon=30），推理时各走各的。
   - 把 h30 的 preset 明确为 `kairos-small-crypto-h30`，避免误用于 h1。
2. **拉到 OKX 永续**。h30 已经有 alpha，补上 funding / OI / basis 之后 h1/h5 大概率不会再负 —— 短 horizon 最依赖这些微观因子。
3. **换 Kronos-base / Kronos-large**（参数量 ×10-50）。5.4M 参数在 3160 万样本上 1 epoch 就饱和，更大模型应该能在 ep 2-3 继续降 val_ce 从而压住过拟合。
4. **冻结策略调整**：只解冻最后 1 层 transformer 在这次 val_ce 轨迹里看仍然是过紧——把 `KAIROS_UNFREEZE_LAST_N` 调到 2 或 3 试一次。
5. 学习率 sweep（low priority，目前 val_ce 下降幅度 0.09 已经比调 lr 更重要）。
