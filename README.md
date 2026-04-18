# Kairos

> 抓住 **正确的时机**。
>
> **Kairos** 是基于 [Kronos](https://github.com/shiyu-coder/Kronos) 基础模型的**多市场微调 + 部署工具箱**。
> 开箱即用的数据采集、因子工程、外生通道扩展、分位回归头、跨市场回测、HuggingFace 一键上传和 FastAPI 推理服务。

<p align="center">
<em>Kronos 管时间 · Kairos 管机会</em>
</p>

---

## ✨ 特性

- **多市场数据层** — 统一 `MarketAdapter` 抽象，内置 A 股（akshare 多源 fallback）、加密货币（ccxt OKX 永续 / Binance Vision 镜像）两个 adapter，可扩展到 Binance / Bybit / 外汇 / 黄金等
- **共享因子 schema** — 24 维通用因子 + 8 维市场专属因子 = 固定 32 维 `EXOG_COLS`，**同一个模型 checkpoint 跨市场通用**，无未来信息泄漏
- **模型** — `KronosWithExogenous`：在 Kronos 之上加外生变量旁路通道 + 分位回归头，**完全兼容预训练权重**（147 层里 136 层直接 reuse）
- **训练** — `torchrun` 启动的 DDP 训练，渐进解冻 + 早停 + OneCycleLR + pinball loss；全量 env override（`KAIROS_BATCH_SIZE` 等）无需改代码就能调参
- **回测** — `backtest_ic` 直接吃 `meta.json` 自动推 market/freq，支持 `--baseline` 拿 Kronos 原权重做对比，输出 overall / by-date / by-hour 三种 bucket 的 IC / Rank-IC / ICIR
- **部署** — 一键 push 到 Hugging Face Hub + FastAPI 实时推理服务

---

## 📈 核心成果

### 加密货币 1-min 微调（2026-04-17）

**数据**：BTC/USDT + ETH/USDT 1-min K 线，2024-01 ~ 2026-04（30 万条 test 样本）；**硬件**：单卡 RTX 5090；**耗时**：10 分 18 秒；**成本**：约 ¥3-5。

| horizon | model | hit_rate | rank_ic | ICIR |
|---|---|---|---|---|
| h1  | baseline  | 49.74% | +0.022 | +0.297 |
| h1  | finetuned | 49.53% | -0.012 | +0.029 |
| h5  | baseline  | 49.72% | -0.007 | +0.093 |
| h5  | finetuned | 50.51% | +0.010 | +0.060 |
| **h30** | **baseline**  | 50.98% | +0.018 | +0.039 |
| **h30** | **finetuned** | **51.68%** | **+0.050** | **+0.325** |

- **h30（对齐 preset `return_horizon=30`）rank-IC 提升 184%，ICIR 提升 743%，首次跨过 0.3 的"可入组合"阈值**
- h1 / h5 基本打平：binance_vision 镜像没有 funding / OI / basis，crypto 特有因子被 pad 为 0，短期预测力被显著削弱
- 完整流程、踩坑记录、一键复现命令见 [docs/CRYPTO_BTC_ETH_RUN.md](docs/CRYPTO_BTC_ETH_RUN.md)

### A 股日线（对比基线）

训了两版（time-split v1 + interleave-split v2），test IC 都为负，结论：在现成 EXOG schema 下 A 股日线信号偏弱。下一步方向（调权重冻结策略、改监督信号、换到分钟级）写在 [docs/TUNING_PLAYBOOK.md](docs/TUNING_PLAYBOOK.md)。

---

## 🏗️ 架构

### 数据流水线（多市场）

```
 ┌────────────┐   ┌──────────────┐   ┌───────────────────┐
 │ akshare /  │   │ ccxt (OKX    │   │ data-api.binance  │
 │ 东财 /腾讯 │   │ 永续,默认)   │   │ .vision (现货镜像)│   … 其他 adapter
 │ 新浪       │   │              │   │                   │
 └─────┬──────┘   └──────┬───────┘   └─────────┬─────────┘
       │ ashare          │ crypto (okx)        │ crypto (binance_vision)
       ▼                 ▼                     ▼
 ┌─────────────────────────────────────────────────────────┐
 │           MarketAdapter 抽象（kairos.data.markets）      │
 │  • FetchTask / universe / fetch_ohlcv / MARKET_EXOG_COLS │
 └──────────────────────┬──────────────────────────────────┘
                        │   kairos-collect  (--market)
                        ▼
              raw/{market}/{freq}/<symbol>.parquet
                        │
                        ▼
     ┌──────────────────────────────────────────────┐
     │ kairos.data.common_features   24 维通用因子  │
     │ adapter.market_features        8 维市场因子  │
     │ ──────────────── = EXOG_COLS (固定 32 维) ── │
     │ kairos.data.prepare_dataset   (含 meta.json) │
     └────────────────────┬─────────────────────────┘
                          │
                          ▼
       finetune/data/<name>/{train,val,test}_data.pkl + exog_*.pkl + meta.json
```

### 训练 + 回测 + 部署

```
 ┌────────────────────────────┐    ┌────────────────────────────┐
 │ kairos.training            │    │ kairos.training            │
 │ .train_tokenizer  (可选)   │    │ .backtest_ic               │
 │ .train_predictor           │    │  --baseline  vs  --ckpt    │
 │  ├── preset_for(name)      │    │  自动从 meta.json 推       │
 │  │    ashare-daily /       │    │  market / freq / exog      │
 │  │    crypto-1min          │    │  输出 overall + by-date +  │
 │  ├── DDP (torchrun)        │    │  by-hour 三档 bucket       │
 │  ├── 渐进解冻 + OneCycleLR │    └──────────────┬─────────────┘
 │  ├── 早停 patience=3       │                   │
 │  └── KAIROS_* env 覆盖参数 │                   ▼
 └──────────────┬─────────────┘          artifacts/backtest_*.json
                │
                ▼
   artifacts/checkpoints/predictor/checkpoints/best_model/
                │
       ┌────────┴─────────┐
       ▼                  ▼
 kairos.deploy      kairos.deploy
 .push_to_hf        .serve
 (HF Hub)           (FastAPI /predict)
```

### 为什么能一套 checkpoint 跨市场

- `kairos.data.common_features` 固定给出 **24 维通用因子**（动量 / 波动率 / 均线偏离 / 量价 / 蜡烛形态 / 日历编码）。
- 每个市场 adapter 额外给出 **8 维市场专属因子**（A 股：换手 / 相对指数；crypto：funding / OI / basis / btc_dominance / hour_sin,cos）。
- 两者相加 **严格等于 32 维 `EXOG_COLS`**，Phase 2 架构硬约束（`build_features` 直接 assert）。
- 模型侧 `n_exog=32` 对所有市场一致；要替换因子就占 pad slot 或换掉某个 slot，**永远不扩维度**。
- 结果：同一个 checkpoint 可以在 A 股 / crypto / 未来的外汇黄金上加载运行，权重迁移零成本。

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/Shadowell/Kairos.git
cd Kairos
pip install -e '.[serve,train]'
```

要求：Python ≥ 3.10，PyTorch ≥ 2.0。

### 1. 采集数据

Kairos 的数据层通过 `--market` 参数选择市场 adapter（默认 `ashare`）：

```bash
# A 股（默认，原有行为不变）
kairos-collect --universe csi300 --freq daily \
    --start 2018-01-01 --adjust qfq --out ./raw/daily

# 指数数据，用于相对收益因子
kairos-collect --universe 000300 --freq daily --adjust qfq --out ./raw/index

# 加密货币（OKX USDT 永续）—— 需要先装 crypto 额外依赖
pip install -e '.[crypto]'
kairos-collect --market crypto \
    --universe "BTC/USDT:USDT,ETH/USDT:USDT" \
    --freq 1min --start 2023-01-01 --end 2025-01-01 \
    --out ./raw/crypto/1min --workers 1
```

完整加密货币工作流（代理、API key、自定义交易所）见 [docs/CRYPTO_GUIDE.md](docs/CRYPTO_GUIDE.md)。

### 2. 生成训练集

```bash
# A 股（日线，传统 time-split）
kairos-prepare \
    --raw ./raw/daily \
    --raw-index ./raw/index/000300.parquet \
    --train 2018-01-01:2023-12-31 \
    --val   2024-01-01:2024-12-31 \
    --test  2025-01-01:2026-04-17 \
    --out   ./artifacts/datasets

# crypto 1min（interleave-split，更适合高频）
kairos-prepare --market crypto \
    --raw ./raw/crypto/1min \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min
```

### 3. 训练（单卡即可）

```bash
# 方法 1：改 preset（推荐）
export KAIROS_PRESET=crypto-1min        # 或 ashare-daily
export KAIROS_DATASET=./finetune/data/crypto_1min
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# 方法 2：临时用 env 覆盖单个参数（无需改代码）
KAIROS_BATCH_SIZE=32 KAIROS_LR=5e-6 \
    torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# 可选：先微调 Tokenizer（通常不必，直接用 NeoQuasar 官方版即可）
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_tokenizer
```

### 4. 回测对比

```bash
# baseline = Kronos 原权重 + 随机初始化的 exog / return head
python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min \
    --horizons 1,5,30 --out artifacts/backtest_baseline.json

# finetuned
python -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min \
    --horizons 1,5,30 --out artifacts/backtest_finetuned.json
```

### 5. 推到 Hugging Face

```bash
huggingface-cli login
kairos-push-hf \
    --tokenizer-ckpt  ./artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --predictor-ckpt  ./artifacts/checkpoints/predictor/checkpoints/best_model \
    --repo-tokenizer  your-user/Kronos-Tokenizer-ashare \
    --repo-predictor  your-user/Kronos-small-ashare \
    --predictor-class ext --private
```

### 6. 起 FastAPI 服务

```bash
kairos-serve \
    --tokenizer your-user/Kronos-Tokenizer-ashare \
    --predictor your-user/Kronos-small-ashare \
    --device cuda:0 --port 8000

# 调用
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "symbol": "600977", "lookback": 400, "pred_len": 20,
  "T": 0.6, "top_p": 0.9, "sample_count": 5
}'
```

---

## 🧬 三种模型改造方案

Kairos 默认实现了**方案 A + 方案 C**，开箱即用。详见 [docs/TUNING_PLAYBOOK.md](docs/TUNING_PLAYBOOK.md)。

| 方案 | 做法 | 成本 | 复用预训练 |
|---|---|---|---|
| **A：外生旁路通道** | 32 维因子 `Linear→SiLU→Linear→gate` 加到 token embedding | 低 | ✅ 完全兼容 |
| **B：Tokenizer 重训** | 扩 `d_in` 为 12~20，重训 Tokenizer 和 Predictor | 高（~¥2-5k） | ❌ |
| **C：分位回归头** | 末层 hidden 接分位头，用 pinball loss | 低 | ✅ |

---

## 📚 文档索引

| 文档 | 说明 |
|---|---|
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | 术语速查 —— K 线 / Transformer / IC / 分位回归，带例子解释。**第一次接触这些名词先看这里** |
| [`docs/TUNING_PLAYBOOK.md`](docs/TUNING_PLAYBOOK.md) | 调参手册 v1→v2，云成本对比，避坑指南 |
| [`docs/AUTODL_GUIDE.md`](docs/AUTODL_GUIDE.md) | 本地 Mac → AutoDL 云端 GPU 的完整租卡训练流程 |
| [`docs/CRYPTO_GUIDE.md`](docs/CRYPTO_GUIDE.md) | 加密货币数据层、OKX/Binance/Binance-Vision 配置、交易所扩展指南 |
| [`docs/CRYPTO_BTC_ETH_RUN.md`](docs/CRYPTO_BTC_ETH_RUN.md) | **2026-04-17 BTC+ETH 1min 端到端跑通记录** —— 命令、坑、IC 对比结果、一键复现清单 |
| [`AGENTS.md`](AGENTS.md) | 仓库操作手册（给 AI agent 和人类协作者看） |

---

## 📂 项目结构

```
Kairos/
├── README.md / LICENSE / pyproject.toml / requirements.txt
├── AGENTS.md                       ← AI coding agent 的仓库操作手册
├── docs/
│   ├── GLOSSARY.md                 ← 术语速查（新手先看这个）
│   ├── TUNING_PLAYBOOK.md          ← 详细调优手册（云成本对比、避坑指南）
│   ├── AUTODL_GUIDE.md             ← AutoDL 租卡训练端到端流程
│   ├── CRYPTO_GUIDE.md             ← 加密货币数据层 & 交易所扩展指南
│   └── CRYPTO_BTC_ETH_RUN.md       ← BTC+ETH 1min 端到端跑通记录 (2026-04-17)
├── kairos/                         ← Python 包
│   ├── __init__.py                 ← 顶层 re-export
│   ├── data/
│   │   ├── collect.py              ← 多市场 CLI dispatcher
│   │   ├── markets/                ← MarketAdapter 抽象 + ashare / crypto 实现
│   │   │   └── crypto_exchanges/   ← ccxt 封装：okx / binance_vision / ...
│   │   ├── common_features.py      ← 24 维通用因子
│   │   ├── features.py             ← 组装 common + adapter 专属 = 32 维
│   │   └── prepare_dataset.py      ← 生成 train/val/test.pkl + meta.json
│   ├── models/
│   │   └── kronos_ext.py           ← KronosWithExogenous + QuantileReturnHead
│   ├── training/
│   │   ├── config.py               ← TrainConfig + preset_for(name)
│   │   ├── dataset.py              ← KronosDataset（跨市场通用）
│   │   ├── train_tokenizer.py
│   │   ├── train_predictor.py      ← 支持 KAIROS_* env 覆盖超参
│   │   └── backtest_ic.py          ← IC / Rank-IC / ICIR，支持 --baseline
│   ├── deploy/
│   │   ├── push_to_hf.py
│   │   └── serve.py
│   ├── utils/
│   │   └── training_utils.py       ← DDP / 种子 / 工具
│   └── vendor/kronos/              ← 官方 Kronos 模型源码（vendored）
├── examples/
│   └── inference_quickstart.py
└── tests/
    └── test_features.py
```

---

## 📋 CLI 速查

| 命令 | 作用 |
|---|---|
| `kairos-collect --market {ashare,crypto}` | 多市场 K 线采集（akshare / ccxt） |
| `kairos-prepare --market {ashare,crypto}` | 生成 train/val/test pkl + `meta.json`，支持 time-split / interleave-split |
| `kairos-train-tokenizer` | 微调 Tokenizer（可选，通常直接用官方版） |
| `kairos-train-predictor` | 微调 Predictor（方案 A+C）；读 `KAIROS_PRESET / KAIROS_DATASET / KAIROS_*` env |
| `python -m kairos.training.backtest_ic` | 回测：`--baseline` vs `--ckpt`，IC / Rank-IC / ICIR |
| `kairos-push-hf` | 上传 checkpoint 到 HuggingFace Hub |
| `kairos-serve` | 起 FastAPI 推理服务 |

### 常用 env 变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `KAIROS_PRESET` | `ashare-daily` | `preset_for(name)` 里注册的预设名，e.g. `crypto-1min` |
| `KAIROS_DATASET` | preset 里的路径 | 训练 / 回测数据目录（`kairos-prepare` 的 `--out`） |
| `KAIROS_BATCH_SIZE` / `KAIROS_LR` / `KAIROS_EPOCHS` / `KAIROS_NUM_WORKERS` | preset 默认 | 不改代码就能扫参；完整列表见 `train_predictor.py` |
| `KAIROS_SMOKE=1` | — | 本地 CPU smoke test 用，50 step × batch 4 |
| `HF_ENDPOINT=https://hf-mirror.com` | — | 中国大陆推荐，走 hf-mirror |

---

## 📜 许可

MIT License · 参见 [LICENSE](LICENSE)。

本项目在 `kairos/vendor/kronos/` 下 vendor 了 [Kronos](https://github.com/shiyu-coder/Kronos) 原始模型代码，同为 MIT 协议。

## 🙏 致谢

- [Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739) — 本项目的模型基础
- [akshare](https://github.com/akfamily/akshare) — 数据源
- [Hugging Face](https://huggingface.co/) — 模型托管
