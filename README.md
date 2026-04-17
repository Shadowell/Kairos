# Kairos

> 抓住 **正确的时机**。
>
> **Kairos** 是面向 A 股场景的 [Kronos](https://github.com/shiyu-coder/Kronos) 基础模型的**微调 + 部署工具箱**。
> 开箱即用的数据采集、因子工程、外生通道扩展、分位回归头、HuggingFace 一键上传和 FastAPI 推理服务。

<p align="center">
<em>Kronos 管时间 · Kairos 管机会</em>
</p>

---

## ✨ 特性

- **数据** — 基于 akshare 的 A 股日/分钟 K 线采集（多源 + 断点续传 + 日度累积）
- **因子** — 32 维技术/量价/日历/相对指数因子，无未来信息泄漏
- **模型** — `KronosWithExogenous`：在 Kronos 之上加外生变量旁路通道 + 可选分位回归头，**完全兼容预训练权重**
- **训练** — `torchrun` 启动的 DDP 训练，支持渐进解冻、pinball loss
- **部署** — 一键 push 到 Hugging Face Hub + FastAPI 实时推理服务
- **CLI** — `kairos-collect / kairos-prepare / kairos-train-* / kairos-push-hf / kairos-serve`

---

## 🏗️ 架构

```
                ┌──────────────────┐
                │  akshare / 东财  │  → 日线全历史 + 5/1min 日度累积
                └────────┬─────────┘
                         │
           kairos.data.collect (每日 cron)
                         │
                         ▼
          ┌─────────────────────────────┐
          │  raw/{freq}/<symbol>.parquet│
          └────────┬────────────────────┘
                   │
          kairos.data.features         ← 技术指标、量价、相对指数
                   │
          kairos.data.prepare_dataset  ← 生成 train/val/test.pkl + exog_*.pkl
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ kairos.training.train_tokenizer (可选)    │
    │ kairos.training.train_predictor (方案A+C) │
    └─────────────────┬────────────────────────┘
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
kairos.deploy.push_to_hf      kairos.deploy.serve
(Hugging Face Hub)            (FastAPI /predict)
```

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/Shadowell/Kairos.git
cd Kairos
pip install -e '.[serve,train]'
```

要求：Python ≥ 3.10，PyTorch ≥ 2.0。

### 1. 采集 A 股日线（CSI300 全历史）

```bash
kairos-collect --universe csi300 --freq daily \
    --start 2018-01-01 --adjust qfq --out ./raw/daily

# 指数数据，用于相对收益因子
kairos-collect --universe 000300 --freq daily --adjust qfq --out ./raw/index
```

### 2. 生成训练集

```bash
kairos-prepare \
    --raw ./raw/daily \
    --raw-index ./raw/index/000300.parquet \
    --train 2018-01-01:2023-12-31 \
    --val   2024-01-01:2024-12-31 \
    --test  2025-01-01:2026-04-17 \
    --out   ./artifacts/datasets
```

### 3. 训练（单卡即可）

```bash
# 可选：先微调 Tokenizer（通常不必，直接用 NeoQuasar 官方版即可）
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_tokenizer

# 训 Predictor（方案 A 外生通道 + 方案 C 分位回归头，默认开启）
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
```

### 4. 推到 Hugging Face

```bash
huggingface-cli login
kairos-push-hf \
    --tokenizer-ckpt  ./artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --predictor-ckpt  ./artifacts/checkpoints/predictor/checkpoints/best_model \
    --repo-tokenizer  your-user/Kronos-Tokenizer-ashare \
    --repo-predictor  your-user/Kronos-small-ashare \
    --predictor-class ext --private
```

### 5. 起 FastAPI 服务

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

本地 Mac（无 CUDA）→ AutoDL 云端 GPU 的完整租卡训练流程见 [docs/AUTODL_GUIDE.md](docs/AUTODL_GUIDE.md)。

> 📖 **第一次接触这些名词？** 先读 [docs/GLOSSARY.md](docs/GLOSSARY.md) —— 从 K 线到 Transformer 到 IC，全部术语带例子解释。

| 方案 | 做法 | 成本 | 复用预训练 |
|---|---|---|---|
| **A：外生旁路通道** | 32 维因子 `Linear→SiLU→Linear→gate` 加到 token embedding | 低 | ✅ 完全兼容 |
| **B：Tokenizer 重训** | 扩 `d_in` 为 12~20，重训 Tokenizer 和 Predictor | 高（~¥2-5k） | ❌ |
| **C：分位回归头** | 末层 hidden 接分位头，用 pinball loss | 低 | ✅ |

---

## 📂 项目结构

```
Kairos/
├── README.md / LICENSE / pyproject.toml / requirements.txt
├── docs/
│   ├── GLOSSARY.md                 ← 术语速查（新手先看这个）
│   ├── TUNING_PLAYBOOK.md          ← 详细调优手册（云成本对比、避坑指南）
│   └── AUTODL_GUIDE.md             ← AutoDL 租卡训练端到端流程
├── kairos/                         ← Python 包
│   ├── __init__.py                 ← 顶层 re-export
│   ├── data/
│   │   ├── collect.py              ← akshare 采集
│   │   ├── features.py             ← 32 维因子工程
│   │   └── prepare_dataset.py      ← 生成 Kronos pkl
│   ├── models/
│   │   └── kronos_ext.py           ← KronosWithExogenous + QuantileReturnHead
│   ├── training/
│   │   ├── config.py               ← TrainConfig
│   │   ├── dataset.py              ← AShareKronosDataset
│   │   ├── train_tokenizer.py
│   │   └── train_predictor.py
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
| `kairos-collect` | 采集 A 股 K 线 |
| `kairos-prepare` | 生成训练 pkl |
| `kairos-train-tokenizer` | 微调 Tokenizer（可选） |
| `kairos-train-predictor` | 微调 Predictor（方案 A+C） |
| `kairos-push-hf` | 上传到 HuggingFace |
| `kairos-serve` | 起推理服务 |

---

## 📜 许可

MIT License · 参见 [LICENSE](LICENSE)。

本项目在 `kairos/vendor/kronos/` 下 vendor 了 [Kronos](https://github.com/shiyu-coder/Kronos) 原始模型代码，同为 MIT 协议。

## 🙏 致谢

- [Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739) — 本项目的模型基础
- [akshare](https://github.com/akfamily/akshare) — 数据源
- [Hugging Face](https://huggingface.co/) — 模型托管
