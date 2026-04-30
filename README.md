# Kairos

Kairos 是基于 [Kronos](https://github.com/shiyu-coder/Kronos) 时间序列基础模型的加密货币微调与评测工具箱。当前只聚焦两类标的：**现货** 和 **USDT 本位永续合约**。

## 项目作用

- 采集 OKX 兼容的加密货币 OHLCV 数据，覆盖现货和永续合约。
- 构建固定 32 维外生通道：24 维通用 OHLCV 因子 + 8 维 crypto 因子。
- 训练 `KronosWithExogenous` predictor checkpoint，并用 IC / Rank-IC / ICIR 回测评估。
- 支持把 checkpoint 推送到 Hugging Face，并通过 FastAPI 用调用方传入的 OHLCV bars 做预测服务。

## 当前状态

下面是目前可作为参考的公开 predictor 结果。报告结果时必须看相对 baseline 的增量，不要只看绝对 IC。

| 实验 | 标的范围 | h30 Rank-IC | h30 ICIR | 说明 |
| --- | --- | ---: | ---: | --- |
| BTC/ETH 2 年现货 | 2 个币 | `+0.050` | `+0.325` | 第一个可用 crypto signal |
| Top100 1 年现货 | 100 个币 | `+0.030` | `+0.454` | 更大 universe 提升稳定性 |
| BTC/ETH 2 年现货 + `Kronos-base` | 2 个币 | `+0.076` | `+0.484` | 当前最好公开结果 |

OKX 永续合约多通道路径已经接通，但仍处在实验阶段。第一次 Top10 30 天永续实验应视为复盘样本，不应视为生产结果。

## 公开模型

| Repo | 基座模型 | 数据 | 用途 |
| --- | --- | --- | --- |
| [`Shadowell/Kairos-small-crypto`](https://huggingface.co/Shadowell/Kairos-small-crypto) | [`NeoQuasar/Kronos-small`](https://huggingface.co/NeoQuasar/Kronos-small) | BTC/USDT + ETH/USDT，1 分钟 | 公开 small predictor checkpoint |
| [`Shadowell/Kairos-base-crypto`](https://huggingface.co/Shadowell/Kairos-base-crypto) | [`NeoQuasar/Kronos-base`](https://huggingface.co/NeoQuasar/Kronos-base) | BTC/USDT + ETH/USDT，1 分钟 | 公开 base predictor checkpoint |

```python
from kairos.models import KronosWithExogenous

model = KronosWithExogenous.from_pretrained("Shadowell/Kairos-base-crypto")
```

## 特征结构

外生向量固定为 32 维：

- 通用 OHLCV 因子 24 维：收益、波动率、成交量/成交额、振幅、VWAP 和保留 pad。
- Crypto 因子 8 维：`market_ret_1`、`market_vol_20`、`hour_sin`、`hour_cos`、`funding_rate`、`funding_rate_z`、`oi_change`、`basis`。

现货数据使用前 4 个通用 crypto 因子，合约专属列填 0。永续合约数据可以额外接入 OKX 的 funding、OI、现货-合约 basis sidecar。

## 快速开始

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,serve]'
```

### 采集 OKX 现货

```bash
kairos-collect --market-type spot \
  --universe "BTC/USDT,ETH/USDT" \
  --freq 1min --start 2026-04-01 --end 2026-04-30 \
  --out ./raw/crypto/okx_spot_btc_eth_1min --workers 1
```

### 采集 OKX 永续合约

```bash
kairos-collect --market-type swap \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" \
  --freq 1min --start 2026-04-01 --end 2026-04-30 \
  --out ./raw/crypto/okx_swap_btc_eth_1min --workers 1 \
  --crypto-extras funding,open_interest,spot,reference
```

### 打包数据集

```bash
kairos-prepare --market crypto --market-type swap \
  --raw ./raw/crypto/okx_swap_btc_eth_1min \
  --train 2026-04-01:2026-04-20 \
  --val 2026-04-21:2026-04-25 \
  --test 2026-04-26:2026-04-30 \
  --out ./finetune/data/crypto_swap_btc_eth_1min
```

### 训练和回测

```bash
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=./finetune/data/crypto_swap_btc_eth_1min
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
  --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_baseline.json

python -m kairos.training.backtest_ic --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
  --preset crypto-1min --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_finetuned.json
```

## 服务接口

`kairos-serve` 不负责抓交易所行情。`/predict` 接口接收 JSON body，字段包括 `symbol`、`market_type`、`freq`，以及 `bars` 数组。`bars` 每一项包含 `datetime`、`open`、`high`、`low`、`close`、`volume`，以及可选的 `amount`。

## 仓库结构

```text
kairos/
  data/       # OKX 兼容采集、extras、特征打包
  models/     # KronosWithExogenous 和 quantile return head
  training/   # predictor/tokenizer 训练与 IC 回测
  deploy/     # Hugging Face 推送和 FastAPI 服务
  vendor/     # Kronos 源码快照
docs/         # 指南、实验记录、计划书、复盘
examples/     # quickstart 脚本和 universe 快照
tests/        # pytest smoke 和特征测试
```

## 下一步阅读

| 目标 | 文档 |
| --- | --- |
| 浏览文档地图 | [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) |
| 理解 OKX 现货/永续外生因子 | [docs/CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md](docs/CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md) |
| 查看 5/15/30 统一窗口因子提案 | [docs/CRYPTO_5_15_30_FACTOR_SCHEMA_PROPOSAL.md](docs/CRYPTO_5_15_30_FACTOR_SCHEMA_PROPOSAL.md) |
| 解释 IC 回测 | [docs/BACKTEST_IC_INTERPRETATION_GUIDE.md](docs/BACKTEST_IC_INTERPRETATION_GUIDE.md) |
| 远程训练 | [docs/AUTODL_REMOTE_TRAINING_GUIDE.md](docs/AUTODL_REMOTE_TRAINING_GUIDE.md) |
| 阅读当前最佳实验 | [docs/CRYPTO_TOP100_1Y_SPOT_RUN.md](docs/CRYPTO_TOP100_1Y_SPOT_RUN.md) |

## 许可证

MIT。
