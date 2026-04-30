# OKX 现货与永续合约外生因子训练计划

最后更新：2026-04-30。

## 1. 范围

Kairos 当前只聚焦 crypto：

- **现货**：OKX `SPOT` 标的，例如 `BTC-USDT`，在 ccxt 中表示为 `BTC/USDT`。
- **永续合约**：OKX 线性 USDT 本位 `SWAP` 标的，例如 `BTC-USDT-SWAP`，在 ccxt 中表示为 `BTC/USDT:USDT`。
- 不在范围内：股票、期权、币本位/反向合约、账户私有数据、下单执行数据、非 OKX 的交易所特有信号。

训练目标仍然是 Kronos predictor 微调，1 分钟 K 线优先用 h30 IC / Rank-IC / ICIR 评估。

## 2. 连接性检查

本地 macOS 访问 `www.okx.com` 超时。备用服务器 `root@47.79.36.92` 可以访问 OKX 公共接口，但请求需要使用 HTTP/1.1 和类似浏览器的 `User-Agent`。不带这些 header 的普通 Python urllib 请求会收到 HTTP 403。

已在备用服务器验证以下公共接口：

| 接口 | 现货 | 永续 | 可用字段 |
| --- | --- | --- | --- |
| `/api/v5/public/instruments` | 是 | 是 | instrument 元数据、tick size、lot size，合约额外有 contract 元数据 |
| `/api/v5/market/ticker` | 是 | 是 | last、bid/ask 价格和数量、24h high/low/open、24h volume |
| `/api/v5/market/history-candles` | 是 | 是 | timestamp、open、high、low、close、volume、quote volume、confirm |
| `/api/v5/market/books` | 是 | 是 | 买卖盘深度快照 |
| `/api/v5/public/funding-rate` | 否 | 是 | 当前/下一期资金费率、premium、impact value |
| `/api/v5/public/funding-rate-history` | 否 | 是 | 历史资金费率结算值 |
| `/api/v5/public/open-interest` | 否 | 是 | 当前 OI、OI 币种、OI USD |
| `/api/v5/market/mark-price-candles` | 否 | 是 | mark price OHLC |
| `/api/v5/market/index-candles` | 可作参考 | 是 | index price OHLC |
| `/api/v5/public/price-limit` | 否 | 是 | 合约买入/卖出限价 |
| `/api/v5/rubik/stat/contracts/open-interest-volume` | 否 | 是 | 按币种聚合的 OI 和 volume 历史 |
| `/api/v5/rubik/stat/contracts/long-short-account-ratio` | 否 | 是 | 按币种聚合的多空账户比例 |

实现影响：OKX ccxt adapter 已设置类似浏览器的 headers。正式采集仍应放在能稳定访问 OKX 的网络环境中运行。

## 3. 因子可用性矩阵

| 因子族 | 特征 | 现货 | 永续 | 数据源 | 是否进入核心训练 |
| --- | --- | --- | --- | --- | --- |
| 通用 OHLCV | returns、volatility、volume、amount、range、VWAP | 是 | 是 | history candles | 是 |
| 通用 crypto | `market_ret_1` | 是 | 是 | BTC/USDT reference close，缺失时回退到自身 close | 是 |
| 通用 crypto | `market_vol_20` | 是 | 是 | BTC/USDT reference close，缺失时回退到自身 close | 是 |
| 通用 crypto | `hour_sin`、`hour_cos` | 是 | 是 | timestamp | 是 |
| 现货特有 | 现货 bid/ask spread | 是 | 否 | ticker/books 快照 | 暂不进入离线核心训练 |
| 现货特有 | 现货盘口深度不平衡 | 是 | 否 | books 快照 | 暂不进入离线核心训练 |
| 合约特有 | `funding_rate` | 否 | 是 | funding-rate-history | 是 |
| 合约特有 | `funding_rate_z` | 否 | 是 | funding-rate-history | 是 |
| 合约特有 | `oi_change` | 否 | 是 | open-interest / Rubik OI | 有历史覆盖时使用 |
| 合约特有 | `basis` | 否 | 是 | swap close vs spot close，或 mark/index | 是 |
| 合约特有 | premium / mark-index spread | 否 | 是 | funding-rate current、mark/index candles | v2 候选 |
| 合约特有 | long/short account ratio | 否 | 是 | Rubik stats | v2 候选 |
| 合约特有 | price-limit distance | 否 | 是 | price-limit endpoint | v2 候选 |

## 4. 选定的 32 维训练结构

Kairos 保持固定外生维度，避免改变模型结构：

- 24 维通用因子来自 `kairos.data.common_features`。
- 8 维 crypto 因子来自 `CryptoAdapter.MARKET_EXOG_COLS`。

当前 crypto 因子块：

| 槽位 | 特征 | 类型 | 现货填充策略 | 永续填充策略 |
| ---: | --- | --- | --- | --- |
| 1 | `market_ret_1` | 通用 | BTC/USDT reference close 或自身 close | BTC/USDT reference close 或自身 close |
| 2 | `market_vol_20` | 通用 | BTC/USDT reference close 或自身 close | BTC/USDT reference close 或自身 close |
| 3 | `hour_sin` | 通用 | 由时间戳计算 | 由时间戳计算 |
| 4 | `hour_cos` | 通用 | 由时间戳计算 | 由时间戳计算 |
| 5 | `funding_rate` | 合约特有 | `0.0` | funding history 前向填充 |
| 6 | `funding_rate_z` | 合约特有 | `0.0` | funding rolling z-score |
| 7 | `oi_change` | 合约特有 | `0.0` | OI 快照 log-change |
| 8 | `basis` | 合约特有 | `0.0` | swap close / spot close - 1 |

选择这个结构的原因：

- 现货和永续可以共用同一 32 维输入结构。
- 所有因子只使用 `t` 时刻及以前的信息，避免未来信息泄漏。
- 在没有稳定历史盘口快照采集器之前，不把 order book 因子放进离线核心训练。
- 不把现货专属因子放进共享 32 维 schema，避免降低现货和永续模型之间的可比性。

## 5. 数据采集计划

### 现货数据集

命令模板：

```bash
kairos-collect --market-type spot \
  --universe "BTC/USDT,ETH/USDT" \
  --freq 1min --start <START> --end <END> \
  --out ./raw/crypto/okx_spot_<run_name> --workers 1 \
  --crypto-extras reference
```

适用场景：

- BTC/ETH 受控现货实验。
- TopN 现货 universe 实验。
- 与早期 Binance Vision 现货实验做 baseline 对照。

### 永续合约数据集

命令模板：

```bash
kairos-collect --market-type swap \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" \
  --freq 1min --start <START> --end <END> \
  --out ./raw/crypto/okx_swap_<run_name> --workers 1 \
  --crypto-extras funding,open_interest,spot,reference
```

适用场景：

- 永续合约 predictor 实验。
- 现货-合约 basis 实验。
- 对比 OHLCV-only 与多通道 sidecar 的消融实验。

OKX 历史数据限制：

- Funding history 可用，但不是无限保留；过老窗口可能为空。
- Public OI 的历史覆盖比 K 线更受限制。长周期训练如果没有自建快照积累或付费历史源，`oi_change` 可能覆盖不足。
- Order book 是在线快照数据，不能可靠回填；除非明确采集并存储深度快照，否则不应作为离线训练因子。

## 6. 打包与训练计划

现货和永续先分开打包，不要在第一轮正式实验中混合。分开训练更容易判断信号来源。

```bash
kairos-prepare --market crypto --market-type spot \
  --raw ./raw/crypto/okx_spot_<run_name> \
  --train <TRAIN_RANGE> --val <VAL_RANGE> --test <TEST_RANGE> \
  --split-mode interleave --val-ratio 0.15 --block-days 20 \
  --out ./finetune/data/crypto_spot_<run_name>

kairos-prepare --market crypto --market-type swap \
  --raw ./raw/crypto/okx_swap_<run_name> \
  --train <TRAIN_RANGE> --val <VAL_RANGE> --test <TEST_RANGE> \
  --split-mode interleave --val-ratio 0.15 --block-days 20 \
  --out ./finetune/data/crypto_swap_<run_name>
```

训练命令：

```bash
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=./finetune/data/<dataset_name>
unset KAIROS_N_TRAIN_ITER
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor
```

必须评测 baseline 和 finetuned：

```bash
python -m kairos.training.backtest_ic --baseline --preset crypto-1min \
  --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_baseline_<run_name>.json

python -m kairos.training.backtest_ic \
  --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
  --preset crypto-1min --dataset-path "$KAIROS_DATASET" --horizons 30 \
  --out artifacts/backtest_finetuned_<run_name>.json
```

## 7. 实验顺序

1. **现货 BTC/ETH smoke**：验证 OKX 现货采集、打包、本地 smoke training 和 h30 回测链路。
2. **永续 BTC/ETH 多通道 smoke**：验证 funding、OI、spot basis、reference sidecar 在 OKX 有数据覆盖的窗口内非空。
3. **现货 TopN 正式实验**：用 OKX 数据源复现更大 universe 的稳定性收益。
4. **永续 TopN 正式实验**：接入 funding/OI/basis sidecar 训练，并与 OHLCV-only baseline 对比。
5. **消融实验**：永续先禁用 sidecar，再只开 funding，再开 funding+OI+basis，拆分判断哪类外生因子真正贡献信号。

## 8. 验收标准

一个实验只有满足以下条件才可以进入 README 或模型卡：

- `meta.json` 记录 `market=crypto`、`market_type`、`exog_cols`、时间区间和发现的 extras channels。
- 打包阶段确认 32 维 exog columns 符合预期。
- 报告 sidecar 覆盖率：funding 行数、OI 行数、spot basis 行数、reference 行数。
- 同时产出 baseline 和 finetuned 的 h30 IC 报告。
- README/model card 报告相对 baseline 的增量，而不是只报告绝对 IC。
- 如果 test window 小于 15 天，使用 `--aggregation none`，并明确标注为 smoke/noise，不作为正式 ICIR。

## 9. 风险与处理

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| 本地访问 OKX 公共接口受阻 | 采集失败 | 在能访问 OKX 的服务器采集；adapter 保持浏览器式 headers |
| OI 历史稀疏或保留周期短 | `oi_change` 大量为 0 | 报告覆盖率，并做不含 OI 的消融 |
| Funding history 在旧窗口缺失 | funding 列部分为 0 | 选择较近窗口，或明确记录覆盖率 |
| Order book 只能拿快照 | 离线回填不可靠，容易造成数据不一致 | 在有深度快照采集器前，不进入核心 schema |
| 过早混合现货和永续 | 归因不清 | 第一阶段分别训练现货和永续 |
