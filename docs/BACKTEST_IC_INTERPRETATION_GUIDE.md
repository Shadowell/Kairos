# Backtest IC 配置与结果解读指南

> 怎么用 `kairos.training.backtest_ic` 拿到**统计上可信**的 IC / Rank-IC / ICIR，避免被 bucket / stride / horizon 选错带偏。
>
> 这份文档的存在源于 [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) 那次 post-mortem —— 同一个 ckpt + 数据，光是 `--aggregation` 选错，"看起来很好的 ICIR=+1.17" 和 "看起来负迁移的 ICIR=-0.06" 能同时出现在两份报告里。
>
> 术语（IC / Rank-IC / ICIR / hit_rate）见 [`CONCEPTS_AND_GLOSSARY.md`](CONCEPTS_AND_GLOSSARY.md)。

---

## 0. 30 秒速查

| 你想要的 | `--aggregation` | `--horizons` | `--stride` | 看 report 里的 |
|---|---|---|---|---|
| 横截面选股 alpha（生产用） | `date`（n_dates ≥ 15） | 跟 preset `return_horizon` 对齐 | 1 | `by_date_mean.h{H}.rank_ic / icir` |
| pooled "方向预测能力" | `none` | 1, 5, 30 都看 | 1 或 5 | `overall.h{H}.spearman / hit_rate` |
| CPU smoke 验证 | `none` | 1 | 60 + `--per-symbol-limit 50` | `overall` 不为 NaN 即可 |
| 高频/分钟级横截面（≥10 symbols × ≥15 天 test） | `minute` 谨慎用 | 1, 5 | 1 | `by_date_mean`（注意 SE） |

**最重要的反模式**：test 区只 3 天却看 `--aggregation date` 的 ICIR —— 那个数字的统计意义≈ 抛 3 次硬币的方差，无论是 +1.17 还是 -0.6 都不能解读为 alpha 信号。

---

## 1. 输出字段含义

回测 JSON 长这样：

```json
{
  "n_records": 40350,
  "n_symbols": 10,
  "date_range": ["2026-04-17 04:16:00", "2026-04-19 23:30:00"],
  "overall": {
    "h1":  {"pearson": ..., "spearman": ..., "hit_rate": ..., "n": ...},
    "h5":  {...},
    "h30": {...}
  },
  "by_date_mean": {
    "h1":  {"ic": ..., "rank_ic": ..., "icir": ..., "n_dates": ..., "bucket": "..."},
    "h5":  {...},
    "h30": {...}
  }
}
```

### 1.1 `overall`（pooled）

把所有 `(score, return)` 对扔到一起算单个 Pearson / Spearman / hit_rate。

- **优点**：n 大（几万到几百万），统计 SE 小，p-value 直接可信。
- **缺点**：信号源混了（不同时刻、不同 symbol），不反映"在某一时刻 rank 这一组 symbol 的能力"。
- **什么时候看**：sanity check（baseline 应该接近 0，finetuned 显著偏离 0），或 test 区太短没法做时间聚合时**唯一**可信的数字。

### 1.2 `by_date_mean`（cross-sectional → 时间平均）

按 `bucket` 把样本分组（每天 / 每小时 / 每分钟一个 bucket），每 bucket 内独立算 cross-sectional IC，再对 buckets 取平均：
- `ic`：每 bucket 的 Pearson IC 的平均
- `rank_ic`：每 bucket 的 Spearman IC 的平均
- `icir`：`mean(IC) / std(IC)` —— 信息比率
- `n_dates`：non-NaN 的 bucket 数

- **优点**：直接对应"我每天/每小时给出排名，长期下来 rank 的能力"，是组合化使用最看重的指标。
- **缺点**：每 bucket 内样本数 = 该时刻有多少个 symbol；如果只有 2-3 个 symbol，bucket IC 几乎是噪声。

---

## 2. `--aggregation` 怎么选

```python
# kairos/training/backtest_ic.py L53-63
_BUCKET_ALIASES = {
    "date": "date",  "day": "date",  "daily": "date",
    "hour": "hour",  "hourly": "hour",
    "minute": "minute", "minutely": "minute",
    "none": "none",  "pool": "none",
}
```

`auto` 当前实现永远返回 `date`，对**短 test 区**不友好（[CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md §8.3](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)）。

### 选择决策树

```
n_test_days < 5
    └─ 用 --aggregation none，看 overall（pooled），忽略 by_date_mean
n_test_days ≥ 15 且 n_symbols ≥ 5
    └─ 默认 --aggregation date（最常用）
n_test_days ≥ 5 且 freq=1min 且 n_symbols ≥ 10
    └─ 可选 --aggregation hour 或 minute（每 bucket 样本 ≥ 10 才有意义）
```

### 2.1 每个 bucket 的最少样本量

| n_per_bucket | 单 bucket Pearson IC 的标准误 | 评价 |
|---|---|---|
| 2 | 不可计算（需 ≥3） | NaN |
| 3 | ~0.71 | 完全是噪声 |
| 5 | ~0.50 | 极不稳 |
| 10 | ~0.35 | 噪声大；多 bucket 平均后能用 |
| 30 | ~0.19 | OK |
| 100 | ~0.10 | 好 |
| 1000 | ~0.032 | 非常稳 |

经验法则：每 bucket 样本数 ≥ 30 才能开始相信单 bucket 的 IC；< 10 时只看 `mean(IC)` 不要看 `ICIR`（标准差被噪声主导，ICIR 是分母上的噪声放大）。

### 2.2 总 bucket 数（n_dates）

| n_dates | `mean(IC)` 的 SE（假设单 bucket SE = 0.1） | ICIR 是否可信 |
|---|---|---|
| 3 | 0.058 | ❌ 完全不可信（[`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §7.2 的 +1.17/+0.06 都是噪声） |
| 10 | 0.032 | ⚠️ 勉强 |
| 30 | 0.018 | ✅ 可以挂模型选择上 |
| 100+ | 0.010 | ✅ 可以做 paper / 上线决策 |

---

## 3. `--stride` 怎么选

`--stride N` 表示每 N 根 bar 取一个起始窗口。

### 3.1 trade-off

- **stride=1**：所有 bar 都做起点，n_records 最大，IC 估计 SE 最小，但慢且相邻样本相关性高（autocorrelated）。
- **stride > 1**：n_records 缩小 N 倍，单 bucket SE 放大 √N 倍，但 wallclock 也快 N 倍。

### 3.2 经验

| 场景 | 推荐 stride |
|---|---|
| 全量 GPU 回测（生产） | 1 |
| 快速迭代 / 多次 sweep | 5-10 |
| Top100 × 1 年 1min（[`CRYPTO_TOP100_1Y_SPOT_RUN.md`](CRYPTO_TOP100_1Y_SPOT_RUN.md)） | 10（stride=1 估 8h+，stride=10 估 45min） |
| CPU smoke | 60 + `--per-symbol-limit 50` |

### 3.3 `--per-symbol-limit` 的坑

`--per-symbol-limit N` 给每个 symbol 等距抽 N 个起点。**问题**：抽出的时间戳 symbol 之间不对齐，每 bucket 只剩 1-2 个 symbol，cross-sectional IC 全 NaN。

- ✅ smoke 时只看 `--aggregation none` 的 `overall`
- ✅ 想保留 bucket 对齐：用 `--stride 60`（所有 symbol 共用同一组偏移），别用 `--per-symbol-limit`

---

## 4. `--horizons` 怎么选

预设 `crypto-1min` 训练时 `return_horizon=30`，pinball loss 的 target 是 `close[t+k+1] - close[t]` (k=0..29) 的 cumulative diff —— 量纲随 k 线性增大，**k=29 主导整个 loss**，模型实际只优化"未来第 30 步"。

| h | 监督强度 | 预期 IC 表现 |
|---|---|---|
| 1, 5 | 弱（cumulative diff 在 k=0,4 上 loss 项小一个数量级） | 接近 0 或 ~baseline |
| 30 | 强（与 `return_horizon` 对齐） | 主信号 |
| 60+ | 完全外推 | 噪声 |

**结论**：用 `crypto-1min` preset 训出来的 ckpt，**回测时把 `--horizons` 设为包含 30**（如 `1,5,30`），主看 h30；h1 / h5 仅作 sanity（如果 h1 显著负、h30 显著正，说明模型把短 horizon 当成了"长 horizon 的反向预警"，正常）。

如果你换了 preset，把 `--horizons` 跟 `cfg.return_horizon` 对齐。

详细的训练 target 设计问题见 [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §8.2 和 [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) §8.

---

## 5. 必须跑的 `--baseline` 对照

`--baseline` 模式加载 Kronos-small 原权重 + **随机初始化的 exog encoder + return head**，输出的 score 是"hidden state 经过随机 fc 之后"的值。

### 5.1 为什么 baseline 不是 0

直觉上随机 head IC 应该 ≈ 0，但实际 baseline 在 pooled 上经常有 |IC| > 0.02（[`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §7.3）：

> Kronos transformer 主干（136 层 reuse）在 hidden state 里已经编码了"未来分布"的方向信息；random fc 会以一个固定的随机投影把它映射到 score 空间，**这个投影对所有 (score, return) 对是一致的**，所以 pooled 时能蹭到一些方向 IC。

实践含义：**只看 finetuned 的绝对 IC 没意义，必须看 finetuned - baseline 的 Δ**。

### 5.2 推荐的两次回测调用

```bash
# 1. baseline
python -m kairos.training.backtest_ic --baseline \
    --preset <your-preset> \
    --dataset-path <dataset> \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/<run>/backtest_baseline.json

# 2. finetuned
python -m kairos.training.backtest_ic \
    --ckpt <ckpt path> \
    --preset <your-preset> \
    --dataset-path <dataset> \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/<run>/backtest_finetuned.json
```

然后用对比脚本（参考 [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §7 末尾的对比表生成器）拉一张 baseline / finetuned / Δ 三列表。

### 5.3 Sanity regression：每次大改完代码跑一遍

为了确保你改 `train_predictor.py` / `backtest_ic.py` / `kronos_ext.py` 没破坏既有 alpha，固定跑这一步：

```bash
# 加载老 BTC/ETH 数据 + 老 ckpt
python -u -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model_btceth_backup \
    --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --aggregation date --stride 5 \
    --out artifacts/sanity/btceth_$(date +%F).json
```

预期 h30 by_date_mean rank_ic ≈ +0.024（stride=5），ICIR ≈ +0.15。差异 > 50% 就要去查 git log。

---

## 6. 常见误读案例

### 6.1 "ICIR=+1.17 太好了！"

实际是 `n_dates=3`。3 个 IC 的标准差几乎完全是噪声方差。**先看 n_dates，再看 ICIR**。

### 6.2 "baseline 的 h30 ICIR=+0.42 说明 Kronos 原权重就有 alpha"

random head + Kronos hidden 在 100 个 symbol × 78 天的尺度下能凑出虚高 ICIR（[`CRYPTO_TOP100_1Y_SPOT_RUN.md`](CRYPTO_TOP100_1Y_SPOT_RUN.md) §7.5）。**只看 Δ（finetuned - baseline）**。

### 6.3 "finetuned 的 IC 是 +0.003，模型有用！"

p-value 不显著就别声称有用。在 n=40k 上，Spearman IC ≥ 0.01 才大概率 p < 0.05；Spearman IC ≥ 0.02 才稳过。

### 6.4 "by_date_mean 的 IC 是负的，模型废了"

可能是 bucket 选错了（n_per_bucket 太小，IC 噪声大）。先用 `--aggregation none` 看 pooled，pooled 也是负的才是真负。

### 6.5 "h1 是负 IC，模型预测反了"

如果 preset `return_horizon=30`，h1 / h5 是模型没被监督的 horizon，被 cumulative-diff target 主导后甚至可能学成 h30 的反向信号。这是 loss 设计的副作用，不是模型 buggy。**只信 horizon 与 `return_horizon` 对齐的那档**。

---

## 7. 即将做的代码改进（TODO，未实施）

按 ROI 排：

1. **`_BUCKET_ALIASES.auto` 增加 `n_dates < 10` 时降级到 pooled 的逻辑**，并在 stdout 打印 warning。
2. **`backtest_ic` 的 report JSON 加 `n_per_bucket_avg` 字段**，让 reader 一眼看出 bucket IC 是否可信。
3. **`dataset.py` 在 print pool 时加 `using {pct:.1%} of pool`**，pct < 5% 时 warning（避免 [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §8.1 那种 KAIROS_N_TRAIN_ITER 残留陷阱）。
4. **训练 pinball target 量纲**改成 raw log-return + per-k normalization，让 h1/h5 也有真实监督信号。

实施前请先在 [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) §8 留 issue 记录、跑一次 §5.3 sanity regression 确保不破坏现有 alpha。
