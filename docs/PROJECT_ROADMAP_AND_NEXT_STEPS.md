# Kairos 项目路线图与下一步计划

> 这份文档承接根 README 里原来的“下一步计划”部分，专门记录**当前研究优先级、验收标准和工时估算**。
> 目标是让 README 回到“项目入口”，把研究待办留在这里单独维护。

---

## 0. 当前判断

截至 2026-04-21，项目的研究状态可以概括成三点：

- `crypto-1min + h30` 已经证明有 alpha，尤其是 `Top100` 和 `Kronos-base` 结果说明路线本身成立。
- 短 horizon（`h1/h5`）仍然不稳定，问题主要在训练目标设计和现货数据缺少微观结构因子。
- 永续多通道（funding / OI / basis）链路已打通，但第一次实盘实验因为配置残留和评测窗口太短，结论无效，需要按更严格的流程重跑。

优先级因此非常明确：

1. 先修会污染实验结论的 bug 和流程坑。
2. 再扩永续数据，把非零微观结构因子真正用于训练。
3. 最后才考虑更大的模型和更重的 tokenizer 路线。

---

## 1. Tier 1：先修会误导结论的问题

### 1.1 修 bug / 已知坑

| # | 事项 | 完成判定 | 成本 | 依据 |
|---|---|---|---|---|
| D1 | **Top10 30d perp 重跑**：清掉 `KAIROS_N_TRAIN_ITER=5000` 残留，用默认 50000 samples/epoch 重训 10 epoch | finetuned rank-IC > baseline；`val_ce` 降幅 > 0.01 | 0.3h 训练 + 0.1h 回测 | [CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) |
| D2 | **扩 perp test 区到 ≥ 15 天**：避免 `n_dates=3` 这种完全不可靠的 ICIR | 新数据集 `meta.json` 里 test 区 ≥ 15 天；回测 `n_dates ≥ 15` | 0.2h 打包 + 0.1h 回测 | 同上 |
| B1 | **修 `backtest_ic` 的 `auto` bucket 逻辑**：当 `n_dates < 10` 时自动降级到 `none` 并打 warning | 新增回归测试；`auto` 在小 bucket 数下不会再给伪高 ICIR | 0.5h 代码 | [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md) |
| B2 | **修蜡烛形态 `h == l` 边缘情况**：静止 bar 不再被分母钳死为 0 | 测试覆盖 `h == l`；微观结构列在静止 bar 上行为可解释 | 0.5h 代码 | 见 2026-04-20 对话诊断记录 |
| B3 | **修训练 target 的量纲不一致**：统一成 per-step log-return + `sqrt(k+1)` 归一化 | h1/h5 IC 不再系统性失真；h30 不显著退化 | 1h 代码 + 2h 重训验证 | [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md) §8.2 |

这部分的目标不是“做新结果”，而是**确保之后的结果可信**。

---

## 2. Tier 2：把永续这条线做成可复用能力

### 2.1 结构性改进

| # | 事项 | 完成判定 | 成本 | 依据 |
|---|---|---|---|---|
| S1 | **Top100 × 90d OKX 永续**：首次在 perp 上凑足 ≈ 1300 万样本 | h30 rank-IC ≥ +0.030；funding / basis 非零覆盖率 > 95% | 0.5h 采集 + 1h 训练 + 0.5h 回测 | ROI 最高的扩容路径 |
| S2 | **crypto-1min-short preset**：新增 `return_horizon=5` 的 preset，专门验证短 horizon | 短 horizon IC > 同数据的 h30 preset；至少在 BTC/ETH 上成立 | 0.5h preset + 1h 重训 + 回测 | 永续优势理论上应在分钟级更显著 |
| S3 | **Funding / basis 做 regime 异常标签**：把极端状态显式编码，而不是全交给模型隐式学习 | regime=1 时的 hit_rate 明显高于 regime=0 | 2h 代码 + 重训 + 回测 | 前期讨论里的“防御信号 vs 进攻信号” |

这部分的目标是把“现货上已经成立的 h30 alpha”，迁移到更有信息密度的永续数据上。

---

## 3. Tier 3：长期方向

| # | 事项 | 完成判定 | 成本 | 阻塞点 |
|---|---|---|---|---|
| L1 | **OI 实时采集 cron**：持续落盘 `oi_change` 原始流 | 连续 4 周无数据空洞；能回放出非零 OI 特征 | 0.5h 写脚本 + 4 周 wall clock | OKX 历史 OI API 只回溯 ~8 小时 |
| L2 | **Kronos-base / 更大模型**：验证容量是不是瓶颈 | 同数据上 `val_ce` 降 > 0.05 或 h30 IC +0.01 | 3-4h 训练 + 回测 | — |
| L3 | **接入 Coinglass / 第三方数据**：突破 funding / OI 历史深度限制 | 能构造 Top100 × 1 年的非零 funding/OI 数据集 | 付费 + 1-2d adapter 开发 | 预算、数据源选型 |
| L4 | **A 股分钟级**：验证 A 股信号是否只是在日线频率上太弱 | 至少一个 test 区有 h5 rank-IC > +0.02 | 0.5d 采集 + 2h 训练 | akshare 分钟历史深度有限 |

---

## 4. 当前明确不做的事

- **先上方案 B（重训 tokenizer）**
  在方案 A 的 exog slot 还没充分利用之前，成本过高，收益也不够确定。
- **同时开很多新 market adapter**
  当前更大的问题不是市场数量，而是把现有 crypto 永续链路做稳。

---

## 5. 维护方式

后续更新这份文档时，遵循两个规则：

- 新待办必须写清楚：`目标 / 完成判定 / 估计成本 / 依据`
- 已完成条目不要直接删掉，应该标注完成日期或对应 commit，保留研究轨迹

相关文档：

- [CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
