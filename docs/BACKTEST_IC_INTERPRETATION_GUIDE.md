# IC 回测配置与结果解释

本指南适用于 `methodology.version = 2` 的回测报告。术语见
[`CONCEPTS_AND_GLOSSARY.md`](CONCEPTS_AND_GLOSSARY.md)。

## 1. 新旧结果的边界

2026-10-05 的统一契约修复同时改变了收益目标、预测锚点、baseline 和截面统计：

- 新模型在最后一个可见 bar `t` 预测 `log(close[t+h]) - log(close[t])`，`h=1..H`。
  标签来自原始价格；历史归一化和截断不影响标签。
- 回测与 HTTP 共用 `KairosPredictor`。历史长度、频率、外生列顺序和期限来自模型包，
  不由当前 `TrainConfig` 默认值覆盖。旧模型包不能自动解释为 log-return 模型。
- baseline 使用原版 Kronos 的自回归价格预测，再转为相同锚点的 log-return。
  不再初始化随机外生编码器或收益头。
- 横截面先在**同一 UTC 时间戳、至少三个不同币种**之间计算，再对日/小时内的 IC 求均值。
  不把一天内所有分钟样本混成一个截面。

历史 BTC/ETH、Top100、Top10 perp 实验及其报告保留，属于旧评测口径。
旧版 h30 IC、ICIR 与 random-head baseline 的差值不能与新版指标直接比较，
也不能作为新版回归测试的预期值。两个币种的新报告不会产生横截面 IC。
需要比较新旧训练方案时，应按新契约重新训练、在同一测试集重新评估。

## 2. 报告字段

| 字段 | 含义 |
| --- | --- |
| `pooled.hH` | 全部币种和时刻混合的 Pearson、Spearman、方向命中率；仅用于总体诊断 |
| `time_series.SYMBOL.hH` | 单个币种跨时间的相关性，区别于截面排序能力 |
| `cross_sectional.hH.timestamps` | 每个有效时间戳的 IC、Rank-IC、不同币种数 |
| `cross_sectional.hH.buckets` | 每日/小时/分钟内有效时间戳 IC 的等权均值 |
| `cross_sectional.hH.summary` | 各有效时间桶的等权均值、ICIR、样本数量 |
| `overall` | `pooled` 的兼容别名 |
| `by_date_mean.hH` | `cross_sectional.hH.summary` 的兼容别名，已改为新统计口径 |
| `seed`, `model`, `evaluation`, `methodology` | 种子、模型/tokenizer 来源、采样参数和方法版本 |

`summary.icir = mean(bucket IC) / sample_std(bucket IC)`，标准差使用 `ddof=1`。
只有一个有效桶、所有桶 IC 相同、常量预测/真值、样本不足或没有连续窗口时，
相应指标为 JSON `null`，不会制造无限值、NaN 或虚假的 ICIR。
`n_dates` 是兼容字段，表示有效时间桶数；优先读含义明确的 `n_buckets`、
`n_timestamps` 和 `n_symbols_mean`。

每个截面内 `(symbol, timestamp)` 必须唯一。缺少外生数据、列错序、时间错位或非有限值
会明确报错；不再补零、截断列或悄悄跳过该币种。
若模型包明确声明 `use_exog=false`，回测不读取外生 sidecar，并向共享推理器传入等长的
空外生输入；原版 Kronos 同样不依赖该文件。
主通道及未来标签必须连续同频，窗口不会跨缺口或 split 中断。

训练包回测必须提供数据集 `meta.json`，至少含 `market`、`freq`、`exog_cols`。
模型包声明非空 `market_type` 时，数据集也必须显式声明相同值；已有 `feature_cols`
同样必须匹配。缺失来源信息的旧数据集需重新运行 `kairos-prepare`，不根据数据形状猜测
现货/永续或频率。已声明的 `schema_version` 只接受整数 `2`；未声明版本的旧 metadata
仍须满足以上字段要求。原版 baseline 可接受无 metadata 的旧数据集，频率等配置由
调用者的 preset/`TrainConfig` 明确指定。

## 3. 聚合与样本数量

| 参数 | 处理方式 | 使用场景 |
| --- | --- | --- |
| `--aggregation date` 或 `auto` | 先逐时刻算 IC，再求日均值，再跨日汇总 | 日级稳定性比较 |
| `--aggregation hour` | 先逐时刻算 IC，再求小时均值 | 日内稳定性诊断 |
| `--aggregation minute` | 同分钟内逐时刻 IC 均值 | 分钟数据可保留逐时刻结果 |
| `--aggregation none` | 把所有有效时刻 IC 求均值，视为单桶；ICIR 为 null | 快速诊断；pooled 仍单独报告 |

不同聚合参数不改变 pooled 和单币种时序结果，也不能通过放大时间桶解决币种不足。
三只币是计算门槛，不是统计可信度门槛；应同时查看币种数、有效时刻、测试日期跨度和
不同市场阶段的表现。只有三天的数据不能支持稳定性结论。

报告中的 `pearson_p`、`spearman_p` 是默认独立样本假设下的诊断值。
相邻历史窗口和未来收益通常重叠，并非独立样本；大样本量、小 p 值不等于可交易收益。
正式统计推断需使用适合时间依赖的区块 bootstrap/HAC 等方法，当前报告不提供该修正。
IC 为正或 finetuned 高于 baseline 也不自动证明扣费后的收益。

## 4. 窗口与期限

`--stride N` 在共享 UTC 时间网格上取锚点。不同上市日期的币种不会因独立行偏移而错位。
`--per-symbol-limit N` 在全部候选锚点的并集上等距选出最多 N 个公共时间戳，
每币种仅评估自身可用的选中时刻；缺失历史的币种仍可能减少某些截面的成员数。
这个参数适合 CPU smoke，不应把小样本当作正式实验。

`--horizons` 必须是互不重复的正整数，且不超过模型包的 `return_horizon`。
新训练目标同时监督 `1..H`，不再把 h1/h5 宣称为“未监督”，也不允许越过 H 外推。
不同期限仍可能具有不同噪声与学习难度，应分别报告。

## 5. baseline 与多种子比较

baseline 默认 tokenizer 为 `cfg.pretrained_tokenizer_path`，不会自动拾取本地 tokenizer
训练产物。`--predictor`、`--tokenizer` 可显式指定原版模型资源；fine-tuned 模式使用模型包
绑定的 tokenizer，并校验内容。为了公平比较，应使用相同测试数据、历史长度、频率、
期限、stride、limit，并记录这些配置。

```bash
# 原版预测包含随机 token 采样，重复运行并保留每个种子的报告。
python -m kairos.training.backtest_ic --baseline \
    --preset crypto-1min --dataset-path <dataset> \
    --horizons 1,5,30 --aggregation date --seeds 11,29,47 \
    --out artifacts/<run>/backtest_baseline.json

# 新版训练包的直接收益分位数推理。
python -m kairos.training.backtest_ic --ckpt <version-2-bundle> \
    --dataset-path <dataset> --horizons 1,5,30 --aggregation date --seed 11 \
    --out artifacts/<run>/backtest_finetuned.json
```

单种子报告含 `seed`；默认种子为 100。`--seeds` 仅用于原版随机 baseline，输出
`runs` 全部报告与 `seed_summary` 各指标的均值、样本标准差及有效种子数。
固定 Python、NumPy、PyTorch 随机状态，并开启 PyTorch 确定性算法；复现范围是相同软件、
设备、batch size 和排序后的数据。不同 GPU/软件版本不承诺位级一致。
不要只挑选最有利的 baseline 种子，应比较整体分布与 finetuned 的差值。

## 6. 验证与历史复盘

离线回归执行 `python -m pytest tests/test_backtest_contract.py -q`，覆盖末根历史锚点、
原始价格 log-return、真实原版 Kronos 小模型生成、多种子复现、严格外生契约、共享采样网格、
真实截面与 pooled 分离，以及空/常量结果。
完整交付还需新模型训练、保存、重载、回测与 HTTP 的同契约集成验证。

历史故障背景可参考
[`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)，
训练流程见 [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md)。
历史报告中的数值及旧版说明用于追溯，不是新版评测的验收标准。
