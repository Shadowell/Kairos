# Kairos 文档导航

> 这份文档解决一个问题：**你现在要做的事，对应应该看哪篇文档。**
> 根 README 只讲项目概览；这里按任务把所有文档重新归类。

---

## 1. 第一次进入仓库

按这个顺序读：

1. [CONCEPTS_AND_GLOSSARY.md](CONCEPTS_AND_GLOSSARY.md)
   先统一术语。IC、Rank-IC、ICIR、teacher forcing、interleave split 这些词都在这里。
2. [README.md](../README.md)
   看项目定位、当前结果、公开模型、最短上手路径。
3. [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
   如果你打算真正跑训练，这是远端 GPU 的标准工作流。

---

## 2. 按任务找文档

### 我想理解模型 / 指标 / 训练逻辑

- [CONCEPTS_AND_GLOSSARY.md](CONCEPTS_AND_GLOSSARY.md)
  术语、概念、指标的统一解释。
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
  训练参数怎么调，哪些坑最常见，怎么定位问题。
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
  回测参数怎么选，报告里的指标怎么读，哪些结论不可信。

### 我想跑数据采集 / 打包 / 训练

- [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
  从本地开发到远端 GPU 训练、回传 checkpoint 的完整流程。
- [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md)
  crypto 数据源、交易所、代理、网络限制、扩展接入。
- [CRYPTO_BTC_ETH_TOKENIZER_RUN.md](CRYPTO_BTC_ETH_TOKENIZER_RUN.md)
  tokenizer 微调和评测的完整流程。

### 我想看已经做过哪些实验

- [CRYPTO_BTC_ETH_2Y_SPOT_RUN.md](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md)
  BTC/ETH 两币两年现货 predictor 基线实验。
- [CRYPTO_TOP100_1Y_SPOT_RUN.md](CRYPTO_TOP100_1Y_SPOT_RUN.md)
  Binance Spot Top100 一年 predictor 扩容实验。
- [CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)
  OKX 永续 Top10 30 天实验复盘，重点是失败原因和诊断方法。

### 我想知道接下来该做什么

- [PROJECT_ROADMAP_AND_NEXT_STEPS.md](PROJECT_ROADMAP_AND_NEXT_STEPS.md)
  当前路线图、优先级、验收标准、工时估算。
- [CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md)
  永续多通道数据改造的专项计划。

---

## 3. 按角色找文档

### 研究 / 策略视角

- [README.md](../README.md)
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
- [PROJECT_ROADMAP_AND_NEXT_STEPS.md](PROJECT_ROADMAP_AND_NEXT_STEPS.md)

### 工程 / 训练视角

- [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
- [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md)

### AI coding agent / 仓库维护视角

- [AGENTS.md](../AGENTS.md)
  仓库规则、提交要求、目录约定、已知坑。

---

## 4. 文档分层约定

后续新增文档时，按下面规则放置：

- `README.md`
  只放项目概览、核心结果、上手入口、公开模型，不再堆放细节流程和路线图。
- `DOCUMENTATION_INDEX.md`
  只做导航，不承载详细技术内容。
- `*_GUIDE.md`
  讲“怎么做”，偏操作手册。
- `*_PLAYBOOK.md`
  讲“如何调优/排障”，偏经验总结。
- `*_RUN.md`
  讲一次成功实验的完整记录。
- `*_POSTMORTEM.md`
  讲失败实验的复盘和根因分析。
- `*_PLAN.md`
  讲未完成的改造方案和设计取舍。

这套命名的目标是：**看到文件名就知道这篇文档是指南、实验记录、计划书，还是复盘。**
