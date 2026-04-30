# Kairos 文档索引

> 这个文档只回答一个问题：**当前任务应该先读哪份文档**。
> 根目录 README 只放项目概览；这里按任务和角色重新组织所有文档入口。

---

## 1. 第一次进入仓库

推荐阅读顺序：

1. [CONCEPTS_AND_GLOSSARY.md](CONCEPTS_AND_GLOSSARY.md)
   先统一术语。IC、Rank-IC、ICIR、teacher forcing、interleave split 等概念都在这里。
2. [README.md](../README.md)
   看项目定位、当前结果、公开模型和最短上手路径。
3. [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
   如果要实际训练，这是远程 GPU 的标准流程。

---

## 2. 按任务查找

### 理解模型、指标或训练逻辑

- [CONCEPTS_AND_GLOSSARY.md](CONCEPTS_AND_GLOSSARY.md)
  统一术语、模型输入输出、预测 token、IC 指标、外生因子等基础概念。
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
  如何设置 IC 回测、如何判断 h30 结果是否可信、为什么要看 baseline 差值。
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
  训练调参、常见问题定位、过拟合/负迁移处理。

### 采集、打包或训练数据

- [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
  从本地开发到远程 GPU 训练、checkpoint 回传的完整流程。
- [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md)
  crypto 数据源、交易所、代理、网络限制和扩展接入。
- [CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md](CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md)
  OKX 现货/永续因子可用性、选定外生 schema 和训练计划。
- [CRYPTO_BTC_ETH_TOKENIZER_RUN.md](CRYPTO_BTC_ETH_TOKENIZER_RUN.md)
  tokenizer 微调和评测记录。

### 部署或调用预测 HTTP API

- [SERVE_HTTP_API.md](SERVE_HTTP_API.md)
  `kairos-serve` 的 JSON 请求/响应结构：`POST /predict` 和 `GET /health`。

### 查看已经做过的实验

- [CRYPTO_BTC_ETH_2Y_SPOT_RUN.md](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md)
  BTC/ETH 两年现货 predictor 基线实验。
- [CRYPTO_TOP100_1Y_SPOT_RUN.md](CRYPTO_TOP100_1Y_SPOT_RUN.md)
  Top100 一年现货 predictor 扩展实验。
- [CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)
  OKX 永续 Top10 30 天实验复盘，重点是失败原因和诊断方法。

### 判断下一步做什么

- [PROJECT_ROADMAP_AND_NEXT_STEPS.md](PROJECT_ROADMAP_AND_NEXT_STEPS.md)
  当前路线图、优先级、验收标准和时间估计。
- [CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md](CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md)
  当前 OKX 现货/永续外生因子训练详细计划。
- [CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md)
  永续合约多通道数据改造的早期专项计划。

---

## 3. 按角色查找

### 研究 / 策略视角

- [README.md](../README.md)
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
- [PROJECT_ROADMAP_AND_NEXT_STEPS.md](PROJECT_ROADMAP_AND_NEXT_STEPS.md)
- [CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md](CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md)

### 工程 / 训练视角

- [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
- [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md)
- [CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md](CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md)
- [SERVE_HTTP_API.md](SERVE_HTTP_API.md)

### AI coding agent / 仓库维护

- [AGENTS.md](../AGENTS.md)
  仓库规则、提交要求、目录约定和已知坑。

---

## 4. 文档分层约定

后续新增文档时按以下规则命名和放置：

- `README.md`
  只放项目概览、核心结果、上手入口和公开模型，不堆详细流程和路线图。
- `DOCUMENTATION_INDEX.md`
  只做导航，不承载详细技术内容。
- `*_GUIDE.md`
  偏操作手册，回答“怎么做”。
- `*_PLAYBOOK.md`
  偏调参和排障经验。
- `*_RUN.md`
  一次具体实验记录，必须包含数据范围、命令、指标和结论。
- `*_PLAN.md`
  尚未完成或准备执行的计划书。
- `*_POSTMORTEM.md`
  失败实验复盘，重点写根因、证据和避免再犯的方法。

命名目标：**看到文件名就能判断它是指南、实验记录、计划书还是复盘**。
