# Kairos Documentation Index

> This document answers one question: **which document you should read for the task you want to do right now.**
> The root README only covers the project overview; this page reorganizes all documents by task.

---

## 1. First Time In The Repository

Recommended reading order:

1. [CONCEPTS_AND_GLOSSARY.md](CONCEPTS_AND_GLOSSARY.md)
   First unify the terminology. The words IC, Rank-IC, ICIR, teacher forcing, interleave split are all here.
2. [README.md](../README.md)
   Look at the project positioning, current results, public models, and the shortest getting started path.
3. [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
   If you plan to actually run training, this is the standard workflow for remote GPUs.

---

## 2. Find documents by task

### I want to understand the model, metrics, or training logic

- [CONCEPTS_AND_GLOSSARY.md](CONCEPTS_AND_GLOSSARY.md)
  Unified explanation of terms, concepts, and indicators.
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
  How to adjust training parameters, which pitfalls are the most common, and how to locate problems.
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
  How to choose backtest parameters, how to read the indicators in the report, and which conclusions are not trustworthy.

### I want to run data collection / packaging / training

- [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
  The complete process from local development to remote GPU training and checkpoint return.
- [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md)
  crypto data source, exchange, proxy, network restrictions, extended access.
- [CRYPTO_BTC_ETH_TOKENIZER_RUN.md](CRYPTO_BTC_ETH_TOKENIZER_RUN.md)
  The complete process of tokenizer fine-tuning and evaluation.

### I want to deploy or call the prediction HTTP API

- [SERVE_HTTP_API.md](SERVE_HTTP_API.md)
  JSON request and response schema for `kairos-serve` (`POST /predict`, `GET /health`).

### I want to see what experiments have been done

- [CRYPTO_BTC_ETH_2Y_SPOT_RUN.md](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md)
  BTC/ETH two-year spot predictor baseline experiment for two currencies.
- [CRYPTO_TOP100_1Y_SPOT_RUN.md](CRYPTO_TOP100_1Y_SPOT_RUN.md)
  Binance Spot Top100 one-year predictor expansion experiment.
- [CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)
  OKX perpetual Top10 30-day experiments post-mortem, focusing on the causes of failure and diagnostic methods.

### I want to know what to do next

- [PROJECT_ROADMAP_AND_NEXT_STEPS.md](PROJECT_ROADMAP_AND_NEXT_STEPS.md)
  Current roadmap, priorities, acceptance criteria, and time estimates.
- [CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md](CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md)
  A special plan for perpetual multi-channel data transformation.

---

## 3. Find documents by role

### Research / Strategic Perspective

- [README.md](../README.md)
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
- [PROJECT_ROADMAP_AND_NEXT_STEPS.md](PROJECT_ROADMAP_AND_NEXT_STEPS.md)

### Engineering / Training Perspective

- [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md)
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
- [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md)
- [SERVE_HTTP_API.md](SERVE_HTTP_API.md)

### AI coding agent / repository maintenance

- [AGENTS.md](../AGENTS.md)
  Repository rules, submission requirements, directory conventions, and known pitfalls.

---

## 4. Documents layering convention

When adding documents later, place them according to the following rules:

- `README.md`
  Only put the project overview, core results, getting started entry, and public model, and no longer stack the detailed process and roadmap.
- `DOCUMENTATION_INDEX.md`
  It is only for navigation and does not carry detailed technical content.
- `*_GUIDE.md`
  Talking about "how to do it" is more about operation manual.
- `*_PLAYBOOK.md`
  Talking about "how to tune/troubleshoot" is more about summarizing experience.
- `*_RUN.md`
  Tell a complete record of a successful experiment.
- `*_POSTMORTEM.md`
  Talk about post-mortem and root cause analysis of failed experiments.
- `*_PLAN.md`
  Talk about unfinished renovation plans and design choices.

The goal of this set of naming is: **When you see the file name, you can know whether the documents are a guide, a run log, a plan, or a post-mortem. **
