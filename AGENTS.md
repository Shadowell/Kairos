# AGENTS.md

> This document is the **repository operations manual for AI coding agents** such as Cursor, Claude Code, and Codex.
> Human collaborators can read it too, but the primary audience is the agent. Read it before starting any new task.

---

## 0. TL;DR — 最重要的 5 条

1. **Working directory**: `/Users/jie.feng/wlb/Kairos`, remote `origin = https://github.com/Shadowell/Kairos.git`, main branch `main`.
2. **`git add -A && git commit && git push`** immediately after modifying the code, there is no need to ask for user consent again (see §6).
3. **Answers must be in Chinese** (User Rules).
4. **File operation**: Use Read/Grep/Glob for reading, StrReplace/Write for editing, **Do not use `cat/sed/awk/echo >`** instead.
5. **Retraining/Long Task**: Run in remote AutoDL by default, do not try GPU training on local macOS; see `docs/AUTODL_REMOTE_TRAINING_GUIDE.md` for details.

---

## 1. 仓库定位

Kairos 是基于 [Kronos](https://github.com/shiyu-coder/Kronos) 的 **crypto 现货与永续合约微调/部署工具箱**：

- **数据采集** — `kairos.data.collect` + `kairos.data.markets.crypto`，覆盖 OKX 兼容现货和 USDT 本位永续合约
- **特征工程** — `kairos.data.common_features`（24 维通用因子）+ `adapter.market_features`（8 维市场因子）= 固定 32 维 `EXOG_COLS`，要求**无未来信息泄漏**
- **数据打包** — `kairos.data.prepare_dataset`，支持 time split / interleave split，并始终写入 `meta.json`
- **模型** — `kairos.models.KronosWithExogenous`，即 Kronos + 外生通道 + 分位数收益头；`n_exog=32` 固定
- **训练** — `kairos.training.train_predictor`，支持 DDP、渐进解冻、early stopping，以及 `kairos.training.config.preset_for("crypto-1min")` 等 preset
- **评测** — `kairos.training.backtest_ic`，支持 IC / Rank-IC / ICIR，支持 `--aggregation date/hour/minute/none`，并从 `meta.json` 恢复 market/frequency
- **部署** — `kairos.deploy.push_to_hf` / `kairos.deploy.serve`

完整术语和背景见 `docs/CONCEPTS_AND_GLOSSARY.md`。
---

## 2. 目录约定

```
Kairos/
├── kairos/                       # Source code (only Python package)
│   ├── data/                     # 采集 / 特征 / 数据打包
│   │   ├── collect.py            # crypto 采集 CLI（kairos-collect）
│   │   ├── common_features.py    # 24 维通用因子
│   │   ├── crypto_extras.py      # funding/OI/spot/reference sidecar 读写
│   │   ├── features.py           # 拼接通用因子 + crypto 因子 = 32 维
│   │   ├── prepare_dataset.py    # 生成 train/val/test pkl + meta.json
│   │   └── markets/              # MarketAdapter 抽象 + crypto adapter
│   │       └── crypto_exchanges/ # 交易所后端：okx/binance_vision/…
│   ├── models/                   # KronosWithExogenous + QuantileReturnHead
│   ├── training/                 # tokenizer/predictor 训练、评测、回测、数据集配置
│   ├── deploy/                   # 推送 HF / 服务
│   ├── utils/                    # training_utils 等
│   └── vendor/                   # 第三方 vendored 代码（Kronos 源码快照）
├── docs/                         # 文档：导航、指南、实验记录、计划、复盘
│   ├── DOCUMENTATION_INDEX.md                # 文档导航
│   ├── PROJECT_ROADMAP_AND_NEXT_STEPS.md    # 路线图、优先级、验收标准
│   ├── CONCEPTS_AND_GLOSSARY.md             # 术语和核心概念
│   ├── TRAINING_TUNING_PLAYBOOK.md          # 训练调参和排障
│   ├── BACKTEST_IC_INTERPRETATION_GUIDE.md  # IC 回测配置与结果解释
│   ├── AUTODL_REMOTE_TRAINING_GUIDE.md      # 远程 GPU 训练与 checkpoint 回传
│   ├── CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md # crypto 数据源/交易所/网络配置
│   ├── CRYPTO_BTC_ETH_2Y_SPOT_RUN.md        # BTC+ETH 两年现货 predictor 实验记录
│   ├── CRYPTO_TOP100_1Y_SPOT_RUN.md         # Top100 一年现货 predictor 实验记录
│   ├── CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md # OKX 现货/永续外生因子训练计划
│   ├── CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md # OKX 永续多通道改造计划
│   ├── CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md # OKX 永续 Top10 30 天实验复盘
│   ├── CRYPTO_BTC_ETH_TOKENIZER_RUN.md      # BTC+ETH tokenizer 微调和评测记录
│   └── SERVE_HTTP_API.md                    # kairos-serve JSON 请求/响应
├── scripts/                      # 运维/CI 辅助脚本
│   ├── autodl_bootstrap.sh       # AutoDL 一键初始化
│   ├── package_and_upload.sh     # 打包并上传到 AutoDL
│   └── smoke_crypto_extras.py    # crypto extras 离线 smoke test
├── examples/                     # 快速上手示例
│   ├── inference_quickstart.py
│   └── crypto_top100_universe.md # Top100 固定列表（2026-04-20 快照）
├── tests/                        # pytest 测试
├── raw/                          # 原始 parquet（不入库，见 .gitignore）
├── finetune/data/                # 打包后的 pkl（不入库）
├── artifacts/                    # checkpoint/backtest_report.json（不入库）
├── pyproject.toml                # 包定义 + CLI 入口
├── requirements.txt              # 核心依赖快照
├── .env.example                  # 环境变量模板（API key/proxy/HF）
├── LICENSE                       # MIT
├── README.md
└── AGENTS.md                     # this document
```

**Do not commit files to `raw/` / `finetune/data/` / `artifacts/` / `.venv/`. **

---

## 3. 常用命令速查

### 环境
```bash
# 本地 macOS：只做数据采集、打包和 smoke test
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "numpy<2" scipy      # numpy 2.x is not compatible with torch
```

### 数据链路
```bash
# 1a) 采集 OKX 现货
kairos-collect --market-type spot \
  --universe "BTC/USDT,ETH/USDT" --freq 1min \
  --start 2026-04-01 --end 2026-04-30 \
  --out ./raw/crypto/okx_spot_btc_eth_1min --workers 1

# 1b) 采集 OKX 永续合约
kairos-collect --market-type swap \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" --freq 1min \
  --start 2026-04-01 --end 2026-04-30 \
  --out ./raw/crypto/okx_swap_btc_eth_1min --workers 1 \
  --crypto-extras funding,open_interest,spot,reference
# 办公网降级通道：Binance public mirror，只能拿现货 K 线，无 funding/OI/basis
kairos-collect --market crypto --exchange binance_vision \
  --universe "BTC/USDT,ETH/USDT" --freq 1min \
  --start 2024-01-01 --end 2024-02-01 \
  --out ./raw/crypto/bv_1min --workers 1

# 2) 打包数据集（v2 默认 interleave split）
kairos-prepare --raw ./raw/daily --out ./finetune/data/processed_datasets \
  --train 2018-01-01:2023-12-31 --val 2024-01-01:2024-12-31 \
  --test 2025-01-01:2026-04-17 \
  --split-mode interleave --val-ratio 0.15 --block-days 20 --seed 42
```

### 训练/评测
```bash
# 本地 CPU smoke test
KAIROS_SMOKE=1 python -m kairos.training.train_predictor

# AutoDL GPU，详见 docs/AUTODL_REMOTE_TRAINING_GUIDE.md
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# IC 回测（baseline 对比）
python -m kairos.training.backtest_ic --baseline --horizons 1,5 \
  --out artifacts/backtest_baseline.json
python -m kairos.training.backtest_ic --ckpt artifacts/best_model --horizons 1,5 \
  --out artifacts/backtest_finetuned.json

# tokenizer 独立微调与评测，详见 docs/CRYPTO_BTC_ETH_TOKENIZER_RUN.md
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_tokenizer
python -m kairos.training.eval_tokenizer --baseline --preset crypto-1min \
  --dataset-path ./finetune/data/crypto_1min_btc_eth \
  --out artifacts/tokenizer_eval_baseline.json
python -m kairos.training.eval_tokenizer \
  --ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
  --preset crypto-1min \
  --dataset-path ./finetune/data/crypto_1min_btc_eth \
  --out artifacts/tokenizer_eval_finetuned.json
```

### Git
```bash
git status
git log --oneline -10
git add -A && git commit -m "..." && git push
```

---

## 4. 代码规范

- **Python 3.10+**, follow the style of existing files (don't reformat the entire file without forcing the introduction of black/ruff).
- **No nonsense comments**. `# import pandas`, `# return results` will be deleted directly. Only comment when expressing intent/tradeoffs/constraints.
- **Do not write emoji** unless the user actively requests it in README/docs.
- **Type annotation**: Add new functions as much as possible; when changing existing functions, just keep the style consistent, and don't make major changes just to add type hints.
- **No future information leakage**: Any new factors put into `kairos/data/features.py` must only use data from `t` time and before; the z-score window in the packaging phase is fixed to 60 days rolling.
- **CLI 入口**：新增可执行脚本时，必须在 `pyproject.toml` 的 `[project.scripts]` 注册，并使用 `kairos-` 前缀命名。

---

## 5. 测试和验证

- Change `kairos/data/*` → Run `kairos-prepare --max-symbols 5 --dry-run` (or the smallest subset) to confirm that it is not exploded.
- Change `kairos/training/*` → Run `KAIROS_SMOKE=1 python -m kairos.training.train_predictor` local CPU first.
- Change `kairos/models/*` → Load Kronos-small weights at least once to verify that state_dict matches.
- **No CI**, the agent is responsible for local verification. If there is `pytest`, run `pytest -x`.

---

## 6. Git 提交规则（重要）

### 6.1 触发时机
**As long as a "logically complete change" is completed - such as fixing a bug, adding a document, adjusting a set of super parameters - commit + push immediately without asking the user. **

例外情况如下，遇到这些情况必须先停下询问用户：
- When `git push --force` / `--force-with-lease` is required;
- 当需要切换到 `main` 以外分支、创建新分支或合并 PR 时；
- When the commit will contain files that look like **key, token, API key, `.env`, `credentials.*`**;
- When the user explicitly says "Don't push it yet" / "I want to see it".

### 6.2 Commit message 格式
- **English, imperative sentence, capitalized, no period**, refer to existing history:
  ```
  Fix repo URLs to Shadowell/Kairos and add package init placeholders
  Harden training loop: data fallbacks, interleave split, early-stop, IC backtest
  Add AutoDL guide and glossary; cross-link from README/playbook
  ```
- One-line title ≤ 72 characters; if necessary, leave a blank line and then write the body.
- **One commit does one thing**, don't stuff "bug fixes + refactoring + adding documents" together.

### 6.3 标准动作
```bash
git status                          # 先确认状态
git add -A
git commit -m "<imperative subject>"
git push                            # 推送到 origin/main
git status                          # 确认 clean 且已同步
```
为避免转义问题，使用 HEREDOC 形式：
```bash
git commit -m "$(cat <<'EOF'
Subject line here

Optional body explaining *why*, not *what*.
EOF
)"
```

### 6.4 禁止事项
- ❌ `git commit --amend` The commit that has been pushed.
- ❌ `git push --force` to `main`.
- ❌ `git config` Change the global/repository configuration.
- ❌ `git rebase -i` / `git add -i` (Interactive commands cannot be run).
- ❌ Submit `raw/` / `artifacts/` / `finetune/data/` / `.venv/` / `*.pkl` / `*.parquet` / checkpoint.
- ❌ Submit any file containing a key: `.env` / `*.secret` / `*.pem` / `*.key` / `secrets/` (`.gitignore` has been intercepted, but take a second look at `git status` before doing it).

### 6.5 密钥约定
- All API keys/tokens are passed through **environment variables**, with names in the form `OKX_API_KEY` / `OKX_API_SECRET` / `OKX_API_PASSPHRASE` / `BINANCE_API_KEY` / `HF_TOKEN`.
- During development, fill in the value into `.env` (git-ignored) and load it with `set -a; source .env; set +a`.
- The public template is in `.env.example` (tracked), and the template is updated simultaneously when adding required variables.
- Public market data (K line / funding / OI) does not require any key, and the crypto adapter uses anonymous requests by default.

---

## 7. 已知坑和处理方法

|symptom|root cause|deal with|
|---|---|---|
|`kairos-collect` stuck with no output|Dongcai API is restricted|Existing fallback → tencent → sina; `--workers 1`|
|`kairos-prepare` reported `Parquet magic bytes not found`|macOS `._*` metadata file|The packaging script has been filtered `startswith("._")`; used for transmission `COPYFILE_DISABLE=1 tar --no-xattrs`|
|`kairos-prepare` were all dropped|`amount` List all NaN|Fallbacked to `close * volume`|
| `ModuleNotFoundError: kairos` |venv is not activated or not `pip install -e .`| `source .venv/bin/activate && pip install -e .` |
|AutoDL stuck at `loading Kronos-Tokenizer-base`|`http_proxy` and `HF_ENDPOINT=hf-mirror.com` conflict|`unset http_proxy https_proxy`; Pre-cache `huggingface-cli download` in advance|
|`numpy` 2.x causes torch to crash|version conflict| `pip install "numpy<2"` |
|DDP cannot run on CPU|default nccl|`training_utils.setup_ddp` Automatically cut gloo|
|办公网无法访问 OKX/Binance 主站|GFW + 公司白名单|使用 `--exchange binance_vision` 走 `data-api.binance.vision` 现货镜像；只能拉现货 K 线，无 funding/OI/basis|
|AutoDL is blocked by default `api.okx.com` / `fapi.binance.com`|DNS Pollution + Squid only whitelist github/hf for `/etc/network_turbo`|Install mihomo (Clash Meta) + airport subscription on AutoDL; get YAML from `flag=meta`; pre-download `Country.mmdb` + `GeoSite.dat` + `geoip.dat` and put it in the `-d` directory; the default GLOBAL is `DIRECT`, you need to use `PUT /proxies/GLOBAL` to switch to the specific node (the US node is the fastest measured, ~1.1s/req). See `docs/CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md` for complete steps|
|永续实验中 `funding_rate` / `oi_change` / `basis` 全为 0|没有采集 `--crypto-extras`，sidecar parquet 缺失，或 OKX 没有返回对应历史覆盖|报告 sidecar 覆盖率，并用 `--crypto-extras funding,open_interest,spot,reference` 重新采集；见 `docs/CRYPTO_OKX_SPOT_PERP_EXOGENOUS_PLAN.md`|
|The parquet time range offset pulled out by `binance_vision`|`_to_unix_ms` Use naive local time to convert to UTC|Expected behavior, does not affect training for 24/7 crypto; if you really want accurate UTC date boundary, just manually transfer the complete ISO time|
|The native macOS `torchrun --standalone` is stuck in a pile of `IPv6 ... gai error: 8` warnings for a long time|macOS fails to resolve the local hostname to IPv6, and the rendezvous server of `torchrun` hangs on the hostname and times out.|Single card/native machine does not use torchrun for smoke, just `MASTER_ADDR=127.0.0.1 MASTER_PORT=295xx WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python -m kairos.training.train_predictor`; AutoDL/GPU machine still uses torchrun normally.|
|When smoke `OneCycleLR` throws `ZeroDivisionError: float division by zero`|`total_steps = epochs * steps_per_epoch` If it is too small, `int(pct_start * total_steps)` degenerates to 0 and the phase boundaries coincide.|`KAIROS_SMOKE=1` has set `n_train_iter` to 200 and `warmup_pct=0.2` to ensure `total_steps ≥ ~50`; this lower limit must also be observed when customizing smoke|
|`backtest_ic --per-symbol-limit` After running `by_date_mean.ic` all are `NaN`|Each symbol is independently and equidistantly sampled, and the timestamps extracted are not aligned → there is only 1 record in each bucket, and the cross-sectional correlation coefficient cannot be calculated.|Smoke can use `--aggregation none` to see the overall; to check the bucket IC on a small number of symbols, either `--stride 60` let all symbols use the same set of offsets, or directly run the GPU with full stride=1|
| `ccxt.base.errors.InvalidProxySettings: okx you have multiple conflicting proxy settings(httpProxy,httpsProxy)` |`check_proxy_settings` with ccxt ≥ 4.5 is not allowed to set `http_proxy` + `https_proxy` at the same time; earlier versions of OKX adapter have both blocked|`kairos/data/markets/crypto_exchanges/okx.py` Now only set up `https_proxy` (commit `9e33a2f`), OKX uses all HTTPS; when writing a new adapter, be careful to only leave the https side.|
|`[<sym>] funding fetch failed: 'timestamp'` or OI `50030 Illegal time range`|The `since` kwarg of ccxt is ignored by the server on the OKX funding-history / OI interface → the latest data returned is filtered to empty by `[start_ms,end_ms)` → subsequent `df["timestamp"]` KeyError|adapter is now changed to `params={"after": cursor}` to check funding, `params={"begin":...,"end":...}` to check OI (commit `05b8595`); at the same time, empty frame retains `funding_rate`/`open_interest` columns to avoid KeyError|
|OKX funding / OI 历史窗口较短|OKX API 有硬保留限制：funding-rate-history 约 90 天；contracts/open-interest-history 只返回较短近期窗口，`after` cursor 对该端点不一定生效|**funding**：近 90 天通常可用于训练，旧窗口为空要接受或改用 Coinglass；**OI**：长窗口需要自己实时订阅累积或使用付费历史源，短窗口 smoke 可用，但必须在文档中标注覆盖率|
|`kairos-prepare --train 2026-04-13:2026-04-13` Each symbol is packaged into only **1 line**|`_slice` uses `(datetime >= start) & (datetime <= end)`, `end="2026-04-13"` is parsed into `2026-04-13 00:00:00`, and the minute-level data only has one hit at 00:00|For minute-level data `--train/--val/--test`, you need to pass "next day" as end (`2026-04-13:2026-04-14` means covering the whole day from 04-13), or pass the complete ISO timestamp (be careful not to have 3 colons, `parse_range` use `:` to hard cut)|
|`kairos-prepare --train "2026-04-13 08:00:2026-04-14 08:00"` reported `too many values to unpack (expected 2)`|`parse_range` Directly `split(":")`, the ISO time with `HH:MM` has too many colons|The current solution is to avoid writing ISO time in the CLI and return to the daily granularity `YYYY-MM-DD:YYYY-MM-DD`; a better solution is to subsequently change parse to rsplit or change the separator|
|The training log shows `[TRAIN] pool=327610, using 5000/epoch.`, val_ce only dropped by 0.006 after 10 epochs, and negative migration occurred in backtest|`KAIROS_N_TRAIN_ITER=5000` was left in the previous mini run, but was not cleared in the official run → Only 5000 samples are randomly selected per epoch = 1.5% of the pool|**Before officially running `unset KAIROS_N_TRAIN_ITER`** let it run default 50000; self-check to see if the proportion of `using Y/X` is ≥ 5%. See `docs/CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md` §8.1 for details|
|`backtest_ic --aggregation date` Output `n_dates: 3, icir: +1.17` (looks good but something is wrong)|The test area only has 3 days → the date bucket has only 3 ICs to calculate mean/std, and the ICIR is completely noise|If the test area is < 5 days, use `--aggregation none` to view `overall.spearman`; if the test area is ≥ 15 days, use `by_date_mean`. For the complete decision tree, see `docs/BACKTEST_IC_INTERPRETATION_GUIDE.md` §2|
|`--baseline` ran out h30 ICIR=+0.42, looking at the original weight of Kronos, there is alpha|Random head + Kronos hidden can produce an artificially high ICIR under the scale of 100 symbols × 78 days|**MUST** report both baseline and finetuned, looking at Δ rather than absolute values. See `docs/BACKTEST_IC_INTERPRETATION_GUIDE.md` §5 for details|
|The training ckpt is overwritten (such as the new perp run overwriting the last spot ckpt)|`train_predictor.py` defaults to `artifacts/checkpoints/predictor/checkpoints/best_model/` without run name|`cp -r best_model best_model_<run-name>_backup` before running a new run; next time it is best to change the best_model writing method to the hash/timestamp subdirectory|

See `docs/AUTODL_REMOTE_TRAINING_GUIDE.md`'s "Common Pitfalls" section for more, and `docs/CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`'s complete post-mortem.

---

## 8. 当前训练/回测基线

### crypto 1min(crypto-1min)

| run | universe | h30 rank-IC (finetuned) | h30 ICIR |Details|
|---|---|---|---|---|
| BTC+ETH 2y spot |2 coins × 2 years| **+0.050** | **+0.325** | `docs/CRYPTO_BTC_ETH_2Y_SPOT_RUN.md` |
| Top100 1y spot |100 coins × 1 year| +0.030 | **+0.454** | `docs/CRYPTO_TOP100_1Y_SPOT_RUN.md` |
| Top10 30d perp ⚠️ |10 coins × 30 days|+0.016 (n=3 noise)| +0.06 |`docs/CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md` (negative migration post-mortem)|

h30 is currently the only valid horizon (preset `return_horizon=30` aligned), h1/h5 is not really supervised due to the training target dimension design, and the IC is close to 0 or reverse. For improvement directions, see `docs/BACKTEST_IC_INTERPRETATION_GUIDE.md` §4 and `docs/TRAINING_TUNING_PLAYBOOK.md` §8.2.

### BSQ Tokenizer（crypto-1min, Kairos-base-crypto）

当前只完成了代码和本地 CPU smoke（50 steps / M1 CPU / 17 s）：`recon_mse_full` 从 baseline 的 0.00557 降到 0.00518（-7%），证明链路可用。**正式实验必须跑 AutoDL，完整步骤见 `docs/CRYPTO_BTC_ETH_TOKENIZER_RUN.md`**。
Tokenizer training ≈ 4M parameters, batch=50, epochs=15 + patience=3, lr=2e-4, fully unfrozen, expected to be completed in 5–10 minutes on 5090.

修改超参数时，统一改 `kairos/training/config.py` 的 `TrainConfig`。不要把数字硬编码进 `train_predictor.py`。跨市场/频率参数组合通过 `preset_for(name)` 管理；新增市场或频率时，同步写入 `_PRESETS`，不要让调用方手写散落的 dict。

### 架构不变量（Phase 2）

1. `len(COMMON_EXOG_COLS) + len(adapter.MARKET_EXOG_COLS) == 32`——This must be maintained when adding a new adapter, `build_features` will directly assert and throw an error.
2. `n_exog` on the model side is fixed at 32, and does not follow the adapter. If you want to add new factors, occupy pad or replace a certain slot, and do not expand the dimension.
3. Optional venue dependencies must fail gracefully at import time; command-line errors should happen only when the selected exchange is instantiated.
4. `kairos-prepare` The output directory must contain `meta.json`, otherwise the downstream `backtest_ic --dataset-path ...` cannot restore market/freq.

---

## 9. 文档维护

- README 是对外门面，只放精炼信息；详细过程写到 `docs/`。
- 新增 `docs/*.md` 时，必须同步更新 `docs/DOCUMENTATION_INDEX.md`、README 的文档入口，以及 `AGENTS.md` §2 的目录列表。
- Check `docs/CONCEPTS_AND_GLOSSARY.md` for terms first, and fill in any missing terms; do not define the same term repeatedly in multiple documents.

---

## 10. 本文档自身维护

- The **long-term agreement** given by the user in the conversation (for example, Article 6.1 of this document is extracted from the user's "commit + push after each modification") should be settled here.
- Changing `AGENTS.md` itself also applies to the rules in Section 6: commit + push immediately after making changes.
