# AGENTS.md

> 本文件是 **给 AI coding agent（Cursor / Claude Code / Codex 等）看的仓库操作手册**。  
> 人类协作者也欢迎读，但首要读者是 agent。请在每次开始一个新任务前先读一遍本文件。

---

## 0. TL;DR — 最重要的 5 条

1. **工作目录**：`/Users/jie.feng/wlb/Kairos`，远端 `origin = https://github.com/Shadowell/Kairos.git`，主分支 `main`。
2. **修改完代码后立即 `git add -A && git commit && git push`**，不需要再次征求用户同意（见 §6）。
3. **回答统一用中文**（用户规则）。
4. **文件操作**：读用 Read/Grep/Glob，编辑用 StrReplace/Write，**不要用 `cat/sed/awk/echo >`** 代替。
5. **重训/长任务**：默认在远端 AutoDL 跑，不要在本地 macOS 上尝试 GPU 训练；详见 `docs/AUTODL_GUIDE.md`。

---

## 1. 仓库定位

Kairos 是 [Kronos](https://github.com/shiyu-coder/Kronos) 基础模型在 **A 股场景** 下的微调 + 部署工具箱：

- **数据采集** — `kairos.data.collect`（dispatcher）+ `kairos.data.markets.*`（每个市场一个 adapter，A 股 / 加密 / 将来的外汇黄金）
- **因子工程** — `kairos.data.common_features`（24 维通用）+ `adapter.market_features`（8 维市场专属）= `EXOG_COLS` 32 维，**无未来信息泄漏**
- **数据集打包** — `kairos.data.prepare_dataset`（time-split / interleave-split；`--market` 切 adapter；落盘 `meta.json`）
- **模型** — `kairos.models.KronosWithExogenous`（Kronos + 外生通道 + 分位回归头；`n_exog=32` 对所有市场一致）
- **训练** — `kairos.training.train_predictor`（DDP + 渐进解冻 + 早停）+ `kairos.training.config.preset_for("crypto-1min")` 等预设
- **评估** — `kairos.training.backtest_ic`（IC / Rank-IC / ICIR；支持 `--aggregation date/hour/minute/none`，从 `meta.json` 自动推 market/freq）
- **部署** — `kairos.deploy.push_to_hf` / `kairos.deploy.serve`

完整术语与背景见 `docs/GLOSSARY.md`。

---

## 2. 目录约定

```
Kairos/
├── kairos/                     # 源代码（唯一的 Python 包）
│   ├── data/                   # collect (dispatcher) / features / prepare_dataset
│   │   ├── markets/            # MarketAdapter 抽象 + ashare / crypto / ...
│   │   │   └── crypto_exchanges/   # ccxt 封装，每个交易所一个文件（okx / 将来的 binance/bybit）
│   ├── models/                 # KronosWithExogenous 等
│   ├── training/               # train_predictor / backtest_ic / dataset / config
│   ├── deploy/                 # push_to_hf / serve
│   ├── utils/                  # training_utils 等
│   └── vendor/                 # 第三方 vendored 代码（如 kronos 源码镜像）
├── docs/
│   ├── AUTODL_GUIDE.md         # 远端 GPU 训练完整手册
│   ├── TUNING_PLAYBOOK.md      # 调参手册 v1→v2
│   ├── GLOSSARY.md             # 术语表（新手友好）
│   ├── CRYPTO_GUIDE.md         # 加密货币数据层 & 交易所扩展指南
│   ├── CRYPTO_BTC_ETH_RUN.md   # 2026-04-17 BTC+ETH 1min 端到端跑通记录
│   ├── CRYPTO_TOP100_RUN.md    # 2026-04-20 Binance Spot Top100 1min 端到端跑通记录
│   └── CRYPTO_PERP_PLAN.md     # OKX 永续多通道（funding/OI/basis）改造计划书
├── raw/                        # 原始 parquet（不入库，见 .gitignore）
├── finetune/data/processed_datasets/   # 打包后的 *.pkl（不入库）
├── artifacts/                  # checkpoint / backtest_report.json（不入库）
├── tests/                      # pytest 测试
├── pyproject.toml              # 包定义 + CLI 入口
├── README.md
└── AGENTS.md                   # 本文件
```

**不要往 `raw/` / `finetune/data/` / `artifacts/` / `.venv/` 里 commit 文件。**

---

## 3. 常用命令速查

### 环境
```bash
# 本地（macOS）——仅做数据采集、打包、smoke test
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "numpy<2" scipy      # numpy 2.x 与 torch 不兼容
```

### 数据流水线
```bash
# 1a) 采集 — A 股（默认 market=ashare，workers=1，mini_racer 线程不安全）
kairos-collect --universe csi300 --freq daily \
  --start 2018-01-01 --end 2026-04-17 --out ./raw/daily --workers 1

# 1b) 采集 — 加密货币（需要先装 crypto extras）
pip install -e '.[crypto]'
# 默认 OKX 永续（需要代理或直连 OKX；有 funding/OI/basis）
kairos-collect --market crypto \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" --freq 1min \
  --start 2023-01-01 --end 2025-01-01 \
  --out ./raw/crypto/1min --workers 1 \
  --proxy "${HTTPS_PROXY:-}"
# 办公网下的降级通道（Binance 公共镜像，只有现货，没有 funding/OI/basis）
kairos-collect --market crypto --exchange binance_vision \
  --universe "BTC/USDT,ETH/USDT" --freq 1min \
  --start 2024-01-01 --end 2024-02-01 \
  --out ./raw/crypto/bv_1min --workers 1

# 2) 打包（v2 默认 interleave split）
kairos-prepare --raw-dir ./raw/daily \
  --out-dir ./finetune/data/processed_datasets \
  --split-mode interleave --val-ratio 0.15 --block-days 20 --seed 42
```

### 训练 / 评估
```bash
# 本地 CPU smoke test（快速验证链路）
KAIROS_SMOKE=1 python -m kairos.training.train_predictor

# AutoDL GPU（详见 docs/AUTODL_GUIDE.md）
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# 回测 IC（baseline 对比）
python -m kairos.training.backtest_ic --baseline --horizons 1,5 \
  --out artifacts/backtest_baseline.json
python -m kairos.training.backtest_ic --ckpt artifacts/best_model --horizons 1,5 \
  --out artifacts/backtest_finetuned.json
```

### Git
```bash
git status
git log --oneline -10
git add -A && git commit -m "..." && git push
```

---

## 4. 代码规范

- **Python 3.10+**，遵循现有文件的风格（不强制引入 black/ruff 之前不要顺手 reformat 整个文件）。
- **不加废话注释**。`# 导入 pandas`、`# 返回结果` 这类直接删。只在表达**意图/权衡/约束**时加注释。
- **不写 emoji**，除非用户在 README/docs 里主动要求。
- **类型标注**：新写函数尽量加；改动现有函数时保持风格一致即可，不要为了加 type hint 而大改。
- **禁止未来信息泄漏**：任何放进 `kairos/data/features.py` 的新因子都必须只用 `t` 时刻及以前的数据；打包阶段的 z-score 窗口固定为 60 天 rolling。
- **CLI 入口**：新加可执行脚本时，在 `pyproject.toml` 的 `[project.scripts]` 下注册，命名前缀 `kairos-`。

---

## 5. 测试与验证

- 改动 `kairos/data/*` → 跑一次 `kairos-prepare --max-symbols 5 --dry-run`（或最小子集）确认没炸。
- 改动 `kairos/training/*` → 先 `KAIROS_SMOKE=1 python -m kairos.training.train_predictor` 本地 CPU 跑通。
- 改动 `kairos/models/*` → 至少加载一次 Kronos-small 权重验证 state_dict 对得上。
- **没有 CI**，agent 自己负责本地验证。有 `pytest` 就跑 `pytest -x`。

---

## 6. Git 提交规则（重要）

### 6.1 触发时机
**只要完成了一个"逻辑完整的改动"——例如修了一个 bug、加了一个文档、调了一组超参——就立即 commit + push，不需要再问用户。**

例外（这些情况必须先停下问用户）：
- 需要 `git push --force` / `--force-with-lease` 时；
- 改到 `main` 以外的分支 / 新建分支 / 合并 PR 时；
- commit 会包含看起来像**密钥、token、API key、`.env`、`credentials.*`** 的文件时；
- 用户明确说 "先别推" / "我要看看" 时。

### 6.2 Commit message 风格
- **英文、祈使句、首字母大写、不加句号**，参考已有历史：
  ```
  Fix repo URLs to Shadowell/Kairos and add package init placeholders
  Harden training loop: data fallbacks, interleave split, early-stop, IC backtest
  Add AutoDL guide and glossary; cross-link from README/playbook
  ```
- 一行标题 ≤ 72 字符；必要时空一行再写 body。
- **一个 commit 做一件事**，不要把"修 bug + 重构 + 加文档"塞到一起。

### 6.3 标准动作
```bash
git status                          # 先看一眼
git add -A
git commit -m "<imperative subject>"
git push                            # 推到 origin/main
git status                          # 确认 clean + "up to date"
```
用 HEREDOC 形式避免转义问题：
```bash
git commit -m "$(cat <<'EOF'
Subject line here

Optional body explaining *why*, not *what*.
EOF
)"
```

### 6.4 不要做的事
- ❌ `git commit --amend` 已经 push 过的 commit。
- ❌ `git push --force` 到 `main`。
- ❌ `git config` 改全局/仓库配置。
- ❌ `git rebase -i` / `git add -i`（交互式命令跑不起来）。
- ❌ 提交 `raw/` / `artifacts/` / `finetune/data/` / `.venv/` / `*.pkl` / `*.parquet` / checkpoint。
- ❌ 提交任何含密钥的文件：`.env` / `*.secret` / `*.pem` / `*.key` / `secrets/`（`.gitignore` 已拦截，但动手前多看一眼 `git status`）。

### 6.5 Secrets 约定
- 所有 API key / token 通过**环境变量**传递，命名形如 `OKX_API_KEY` / `OKX_API_SECRET` / `OKX_API_PASSPHRASE` / `BINANCE_API_KEY` / `HF_TOKEN`。
- 开发时把值填进 `.env`（git-ignored），用 `set -a; source .env; set +a` 加载。
- 公开模板在 `.env.example`（tracked），新增需要的变量时**同时**更新模板。
- 公共行情数据（K 线 / funding / OI）不需要任何 key，crypto adapter 默认走匿名请求。

---

## 7. 已知坑与固定处理方式

| 症状 | 根因 | 处理 |
|---|---|---|
| `kairos-collect` 卡住无输出 | 东财 API 被限流 | 已有 fallback → tencent → sina；`--workers 1` |
| `kairos-prepare` 报 `Parquet magic bytes not found` | macOS `._*` 元数据文件 | 打包脚本已 filter `startswith("._")`；传输用 `COPYFILE_DISABLE=1 tar --no-xattrs` |
| `kairos-prepare` 全被 drop | `amount` 列全 NaN | 已 fallback 为 `close * volume` |
| `ModuleNotFoundError: kairos` | 没激活 venv 或没 `pip install -e .` | `source .venv/bin/activate && pip install -e .` |
| AutoDL 卡在 `loading Kronos-Tokenizer-base` | `http_proxy` 和 `HF_ENDPOINT=hf-mirror.com` 冲突 | `unset http_proxy https_proxy`；提前 `huggingface-cli download` 预缓存 |
| `numpy` 2.x 导致 torch 崩 | 版本冲突 | `pip install "numpy<2"` |
| DDP 在 CPU 上跑不起来 | 默认 nccl | `training_utils.setup_ddp` 已自动切 gloo |
| 办公网封了 OKX / Binance 主站 | GFW + 公司白名单 | 用 `--exchange binance_vision` 走 `data-api.binance.vision` 现货镜像，只能拉现货 K 线（没有 funding/OI/basis） |
| AutoDL 默认走不通 `api.okx.com` / `fapi.binance.com` | DNS 污染 + `/etc/network_turbo` 的 Squid 只白名单 github/hf | 在 AutoDL 上装 mihomo (Clash Meta) + 机场订阅；`flag=meta` 取 YAML；预下载 `Country.mmdb` + `GeoSite.dat` + `geoip.dat` 放到 `-d` 目录；GLOBAL 默认是 `DIRECT`，要用 `PUT /proxies/GLOBAL` 切到具体节点（US 节点实测最快，~1.1s/req）。完整步骤见 `docs/CRYPTO_PERP_PLAN.md` |
| `--market crypto` 跑出来 `funding_rate` / `oi_change` / `basis` / `btc_dominance` 四列全是 0 | `kairos-collect` 只采 OHLCV，`prepare_dataset` 不传 `extras`，adapter 的 `_align_series` 对缺失 series 就 fillna(0) | 要么接受（当前 BTC/ETH + Top100 现货 run 都是如此），要么走 `docs/CRYPTO_PERP_PLAN.md` 的多通道改造 |
| 用 `binance_vision` 拉出来的 parquet 时间范围偏移 | `_to_unix_ms` 用 naive local time 转 UTC | 预期行为，对 24/7 crypto 不影响训练；真要精确 UTC 日界就手动传完整 ISO 时间 |
| 本机 macOS `torchrun --standalone` 长时间卡在一堆 `IPv6 ... gai error: 8` 警告 | macOS 对本机 hostname 解析到 IPv6 失败，`torchrun` 的 rendezvous server 挂在主机名上等超时 | 单卡/本机 smoke 不用 torchrun，直接 `MASTER_ADDR=127.0.0.1 MASTER_PORT=295xx WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python -m kairos.training.train_predictor`；AutoDL/GPU 机仍正常用 torchrun |
| smoke 时 `OneCycleLR` 抛 `ZeroDivisionError: float division by zero` | `total_steps = epochs * steps_per_epoch` 过小时 `int(pct_start * total_steps)` 退化为 0，阶段边界重合 | `KAIROS_SMOKE=1` 已经把 `n_train_iter` 设到 200、`warmup_pct=0.2`，保证 `total_steps ≥ ~50`；自定义 smoke 时也要守这一下限 |
| `backtest_ic --per-symbol-limit` 跑完 `by_date_mean.ic` 全是 `NaN` | 每个 symbol 独立等距抽样，抽到的时间戳两两不对齐 → 每个 bucket 只 1 条记录，横截面相关系数算不出来 | smoke 用 `--aggregation none` 看 overall 即可；要在少量 symbol 上验 bucket IC，要么 `--stride 60` 让所有 symbol 用同一套偏移，要么直接 GPU 全量 stride=1 跑 |

更多见 `docs/AUTODL_GUIDE.md` 的 "常见坑" 一节。

---

## 8. 训练 / 回测当前基线

- **v1**（time-split，2024 全年验证）→ 过拟合，test IC 为负。
- **v2**（interleave-split + 降低 lr + 加大 quantile_weight + 早停）→ val_ce 改善，但 test IC 仍为负。
- 结论：监督信号与 A 股未来收益相关性弱；下一步方向写在 `docs/TUNING_PLAYBOOK.md`。

改超参时统一改 `kairos/training/config.py` 的 `TrainConfig`，不要把数字硬编码进 `train_predictor.py`。跨市场的参数组合走 `preset_for(name)`——新建市场/频率时**同步**在 `_PRESETS` 里加一条，而不是让调用方自己拼 dict。

### 架构不变式（Phase 2）

1. `len(COMMON_EXOG_COLS) + len(adapter.MARKET_EXOG_COLS) == 32`——加新 adapter 必须保持这一点，`build_features` 会直接 assert 抛错。
2. 模型侧的 `n_exog` 固定 32，不要跟随 adapter 动；要加新因子就占 pad 或换掉某个 slot，不要扩维度。
3. 新 adapter 必须能在**不加 `[crypto]` 这类可选依赖**的环境里 import 失败后被 `kairos/data/markets/__init__.py` 里的 `try/except ImportError` 吞掉，不能把 A 股主路径搞崩。
4. `kairos-prepare` 产出的目录里必须有 `meta.json`，否则下游 `backtest_ic --dataset-path ...` 无法恢复 market/freq。

---

## 9. 文档维护

- README 是对外门面，改动要精炼；细节写进 `docs/`。
- 新增 `docs/*.md` 时，在 README 目录和 `AGENTS.md` §2 都加一行链接。
- 术语先查 `docs/GLOSSARY.md`，缺了就补；不要在多个文档里重复定义同一个术语。

---

## 10. 本文件本身的维护

- 用户在对话里给出的**长期约定**（比如本文件第 6.1 条就是从用户的 "每次修改之后 commit + push" 提炼的）应该沉淀到这里。
- 改 `AGENTS.md` 本身也适用第 6 节规则：改完立刻 commit + push。
