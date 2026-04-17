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

- **数据采集** — `kairos.data.collect`（akshare 多源 fallback）
- **因子工程** — `kairos.data.features`，32 维 `EXOG_COLS`，**无未来信息泄漏**
- **数据集打包** — `kairos.data.prepare_dataset`（time-split / interleave-split）
- **模型** — `kairos.models.KronosWithExogenous`（Kronos + 外生通道 + 分位回归头）
- **训练** — `kairos.training.train_predictor`（DDP + 渐进解冻 + 早停）
- **评估** — `kairos.training.backtest_ic`（IC / Rank-IC / ICIR）
- **部署** — `kairos.deploy.push_to_hf` / `kairos.deploy.serve`

完整术语与背景见 `docs/GLOSSARY.md`。

---

## 2. 目录约定

```
Kairos/
├── kairos/                     # 源代码（唯一的 Python 包）
│   ├── data/                   # collect / features / prepare_dataset
│   ├── models/                 # KronosWithExogenous 等
│   ├── training/               # train_predictor / backtest_ic / dataset / config
│   ├── deploy/                 # push_to_hf / serve
│   ├── utils/                  # training_utils 等
│   └── vendor/                 # 第三方 vendored 代码（如 kronos 源码镜像）
├── docs/
│   ├── AUTODL_GUIDE.md         # 远端 GPU 训练完整手册
│   ├── TUNING_PLAYBOOK.md      # 调参手册 v1→v2
│   └── GLOSSARY.md             # 术语表（新手友好）
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
# 1) 采集（workers=1，mini_racer 线程不安全）
kairos-collect --universe csi300 --freq daily \
  --start 2018-01-01 --end 2026-04-17 --out ./raw/daily --workers 1

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
- ❌ 提交 `raw/` / `artifacts/` / `.venv/` / `*.pkl` / `*.parquet` / checkpoint。

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

更多见 `docs/AUTODL_GUIDE.md` 的 "常见坑" 一节。

---

## 8. 训练 / 回测当前基线

- **v1**（time-split，2024 全年验证）→ 过拟合，test IC 为负。
- **v2**（interleave-split + 降低 lr + 加大 quantile_weight + 早停）→ val_ce 改善，但 test IC 仍为负。
- 结论：监督信号与 A 股未来收益相关性弱；下一步方向写在 `docs/TUNING_PLAYBOOK.md`。

改超参时统一改 `kairos/training/config.py` 的 `TrainConfig`，不要把数字硬编码进 `train_predictor.py`。

---

## 9. 文档维护

- README 是对外门面，改动要精炼；细节写进 `docs/`。
- 新增 `docs/*.md` 时，在 README 目录和 `AGENTS.md` §2 都加一行链接。
- 术语先查 `docs/GLOSSARY.md`，缺了就补；不要在多个文档里重复定义同一个术语。

---

## 10. 本文件本身的维护

- 用户在对话里给出的**长期约定**（比如本文件第 6.1 条就是从用户的 "每次修改之后 commit + push" 提炼的）应该沉淀到这里。
- 改 `AGENTS.md` 本身也适用第 6 节规则：改完立刻 commit + push。
