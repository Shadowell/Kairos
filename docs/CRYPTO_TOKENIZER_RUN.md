# Crypto BSQ Tokenizer 微调（Kronos-Tokenizer-base → Kairos-base-crypto）

> 参考之前跑 [`Shadowell/Kairos-small-crypto`](https://huggingface.co/Shadowell/Kairos-small-crypto)
> （predictor 微调）的完整流程，这次**把同一套 BTC/USDT + ETH/USDT 2 年 1min
> 数据**拿来微调 `NeoQuasar/Kronos-Tokenizer-base`（BSQ tokenizer），评测重构误差，
> 再推到 `Shadowell/Kairos-base-crypto`。
>
> 相关文档：
> - [`AUTODL_GUIDE.md`](AUTODL_GUIDE.md) — AutoDL 通用租卡训练手册
> - [`CRYPTO_BTC_ETH_RUN.md`](CRYPTO_BTC_ETH_RUN.md) — 同一批数据的 predictor run，复用那里的 §1–§4 环境 + 采集 + 打包步骤
> - [`TUNING_PLAYBOOK.md`](TUNING_PLAYBOOK.md) — 调参手册

---

## 0. 为什么单独微调 tokenizer

之前 `Shadowell/Kairos-small-crypto` 只动了 Kronos-small (predictor) 的最后 1 层 + exog/return head，
**tokenizer 还在用 `NeoQuasar/Kronos-Tokenizer-base` 的官方权重**，它是在 A 股 / 美股日线上训练的，
对 BTC/USDT 1min 这种高频、波动更剧烈的 bar 分布**不是最优的**：

| 症状 | 说明 |
|---|---|
| Codebook 利用率低 | 1024-vocab 的 s1/s2 码本，实际只用到 200 多个 (~23%)；剩下的 slot 空转 |
| 重构 MSE 不小 | 本地 smoke 上 baseline 的 `recon_mse_full ≈ 0.0056`，per-channel MAE ~5.5%；意味着 predictor 从开始就在做有损压缩的 token 上学 |
| 下游 IC 天花板被压 | tokenizer 漏掉的信号 predictor 再也学不回来 |

Fine-tune 一遍 BSQ tokenizer 有望：
1. **重构 MSE 降 30–60%**（整个模型都在训，而且只训 4M 参数，拟合很轻）；
2. **codebook 利用率起来**（crypto 的 bar 分布比 A 股日线宽）；
3. **下游 predictor 的 IC 天花板抬一截**（只在我们后续把新 tokenizer + Kronos-small 一起重训 predictor 时才会体现；当前 Kairos-small-crypto 是**配旧 tokenizer** 微调的）。

---

## 1. 前置：数据和环境

如果你已经按 [`CRYPTO_BTC_ETH_RUN.md`](CRYPTO_BTC_ETH_RUN.md) §1–§4 跑完一次（本地采 BTC+ETH 2y 1min
parquet，或者已经打包好 `finetune/data/crypto_1min_btc_eth/`），**直接跳到 §3**。

否则按那份文档的 TL;DR：

```bash
# 在 AutoDL 上
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

# 采集（Binance Vision 直连，~11 min）
kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2024-01-01 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_btc_eth --workers 1

# 打包（~10 s）
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_btc_eth \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_btc_eth
```

---

## 2. 本地 CPU Smoke（可选但推荐）

开长跑前最好本地先把链路跑一遍。macOS 本机不需要 CUDA；只要能加载 Kronos-Tokenizer-base 权重即可。

### 2.1 准备 30 天 smoke 数据

```bash
cd /Users/jie.feng/wlb/Kairos && source .venv/bin/activate

# 如果只有 2y 数据，先拉一份 1mo 的 mini 集
kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2026-03-17 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_1mo --workers 1

kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_1mo \
    --train 2026-03-17:2026-04-01 \
    --val   2026-04-01:2026-04-08 \
    --test  2026-04-08:2026-04-15 \
    --split-mode interleave --val-ratio 0.15 --block-days 3 --seed 42 \
    --out ./finetune/data/smoke_crypto_tokenizer
```

### 2.2 跑 50 步 smoke 训练

注意 macOS 下不要用 `torchrun --standalone`，会在 `IPv6 gai error` 卡住
（AGENTS.md §7）。手动设 DDP env var：

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    KAIROS_SMOKE=1 KAIROS_PRESET=crypto-1min \
    KAIROS_DATASET=./finetune/data/smoke_crypto_tokenizer \
    python -m kairos.training.train_tokenizer
```

预期输出（RTX 5090 约 6 s，M1 CPU 约 17 s）：

```
[DDP Setup] Global Rank: 0/1, CPU mode (no CUDA detected)
[tokenizer] loading NeoQuasar/Kronos-Tokenizer-base
Tokenizer size: 4.0M
[TRAIN] pool=53282, using 200/epoch.
[VAL] pool=8928, using 40/epoch.
[ep 1/1 step  5/50] lr=1.26e-04 loss=-0.0259
...
--- ep 1: val_recon=0.005329 (0:00:17 / total 0:00:17) ---
[save] best → artifacts/checkpoints/tokenizer/checkpoints/best_model (val_recon=0.005329)
```

> `loss` 是 `(recon + bsq_loss) / 2`；`bsq_loss` 带熵正则项，**在训练良好时常为负**。
> 判断收敛只看 `val_recon`。

### 2.3 Smoke 评测

```bash
python -m kairos.training.eval_tokenizer --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/smoke_crypto_tokenizer \
    --per-symbol-limit 20 --batch-size 16 \
    --out artifacts/tokenizer_eval_baseline_smoke.json

python -m kairos.training.eval_tokenizer \
    --ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/smoke_crypto_tokenizer \
    --per-symbol-limit 20 --batch-size 16 \
    --out artifacts/tokenizer_eval_finetuned_smoke.json
```

Smoke 结果示例（2026-04-21 本地 M1，**50 步训练**，不是正式结果）：

| 指标 | baseline | finetuned (smoke) | Δ |
|---|---|---|---|
| recon_mse_full | 0.005565 | 0.005178 | **-7.0%** |
| recon_mae_full | 0.05498 | 0.05318 | -3.3% |
| s1 codebook util | 23.6% | 23.6% | 0% |
| s2 codebook util | 10.4% | 10.3% | 0% |

只 50 步、40 个 val 窗口，就能看到 MSE 开始降；长跑能到多少 §5 有答案。

### 2.4 Smoke 清理

```bash
rm -rf finetune/data/smoke_crypto_tokenizer artifacts/checkpoints/tokenizer
```

正式训练前一定要清掉 smoke 的 ckpt，否则 `best_model` 会被混用。

---

## 3. AutoDL 正式训练

### 3.1 启动脚本

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
mkdir -p logs

cat > logs/run_train_tokenizer.sh <<'SH'
#!/bin/bash
set -e
cd /root/autodl-tmp/Kairos
source .venv/bin/activate

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth
export KAIROS_NUM_WORKERS=0      # 防 DataLoader 内存线性爆炸（AGENTS.md §7）
# 可选：显存紧张就开小 batch
# export KAIROS_BATCH_SIZE=32

torchrun --standalone --nproc_per_node=1 -m kairos.training.train_tokenizer
SH
chmod +x logs/run_train_tokenizer.sh

nohup bash logs/run_train_tokenizer.sh > logs/train_tokenizer.log 2>&1 &
echo $! > logs/train_tokenizer.pid
tail -f logs/train_tokenizer.log
```

### 3.2 配置说明

| 项 | 值（来自 `preset_for("crypto-1min")`） | 说明 |
|---|---|---|
| `lookback_window` | 256 min | Tokenizer 一次压缩的窗口大小 |
| `predict_window` | 32 min | Tokenizer 这里不用，但 dataset 一并取了 `lookback + predict + 1` 行 |
| `batch_size` | 50 | 5090 32GB 够用；4090 24GB 可降到 32 |
| `tokenizer_learning_rate` | 2e-4 | 不同于 predictor 的 5e-6：整个模型都在训，不做冻结 |
| `epochs` | 15 | 最大轮数 |
| `patience` | 3 | 连续 3 epoch val_recon 无改善就早停 |
| `n_train_iter` | 50000 | 每 epoch 取 5 万样本（= 1000 step × batch 50） |
| `warmup_pct` | 0.1 | OneCycleLR 热身比例（predictor 用 0.03，tokenizer 略大） |
| `accumulation_steps` | 1 | tokenizer 轻量，不需要 accumulate |

### 3.3 训练曲线预期

| epoch | 预计 val_recon (full)† | 说明 |
|---|---|---|
| baseline (ep 0) | 0.0055 左右 | 仅加载 Kronos-Tokenizer-base 权重跑 val；smoke 实测 |
| ep 1 | 0.003 — 0.004 | 第 1 个 epoch 大幅下降 |
| ep 3-5 | 0.0020 — 0.0030 | 最佳 checkpoint 多数在这里出现 |
| ep 6+ | 停在 patience=3 | val 震荡，早停 |

† **范围是估算**，真实结果以本次 run 为准。smoke 50 步已经看到 -7%，长跑应该能走到 -40 ~ -60%。

每 epoch ~1 分钟（5090：50000 样本 / batch 50 = 1000 step × ~60 ms/step），
4-6 epoch 早停 → **总 wall time ~5-10 分钟**，比 predictor run (10 分 18 秒) 还快。

---

## 4. 评测（baseline vs finetuned）

两次跑 `eval_tokenizer`，一次 baseline（`NeoQuasar/Kronos-Tokenizer-base`），
一次 `--ckpt best_model`：

```bash
# 全量评测（RTX 5090：每次约 1-2 分钟）
python -m kairos.training.eval_tokenizer --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --batch-size 128 \
    --out artifacts/tokenizer_eval_baseline.json

python -m kairos.training.eval_tokenizer \
    --ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --batch-size 128 \
    --out artifacts/tokenizer_eval_finetuned.json
```

### 4.1 指标含义

| 字段 | 含义 | 越好的方向 |
|---|---|---|
| `recon_mse_full` | 用全 codebook (s1+s2) 解码后的平均 MSE（标准化空间） | ↓ |
| `recon_mae_full` | 同上，MAE | ↓ |
| `recon_mse_pre_s1_only` | 只用 s1 (前半 codebook) 解码的 MSE | ↓ |
| `bsq_loss_mean` | 训练时用的 BSQ 正则项（可为负） | — |
| `per_channel_mse` | 按 OHLCVA 6 通道切开的 MSE | ↓ |
| `codebook.s1.utilization` | s1 码表 unique 使用占比（0-1） | ↑ |
| `codebook.s1.entropy_bits` | s1 token 分布的 Shannon 熵（比特） | 接近 `entropy_max_bits` 更好 |
| `codebook.s2.*` | s2 的同样指标 | |

### 4.2 结果汇总模板

跑完把两份 JSON 的数字填进下表，顺便做成 `artifacts/tokenizer_eval_summary.md`，
push HF 时用 `--metrics-file` 把它嵌进 README：

```markdown
## Results on test set (2026-01-01 ~ 2026-04-16, ~304k 1-min bars)

| metric | baseline | finetuned | Δ |
|---|---|---|---|
| recon_mse_full | 0.00xx | 0.00xx | -xx% |
| recon_mae_full | 0.0xx | 0.0xx | -xx% |
| recon_mse_pre_s1_only | 0.0xx | 0.0xx | -xx% |
| s1 codebook utilization | xx.x% | xx.x% | +x pp |
| s2 codebook utilization | xx.x% | xx.x% | +x pp |
| s1 entropy (bits) | x.xx / 10.00 | x.xx / 10.00 | +x.xx |
| s2 entropy (bits) | x.xx / 10.00 | x.xx / 10.00 | +x.xx |

Per-channel MSE drop (finetuned vs baseline):

| channel | baseline | finetuned | Δ |
|---|---|---|---|
| open   | 0.00xx | 0.00xx | -xx% |
| high   | 0.00xx | 0.00xx | -xx% |
| low    | 0.00xx | 0.00xx | -xx% |
| close  | 0.00xx | 0.00xx | -xx% |
| vol    | 0.00xx | 0.00xx | -xx% |
| amt    | 0.00xx | 0.00xx | -xx% |
```

生成 summary 的一行脚本：

```bash
python - <<'PY' > artifacts/tokenizer_eval_summary.md
import json, pathlib
base = json.loads(pathlib.Path("artifacts/tokenizer_eval_baseline.json").read_text())
fine = json.loads(pathlib.Path("artifacts/tokenizer_eval_finetuned.json").read_text())
m_b, m_f = base["metrics"], fine["metrics"]
cb_b, cb_f = base["codebook"], fine["codebook"]
def pct(b, a): return f"{(a - b) / b * 100:+.1f}%" if b else "—"
lines = [
    f"## Results on test set ({base['n_windows']:,} windows, {base['n_symbols']} symbols)",
    "",
    "| metric | baseline | finetuned | Δ |",
    "|---|---|---|---|",
    f"| recon_mse_full | {m_b['recon_mse_full']:.5f} | {m_f['recon_mse_full']:.5f} | {pct(m_b['recon_mse_full'], m_f['recon_mse_full'])} |",
    f"| recon_mae_full | {m_b['recon_mae_full']:.5f} | {m_f['recon_mae_full']:.5f} | {pct(m_b['recon_mae_full'], m_f['recon_mae_full'])} |",
    f"| recon_mse_pre_s1_only | {m_b['recon_mse_pre_s1_only']:.5f} | {m_f['recon_mse_pre_s1_only']:.5f} | {pct(m_b['recon_mse_pre_s1_only'], m_f['recon_mse_pre_s1_only'])} |",
    f"| s1 codebook util | {cb_b['s1']['utilization']*100:.1f}% | {cb_f['s1']['utilization']*100:.1f}% | {(cb_f['s1']['utilization']-cb_b['s1']['utilization'])*100:+.1f} pp |",
    f"| s2 codebook util | {cb_b['s2']['utilization']*100:.1f}% | {cb_f['s2']['utilization']*100:.1f}% | {(cb_f['s2']['utilization']-cb_b['s2']['utilization'])*100:+.1f} pp |",
    f"| s1 entropy | {cb_b['s1']['entropy_bits']:.2f}/{cb_b['s1']['entropy_max_bits']:.2f} | {cb_f['s1']['entropy_bits']:.2f}/{cb_f['s1']['entropy_max_bits']:.2f} | {cb_f['s1']['entropy_bits']-cb_b['s1']['entropy_bits']:+.2f} bits |",
    f"| s2 entropy | {cb_b['s2']['entropy_bits']:.2f}/{cb_b['s2']['entropy_max_bits']:.2f} | {cb_f['s2']['entropy_bits']:.2f}/{cb_f['s2']['entropy_max_bits']:.2f} | {cb_f['s2']['entropy_bits']-cb_b['s2']['entropy_bits']:+.2f} bits |",
    "",
    "Per-channel MSE drop:",
    "",
    "| channel | baseline | finetuned | Δ |",
    "|---|---|---|---|",
]
for ch in ["open", "high", "low", "close", "vol", "amt"]:
    b, f_ = m_b["per_channel_mse"][ch], m_f["per_channel_mse"][ch]
    lines.append(f"| {ch} | {b:.5f} | {f_:.5f} | {pct(b, f_)} |")
print("\n".join(lines))
PY
cat artifacts/tokenizer_eval_summary.md
```

---

## 5. 推送到 HuggingFace

```bash
export HF_TOKEN=<你的 token>   # 从 https://huggingface.co/settings/tokens 拿

kairos-push-hf \
    --tokenizer-ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --repo-tokenizer Shadowell/Kairos-base-crypto \
    --market-tag crypto \
    --metrics-file artifacts/tokenizer_eval_summary.md
```

推送后 README 会自动嵌入 `artifacts/tokenizer_eval_summary.md` 的内容。

### 5.1 Dry-run 预览

如果想先看一眼 card 长什么样：

```bash
kairos-push-hf \
    --tokenizer-ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --repo-tokenizer Shadowell/Kairos-base-crypto \
    --market-tag crypto \
    --metrics-file artifacts/tokenizer_eval_summary.md \
    --dry-run
```

### 5.2 验证（加载回来试一次）

```bash
python - <<'PY'
from kairos.vendor.kronos import KronosTokenizer
import torch
tok = KronosTokenizer.from_pretrained("Shadowell/Kairos-base-crypto").eval()
x = torch.randn(2, 128, 6)   # [B, T, 6-dim OHLCVA]
(z_pre, z), bsq_loss, quant, idx = tok(x)
print("recon shape:", z.shape, "s1_idx:", idx[0].shape, "s2_idx:", idx[1].shape)
print("recon MSE vs random input:", ((z - x) ** 2).mean().item())
PY
```

---

## 6. TL;DR 一键复现命令清单

假设你已经按 `CRYPTO_BTC_ETH_RUN.md` §1–§4 把 AutoDL 环境 + BTC+ETH 2y 打包数据
搞定（`finetune/data/crypto_1min_btc_eth/` 齐了）。

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET=/root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth
export KAIROS_NUM_WORKERS=0
mkdir -p logs

# 1. 训练（~5-10 min）
nohup torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_tokenizer \
    > logs/train_tokenizer.log 2>&1 &
echo $! > logs/train_tokenizer.pid
tail -f logs/train_tokenizer.log    # Ctrl-C 不会杀进程

# 2. 等它 early-stop，然后 evaluate（各 ~1-2 min）
python -m kairos.training.eval_tokenizer --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --batch-size 128 --out artifacts/tokenizer_eval_baseline.json

python -m kairos.training.eval_tokenizer \
    --ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --batch-size 128 --out artifacts/tokenizer_eval_finetuned.json

# 3. 生成对比 summary
python - <<'PY' > artifacts/tokenizer_eval_summary.md
# (上面 §4.2 里的脚本)
PY

# 4. 推 HF
export HF_TOKEN=<your_token>
kairos-push-hf \
    --tokenizer-ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --repo-tokenizer Shadowell/Kairos-base-crypto \
    --market-tag crypto \
    --metrics-file artifacts/tokenizer_eval_summary.md
```

**端到端 ~15 分钟**（训练 + 两次评测 + 推送），5090 成本 ≈ ¥1。

---

## 7. 常见坑（新增）

在 AGENTS.md §7 的基础上，tokenizer 单独有几个专属坑：

| 症状 | 根因 | 处理 |
|---|---|---|
| 训练 loss 一直是负数看着吓人 | `bsq_loss` 带熵正则项，训练良好时常为负；总 loss = (recon + bsq)/2 也会跟着下去 | **不看 loss，看 val_recon**；val_recon 一定是 ≥ 0 的 |
| val_recon 开 ep 1 就突然跳 | OneCycleLR warmup 在跑；tokenizer lr=2e-4 比 predictor 大 40 倍，早期会漂一下 | 等到 ep 2-3，patience=3 已经涵盖 |
| ep 1 val_recon 比 baseline 还高 | warmup 阶段 + 还没适应 crypto 分布 | 正常；别提前 Ctrl-C |
| `save_pretrained` 保存出来的 ckpt 推 HF 加载不回来 | 用了 DDP 没解包 | `train_tokenizer._train` 里已经 `model.module if hasattr(model, "module") else model` 兜底 |
| eval 出来 `codebook.s1.utilization = 1.0` | 数据量太小 / 乱序 | 确认 `n_windows` 至少 1000+；smoke 40 窗口是不够看的 |
| 推 HF 报 `Repository not found for url: ...Kairos-base-crypto` | 还没建 repo | `--token <HF_TOKEN>` 传了就会自动 `create_repo(exist_ok=True)`，多半是 token 过期 |
| eval 时 `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | `--dataset-path` 指向 A 股 daily 包，但 preset 用了 crypto-1min（lookback=256） | 让 `--dataset-path` 和 `--preset` 对得上；或者让 `eval_tokenizer` 从 `meta.json` 自动推 |

---

## 8. 后续方向

1. **用新 tokenizer 重训 Kairos-small-crypto**：保持同样的 preset 和数据，只把
   `config.pretrained_tokenizer_path` 换成 `Shadowell/Kairos-base-crypto`。预期
   h30 rank-IC / ICIR 能在原基础上再抬 10-30%（tokenizer 保真度提升 + codebook
   利用率提升的叠加效应）。
2. **Kronos-Tokenizer-2k（更大码本）微调**：Kronos 还有个 2k tokenizer（s1/s2
   各 11 bits），词表翻倍，理论上能压更多 crypto 特有的 regime。本仓库的
   `train_tokenizer.py` 已经通用，只要改 `cfg.pretrained_tokenizer_path` 就能跑。
3. **扩到 Top100 crypto**：数据量 × 50，tokenizer 的 codebook 利用率会再涨一截；
   参考 `docs/CRYPTO_TOP100_RUN.md` 的宇宙。
