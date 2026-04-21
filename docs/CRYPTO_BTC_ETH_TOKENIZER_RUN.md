# Crypto BTC/ETH Tokenizer fine-tuning and evaluation records

> Reference before running [`Shadowell/Kairos-small-crypto`](https://huggingface.co/Shadowell/Kairos-small-crypto)
> (Predictor fine-tuning) complete process, this time** put the same set of BTC/USDT + ETH/USDT 2 years 1min
> The data** is used for fine-tuning `NeoQuasar/Kronos-Tokenizer-base` (BSQ tokenizer) to evaluate the reconstruction error.
> Then push it to `Shadowell/Kairos-base-crypto`.
>
> Related documents:
> - [`AUTODL_REMOTE_TRAINING_GUIDE.md`](AUTODL_REMOTE_TRAINING_GUIDE.md) — AutoDL common card rental training manual
> - [`CRYPTO_BTC_ETH_2Y_SPOT_RUN.md`](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md) — Predictor run of the same batch of data, reusing the §1–§4 environment + collection + packaging steps there
> - [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) — Parameter Manual

---

## 0. Why fine-tuning tokenizer alone

Before `Shadowell/Kairos-small-crypto` only moved the last layer of Kronos-small (predictor) + exog/return head,
**tokenizer is still using `NeoQuasar/Kronos-Tokenizer-base`’s official weight**, which is trained on A-shares / US stock daily online,
The bar distribution of BTC/USDT 1min, which is high-frequency and fluctuates more violently, is not optimal:

|symptom|illustrate|
|---|---|
|Codebook utilization is low|Only more than 200 (~23%) of the s1/s2 codebooks of 1024-vocab are actually used; the remaining slots are idle|
|Refactoring MSE is not small|The `recon_mse_full ≈ 0.0056` of baseline on local smoke, per-channel MAE ~5.5%; means that the predictor has been doing lossy compression token school from the beginning|
|The downstream IC ceiling is pressed|Signals missed by the tokenizer can never be learned by the predictor|

Fine-tune one pass of the BSQ tokenizer is expected to:
1. **Reconstruction MSE dropped by 30–60%** (the entire model is being trained, and only 4M parameters are trained, so the fitting is very light);
2. **codebook utilization is up** (crypto’s bar distribution is wider than A-shares daily line);
3. **The IC ceiling of the downstream predictor has been raised** (this will only be reflected when we subsequently retrain the predictor with the new tokenizer + Kronos-small; currently Kairos-small-crypto is **equipped with the old tokenizer** fine-tuning).

---

## 1. Prerequisite: data and environment

If you have already completed one run by pressing [`CRYPTO_BTC_ETH_2Y_SPOT_RUN.md`](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md) §1–§4 (local purchase BTC+ETH 2y 1min
parquet, or already packaged `finetune/data/crypto_1min_btc_eth/`), **jump directly to §3**.

Otherwise, follow the TL;DR of that document:

```bash
# On AutoDL
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

# Collection (Binance Vision direct connection, ~11 min)
kairos-collect --market crypto --exchange binance_vision \
    --universe "BTC/USDT,ETH/USDT" --freq 1min \
    --start 2024-01-01 --end 2026-04-17 \
    --out ./raw/crypto/bv_1min_btc_eth --workers 1

# Pack (~10 s)
kairos-prepare --market crypto \
    --raw ./raw/crypto/bv_1min_btc_eth \
    --train 2024-01-01:2025-12-31 \
    --val   2024-01-01:2025-12-31 \
    --test  2026-01-01:2026-04-17 \
    --split-mode interleave --val-ratio 0.15 --block-days 7 --seed 42 \
    --out ./finetune/data/crypto_1min_btc_eth
```

---

## 2. Local CPU Smoke (optional but recommended)

It is best to run the link locally before starting a long run. CUDA is not required natively on macOS; only the Kronos-Tokenizer-base weights can be loaded.

### 2.1 Prepare 30 days of smoke data

```bash
cd /Users/jie.feng/wlb/Kairos && source .venv/bin/activate

# If you only have 2y data, first pull a 1mo mini set
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

### 2.2 Run 50 steps smoke training

Note: do not use `torchrun --standalone` under macOS, it will get stuck at `IPv6 gai error`
(AGENTS.md §7). Manually set DDP env var:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    KAIROS_SMOKE=1 KAIROS_PRESET=crypto-1min \
    KAIROS_DATASET=./finetune/data/smoke_crypto_tokenizer \
    python -m kairos.training.train_tokenizer
```

Expected output (~6 s for RTX 5090, ~17 s for M1 CPU):

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

> `loss` is `(recon + bsq_loss) / 2`; `bsq_loss` has an entropy regular term, which is often negative when the training is good.
> To judge convergence, only look at `val_recon`.

### 2.3 Smoke Review

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

Smoke result example (2026-04-21 local M1, **50 step training**, not official result):

|index| baseline | finetuned (smoke) | Δ |
|---|---|---|---|
| recon_mse_full | 0.005565 | 0.005178 | **-7.0%** |
| recon_mae_full | 0.05498 | 0.05318 | -3.3% |
| s1 codebook util | 23.6% | 23.6% | 0% |
| s2 codebook util | 10.4% | 10.3% | 0% |

With only 50 steps and 40 val windows, you can see that the MSE begins to decrease; how far can you reach in long-distance running? §5 has the answer.

### 2.4 Smoke Cleanup

```bash
rm -rf finetune/data/smoke_crypto_tokenizer artifacts/checkpoints/tokenizer
```

Be sure to clear smoke's ckpt before formal training, otherwise `best_model` will be mixed.

---

## 3. AutoDL formal training

### 3.1 Startup script

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
export KAIROS_NUM_WORKERS=0      # Prevent DataLoader memory linear explosion (AGENTS.md §7)
# Optional: If the video memory is tight, run a small batch
# export KAIROS_BATCH_SIZE=32

torchrun --standalone --nproc_per_node=1 -m kairos.training.train_tokenizer
SH
chmod +x logs/run_train_tokenizer.sh

nohup bash logs/run_train_tokenizer.sh > logs/train_tokenizer.log 2>&1 &
echo $! > logs/train_tokenizer.pid
tail -f logs/train_tokenizer.log
```

### 3.2 Configuration instructions

|item|value (from `preset_for("crypto-1min")`)|illustrate|
|---|---|---|
| `lookback_window` | 256 min |Tokenizer window size for one compression|
| `predict_window` | 32 min |Tokenizer is not used here, but dataset also takes the `lookback + predict + 1` line|
| `batch_size` | 50 |5090 32GB is enough; 4090 24GB can be reduced to 32|
| `tokenizer_learning_rate` | 2e-4 |Different from predictor 5e-6: the entire model is being trained and no freezing is done.|
| `epochs` | 15 |Maximum number of rounds|
| `patience` | 3 |Stop early if there is no improvement after 3 consecutive epoch val_recon|
| `n_train_iter` | 50000 |Take 50,000 samples every epoch (= 1000 step × batch 50)|
| `warmup_pct` | 0.1 |OneCycleLR warm-up ratio (predictor uses 0.03, tokenizer is slightly larger)|
| `accumulation_steps` | 1 |tokenizer is lightweight and does not require accumulation|

### 3.3 Training curve expectations

| epoch |expected val_recon (full)†|illustrate|
|---|---|---|
| baseline (ep 0) |Around 0.0055|Only load Kronos-Tokenizer-base weight and run val; smoke actual measurement|
| ep 1 | 0.003 — 0.004 |Big drop in 1st epoch|
| ep 3-5 | 0.0020 — 0.0030 |Most of the best checkpoints appear here|
| ep 6+ |Stop at patience=3|val oscillates, stops early|

† **The range is an estimate**, and the actual results are subject to this run. I have seen -7% in 50 steps of smoke, and I should be able to get to -40 ~ -60% in long distance running.

~1 minute per epoch (5090: 50000 samples/batch 50 = 1000 step × ~60 ms/step),
4-6 epoch early stop → **Total wall time ~5-10 minutes**, faster than predictor run (10 minutes 18 seconds).

---

## 4. Evaluation (baseline vs finetuned)

Run `eval_tokenizer` twice, baseline (`NeoQuasar/Kronos-Tokenizer-base`) once,
Once `--ckpt best_model`:

```bash
# Full evaluation (RTX 5090: about 1-2 minutes each time)
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

### 4.1 Meaning of indicators

|Field|meaning|better direction|
|---|---|---|
| `recon_mse_full` |Average MSE (space normalized) after decoding with full codebook (s1+s2)| ↓ |
| `recon_mae_full` |Same as above, MAE| ↓ |
| `recon_mse_pre_s1_only` |MSE decoded using only s1 (first half of the codebook)| ↓ |
| `bsq_loss_mean` |BSQ regularization term used during training (can be negative)| — |
| `per_channel_mse` |MSE cut by OHLCVA 6-channel| ↓ |
| `codebook.s1.utilization` |s1 code table unique usage ratio (0-1)| ↑ |
| `codebook.s1.entropy_bits` |Shannon entropy of s1 token distribution (bits)|Closer to `entropy_max_bits` better|
| `codebook.s2.*` |The same indicator for s2| |

### 4.2 Result summary template

After running, fill in the two JSON numbers into the table below and make it `artifacts/tokenizer_eval_summary.md`.
Use `--metrics-file` when pushing HF to embed it in the README:

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

A one-line script that generates summary:

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

## 5. Push to HuggingFace

```bash
export HF_TOKEN=<your_token>   # Get it from https://huggingface.co/settings/tokens

kairos-push-hf \
    --tokenizer-ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --repo-tokenizer Shadowell/Kairos-base-crypto \
    --market-tag crypto \
    --metrics-file artifacts/tokenizer_eval_summary.md
```

After pushing, the README will automatically embed the content of `artifacts/tokenizer_eval_summary.md`.

### 5.1 Dry-run preview

If you want to take a look at what card looks like:

```bash
kairos-push-hf \
    --tokenizer-ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --repo-tokenizer Shadowell/Kairos-base-crypto \
    --market-tag crypto \
    --metrics-file artifacts/tokenizer_eval_summary.md \
    --dry-run
```

### 5.2 Verification (load it and try again)

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

## 6. TL;DR One-click reproduction of the command list

Assume that you have packed the AutoDL environment + BTC+ETH 2y according to `CRYPTO_BTC_ETH_2Y_SPOT_RUN.md` §1–§4
Done (`finetune/data/crypto_1min_btc_eth/` is all done).

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

# 1. Training (~5-10 min)
nohup torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_tokenizer \
    > logs/train_tokenizer.log 2>&1 &
echo $! > logs/train_tokenizer.pid
tail -f logs/train_tokenizer.log    # Ctrl-C will not kill the process

# 2. Wait for it to early-stop, then evaluate (~1-2 min each)
python -m kairos.training.eval_tokenizer --baseline --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --batch-size 128 --out artifacts/tokenizer_eval_baseline.json

python -m kairos.training.eval_tokenizer \
    --ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --preset crypto-1min \
    --dataset-path ./finetune/data/crypto_1min_btc_eth \
    --batch-size 128 --out artifacts/tokenizer_eval_finetuned.json

# 3. Generate the comparison summary
python - <<'PY' > artifacts/tokenizer_eval_summary.md
# (Script in §4.2 above)
PY

# 4. Push to Hugging Face
export HF_TOKEN=<your_token>
kairos-push-hf \
    --tokenizer-ckpt artifacts/checkpoints/tokenizer/checkpoints/best_model \
    --repo-tokenizer Shadowell/Kairos-base-crypto \
    --market-tag crypto \
    --metrics-file artifacts/tokenizer_eval_summary.md
```

**End-to-end ~15 minutes** (training + two evaluations + push), 5090 cost ≈ ¥1.

---

## 7. Common pitfalls (new)

Based on AGENTS.md §7, tokenizer has several exclusive pitfalls:

|symptom|root cause|deal with|
|---|---|---|
|Training loss is always negative and looks scary.|`bsq_loss` With entropy regularization term, it is often negative when the training is good; the total loss = (recon + bsq)/2 will also follow|**Don’t look at loss, look at val_recon**; val_recon must be ≥ 0|
|val_recon jumps suddenly after opening ep 1|OneCycleLR warmup is running; tokenizer lr=2e-4 is 40 times larger than predictor and will drift a bit in the early stage.|Wait until ep 2-3, patience=3 has been covered|
|ep 1 val_recon is higher than baseline|Warmup stage + not yet adapted to crypto distribution|Normal; don’t advance Ctrl-C|
|`save_pretrained` The saved ckpt push HF cannot be loaded back|Used DDP but did not unpack it.|`train_tokenizer._train` has already revealed `model.module if hasattr(model, "module") else model`|
|eval comes out `codebook.s1.utilization = 1.0`|The amount of data is too small/out of order|Confirm `n_windows` at least 1000+; smoke 40 window is not enough|
|Push HF to report `Repository not found for url: ...Kairos-base-crypto`|The repo has not been created yet|`--token <HF_TOKEN>` will be automatically `create_repo(exist_ok=True)` after it is passed, most likely the token has expired.|
|eval `RuntimeError: mat1 and mat2 shapes cannot be multiplied`|`--dataset-path` points to the A-shares daily package, but the preset uses crypto-1min (lookback=256)|Make `--dataset-path` and `--preset` match; or let `eval_tokenizer` automatically push from `meta.json`|

---

## 8. Follow-up direction

1. **Retrain Kairos-small-crypto with new tokenizer**: Keep the same preset and data, just change
`config.pretrained_tokenizer_path` is replaced by `Shadowell/Kairos-base-crypto`. expected
h30 rank-IC / ICIR can be improved by 10-30% on the original basis (tokenizer fidelity improvement + codebook
The additive effect of increased utilization).
2. **Kronos-Tokenizer-2k (larger codebook) fine-tuning**: Kronos also has a 2k tokenizer (s1/s2
11 bits each), doubling the vocabulary, theoretically able to suppress more crypto-specific regimes. This repository
`train_tokenizer.py` is already common, just change it to `cfg.pretrained_tokenizer_path` and it will run.
3. **Expansion to Top100 crypto**: Data volume × 50, tokenizer’s codebook utilization will increase further;
Reference to the universe of `docs/CRYPTO_TOP100_1Y_SPOT_RUN.md`.
