# AutoDL remote training and backhaul guide

> The standard process when the local Mac does not have CUDA: complete the data collection + data set preparation locally, run fine-tuning on the AutoDL GPU instance, and then pull the checkpoint back to the local machine for backtesting.
>
> This document-aligned version of the code already contains the following key fixes (make sure you're using the latest code):
>
> - `kairos/data/collect.py`: CSI300 constituent stock listing repair, Dongcai failed and automatically downgraded to Tencent/Sina
> - `kairos/data/prepare_dataset.py`: If `amount` is missing, use `close*volume` to approximate and filter the macOS `._xxx.parquet` companion file
> - `kairos/training/train_predictor.py`: Support CPU mode, `KAIROS_SMOKE=1` fast smoke
> - `kairos/training/config.py`: `dataset_path` points to `finetune/data/processed_datasets`
> - `kairos/utils/training_utils.py`: `setup_ddp` automatically switches the `gloo` backend without CUDA

---

## 0. Route overview

```
[Local Mac] [AutoDL GPU Instance]
1. Collect K-line (raw/)──Package scp──▶ 4. Install dependencies
2. Collection index (raw/index_)──Package scp──▶ 5. Generate pkl data set
3. (Optional) Local smoke test 6. Training predictor (the highlight)
◀──scp pull── 7. Pack artifacts
8. Check IC using local backtest
```

**Time/Cost Estimate** (RTX 5090 32GB / ¥2.93/h):

|step|time|cost|
|---|---|---|
|Create instance + transfer data| ~10 min | - |
|Environment preparation + pip install| ~5 min | ¥0.3 |
| kairos-prepare | ~30 s | - |
|Training 30 epochs| **~1.5–2 h** | **¥5–6** |
|Package download| ~5 min | - |
|**total**| **~2.5 h** | **≈ ¥8** |

---

## 1. Create an AutoDL instance

### 1.1 configuration selection

|item|recommend|illustrate|
|---|---|---|
|Card type| **RTX 5090 32GB × 1** |Kronos-small with 25.3M parameters, a single card is enough|
|CPU/memory|25 cores/90GB|DataLoader num_workers can be enlarged|
|data disk|Free 50GB|Total data + HF cache + checkpoint about 10GB|
|billing|**Pay as you go**|It is the most economical to turn off the phone immediately after running.|

### 1.2 Image selection (**most critical**)

RTX 5090 is **Blackwell architecture (sm_120)** and requires **CUDA 12.8+ / PyTorch 2.7+**. The old image cannot run (`no kernel image available` will be reported).

**Order of Priority**:

1. `PyTorch 2.7.0 / Python 3.12 / CUDA 12.8` ← Preferred
2. `PyTorch 2.5.1 / Python 3.12 / CUDA 12.4` ← Second choice, you need to upgrade after entering the machine:
   ```bash
   pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
   ```

### 1.3 SSH information

After the instance is started, you will see the following in the "Shortcut Tools" of the AutoDL console:

- SSH command: `ssh -p <PORT> root@<HOST>`
- Password: Copy it in "Instance Details"

In the rest of this article, `<HOST>` and `<PORT>` will be used as placeholders, and replace them with your own.

---

## 2. Prepare packets locally (on Mac)

### 2.1 Collect K-line and index (if you haven’t done it yet)

```bash
cd /Users/jie.feng/wlb/Kairos
source .venv/bin/activate

# 300 constituent stocks × 2018-2026 daily line, about 12 minutes
kairos-collect --universe csi300 --freq daily \
    --start 2018-01-01 --end 2026-04-17 \
    --out ./raw/daily --workers 1

# CSI300 index daily line (manually collected, not exposed in the script)
python -c "
import akshare as ak, pandas as pd
from pathlib import Path
df = ak.stock_zh_index_daily(symbol='sh000300')
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'date': 'datetime'})
df = df[(df['datetime'] >= '2018-01-01') & (df['datetime'] <= '2026-04-17')].reset_index(drop=True)
for col in ['amount', 'turnover', 'pct_chg', 'vwap']:
    df[col] = pd.NA
df['vwap'] = df['close']
df = df[['datetime','open','high','low','close','volume','amount','turnover','pct_chg','vwap']]
out = Path('raw/index_daily/000300.parquet')
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print(f'saved {out}: {len(df)} rows')
"
```

### 2.2 Packaging code + data

**Key**: macOS metadata files `._xxx` must be blocked when packaging, otherwise `kairos-prepare` will read them as parquet and report `Parquet magic bytes not found`.

```bash
cd /Users/jie.feng/wlb

# Code package (a few MB)
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/kairos_code.tar.gz \
    --exclude='Kairos/.venv' \
    --exclude='Kairos/__pycache__' \
    --exclude='Kairos/**/__pycache__' \
    --exclude='Kairos/.pytest_cache' \
    --exclude='Kairos/*.egg-info' \
    --exclude='Kairos/.git' \
    --exclude='Kairos/raw' \
    --exclude='Kairos/artifacts' \
    --exclude='._*' --exclude='.DS_Store' \
    Kairos/

# Data package (~20 MB)
cd /Users/jie.feng/wlb/Kairos
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/kairos_raw.tar.gz \
    --exclude='._*' --exclude='.DS_Store' \
    raw/

ls -lh /tmp/kairos_code.tar.gz /tmp/kairos_raw.tar.gz
```

### 2.3 Upload to AutoDL

```bash
scp -P <PORT> /tmp/kairos_code.tar.gz /tmp/kairos_raw.tar.gz \
    root@<HOST>:/root/autodl-tmp/
```

> When the network is unstable, use `rsync` to support breakpoint resumption:
> ```bash
> rsync -avzP -e "ssh -p <PORT>" \
>     /tmp/kairos_code.tar.gz /tmp/kairos_raw.tar.gz \
>     root@<HOST>:/root/autodl-tmp/
> ```

---

## 3. AutoDL environment preparation

Log in via SSH:

```bash
ssh -p <PORT> root@<HOST>
```

### 3.1 Check GPU and CUDA

```bash
nvidia-smi                              # Should see RTX 5090 32GB, CUDA 13.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expectation: 2.7.0+cu128 True NVIDIA GeForce RTX 5090
```

If `torch.cuda.is_available()` is `False` or subsequent training reports `sm_120 not compiled`:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

### 3.2 configure HuggingFace mirror (must be done in China)

```bash
cat >> ~/.bashrc << 'EOF'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
EOF
source ~/.bashrc
mkdir -p /root/autodl-tmp/hf_cache
```

> `HF_HOME` must be placed on the data disk `/root/autodl-tmp/`, not the system disk (it is easy to fill up and may be lost when shutting down).

#### ⚠️ Important: Do not open AutoDL academic acceleration and HF mirroring at the same time

If you have executed `source /etc/network_turbo` before, `http_proxy/https_proxy` will be set to point to the overseas agent of AutoDL. This proxy will bypass your request for `hf-mirror.com` overseas, causing the download to freeze (domestic images are proxied to overseas requests and then returned to China).

**diagnosis**:

```bash
env | grep -i proxy
# If you see http_proxy=http://10.x.x.x:xxxxx, you have been tricked.
```

**Fix** (Choose one of two strategies):

Strategy A: Use hf-mirror to turn off academic acceleration (recommended)

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
env | grep -i proxy    # There should be no output
```

Strategy B: Keep the proxy and add hf-mirror to the no_proxy whitelist

```bash
export no_proxy="$no_proxy,hf-mirror.com"
export NO_PROXY="$no_proxy"
```

**Rules**: Do not use an agent for domestic mirrors (hf-mirror, modelscope, Tsinghua/Alibaba pypi); only overseas original sites (`huggingface.co`, `github.com`) require agents.

### 3.3 Unzip code and data

```bash
cd /root/autodl-tmp
tar xzf kairos_code.tar.gz
cd Kairos
tar xzf ../kairos_raw.tar.gz

# verify
ls raw/daily | grep -v '^\._' | wc -l    # Should be 300
ls raw/index_daily                        # There should be 000300.parquet
```

If the number is wrong, or `ls | grep '^\._'` shows that there is a `._xxx` file:

```bash
find raw -name '._*' -delete
```

### 3.4 Install dependencies

```bash
cd /root/autodl-tmp/Kairos
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e '.[train]'
pip install 'numpy<2'        # Avoid numpy 2.x being incompatible with torch precompiled wheel
```

> If `pip install` is slow to PyPI, use Tsinghua mirror:
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 4. Generate training data set

```bash
cd /root/autodl-tmp/Kairos
source .venv/bin/activate

kairos-prepare \
    --raw ./raw/daily \
    --raw-index ./raw/index_daily/000300.parquet \
    --train 2018-01-01:2023-12-31 \
    --val   2024-01-01:2024-12-31 \
    --test  2025-01-01:2026-04-17 \
    --out   ./finetune/data/processed_datasets
```

Expected output (~15 seconds):

```
Saved train_data.pkl: 298 symbols
Saved val_data.pkl: 299 symbols
Saved test_data.pkl: 299 symbols
Saved exog_train.pkl / exog_val.pkl / exog_test.pkl
```

---

## 5. Training predictor (main course)

### 5.1 Adjust hyperparameters (optional but recommended)

Fine-tuning `kairos/training/config.py` for AutoDL machine with 25-core CPU and 32GB video memory:

```python
num_workers: int = 8          # Original 2 → 8, feeding data faster
batch_size: int = 64          # Original 50 → 64, utilizing 32GB video memory
epochs: int = 30              # Keep, about 1.5-2h
```

### 5.2 Start training (tmux is recommended)

`tmux` allows you to disconnect SSH at any time and train without stopping:

```bash
tmux new -s train
cd /root/autodl-tmp/Kairos && source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_predictor
```

**Exit tmux session**: `Ctrl+B` then `D`. Disconnecting SSH or shutting down the computer will not affect it.

**Reconnect**:

```bash
ssh -p <PORT> root@<HOST>
tmux attach -t train
```

### 5.3 nohup alternative

If you don’t want to install tmux:

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
nohup torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_predictor \
    > train.log 2>&1 &

tail -f train.log
```

### 5.4 Training monitoring

Open a second SSH window:

```bash
# Look at GPU usage (should be stable at >80%)
watch -n 2 nvidia-smi

# View training log
tail -f /root/autodl-tmp/Kairos/train.log

# See checkpoint saving progress
ls -lth /root/autodl-tmp/Kairos/artifacts/checkpoints/predictor/checkpoints/
```

**Normal log snippet**:

```
[ep 1/30 step 100/2000] lr=2.3e-05 loss=3.4218 ce=2.8901
...
--- ep 1: val_ce=2.7834 (0:02:45 / total 0:02:45) ---
[save] best → .../best_model
```

### 5.5 Run smoke first before running formal training (strongly recommended)

Before starting a long run, first use 1 epoch + 40 step to confirm that the entire link is OK:

```bash
KAIROS_SMOKE=1 torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_predictor
```

It is expected that it will take 2-5 minutes to run an epoch and save a checkpoint. No problem, run formal training again (remove `KAIROS_SMOKE=1`).

---

## 6. After training, pull the checkpoint back to the local computer

### 6.1 Packaging on AutoDL

```bash
cd /root/autodl-tmp/Kairos
tar czf /root/autodl-tmp/artifacts.tar.gz artifacts/
ls -lh /root/autodl-tmp/artifacts.tar.gz    # Typically 1-3 GB
```

### 6.2 Download on Mac

```bash
# on Mac
scp -P <PORT> root@<HOST>:/root/autodl-tmp/artifacts.tar.gz ~/Downloads/

cd /Users/jie.feng/wlb/Kairos
tar xzf ~/Downloads/artifacts.tar.gz
ls artifacts/checkpoints/predictor/checkpoints/best_model/
```

### 6.3 Shutdown/Release

- **Shutdown only**: The data disk will be retained and will be restored in 5 seconds when the computer is turned on next time (no more timely charges, but a small storage fee will be charged based on the hard disk)
- **Release instance**: The data disk is **completely cleared**, only do this when no more data is needed.

It is recommended to **shut down** and keep it for a few days to confirm that there is no problem with the checkpoint before releasing it.

---

## 7. Local backtest (on Mac, CPU is enough)

```bash
cd /Users/jie.feng/wlb/Kairos
source .venv/bin/activate

# Run finetune/qlib_test.py or your own backtest script
python finetune/qlib_test.py
```

observe:

- **IC (Information Coefficient)** > 0.03 is marginal; > 0.05 is pretty good
- **RankIC** is more stable than IC
- Stratified backtest curve should be monotonic

---

## 8. Quick check on common pitfalls

|error message|reason|solve|
|---|---|---|
| `Parquet magic bytes not found` |macOS `._xxx` companion files| `find raw -name '._*' -delete` |
| `no kernel image available for execution on device sm_120` |PyTorch version does not support 5090|Install torch 2.7+cu128|
| `ModuleNotFoundError: No module named 'kairos'` |Forgot `pip install -e .` or forgot to activate venv| `source .venv/bin/activate` |
|`git clone` stuck|AutoDL → GitHub link poor|`source /etc/network_turbo` or use mirror `gh-proxy.com`, or from local scp|
|HF download stuck|Direct connection to huggingface.co|Set `HF_ENDPOINT=https://hf-mirror.com`|
|HF is still stuck at `loading NeoQuasar/...` after installing the image|The academic acceleration agent also bypasses mirror requests overseas.|`unset http_proxy https_proxy` or add `hf-mirror.com` to `no_proxy`|
| `CUDA out of memory` |batch_size is too large|Turn down to 32 or 24|
| `RuntimeError: NCCL` |CPU machine misuses nccl backend|Processed automatically, no intervention required|
|`torch.compile` Report sm_120 bug|PyTorch 2.7 is not fully stable with the Blackwell compiler|Disable torch.compile (disabled by default)|

---

## 9. Subsequent incremental synchronization

If the code is changed locally during training, the changes will only be synchronized without retransmitting the data:

```bash
# Local Mac
rsync -avz -e "ssh -p <PORT>" \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='raw' --exclude='artifacts' --exclude='.git' \
    --exclude='._*' --exclude='.DS_Store' \
    /Users/jie.feng/wlb/Kairos/ \
    root@<HOST>:/root/autodl-tmp/Kairos/
```

The same goes for pulling back incremental checkpoints from AutoDL in reverse:

```bash
rsync -avzP -e "ssh -p <PORT>" \
    root@<HOST>:/root/autodl-tmp/Kairos/artifacts/ \
    /Users/jie.feng/wlb/Kairos/artifacts/
```

---

## 10. Tips to save money

1. **Preparation work is all done locally** (collection, packaging, code debugging), the GPU machine is only responsible for training
2. **Smoke first and then officially run**, saving a few hours of bills for "changing a word and restarting"
3. **Pay-as-you-go**, not daily or monthly; shut down immediately after running
4. **Multiple cards are not necessarily cost-effective**: Kronos-small 25M parameter single card 5090 is enough; multiple cards are only used when the memory is full or the data is extremely large
5. **Pay attention to AutoDL’s nighttime discounts** (discounted prices are available during certain periods/some cards)
