# Kairos concept and glossary

> This document explains the nouns that will appear repeatedly in the Kronos fine-tune process in the order of "what we did in this project". Each word comes with specific examples and actual numbers. After reading it, you should be able to understand every sentence in the v1 / v2 training report.

Table of contents:

1. [Data layer: K-line/factor/exogenous variable](#1-data layer)
2. [Model layer: Kronos / Tokenizer / Transformer / quantile regression head] (#2-Model layer)
3. [Training Layer: Supervision Objective/Loss Function/Optimizer/Learning Rate Scheduling](#3-Training Layer)
4. [Evaluation Layer: IC / Rank-IC / ICIR / Hit Rate] (#4-Evaluation Layer)
5. [Common "symptoms": overfitting/distribution shift/leakage](#5-Common symptoms)
6. [Engineering words: ssh/tmux/nohup/scp/HF mirror/torchrun](#6-Engineering words)
7. [The hyperparameters used in our two rounds of training are explained one by one] (#7-The hyperparameters used in our two rounds of training are explained one by one)

---

## 1. Data layer

### 1.1 K-line (Candlestick / OHLCV)

A-shares has a K line for each trading day, containing 6 fields:

| Column name | Meaning |
|---|---|
| `open` | Opening price |
| `high` | Highest price of the day |
| `low` | The lowest price of the day |
| `close`| Closing price |
| `volume` (`vol`) | Number of shares traded |
| `amount` (`amt`) | Transaction amount (yuan) |

Our `raw/daily/000001.parquet` contains the daily lines of Ping An Bank in the past 2009 trading days (2018-01-02 ~ 2026-04-16).

> ⚠️ The K-line returned by some data sources (such as Sina and Tencent) **does not have the amount column**. We made a caveat in `prepare_dataset.py`: `amount ≈ close × volume`, otherwise the downstream `dropna` will clear the entire table.

### 1.2 Factor/Exogenous Feature

In addition to the original K-line, we also manually calculated a bunch of "features" to feed the model, such as:

- **5/10/20/60 Daily Moving Average** (Moving Average)
- **RSI** (relative strength index, measures overbought and oversold)
- **MACD** (fast and slow moving average difference)
- **Volatility** (standard deviation of returns over the last N days)
- **Excess returns relative to CSI300** (difference between individual stocks vs. CSI 300)
-…

These **derived features** are called "Exogenous Features" (`exog` in the code) in the model, with a total of **32 columns** defined in `EXOG_COLS` of `kairos/data/features.py`.

> The word "exogenous" comes from econometrics and refers to variables that "come from outside the model. The model is not responsible for predicting itself, but only uses it as an input." The opposite is the "endogenous variable" (the y that the model itself wants to predict).

### 1.3 Universe (stock pool)

Collection of which stocks to use for training. We use the **CSI 300** (CSI 300), 300 large-cap stocks. In the code `get_universe("csi300")` returns these 300 stock codes.

### 1.4 Parquet

A column-stored binary table format that is much smaller than CSV and faster to read and write. `raw/daily/000001.parquet` is one. Read with `pd.read_parquet()`.

### 1.5 Pickle(.pkl)

Python's serialization format can save any Python object into a binary file. Our `train_data.pkl` stores `Dict[symbol_str, pd.DataFrame]`: each stock corresponds to a K-line DataFrame.

### 1.6 Train / Val / Test set (three points)

Machine learning standard segmentation:

- **Train (training set)**: The model is "visible" and used to update weights
- **Val (validation set / Validation)**: The model is "invisible, but we can see it". It is tested once after each epoch to determine when to stop and which checkpoint to select.
- **Test (test set)**: **Fully isolated**. Only use it to report the final results after everything has been trained. **You cannot** go back and adjust parameters based on the test results, otherwise it would be "using test to adjust parameters" = cheating.

Our two-round cut method:

| Version | Train | Val | Test |
|---|---|---|---|
| v1 | 2018-2023 | **2024 (single year)** | 2025-present |
| v2 | 85% of the blocks from 2018-2024 | 15% of the blocks from 2018-2024 (randomly drawn across New Years) | 2025-present |

The val of v1 is concentrated in 2024, a single year, with a large style deviation (see [Distribution Shift](#52-Distribution Shift-distribution-shift)), which is also one of the main reasons for the failure of v1.

### 1.7 Block-level Interleaved Split (block-level interleaved split)

The cutting method used in v2: first cut all trading days from 2018 to 2024 into consecutive **20-day blocks** (about a month), then randomly select 15% from all blocks as val, and the rest as train.

```
                ┌──────── 2018 ────────┐  ┌──── 2024 ────┐
 blocks: [B001][B002][B003][B004][B005]...[B079][B080][B081]
 assign:   T    V    T    T    V  ...  T    T    V       ← random
```

In this way, val covers various years and various market styles, and will not only see 2024 like v1.

Why not just "daily random 15%"? Because day-by-day random selection will make val and train adjacent days highly correlated (yesterday was in train and today is in val), which is equivalent to **data leakage** in disguise. Press block to ensure at least a 20-day "quarantine" between val and train.

---

## 2. Model layer

### 2.1 Kronos (our fine-tune base model)

NeoQuasar is an open source financial time series pre-training model. The structure is:

```
K line (T×6) ──► Tokenizer ──► token id sequence ──► Transformer ──► Predict the next token
```

Features: It discretizes continuous K lines into tokens (similar to cutting a picture into patches), and then predicts the next token like a language model. We are using **Kronos-small** (~25M parameters).

HuggingFace repository: `NeoQuasar/Kronos-small` / `NeoQuasar/Kronos-Tokenizer-base`

### 2.2 Tokenizer (Tokenizer/Discretizer)

In NLP, the tokenizer cuts "I love you" into three token ids: `[I, love, you]`.
In Kronos, the tokenizer maps the 6-dimensional continuous vector `[open, high, low, close, vol, amt]` of each K line into **two integer tokens** (`s1_id`, `s2_id`) - so it is called **Hierarchical** (two-level) tokenization.

When we train, the **tokenizer weights are frozen** and only use it for preprocessing; what is really being learned is the Transformer part.

### 2.3 Transformer

The core structure of modern large models. Without going into details, just understand:

- Consists of multiple **Transformer blocks (layers)** stacked
- Each layer is mainly **self-attention + feedforward**
- Kronos-small is **8 layers**
- The deeper the layer, the closer it is to the output, and the learned features are more "task relevant"
- The shallower the layer, the closer it is to the input and the more "common" the learned features are.

### 2.4 Embedding

Convert discrete token ids into continuous vectors. For example token id `37` → `[0.12, -0.4, 0.8, ...]` (`d_model` dimension). `d_model=256` in Kronos.

### 2.5 Freeze/Unfreeze

**Freeze a layer** = The weights of this layer are **not updated** during training (`requires_grad = False`).

Why freeze? The base model is pre-trained with billions of pieces of data, and common knowledge is shallow. We only have 300 shares × 7 years of data. If the entire model is fully trained, this common knowledge will be easily "washed away" (catastrophic forgetting).

So the approach is: **Freeze the shallow layers, and only unfreeze the last few layers + our newly added head**. This is what `unfreeze_last_n` means.

v1: `unfreeze_last_n=2` → Unfreeze the last 2 layers transformer + exog encoder + return head
v2: `unfreeze_last_n=1` → unfreeze last 1 layer + new header (more conservative)

### 2.6 Exogenous Encoder (exogenous encoder)

Kronos originally only took 6-dimensional OHLCV. We added a **bypass channel** in `kairos/models/kronos_ext.py`: the 32-dimensional exog feature mentioned in 1.2 above is first projected into a `Linear → SiLU → Linear` into the same 256-dimensional dimension as the main channel, and then added to the token embedding.

```
  OHLCV token ──► embedding (256-d) ──┐
                                      +──► Transformer
  Exog (32-d) ──► linear proj (256-d)─┘
```

The advantage of this is that it retains the transferability of Kronos’ original pre-trained weights and stuffs our additional features into it.

The gate `gate` is initialized with zero, so the contribution of exog is 0 at the beginning of training, which is equivalent to the original Kronos; during the training process, the model decides whether to activate this bypass.

### 2.7 Return Head (quantile regression head)

We add an **extra output header** to the last hidden state layer of Transformer. What it does:

- Given the hidden state at the current time t → predict the **return quantiles** for **h=5 days** in the future (9 quantiles: 0.1, 0.2, ..., 0.9)

General regression only outputs the "median" point. Quantile regression predicts 9 more quantiles → directly gives the predicted confidence interval. The loss used for training is **pinball loss** (quantile loss).

During the backtest, we only take the median (5th quantile = 0.5) as the "next step revenue prediction value", and then calculate IC with the real revenue.

---

## 3. Training layer

### 3.1 Epoch / Step / Batch

- **Batch** (batch): A set of samples used in one forward + back propagation. `batch_size=50` means feeding the model 50 windows at a time.
- **Step** (step): After completing a "forward + reverse + parameter update", it is called a step.
- **Epoch** (round): "Going through" the training set is called an epoch.

Our `n_train_iter=50000` means: **Randomly draw 50,000 samples from the training pool for each epoch** (instead of running the entire pool, the training pool actually has 360,000 samples). So each epoch = 50000 / 50 = **1000 step**.

### 3.2 Sample / Window (Sample / Window)

For time series forecasting, a "sample" is a sliding window:

```
Daily line of a certain stock───────────────────────────►
        ↑                   ↑
[lookback=90 days] [predict=10 days]
└──── A sample ────┘
These 100 days are input into the model together, and the model needs to predict future tokens.
```

So 360,000 samples ≠ 360,000 dates, but "300 shares × ~1200 effective sliding starting points per share".

### 3.3 Teacher Forcing

Standard practice when training language models. The model predicts token by token:

- When predicting the t+1th token, **the context input to the model uses the "real" token before t**, rather than the token predicted by the model itself in the previous step.
- This way the training is more stable and converges faster
- Teacher forcing cannot be used during inference (because the true value is unknown), and you can only use your own predictions, so there is a training-inference distribution difference.

### 3.4 Loss function (Loss)

A scalar measure of "how much the model predictions differ from the true value." Training goal = make loss smaller and smaller.

Our loss is the weighted sum of two parts:

```
total_loss = ce_weight * CE_loss  +  quantile_weight * Pinball_loss
```

#### 3.4.1 CE Loss (Cross-Entropy, cross entropy)

Used for "the prediction accuracy of the model spitting out token id": Give `vocab_size` an option, and the model wants to say which token is the next one.

The `val_ce=4.57` seen in v1 / `val_ce=2.95` in v2 are all this value.

Intuitive understanding:
- `ce = ln(vocab_size)` means complete guessing (uniform distribution)
- Kronos's s1 vocab_size = 1024, guessing = ln(1024) ≈ **6.93**
- Our v2 value of 2.95 means that the model "narrows the possible options to approximately $e^{2.95} ≈ 19$"

#### 3.4.2 Pinball Loss (point loss)

Used for quantile regression head. For each quantile q ∈ [0.1, 0.9], the cost of penalizing "predictions that are smaller than the true value" and "predicting that is larger than the true value" is different:

```
loss_q = max(q·(y - ŷ), (q-1)·(y - ŷ))
```

The sum of the losses across all 9 quantiles is the total pinball.

### 3.5 Optimizer

How to use the gradient of loss to update the weight algorithm. We use **AdamW** - the weighted attenuation version of Adam, which is the most common choice now.

Key parameters:

- **learning rate (learning rate `lr`)**: the "step size" of parameter update at each step. Too big → unstable oscillation; too small → slow to learn.
- **weight decay**: Regularization term, which pulls the weight a little toward 0 at each step to prevent overfitting.
- **betas (β1, β2)**: Momentum parameters, generally use the default (0.9, 0.95).

We use `lr=4e-5` for v1, and drop v2 to `lr=5e-6` (reduced by 8 times), because v1 sees val_ce climbing from ep1, indicating that the newly initialized head gradient is too large and is blown away.

### 3.6 Learning Rate Schedule (learning rate schedule)

lr is not kept constant during the training process, but changes according to rules. We use **OneCycleLR**:

```
lr ──┐   ← ── warmup: from lr/10 to max_lr
     │ ╱╲
     │╱  ╲______  ← ── cosine: slowly decreases from max_lr to close to 0
     0──────────► step
        ↑
pct_start (warmup as a proportion of total training)
```

- **warmup**: Do not use the maximum lr directly at the beginning, but slowly increase it from `lr/10`. The reason is: the weight of the newly initialized layer (exog, return_head) is random, and it is easy to use large lr directly.
- **pct_start**: Warmup proportion. v1 = 3% (too short, the new head will start cosine decline before it is stable), v2 = 10% (warmup is gentler).

### 3.7 early stopping(Early Stopping)

During the training process, if **val loss does not decrease** for N consecutive epochs, it will stop early, no matter how many epochs are left.

- `patience`: tolerate a few epochs before stopping. We use `patience=3` for v2.
- Benefits: Save computing power; avoiding continued running will only make it worse with training (overfitting).
- What is retained after training is "the checkpoint of the epoch with the lowest val loss in history", not the last one.

v2 actually happens:
```
ep1 val=2.97 [save]
ep2 val=2.95 [save best]  ← ── This is what is saved
ep3 val=3.02 [patience 1/3]
ep4 val=3.31 [patience 2/3]
ep5 val=3.72 [patience 3/3 → stop]
```

### 3.8 Checkpoint (checkpoint/ckpt)

File snapshot of model weights. We save it in HuggingFace format:

```
best_model/
├─ config.json        ← Model architecture description
├─ model.safetensors  ← Weight (97 MB)
└─ README.md
```

It can be loaded directly with `KronosWithExogenous.from_pretrained("best_model/")`.

### 3.9 DDP (DistributedDataParallel, distributed data parallel)

During multi-card training, each GPU runs a part of the batch and then synchronizes the gradient mechanism. Single card training is not used, but the `torchrun --nproc_per_node=1` framework is still used in the code - just world_size=1 is reduced to a single card.

There is a communication library called `nccl` (NVIDIA Collective Communications Library) behind it, which will fall back to `gloo` during CPU training.

### 3.10 DataLoader / Sampler

`DataLoader` is responsible for batch sampling from Dataset → forming batch → feeding to GPU. `DistributedSampler` is a sampler that allows each card to only see a subset of data in a DDP scenario.

### 3.11 Random Seed (random seed)

Numbers that make "random processes reproducible". We use `kairos-prepare --seed 42` to fix block sampling to ensure that you can get the same segmentation if you rerun it.

---

## 4. Evaluation layer

There are only these 4 indicators in the backtest report, and they must be understood.

### 4.1 IC (Information Coefficient, information coefficient)

**IC = Pearson correlation coefficient (model prediction score, true future rate of return)**

```
IC = cov(score, ret) / (std(score) * std(ret))
```

Range [-1, 1]:
- `IC = +1`: The higher the prediction, the greater the future return (perfect positive correlation)
- `IC = 0`: Don’t care
- `IC = -1`: Completely reverse (you can also make money by using it in reverse)

Quantitative industry experience value:
- `|IC| < 0.02`: almost ineffective
- `|IC| = 0.03 ~ 0.05`: There is signal but weak
- `|IC| > 0.05`: already good
- `|IC| > 0.1`: Top in the industry (mostly backtest overfitting)

Our v2's `h1 IC = -0.020` is basically at the level of "invalid + slightly reversed".

### 4.2 Rank-IC (Rank IC / Spearman Correlation)

**Rank-IC = Spearman correlation coefficient (score, ret)** = Pearson after replacing both columns with "rank".

Why this? Because the actual stock selection is concerned with the ranking of "who is higher and who is lower" (selecting the Top K stocks to go long), rather than the absolute size of the predicted value. Rank-IC is more robust to outliers.

Generally `|Rank-IC| > |IC|` indicates that the model can provide ranking signals but the absolute value prediction is inaccurate. On the contrary, it means that the model is biased by outliers.

### 4.3 ICIR (IC Information Ratio)

Calculating the cross-sectional IC for each day (prediction vs true value for 300 stocks on this day), we get 216 daily ICs for 216 trading days. Then:

```
ICIR = mean(daily_IC) / std(daily_IC)
```

Meaning: "IC stability". IC is positive but fluctuates greatly → ICIR is low, indicating unreliability. The real offer strategy looks more at ICIR than IC:

- `ICIR > 0.3`: quite stable factor
- `ICIR > 0.5`: good
- US v2: `ICIR = -0.13`: both negative and unstable

### 4.4 Hit Rate / Directional Accuracy

**The proportion of the model predicting the rising/falling direction is consistent with the reality**. A random guess is 50%.

- `55%`: slightly helpful
- `60%+`: difficult to achieve (taking into account real market noise)

We were at 49-50% both rounds, which is pretty much a wild guess.

### 4.5 Baseline

**"Do Nothing" cross-reference**. The baseline comparison we did:

- `Kronos-small` without fine-tuning (heads randomly initialized) is directly used to predict the test set → `h1 IC = -0.007`
- `v1 fine-tune` → `h1 IC = -0.021` (worse than baseline)
- `v2 fine-tune` → `h1 IC = -0.020` (also worse than baseline)

baseline is a control that must not be skipped! **With baseline, you can know whether your training is "useful" or "worse"**.

---

## 5. Common diseases

### 5.1 Overfitting

The model remembers the training set (including noise), but performs poorly on the validation/test set. Typical symptoms:

```
train_loss  ↓  ↓  ↓  ↓  ↓
val_loss    ↑  ↑  ↑  ↑  ↑   ← This is overfitting
```

Response:
- early stopping
- Decrease learning rate
-Add regularization (dropout/weight decay)
- Reduce capacity (for example, we reduce `unfreeze_last_n` from 2 to 1)
- Add data

### 5.2 Distribution Shift

The **statistical distributions of training data and val/test data are different**.

The most typical manifestation in v1: train covers 2018-2023 (including stock market crashes, trade wars, epidemics, and bull market switching), val is used alone in 2024 (the independent market trend of AI + high dividends), **2024 has structural differences compared to the previous five years**, the rules learned by the model on train do not apply to val, and val_loss starts to climb from ep1.

The interleave split of v2 is to break this offset.

### 5.3 Data Leakage

The Train set "peeps" into future information that should not be seen, resulting in falsely good results during training. Common forms:

- Features used in future data calculations (such as normalization using the entire historical std, leaking information in the test period)
- Improper segmentation of Train/Val (for example, random segmentation based on samples, adjacent days falling into different sets)
- Index data (CSI300) updates lag behind individual stocks, etc.

Our sliding window normalization in v2 only uses the lookback part of the window to calculate mu/sd, and no future information is used, so there is no leakage here.

### 5.4 Catastrophic Forgetting

On the pre-training model, fine-tune is too aggressive (lr is too large, too many epochs, too many layers are released), which will "wash away" the common knowledge learned in pre-training. The performance is that the model after fine-tuning is worse than the baseline.

Both v1 and v2 are suspected of this - the IC after v1 fine-tune is 0.014 worse than the baseline, which is a typical catastrophic forgetting.

---

## 6. Engineering words

### 6.1 SSH / SCP

- **ssh**: Secure remote login to other people’s machines. `ssh -p 30083 root@connect.westd.seetacloud.com` is connected to AutoDL.
- **scp**: ssh-based file transfer. `scp local.py root@host:/remote/path/` Push local files to the remote end.

### 6.2 tmux / nohup

Two ways to continue running the program after ssh is disconnected:

- **tmux**: virtual terminal, which can be attached/detached repeatedly. Recommended, but not installed in the AutoDL image (requires `apt install tmux`).
- **nohup … &**: Detach the process from the current shell and redirect the output to a file. Poor man's version. What we used this time is `nohup bash run_train.sh > train.out 2>&1 &`.

### 6.3 HuggingFace / HF mirror

**HuggingFace**: The world's largest open source model hosting platform (`huggingface.co`). The Kronos model sits right on top.

**HF mirror**: Access to HF from mainland China is slow, `hf-mirror.com` is a domestic mirror. Switch via environment variables:

```
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache   # Model cache directory
```

### 6.4 Proxy Conflict

AutoDL's "academic accelerator" uses the `http_proxy` environment variable to route all traffic through international proxies. As a result, requests going through hf-mirror are also accelerated (= take a long way to foreign servers), causing access to domestic mirrors to slow down to timeout.

Solution: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY` Clear it before running training. We have this line in `run_train.sh`.

### 6.5 torchrun

PyTorch official training launcher, used to pull up DDP. Writing method:

```
torchrun --standalone --nproc_per_node=1 kairos/training/train_predictor.py
```

- `--standalone`: stand-alone mode
- `--nproc_per_node=1`: Start 1 worker process (single card)
- Change to `--nproc_per_node=4` when using multiple cards

### 6.6 venv / pip -e

- **venv** (virtual environment): Python isolation environment. `python -m venv .venv` creates one in the current directory. All pip installs only affect this environment and do not pollute the system Python.
- **pip install -e .**: editable install. Install the current project into venv in "editable" mode, so that your changes to the source code will take effect immediately without reinstalling `pip install` every time.

---

## 7. The hyperparameters used in our two rounds of training are explained one by one.

Corresponds to `TrainConfig` in `kairos/training/config.py`.

```python
# ─── Data ───
dataset_path = "./finetune/data/processed_datasets"  # Dataset pkl directory
lookback_window = 90        # Forecast using last 90 days
predict_window = 10         # Forecast for the next 10 days
max_context = 512           # Maximum context length of Transformer
n_exog = 32                 # exogenous feature dimensions

# ─── Training ───
seed = 100                  # random seed
clip = 5.0                  # Normalized numerical clipping range [-5, 5]
epochs = 15                 # Total epoch upper limit (by patience early stopping)
batch_size = 50             # Feed 50 samples per step
n_train_iter = 50000        # Each epoch draws 50,000 samples from the training pool = 1,000 steps
n_val_iter = 10000          # 10000 samples per epoch val = 200 steps
log_interval = 100          # Print training loss every 100 steps

predictor_learning_rate = 5e-6   # Predictor lr (v2 down to 1/8 of v1)
warmup_pct = 0.10                # Warmup accounts for 10% of total training
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_weight_decay = 0.05    # L2 regular
num_workers = 2             # DataLoader prefetch thread
patience = 3                # val will stop if it does not decrease for 3 consecutive epochs.

# ─── Loss ───
ce_weight = 0.5             # CE loss weight (v2 dropped to 0.5)
quantile_weight = 2.0       # pinball loss weight (v2 rises to 2.0, dominates the gradient)

# ─── Model ───
use_exog = True             # Enable exogenous channels
use_return_head = True      # Enable quantile regression head
return_horizon = 5          # Return head prediction for next 5 days
n_quantiles = 9             # Predict 9 quantiles
unfreeze_last_n = 1         # Only unfreeze the last 1 transformer layer

# ─── Pretrained ───
pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor_path = "NeoQuasar/Kronos-small"
```

---

## Appendix: v1 vs v2 comparison overview

### Let’s make one thing clear first: v1 and v2 run the same model structure.

Both rounds of training** have enabled scheme A (exogenous channel) + scheme C (quantile regression head)** - at the code level, the two switches `use_exog=True` and `use_return_head=True` have not been moved in v1 and v2, and the structure of `KronosWithExogenous` is exactly the same.

The same line of marks can be seen in the training log:

```
[KronosWithExogenous] Loading layer 136/147; newly initializing layer 11 (exog/return_head)
```

"The 136 layers are from the official Kronos-small weights (the base shared by schemes A and C), and the 11th layer is our new addition (`exog_encoder.*` of scheme A + `return_head.*` of scheme C), randomly initialized and waiting to be trained."

> **v1 → v2 only changed the "training settings" (learning rate, loss weight, segmentation, early stopping), without touching the structure. **
> So v1 fails ≠ scenario A/C fails. v2 has stabilized the training but the IC is still negative. The problem lies in the supervision signal itself (next-token CE + differential income pinball), which has no transferable relationship with the income direction in 2025.

### Hyperparameter comparison table

| Dimensions | v1 (failed version) | v2 (improved version) | Differences |
|---|---|---|---|
| **Structure** | | | |
| Plan A (exogenous channel) | ✅ Open | ✅ Open | **Unchanged** |
| Plan C (quantile regression head) | ✅ ON | ✅ ON | **unchanged** |
| **Data** | | | |
| val split | 2024 single year | interleave block level extraction | eliminate single year bias |
| **Loss Weight** (Program C Contribution) | | | |
| `ce_weight` | 1.0 | 0.5 | reduced by half |
| `quantile_weight` | 0.5 | 2.0 | **l 4×** (let scheme C gradient dominate) |
| **Optimizer** | | | |
| Predictor lr | 4e-5 | 5e-6 | **drop 8×** |
| warmup_pct | 3% | 10% | warmer |
| weight_decay | 0.1 | 0.05 | Small lr with small wd |
| **Capacity Control** | | | |
| `unfreeze_last_n` | 2 | 1 | More conservative |
| **Training Scheduling** | | | |
| epochs | 30 | 15 + `patience=3` | plus early stopping |
| **Output** | | | |
| val_ce (best) | **4.57** @ ep1 | **2.95** @ ep2 | ⬇ 35% |
| test h1 IC | **-0.021** | **-0.020** | Almost unchanged |
| test h1 Rank-IC | -0.002 | -0.005 | Almost unchanged |
| **Conclusion** | val is overfitting immediately | val is normal, but test has no signal | The root cause is the supervision target |

### How to read this table

- **The structure remains unchanged, all the changes are training settings** → v2 proves that "the architecture of plan A + C itself runs stably"
- **val_ce from 4.57 → 2.95** → Overfitting is indeed cured (this is a victory for engineering)
- **test IC is still negative** → but it will not help the actual income in 2025, so **it cannot be solved by parameter adjustment, and must be changed from the data or supervision target level**

### How to change it in the next round (moving from the root)

1. **Change the supervision target** (the cheapest): Cut off the CE loss, let the quantile head of plan C **exclusively** own the gradient, and directly change the target to the logarithmic return rate of h=1 (rather than the "normalized close difference"). This way the training signal is directly aligned with the backtest metric.
2. **Change to baseline control** (should be done first): First run a regression of LightGBM + 32-dimensional exog → h1 yield locally to see if the IC can turn positive. **If LightGBM can't run out a signal, it means there is a data problem, and you should get the data; if LightGBM can run out, it means there is a problem with the Kronos fine-tune method, and it is worth continuing to adjust the model. **
3. **Expanded data** (most expensive): 300 shares → full A-shares 5000+; 7 years → 15 years (2010-); plus minute line.

Priority: **Do 2 first to locate the cause**, and then decide on 1 or 3.

---

## Multi-market architecture terminology (new in Phase 2)

| Noun | One sentence explanation | Code entry |
|---|---|---|
| `MarketAdapter` | Each market (A-shares, crypto, forex...) has an adapter responsible for `list_symbols` / `fetch_ohlcv` / `trading_calendar` / `market_features` | `kairos/data/markets/base.py` |
| `CryptoExchange` | There is another layer below the crypto adapter, specific exchange (OKX, Binance...) implementation, unified access through ccxt | `kairos/data/markets/crypto_exchanges/base.py` |
| `COMMON_EXOG_COLS` | 24-dimensional factors of reusable across markets (return, RSI, MACD, volatility, moving average, Bollinger, volume and price, microstructure, pad) | `kairos/data/common_features.py` |
| `MARKET_EXOG_COLS` | 8-dimensional market characteristics contributed by each adapter; A-shares is turnover/calendar/excess returns, crypto is funding/OI/basis/dominance/hourly triangle + pad | Class attributes of each adapter |
| `FeatureContext` | Side channel during feature calculation - A-shares uses it to transmit index K-line, crypto uses it to transmit funding / OI / spot price; adapter can get what it needs | `kairos/data/markets/base.py` |
| `meta.json` | `kairos-prepare` List of data sets placed in the output directory (market / freq / exog_cols / ranges), automatically read on the training and backtest sides | `kairos/data/prepare_dataset.py` |
| `preset_for("crypto-1min")` | Returns a dictionary of `TrainConfig` hyperparameters (`lookback=256`, `horizon=30`, `ce_weight=0.7`...) tuned for 1min crypto | `kairos/training/config.py` |
| `aggregation` (backtest) | The granularity at which cross-sectional IC is aggregated: `date` / `hour` / `minute` / `none` | `kairos/training/backtest_ic.py` |

**Key Invariant**:
- `len(COMMON_EXOG_COLS) + len(adapter.MARKET_EXOG_COLS) == 32`
  Any new adapter must ensure this, otherwise `build_features` will throw an error directly.
- The `n_exog=32` of the model remains unchanged, so the old A-shares checkpoint can be directly loaded by crypto data for ablation (the input shape is the same, but the semantics are different).

---

## Extended reading (within the project)

- [AUTODL_REMOTE_TRAINING_GUIDE.md](AUTODL_REMOTE_TRAINING_GUIDE.md) — The entire deployment process on AutoDL
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md) — Parameter adjustment manual
- [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) — Crypto market access and training guide
- [../artifacts/autodl_run_v2_20260417/train_summary.md](../artifacts/autodl_run_v2_20260417/train_summary.md) — v2 This report
