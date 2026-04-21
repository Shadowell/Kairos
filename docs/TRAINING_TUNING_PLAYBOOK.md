# Kairos Training Participation Troubleshooting Manual

> Intended for readers with no experience tuning Transformer. Each step gives "Why do this / specific commands / common pitfalls".
>
> 📖 **Can’t understand the terminology? ** First go to [CONCEPTS_AND_GLOSSARY.md](CONCEPTS_AND_GLOSSARY.md) to check the meaning of words such as IC, Rank-IC, overfitting, distribution shift, early stopping, teacher forcing, etc.
>
> 🪙 **Want to start the crypto market? ** This manual focuses on A-shares; see [CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md](CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md) for the encrypted version of data collection, default super parameters, and training entrance. Configurations other than core super parameters (`market=crypto`, `freq=1min`, `lookback=256`, `return_horizon=30`) all use `kairos.training.config.preset_for("crypto-1min")`.
>
> The supporting script has been placed at:
>
> - `kairos-collect` —— akshare multi-source A-shares K-line collection
> - `kairos/data/features.py` —— Technical indicators / feature engineering
> - `kairos-prepare` —— Transfer to Kronos training pkl
> - `kairos-push-hf` —— Uploaded to Hugging Face
> - `kairos-serve` —— FastAPI inference service
> - `kairos/models/kronos_ext.py` —— Example of Transformer transformation to extend exogenous variable channel

---

## 0. Why is your current forecast "inaccurate" - locate the bottleneck first

Kronos is an **autoregressive K-line generation model** (decoder-only, similar to GPT). It has only "seen" the 6-dimensional data of OHLCV+amount. In the A-shares scenario, it is likely to encounter three problems:

1. **Domain shift**: The pre-training data is mainly based on 45 exchanges around the world (including encryption, US stocks, and Hong Kong stocks). The A-shares rhythm, price limit, T+1, and turnover rate structures are different.
2. **Serious lack of information**: Only OHLCV has no capital flow, fundamentals, sectors, sentiments, and macros. The model can never learn "why it rises."
3. **Objective function mismatch**: It learns token-level cross entropy (replicating the K-line shape), not "the direction/quantile of the next Δclose". The shape is like ≠ point.

Tuned core path:

```
[A-share exclusive large-scale minutes/day K] → [Add exogenous factors] → [Tokenizer fine-tuning]
→ [Predictor fine-tuning (+ exogenous channel modification)] → [Digit/Direction Calibration Head] → [Backtest Selection]
```

---

## 1. Data collection (A-shares K-line, the smaller the granularity, the better)

### 1.1 akshare minute-level **hard constraints** (must be known first)


|interface|granularity|traceable history|Remark|
| ---------------------------- | --------------- | --------------------- | ---------- |
| `stock_zh_a_hist`            |day/week/month|**Full History** (2005-present)|First choice, pre-reinstatement/post-reinstatement|
| `stock_zh_a_hist_min_em`     | 1min            |Last **5** trading days|data source hard limit|
| `stock_zh_a_hist_min_em`     | 5/15/30/60min   |**Last 3~6 months**|will change over time|
|`stock_zh_a_minute` (Sina)| 1/5/15/30/60min |1min for the past 9 days; 5min for the past 1 year|Sina source, occasionally limited frequency|
| `stock_zh_a_hist_pre_min_em` |1min pre-market and intra-market|same day|for real-time use|


**Conclusion**: If you want to be "as thin as possible" and "have as long a history as possible", there are only two ways to go about the **free** open source stack:

- **Route A (recommended entry)**: Full daily history (~~20 years × 5000 shares ≈ 25 million lines) + 5min in the past 6 months (daily accumulation to 2~~3 years). Training context=512 items, 5min ≈ 1 week window, enough to learn handicap microstructure.
- **Route B (want to be more refined)**: Full history of the daily line + **Starting from today** Daily cron captures 1min, and after half a year of accumulation, there will be a private 1min library.
- **Route C (sufficient budget)**: Use **Tushare Pro (≥2000 points)/Jukuan/Mikang/Wande** instead, and you can directly get the entire history in 1 minute.

The collection script in this manual supports both A/B paths and reserves the `--source tushare` hook.

### 1.2 Use collection script

```bash
# Install dependencies
pip install akshare==1.14.* pandas pyarrow tqdm schedule

# 1. The whole market daily line, the whole history at one time
kairos-collect \
    --universe csi800 \
    --freq daily \
    --start 2015-01-01 \
    --end   2026-04-17 \
    --adjust qfq \
    --out   ./raw/daily

# 2. Whole market 5min (as long as possible, usually 3~6 months)
kairos-collect \
    --universe csi300 \
    --freq 5min \
    --adjust qfq \
    --out   ./raw/5min

# 3. Daily scheduled accumulation (crontab runs once at 15:30)
kairos-collect \
    --universe csi300 --freq 1min \
    --daily-append --out ./raw/1min
```

Output: `./raw/{freq}/{symbol}.parquet`, the fields are unified as:
`datetime, open, high, low, close, volume, amount, turnover, pct_chg, vwap`.

### 1.3 How to choose a stock pool (universe)


|pond|quantity|use|
| --------------- | ------------ | ------------ |
| csi300          | 300          |The cleanest large-cap stocks, the first choice for reference analysis|
| csi500/800/1000 | 500/800/1000 |Small and medium caps, more samples|
| all_ashare      | ~5200        |Practice large models and final fine-tuning|


Recommendation: **First run CSI300 through the entire link, and then expand to CSI800 or the entire market**. The amount of training data increases from a few GB to dozens of GB, and the corresponding cost and duration will also change.

---

## 2. Feature expansion: What should be added beyond the K line?

Kronos natively only eats `[open, high, low, close, volume, amount]` in total 6 dimensions. The following suggestions are given according to the priority of "gain/difficulty of implementation". **The first 3 categories must almost be added**:

### 2.1 Technical aspect (cheap, can be added immediately)

- **Return/logarithmic return**: `log(close_t / close_{t-1})`——Numerical stability, recommended as the first sequence feature
- **Volatility**: ATR(14), Parkinson, RV (realized volatility)
- **Momentum**: ROC(5/10/20), RSI(14), MACD, Stoch(9,3)
- **Moving average/deviation**: MA5/20/60, close/MA - 1, BOLL (N=20, K=2)
- **Volume and Price**: OBV, MFI, AMF (amount/volume is similar to vwap)
- **Market Microstructure**: Daily amplitude `(high-low)/pre_close`, upper shadow/lower shadow ratio, cross star mark
- **Turnover rate**: turnover (one of the most important A-shares characteristic factors)
- **Fund Flow**: Net inflow of main/super large orders (akshare `stock_individual_fund_flow`)

### 2.2 Cross Section/Plate (A-shares Features)

- **Sector Normalized Return**: Constituent Stocks - Industry Index
- **K-day average return of stocks related to the same industry/same concept**
- **Index returns**: CSI300, CSI500, GEM, ChiNext as exogenous conditions
- **Northbound funds** (`stock_hsgt_`* series), margin trading (margin) - significant for large-cap stocks

### 2.3 Fundamentals/events (low frequency, but key to long-term forecasting)

- PE/PB/PS(TTM) `stock_a_indicator_lg`
- Earnings season window mark (5 days before announcement, 5 days after earnings report)
- Dividend ex-rights date mark, trading suspension and resumption mark
- Proportion of large transactions

### 2.4 Calendar/time (already partially available)

Kronos already uses `[minute, hour, weekday, day, month]`. A-shares plus:

- `day_of_month_in_quarter`、`is_quarter_end`
- `is_before_holiday`, `distance_to_holiday` (abnormal liquidity before Spring Festival/National Day and other long holidays)
- `intraday_bucket` (opening/late/midday split, used at minute level)

### 2.5 Macro/Sentiment (optional enhancement)

- USD/CNY, 10Y Treasury bond yield, LIBOR, Brent, VIX
- News/research report sentiment score (requires NLP pipeline, higher difficulty)
- Google Trends / Baidu Index

The supporting `kairos/data/features.py` has achieved most of 2.1 and 2.4, and 2.2 index/sector gains. 2.3 and 2.5 retain the interface.

---

## 3. Model modification: How to make Kronos incorporate these new features

This is the trickiest part. Kronos’ pipeline is:

```
OHLCV(6d) ──Tokenizer──> Discrete token (s1, s2) ──Transformer──> Next token
```

Tokenizer's `d_in=6` is **fixed**. If you directly change its `d_in`, you will have to **retrain Tokenizer and Predictor** from scratch, and all official pre-training weights will be invalidated - not recommended for inexperienced users.

Here are 3 options, ranked from lowest to highest cost:

### 3.1 Option A: retain Tokenizer and bypass fusion of exogenous variables (**strongly recommended for entry**)

Idea: The K line still follows the Tokenizer → token sequence; **exogenous features follow a parallel Embedding**, and are fused inside the Transformer using `add / concat / cross-attention`.

```
 OHLCV ──Tokenizer──> s1,s2 ─┐
                             ├─ + ─► TransformerBlocks ─► heads
Exogenous factor X ──Linear──► x_ext ┘
                             ↑
Timestamp ──TempEmb──► t_emb ───┘
```

**advantage**:

- You can continue to load NeoQuasar pre-trained weights (most parameters are reused)
- The new parameters are only `Linear(F_ext, d_model)` + optional one layer of FFN, fast training
- Does not destroy Kronos’ autoregressive generation capabilities

**Implementation changes (see `kairos/models/kronos_ext.py`)**:

```python
class KronosWithExogenous(Kronos):
    def __init__(self, *args, n_exog=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.exog_proj = nn.Sequential(
            nn.Linear(n_exog, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.exog_norm = RMSNorm(self.d_model)

    def forward(self, s1_ids, s2_ids, stamp=None, exog=None,
                padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if exog is not None:
            x = x + self.exog_norm(self.exog_proj(exog))
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        ...
```

The Dataset side returns `(x_kline, x_stamp, x_exog)` at the same time; freezing `embedding + first N layers transformer` during training and only training `exog_proj + last few layers + head` is called **progressive unfreeze**.

### 3.2 Plan B: Tokenizer retraining, `d_in` expanded to 12~20

Idea: Change Tokenizer's `d_in` from 6 to 6 + k key factors (such as turnover/ma20_norm/atr/rsi), **pre-train from scratch**.

**Advantages**: Exogenous signals directly enter discrete tokens, and information fusion is deeper.
**Disadvantages**: Large-scale pre-training needs to run for dozens of epochs, and personal budget is basically insufficient (see Section 5 Cost).
**Applicable**: Have passed Plan A, want to further improve accuracy, and have 8xA100 or above.

Key changes:

- `kairos/training/config.py`: `feature_list = ['open','high','low','close','vol','amt','turn','atr','rsi','ret']`
- `model/kronos.py`: When the model is instantiated `d_in=len(feature_list)`
- Retrain Tokenizer (from random or from `d_in=6`'s warm-start then + linear adapter)
- Retrain Predictor

### 3.3 Plan C: Hang the "return head" to the Transformer side output

Kronos' `DualHead` outputs a discrete token distribution. What you want is the **direction/income value**, you can **add a new regression head** on the last hidden layer of the transformer:

```python
class ReturnHead(nn.Module):
    def __init__(self, d_model, n_quantiles=9):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_quantiles)   # Output 9 quantiles
        )
    def forward(self, h):
        return self.fc(h)
```

The loss function uses **pinball loss (quantile regression)** or **Huber loss**, which is weighted and merged with the original CE loss. In this way, the model learns to "like a K-line" and "predict the 5% / 50% / 95% quantile of future Δclose" at the same time, and the latter is used directly for trading signals.

> Tips: Backtest you will find that the robustness of **quantile regression + directional head** is much better than the autoregressive generation of "looking at the shape".

### 3.4 My route suggestions for you

```
Week 1: akshare daily collection + factor engineering → run through finetune original version (6 dimensions)
Week 2: Option A: Exogenous channel + progressive unfreezing fine-tuning Predictor
Week 3: Plan C: Additional digit regression head, joint training
Week 4: Backtest parameter adjustment + 5min data access + deployment
Month 2: If you are still not satisfied, consider plan B (retraining Tokenizer)
```

---

## 4. Training: stand-alone & multiple cards & cloud

### 4.1 Minimum Runnable (Local/Mac M-series)

```bash
# Only train Predictor, the batch size is smaller, and only one GPU/MPS is used
export CUDA_VISIBLE_DEVICES=0
cd finetune && torchrun --standalone --nproc_per_node=1 train_predictor.py
# or Mac:
python train_predictor.py  # Need to handle mps device yourself
```

On M2 Max 64GB, Kronos-small (25M) runs csi300 daily line 30 epoch in about 12~~18 hours, and Kronos-mini (4M) takes about 2~~3 hours.

### 4.2 Multiple cards (one machine with N cards)

```bash
torchrun --standalone --nproc_per_node=4 -m kairos.training.train_tokenizer
torchrun --standalone --nproc_per_node=4 -m kairos.training.train_predictor
```

Video memory experience value (bf16):


|Model| context=512, bs=50 | bs=128     |
| ------------------- | ------------------ | ---------- |
| Kronos-mini (4M)    | ~3 GB              | ~6 GB      |
| Kronos-small (25M)  | ~8 GB              | ~16 GB     |
| Kronos-base (102M)  | ~18 GB             | ~32 GB     |
| Kronos-large (500M) | ~36 GB             | OOM on 40G |


### 4.3 Key hyperparameters (adjustable in project `config.py`)


|parameter|Recommendation (A-shares daily line)|illustrate|
| --------------- | --------------- | --------------- |
| lookback_window | 90~128          |About 4~6 months passed K|
| predict_window  | 5~10            |Predict the next 1~2 weeks|
| batch_size      | 50→128          |Press memory to fill up|
| epochs          | 30→50           |There is an early stop available|
| tokenizer_lr    | 2e-4            |Already reasonable|
| predictor_lr    | 4e-5 → **1e-5** |Exogenous channels should be more conservative when entering|
| weight_decay    | 0.1             |Prevent overfitting|
| clip            | 5.0             |Normalized truncation|


---

## 5. Comparison of training costs among cloud vendors (2026-Q1 public price, for reference only)

Based on "**Complete fine-tuning Kronos-small 30 epoch, CSI800 daily line + 5min**" as a benchmark, the experience is about **40 GPU·hour (A100-40G)**.


|platform| GPU          |Price based on volume (¥/h, before tax)|Bidding/preemptive price|40 hour cost per trip|Remark|
| --------------------- | ------------ | -------------- | -------- | --------------------- | ------------------------ |
| **AutoDL**            | RTX 4090 24G | ¥1.8~2.5       | —        | ¥70~100               |The best value for money; small model is enough|
| AutoDL                | A800 80G     | ¥6~8           | —        | ¥240~320              |Large models/long context preferred|
|**Alibaba Cloud PAI-DSW**| V100-32G     | ¥9~12          |¥3~4 (preemption)| ¥360~~480 / ¥120~~160 |Enterprise stability|
|Alibaba Cloud PAI| A100-80G     | ¥28~32         | ¥10      | ¥1120 / ¥400          |new generation flagship|
|**Tencent Cloud**| V100         | ¥10            | —        | ¥400                  | —                        |
|Tencent Cloud| A10          | ¥6             | —        | ¥240                  |Good value for money|
|**Volcano Engine**| A100-40G     | ¥22            | ¥8       | ¥880 / ¥320           |Take advantage|
|Huawei Cloud ModelArts| Ascend 910   | ¥18            | —        | ¥720                  |Domestic card, torch needs MindSpore adaptation|
|**Baidu Qianfan**| A800         | ¥20            | —        | ¥800                  |Lots of coupon activities|
| **Google Colab Pro+** | A100         |$10/mo Subscribe| —        |≈¥72/month|Time slots are limited, suitable for experiments|
| Colab Pro             | T4/L4        | $10/mo         | —        |≈¥72/month|very slow|
| **Lambda Labs**       | A100-80G     | $1.29/h ≈ ¥9.3 | —        | **¥370**              |Billed by the second, stable overseas|
| Lambda                | H100-80G     | $2.49/h ≈ ¥18  | —        | ¥720                  |fastest|
| **vast.ai**           | RTX 4090     | $0.4~0.7/h     | —        | ¥120~200              |Overseas crowdsourcing, pay attention to stability|
| **RunPod**            | A100-80G     | $1.19/h        |Grab $0.79| ¥340 / ¥230           |Serverless options|
| AWS p3.2xlarge        | V100         | $3.06/h        | spot $1  | ¥880 / ¥290           |Old brand, expensive|
| AWS p4d.24xlarge      | 8×A100-40G   | $32/h          | spot $10 | —                     |large scale training|


**Practical suggestions**:

1. **Parameter adjustment stage**: AutoDL 4090 / Colab Pro+, running through the pipeline costs tens of yuan per time.
2. **Formal training**: AutoDL A800 or Lambda A100, single budget ¥300~500.
3. **Large-scale pre-training** (Plan B, Tokenizer re-training): 8×A100 spot instance ~~40 hours, ¥2~~5k.
4. **Domestic Compliance**: If the data does not go out of the country, choose Alibaba/Huoshan/AutoDL, and the training artifacts should also stay in the country before uploading to HF.

> **Tips to save money**: bf16 + `torch.compile()` + `gradient_checkpointing` can double the batch; run through Kronos-mini before training, and then change to small/base; use safetensors to save checkpoints.

---

## 6. Deployment: Turn your fine-tuning model into a service

### 6.1 Upload to Hugging Face Hub

```bash
huggingface-cli login        # Enter your write token
kairos-push-hf \
    --tokenizer-ckpt ./outputs/models/finetune_tokenizer_demo/checkpoints/best_model \
    --predictor-ckpt ./outputs/models/finetune_predictor_demo/checkpoints/best_model \
    --repo-tokenizer your-username/Kronos-Tokenizer-ashare \
    --repo-predictor your-username/Kronos-small-ashare \
    --private
```

Key points: Kronos inherits `PyTorchModelHubMixin`, `save_pretrained()` / `push_to_hub()` are directly available.

### 6.2 Local/server inference service (FastAPI)

```bash
pip install fastapi uvicorn[standard]
kairos-serve \
    --tokenizer your-username/Kronos-Tokenizer-ashare \
    --predictor your-username/Kronos-small-ashare \
    --device cuda:0 --port 8000
```

Call:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "600977",
    "lookback": 400,
    "pred_len": 20,
    "T": 0.6, "top_p": 0.9, "sample_count": 5
  }'
```

Returns the OHLCV in JSON containing the next 20 K's + quantile interval (if you trained the quantile head) + direction probability.

### 6.3 Production suggestions

- **Model hot loading**: Use `accelerate` + `safetensors`, cold start within 3s.
- **Request for joint batch**: `predict_batch` Already supported, `BatchScheduler` will gather once in 30ms.
- **Index prediction cache**: Market indices such as CSI300 are pushed every 5 minutes, and speculative traffic can be directly hit.
- **backtest regression**: Run a small backtest with a fixed seed in CI, RegNet-style monitoring IC/ICIR drift.
- **Risk control bypass**: After the reasoning is completed, do a rationality check (within the price limit and the volume is not negative).

---

## 7. Use of reasoning (complete example)

```python
from model.kronos_ext import KronosWithExogenous
from model import KronosTokenizer, KronosPredictor
from data_pipeline.build_features import build_features
import akshare as ak
import pandas as pd

tokenizer = KronosTokenizer.from_pretrained("your-username/Kronos-Tokenizer-ashare")
model     = KronosWithExogenous.from_pretrained("your-username/Kronos-small-ashare")

predictor = KronosPredictor(model, tokenizer, max_context=512)

# Get the last 400 trading days of 600977
df = ak.stock_zh_a_hist(symbol="600977", period="daily", adjust="qfq").tail(400)
df = df.rename(columns={
"Date": "datetime", "Open": "open", "Close": "close", "Highest": "high",
"Minimum": "low", "Trading volume": "volume", "Trading amount": "amount", "Turnover rate": "turnover"
})
df['datetime'] = pd.to_datetime(df['datetime'])
df = build_features(df)              # Add RSI/ATR/MA etc.

x_df = df[["open","high","low","close","volume","amount"]]
x_ts = df["datetime"]
y_ts = pd.bdate_range(df["datetime"].iloc[-1] + pd.Timedelta(days=1), periods=10)

pred = predictor.predict(x_df, x_ts, y_ts, pred_len=10,
                         T=0.6, top_p=0.9, sample_count=5)
print(pred)
```

---

## 8. Common pitfalls

### 8.1 Data side

1. **Upper and lower limit truncation**: Do not directly treat the daily limit `pct_chg=9.98%` as the normal point in the training set, otherwise the model will be seriously biased. It is recommended to mark `is_limit_up/down` and feed it through exogenous channel.
2. **Resumption method**: Post-resumption will make the historical price become a negative range, **Unify the use of pre-resumption `qfq`**.
3. **Survivor bias**: Only taking the current CSI300 components will be seriously optimistic; **Historical component stock list** (akshare `index_stock_cons_sina` corresponding date) should be used.
4. **Minute data gap**: Lunch break 11:30~13:00 Without K, do not use `pd.date_range` to directly generate y_timestamp.
5. **Regularization**: Kronos does z-score within a single stock window, **not** normalizing the whole market together.
6. **Signal to Trade**: Predicted `close` is not alpha. Calculate `(pred_close_{t+h} / last_close - 1)` and do cross-sectional sorting to achieve normal posture.

### 8.2 Training/backtest side (precipitated from [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) post-mortem)

7. **`KAIROS_N_TRAIN_ITER` Residual → Silent underfitting**.
   - `n_train_iter` is **how many samples are taken per epoch** (not the number of steps), default 50000, mini run is often set to 5000 to verify the link.
   - **Be sure** to clear env before officially running, otherwise there will be an under-fitting artifact of "watching val_ce keep falling but only falling by 0.006". [Top10 30d perp run](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) in this repository has become a negative migration because of this.
   - Self-inspection: You need to be vigilant about `[TRAIN] pool=X, using Y/epoch.` —— `Y/X < 5%` in the training log.
8. **The training target is the cumulative diff of normalized close, and the backtest true value is raw log-return** (the dimensions are inconsistent).
   - The pinball target of `train_predictor.py` is `close_n[t+k+1] - close_n[t]` (k=0..29), and the cumulative diff dimension increases linearly with k → k=29 dominates the entire loss → the model only learns the horizon=`return_horizon` level.
   - This can explain why h1 / h5 of all crypto-1min runs ([BTC/ETH](CRYPTO_BTC_ETH_2Y_SPOT_RUN.md), [Top100](CRYPTO_TOP100_1Y_SPOT_RUN.md), [Top10 perp](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)) do not move or even reverse, and only h30 is valid.
   - Fix direction: raw log-return + per-k normalization, or multiple horizon heads (multitasking). After modification, be sure to run the sanity regression of [`BACKTEST_IC_INTERPRETATION_GUIDE.md`](BACKTEST_IC_INTERPRETATION_GUIDE.md) §5.3.
9. **`backtest_ic --aggregation` Making the wrong choice will make the result completely meaningless**.
   - `auto` currently always goes to `date`, and calculates the ICIR (=noise) of `n_dates < 5` for the data set in the test area < 5 days.
   - The short test area only looks at `overall.spearman` of `--aggregation none`; see [`BACKTEST_IC_INTERPRETATION_GUIDE.md`](BACKTEST_IC_INTERPRETATION_GUIDE.md) §2 Decision Tree for details.
10. **You must look at the baseline to see IC**. Kronos original weight + random head on pooled often|IC|> 0.02, it is meaningless to look at finetuned absolute IC, you must look at Δ. See [`BACKTEST_IC_INTERPRETATION_GUIDE.md`](BACKTEST_IC_INTERPRETATION_GUIDE.md) §5 for details.

---

## 9. What should you do next (you can start today)

1. `pip install akshare pyarrow tqdm fastapi uvicorn huggingface_hub`
2. `kairos-collect --universe csi300 --freq daily --start 2018-01-01 --end 2026-04-17 --out ./raw/daily`
3. `kairos-prepare --raw ./raw/daily --out ./finetune/data/processed_datasets`
4. Change `pretrained_tokenizer_path/predictor_path` in `kairos/training/config.py` to `NeoQuasar/Kronos-Tokenizer-base` / `NeoQuasar/Kronos-small`
5. `torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor` (1 card is also acceptable)
6. Run `finetune/qlib_test.py` (or your own backtest script) to see the IC
7. If IC > 0.03, continue with plan A (exogenous channel) and plan C (digital head)
8. Upload to HF and deploy FastAPI

Good luck with the alchemy.
