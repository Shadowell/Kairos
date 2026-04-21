# Backtest IC configuration and results interpretation guide

> How to use `kairos.training.backtest_ic` to get **statistically credible** IC / Rank-IC / ICIR to avoid being biased by wrong selection of bucket / stride / horizon.
>
> The existence of this document stems from [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) that post-mortem - the same ckpt + data, just because `--aggregation` chose the wrong one, "ICIR=+1.17 that looks good" and "ICIR=-0.06 that looks like negative migration" can appear in two reports at the same time.
>
> Terms (IC / Rank-IC / ICIR / hit_rate) see [`CONCEPTS_AND_GLOSSARY.md`](CONCEPTS_AND_GLOSSARY.md).

---

## 0. 30 second quick search

|what you want| `--aggregation` | `--horizons` | `--stride` |Look in the report|
|---|---|---|---|---|
|Cross-sectional stock picking alpha (for production purposes)| `date`(n_dates ≥ 15) |Align with preset `return_horizon`| 1 | `by_date_mean.h{H}.rank_ic / icir` |
|pooled "Direction prediction ability"| `none` |1, 5, 30 all viewed|1 or 5| `overall.h{H}.spearman / hit_rate` |
|CPU smoke verification| `none` | 1 | 60 + `--per-symbol-limit 50` |`overall` As long as it is not NaN|
|High frequency/minute level cross section (≥10 symbols × ≥15 days test)|`minute` Use with caution| 1, 5 | 1 |`by_date_mean` (note SE)|

**The most important anti-pattern**: Looking at the ICIR of `--aggregation date` when the test zone is only 3 days old - the statistical significance of that number ≈ the variance of 3 coin tosses, neither +1.17 nor -0.6 can be interpreted as an alpha signal.

---

## 1. Output field meaning

backtest JSON looks like this:

```json
{
  "n_records": 40350,
  "n_symbols": 10,
  "date_range": ["2026-04-17 04:16:00", "2026-04-19 23:30:00"],
  "overall": {
    "h1":  {"pearson": ..., "spearman": ..., "hit_rate": ..., "n": ...},
    "h5":  {...},
    "h30": {...}
  },
  "by_date_mean": {
    "h1":  {"ic": ..., "rank_ic": ..., "icir": ..., "n_dates": ..., "bucket": "..."},
    "h5":  {...},
    "h30": {...}
  }
}
```

### 1.1 `overall`(pooled)

Throw all `(score, return)` pairs together and count a single Pearson / Spearman / hit_rate.

- **Advantages**: n is large (tens of thousands to millions), statistical SE is small, and p-value is directly reliable.
- **Disadvantages**: The signal sources are mixed (different times, different symbols), which does not reflect "the ability to rank this group of symbols at a certain time".
- **When to look**: sanity check (baseline should be close to 0, finetuned deviates significantly from 0), or the **only** credible number when the test area is too short to do time aggregation.

### 1.2 `by_date_mean` (cross-sectional → time average)

Group the samples according to `bucket` (one bucket per day/hour/minute), calculate the cross-sectional IC independently in each bucket, and then average the buckets:
- `ic`: Average Pearson IC per bucket
- `rank_ic`: Average Spearman IC per bucket
- `icir`: `mean(IC) / std(IC)` —— Information ratio
- `n_dates`: non-NaN bucket number

- **Advantages**: Directly corresponds to "I give rankings every day/hourly, and the ability to rank in the long run" is the most important indicator for combined use.
- **Disadvantages**: The number of samples in each bucket = how many symbols there are at that moment; if there are only 2-3 symbols, the bucket IC is almost noise.

---

## 2. `--aggregation` How to choose

```python
# kairos/training/backtest_ic.py L53-63
_BUCKET_ALIASES = {
    "date": "date",  "day": "date",  "daily": "date",
    "hour": "hour",  "hourly": "hour",
    "minute": "minute", "minutely": "minute",
    "none": "none",  "pool": "none",
}
```

`auto` The current implementation always returns `date`, which is not friendly to the **short test area** ([CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md §8.3](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)).

### Select decision tree

```
n_test_days < 5
└─ Use --aggregation none to see overall (pooled) and ignore by_date_mean
n_test_days ≥ 15 and n_symbols ≥ 5
└─Default --aggregation date (most commonly used)
n_test_days ≥ 5 and freq=1min and n_symbols ≥ 10
└─ Optional --aggregation hour or minute (only meaningful when samples per bucket are ≥ 10)
```

### 2.1 Minimum sample size per bucket

| n_per_bucket |Standard error of single bucket Pearson IC|evaluate|
|---|---|---|
| 2 |Not computable (requires ≥3)| NaN |
| 3 | ~0.71 |Total noise|
| 5 | ~0.50 |Extremely unstable|
| 10 | ~0.35 |Noisy; can be used after averaging multiple buckets|
| 30 | ~0.19 | OK |
| 100 | ~0.10 |good|
| 1000 | ~0.032 |Very stable|

Rule of thumb: Only when the number of samples per bucket is ≥ 30 can you start to trust the IC of a single bucket; when < 10, only look at `mean(IC)` and not `ICIR` (the standard deviation is dominated by noise, and ICIR is the noise amplification in the denominator).

### 2.2 Total bucket number (n_dates)

| n_dates |SE of `mean(IC)` (assuming single bucket SE = 0.1)|Is ICIR credible?|
|---|---|---|
| 3 | 0.058 |❌ Completely untrustworthy ([`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §7.2’s +1.17/+0.06 are all noise)|
| 10 | 0.032 |⚠️ Barely|
| 30 | 0.018 |✅ You can select models|
| 100+ | 0.010 |✅ Can make paper/online decisions|

---

## 3. `--stride` How to choose

`--stride N` means taking a starting window for every N bars.

### 3.1 trade-off

- **stride=1**: All bars are used as starting points, n_records is the largest, IC estimate SE is the smallest, but it is slow and adjacent samples are highly correlated (autocorrelated).
- **stride > 1**: n_records is reduced by N times, single bucket SE is enlarged by √N times, but the wallclock is also N times faster.

### 3.2 Experience

|scene|Recommended stride|
|---|---|
|Full GPU backtest (production)| 1 |
|Fast iteration/multiple sweeps| 5-10 |
|Top100 × 1 year 1min ([`CRYPTO_TOP100_1Y_SPOT_RUN.md`](CRYPTO_TOP100_1Y_SPOT_RUN.md))|10 (stride=1 estimates 8h+, stride=10 estimates 45min)|
| CPU smoke | 60 + `--per-symbol-limit 50` |

### 3.3 `--per-symbol-limit`’s Pitfalls

`--per-symbol-limit N` Draw N starting points at equal intervals for each symbol. **Problem**: The extracted timestamp symbols are not aligned, there are only 1-2 symbols left in each bucket, and the cross-sectional IC is all NaN.

- ✅ When smoking, only look at `--aggregation none`’s `overall`
- ✅ If you want to preserve bucket alignment: use `--stride 60` (all symbols share the same set of offsets), do not use `--per-symbol-limit`

---

## 4. `--horizons` How to choose

By default, when `crypto-1min` is trained `return_horizon=30`, the target of pinball loss is the cumulative diff of `close[t+k+1] - close[t]` (k=0..29) - the dimension increases linearly with k, **k=29 dominates the entire loss**, and the model actually only optimizes "the 30th step in the future".

| h |Supervision intensity|Expected IC performance|
|---|---|---|
| 1, 5 |Weak (cumulative diff on k=0,4 the loss term is an order of magnitude smaller)|Close to 0 or ~baseline|
| 30 |Strong (aligned with `return_horizon`)|main signal|
| 60+ |Complete extrapolation|noise|

**Conclusion**: Using ckpt trained with `crypto-1min` preset, **set `--horizons` to include 30** (such as `1,5,30`) during the backtest, mainly looking at h30; h1 / h5 are only used for sanity (if h1 is significantly negative and h30 is significantly positive, it means that the model regards the short horizon as a "reverse warning of the long horizon", which is normal).

If you change the preset, align `--horizons` with `cfg.return_horizon`.

For detailed training target design issues, see [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §8.2 and [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) §8.

---

## 5. Must run `--baseline` comparison

`--baseline` Pattern loading Kronos-small original weight + **randomly initialized exog encoder + return head**, the output score is the value of "hidden state after random fc".

### 5.1 Why baseline is not 0

Intuitively, random head IC should be ≈ 0, but in fact, baseline often has pooled|IC| > 0.02([`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §7.3): 

> The Kronos transformer backbone (layer 136 reuse) has encoded the direction information of "future distribution" in the hidden state; random fc will map it to the score space with a fixed random projection. **This projection is consistent for all (score, return) pairs**, so some direction ICs can be caught when pooled.

Practical implications: **It is meaningless to only look at the absolute IC of finetuned. You must look at the Δ** of finetuned - baseline.

### 5.2 Recommended two backtest calls

```bash
# 1. baseline
python -m kairos.training.backtest_ic --baseline \
    --preset <your-preset> \
    --dataset-path <dataset> \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/<run>/backtest_baseline.json

# 2. finetuned
python -m kairos.training.backtest_ic \
    --ckpt <ckpt path> \
    --preset <your-preset> \
    --dataset-path <dataset> \
    --horizons 1,5,30 --aggregation date \
    --out artifacts/<run>/backtest_finetuned.json
```

Then use the comparison script (refer to the comparison table generator at the end of [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §7) to pull a three-table table of baseline / finetuned / Δ.

### 5.3 Sanity regression: Run it every time after major changes to the code

To ensure that your changes to `train_predictor.py` / `backtest_ic.py` / `kronos_ext.py` do not destroy the existing alpha, always run this step:

```bash
# Load old BTC/ETH data + old ckpt
python -u -m kairos.training.backtest_ic \
    --ckpt artifacts/checkpoints/predictor/checkpoints/best_model_btceth_backup \
    --preset crypto-1min \
    --dataset-path /root/autodl-tmp/Kairos/finetune/data/crypto_1min_btc_eth \
    --horizons 1,5,30 --aggregation date --stride 5 \
    --out artifacts/sanity/btceth_$(date +%F).json
```

Expected h30 by_date_mean rank_ic ≈ +0.024 (stride=5), ICIR ≈ +0.15. If the difference is > 50%, check the git log.

---

## 6. Common misunderstanding cases

### 6.1 "ICIR=+1.17 Great!"

It’s actually `n_dates=3`. The standard deviation of 3 ICs is almost entirely the noise variance. **Look at n_dates first, then ICIR**.

### 6.2 "The baseline's h30 ICIR=+0.42 indicates that the original weight of Kronos has alpha"

Random head + Kronos hidden can produce a falsely high ICIR on the scale of 100 symbols × 78 days ([`CRYPTO_TOP100_1Y_SPOT_RUN.md`](CRYPTO_TOP100_1Y_SPOT_RUN.md) §7.5). **Look only at Δ (finetuned - baseline)**.

### 6.3 "The IC of finetuned is +0.003, the model works!"

Don't claim to be useful if the p-value is not significant. On n=40k, Spearman IC ≥ 0.01 will have a high probability of p < 0.05; Spearman IC ≥ 0.02 will be stable.

### 6.4 "The IC of by_date_mean is negative and the model is useless"

It may be that the bucket is selected incorrectly (n_per_bucket is too small, and the IC noise is large). First use `--aggregation none` to see pooled. If pooled is also negative, it is truly negative.

### 6.5 "h1 is a negative IC, and the model prediction is wrong"

If preset `return_horizon=30`, h1 / h5 are unsupervised horizons of the model, and may even learn the reverse signal of h30 after being dominated by the cumulative-diff target. This is a side effect of the loss design, not model buggy. **Only trust the file whose horizon is aligned with `return_horizon`**.

---

## 7. Upcoming code improvements (TODO, not implemented)

Sorted by ROI:

1. **`_BUCKET_ALIASES.auto` The logic of downgrading to pooled when adding `n_dates < 10`**, and printing warning on stdout.
2. **`backtest_ic`’s report JSON adds the `n_per_bucket_avg` field**, allowing the reader to see at a glance whether the bucket IC is trustworthy.
3. **`dataset.py` Add `using {pct:.1%} of pool`** when printing pool, warning when pct < 5% (avoid the KAIROS_N_TRAIN_ITER residual trap like [`CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) §8.1).
4. **Training pinball target dimension** is changed to raw log-return + per-k normalization, so that h1/h5 also has real supervision signals.

Before implementation, please leave an issue record in [`TRAINING_TUNING_PLAYBOOK.md`](TRAINING_TUNING_PLAYBOOK.md) §8 and run §5.3 sanity regression once to ensure that the existing alpha is not destroyed.
