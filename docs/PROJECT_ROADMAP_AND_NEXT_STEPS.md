# Kairos project roadmap and next steps

> This document inherits the original "Next Step Plan" section in the root README and specifically records **current research priorities, acceptance criteria and work hours estimates**.
> The goal is to return the README to the "Project Entry" and leave the research backlog here for separate maintenance.

---

## 0. Current assessment

As of 2026-04-21, the research status of the project can be summarized into three points:

- `crypto-1min + h30` has already shown usable alpha. The Top100 and `Kronos-base` results both support the core direction.
- Short horizon (`h1/h5`) is still unstable. The problem is mainly in the training target design and the lack of microstructure factors in the spot data.
- The perpetual multi-channel path (funding / OI / basis) is now connected, but the first real-market experiment is not reliable because of configuration residue and an evaluation window that was too short. It needs a stricter rerun.

The priorities are therefore very clear:

1. First fix the bugs and process pitfalls that will contaminate the experimental conclusions.
2. Expand the perpetual data and use non-zero microstructure factors for training.
3. Consider larger models and heavier tokenizer routes last.

---

## 1. Tier 1: Prerequisite questions that may lead to misleading conclusions

### 1.1 Bug fixes/Known pitfalls

| # | Matter | Completion Judgment | Cost | Basis |
|---|---|---|---|---|
| D1 | **Top10 30d perp rerun**: Clear the `KAIROS_N_TRAIN_ITER=5000` residue, retrain with the default 50000 samples/epoch 10 epoch | finetuned rank-IC > baseline; `val_ce` drop > 0.01 | 0.3h training + 0.1h backtest | [CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md) |
| D2 | **Expand perp test area to ≥ 15 days**: avoid `n_dates=3` which is completely unreliable ICIR | New data set `meta.json` in test area ≥ 15 days; backtest `n_dates ≥ 15` | 0.2h packaging + 0.1h backtest | Same as above |
| B1 | **Fix the `auto` bucket logic of `backtest_ic`**: when `n_dates < 10`, it will automatically downgrade to `none` and issue a warning | Add a new regression test; `auto` will no longer give falsely high ICIR under a small bucket number | 0.5h code | [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md) |
| B2 | **Fixed candle pattern `h == l` edge case**: stationary bars are no longer clamped to 0 by the denominator | Test coverage for `h == l`; microstructure columns behave explainably on stationary bars | 0.5h code | See 2026-04-20 Dialog Diagnostic Record |
| B3 | **Fix the dimensional inconsistency of the training target**: Unify into per-step log-return + `sqrt(k+1)` normalization | h1/h5 IC is no longer systematically distorted; h30 is not significantly degraded | 1h code + 2h retraining verification | [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md) §8.2 |

The goal of this part is not to "produce new results", but to ensure that subsequent results are credible.

---

## 2. Tier 2: Make the perpetual line reusable

### 2.1 Structural improvements

| # | Matter | Completion Judgment | Cost | Basis |
|---|---|---|---|---|
| S1 | **Top100 × 90d OKX perpetual**: ≈ 13 million samples collected on perp for the first time | h30 rank-IC ≥ +0.030; funding / basis non-zero coverage > 95% | 0.5h acquisition + 1h training + 0.5h backtest | The expansion path with the highest ROI |
| S2 | **crypto-1min-short preset**: Added `return_horizon=5` preset to specifically verify short horizon | Short horizon IC > h30 preset with the same data; at least established on BTC/ETH | 0.5h preset + 1h retraining + backtest | The perpetual advantage should theoretically be more significant at the minute level |
| S3 | **Funding / basis for regime anomaly label**: Explicitly encode extreme states instead of leaving them all to the model to learn implicitly | The hit_rate when regime=1 is significantly higher than regime=0 | 2h code + retraining + backtest | "Defense signal vs offensive signal" in the previous discussion |

The goal of this part is to migrate the "h30 alpha that has been established on spot" to perpetual data with more information density.

---

## 3. Tier 3: Long-term direction

| # | Matter | Completion Determination | Cost | Choking Point |
|---|---|---|---|---|
| L1 | **OI real-time collection cron**: continuous placement of `oi_change` original stream | No data holes for 4 consecutive weeks; non-zero OI features can be played back | 0.5h script writing + 4 weeks wall clock | OKX historical OI API only goes back ~8 hours |
| L2 | **Kronos-base / larger model**: Verify whether capacity is a bottleneck | On the same data, `val_ce` drops > 0.05 or h30 IC +0.01 | 3-4h training + backtest | — |
| L3 | **Access to Coinglass / third-party data**: Breaking through the funding / OI historical depth limit | Able to construct Top100 × 1 year non-zero funding / OI data set | Paid + 1-2d adapter development | Budget, data source selection |
| L4 | **A-shares minute level**: Verify whether the A-shares signal is just too weak on the daily frequency | At least one test zone has h5 rank-IC > +0.02 | 0.5d acquisition + 2h training | akshare minute history depth is limited |

---

## 4. Things you clearly don’t want to do at the moment

- **Go to plan B first (retrain tokenizer)**
  Before the exog slot of option A is fully utilized, the cost is too high and the benefits are not certain enough.
- **Open many new market adapters at the same time**
  The bigger problem now is not the market volume, but stabilizing the existing crypto perpetual links.

---

## 5. Maintenance method

When updating this document later, follow two rules:

- The new to-do must be written clearly: `goal/completion judgment/estimated cost/basis`
- Completed entries should not be deleted directly. The completion date or corresponding commit should be marked to keep the research track.

Related documents:

- [CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md](CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md)
- [TRAINING_TUNING_PLAYBOOK.md](TRAINING_TUNING_PLAYBOOK.md)
- [BACKTEST_IC_INTERPRETATION_GUIDE.md](BACKTEST_IC_INTERPRETATION_GUIDE.md)
