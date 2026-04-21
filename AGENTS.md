# AGENTS.md

> This document is the **repository operations manual for AI coding agents** such as Cursor, Claude Code, and Codex.
> Human collaborators can read it too, but the primary audience is the agent. Read it before starting any new task.

---

## 0. TL;DR — Top 5

1. **Working directory**: `/Users/jie.feng/wlb/Kairos`, remote `origin = https://github.com/Shadowell/Kairos.git`, main branch `main`.
2. **`git add -A && git commit && git push`** immediately after modifying the code, there is no need to ask for user consent again (see §6).
3. **Answers must be in Chinese** (User Rules).
4. **File operation**: Use Read/Grep/Glob for reading, StrReplace/Write for editing, **Do not use `cat/sed/awk/echo >`** instead.
5. **Retraining/Long Task**: Run in remote AutoDL by default, do not try GPU training on local macOS; see `docs/AUTODL_REMOTE_TRAINING_GUIDE.md` for details.

---

## 1. Repository Positioning

Kairos is a **multi-market (A-shares + crypto) fine-tuning and deployment toolbox** for the [Kronos](https://github.com/shiyu-coder/Kronos) base model:

- **Data collection** — `kairos.data.collect` (dispatcher) + `kairos.data.markets.*` (one adapter per market, covering A-shares, crypto, and future extensions such as FX or gold)
- **Feature engineering** — `kairos.data.common_features` (24 common dimensions) + `adapter.market_features` (8 market-specific dimensions) = fixed 32-dimensional `EXOG_COLS`, with **no future-information leakage**
- **Dataset packaging** — `kairos.data.prepare_dataset` (time-split / interleave-split; `--market` switches adapters; always writes `meta.json`)
- **Model** — `kairos.models.KronosWithExogenous` (Kronos + exogenous channel + quantile regression head; `n_exog=32` is fixed across markets)
- **Training** — `kairos.training.train_predictor` (DDP + gradual unfreezing + early stopping) plus presets such as `kairos.training.config.preset_for("crypto-1min")`
- **Evaluation** — `kairos.training.backtest_ic` (IC / Rank-IC / ICIR; supports `--aggregation date/hour/minute/none`, and recovers market/frequency from `meta.json`)
- **Deployment** — `kairos.deploy.push_to_hf` / `kairos.deploy.serve`

See `docs/CONCEPTS_AND_GLOSSARY.md` for full terminology and background.

---

## 2. Directory convention

```
Kairos/
├── kairos/                       # Source code (only Python package)
│   ├── data/                     # collect / features / prepare_dataset
│   │   ├── collect.py            # multi-market CLI dispatcher (kairos-collect)
│   │   ├── common_features.py    # 24-dimensional common factor
│   │   ├── crypto_extras.py      # funding/OI/spot sidecar loading
│   │   ├── features.py           # assembly common + adapter specific = 32 dimensions
│   │   ├── prepare_dataset.py    # Generate train/val/test.pkl + meta.json
│   │   └── markets/              # MarketAdapter abstraction + share/crypto
│   │       └── crypto_exchanges/ # ccxt package: okx/binance_vision/…
│   ├── models/                   # KronosWithExogenous + QuantileReturnHead
│   ├── training/                 # train_tokenizer / train_predictor / eval_tokenizer / backtest_ic / dataset / config
│   ├── deploy/                   # push_to_hf / serve
│   ├── utils/                    # training_utils etc.
│   └── vendor/                   # Third-party vendored code (Kronos source snapshot)
├── docs/                         # All documents (named according to the division of "navigation/guide/experiment/roadmap")
│   ├── DOCUMENTATION_INDEX.md                # documents navigation: find documents by tasks and roles
│   ├── PROJECT_ROADMAP_AND_NEXT_STEPS.md    # Current roadmap, priority, acceptance criteria
│   ├── CONCEPTS_AND_GLOSSARY.md             # Unified explanation of terminology and core concepts
│   ├── TRAINING_TUNING_PLAYBOOK.md          # Training tuning and troubleshooting notes
│   ├── BACKTEST_IC_INTERPRETATION_GUIDE.md  # IC backtest configuration and result interpretation
│   ├── AUTODL_REMOTE_TRAINING_GUIDE.md      # Remote GPU training and checkpoint return workflow
│   ├── CRYPTO_DATA_SOURCE_AND_EXCHANGE_GUIDE.md # crypto data source/exchange/network configuration
│   ├── CRYPTO_BTC_ETH_2Y_SPOT_RUN.md        # BTC+ETH 2 years spot predictor run log
│   ├── CRYPTO_TOP100_1Y_SPOT_RUN.md         # Top100 1 year spot predictor run log
│   ├── CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md # OKX perpetual multi-channel transformation plan
│   ├── CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md # OKX perpetual Top10 30-day experiment post-mortem
│   └── CRYPTO_BTC_ETH_TOKENIZER_RUN.md      # BTC+ETH tokenizer fine-tuning and evaluation records
├── scripts/                      # Operations/CI auxiliary script
│   ├── autodl_bootstrap.sh       # AutoDL one-command initialization (venv + hf mirror + smoke)
│   ├── package_and_upload.sh     # Package + scp to AutoDL
│   └── smoke_crypto_extras.py    # crypto extras offline smoke test
├── examples/                     # Quick getting started example
│   ├── inference_quickstart.py
│   └── crypto_top100_universe.md # Top100 frozen list (2026-04-20 snapshot)
├── tests/                        # pytest tests (5 test_*.py)
├── raw/                          # Original parquet (not stored in the library, see .gitignore)
├── finetune/data/                # Packaged *.pkl (not stored in the library)
├── artifacts/                    # checkpoint/backtest_report.json (not stored in the database)
├── pyproject.toml                # Package definition + CLI entry
├── requirements.txt              # Core dependency snapshot (convenient for `pip install -r`)
├── .env.example                  # Environment variable template (API key/proxy/HF)
├── LICENSE                       # MIT
├── README.md
└── AGENTS.md                     # this document
```

**Do not commit files to `raw/` / `finetune/data/` / `artifacts/` / `.venv/`. **

---

## 3. Quick check of commonly used commands

### environment
```bash
# Local (macOS) - only data collection, packaging, smoke test
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "numpy<2" scipy      # numpy 2.x is not compatible with torch
```

### data pipeline
```bash
# 1a) Collection - A-shares (default market=ashare, workers=1, mini_racer thread is not safe)
kairos-collect --universe csi300 --freq daily \
  --start 2018-01-01 --end 2026-04-17 --out ./raw/daily --workers 1

# 1b) Collection — crypto (crypto extras need to be installed first)
pip install -e '.[crypto]'
# Default OKX perpetual (needs proxy or direct connection to OKX; funding/OI/basis available)
kairos-collect --market crypto \
  --universe "BTC/USDT:USDT,ETH/USDT:USDT" --freq 1min \
  --start 2023-01-01 --end 2025-01-01 \
  --out ./raw/crypto/1min --workers 1 \
  --proxy "${HTTPS_PROXY:-}"
# Downgrade channel under the office network (Binance public image, only spot, no funding/OI/basis)
kairos-collect --market crypto --exchange binance_vision \
  --universe "BTC/USDT,ETH/USDT" --freq 1min \
  --start 2024-01-01 --end 2024-02-01 \
  --out ./raw/crypto/bv_1min --workers 1

# 2) Packaging (v2 defaults to interleave split)
kairos-prepare --raw ./raw/daily --out ./finetune/data/processed_datasets \
  --train 2018-01-01:2023-12-31 --val 2024-01-01:2024-12-31 \
  --test 2025-01-01:2026-04-17 \
  --split-mode interleave --val-ratio 0.15 --block-days 20 --seed 42
```

### training/evaluation
```bash
# Local CPU smoke test (quick verification link)
KAIROS_SMOKE=1 python -m kairos.training.train_predictor

# AutoDL GPU (see docs/AUTODL_REMOTE_TRAINING_GUIDE.md for details)
torchrun --standalone --nproc_per_node=1 -m kairos.training.train_predictor

# backtest IC (baseline comparison)
python -m kairos.training.backtest_ic --baseline --horizons 1,5 \
  --out artifacts/backtest_baseline.json
python -m kairos.training.backtest_ic --ckpt artifacts/best_model --horizons 1,5 \
  --out artifacts/backtest_finetuned.json

# Tokenizer separate fine-tuning + evaluation (see docs/CRYPTO_BTC_ETH_TOKENIZER_RUN.md for details)
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

## 4. Code specifications

- **Python 3.10+**, follow the style of existing files (don't reformat the entire file without forcing the introduction of black/ruff).
- **No nonsense comments**. `# import pandas`, `# return results` will be deleted directly. Only comment when expressing intent/tradeoffs/constraints.
- **Do not write emoji** unless the user actively requests it in README/docs.
- **Type annotation**: Add new functions as much as possible; when changing existing functions, just keep the style consistent, and don't make major changes just to add type hints.
- **No future information leakage**: Any new factors put into `kairos/data/features.py` must only use data from `t` time and before; the z-score window in the packaging phase is fixed to 60 days rolling.
- **CLI Entry**: When adding a new executable script, register it under `[project.scripts]` of `pyproject.toml` and name it with the prefix `kairos-`.

---

## 5. Testing and Verification

- Change `kairos/data/*` → Run `kairos-prepare --max-symbols 5 --dry-run` (or the smallest subset) to confirm that it is not exploded.
- Change `kairos/training/*` → Run `KAIROS_SMOKE=1 python -m kairos.training.train_predictor` local CPU first.
- Change `kairos/models/*` → Load Kronos-small weights at least once to verify that state_dict matches.
- **No CI**, the agent is responsible for local verification. If there is `pytest`, run `pytest -x`.

---

## 6. Git commit rules (important)

### 6.1 Trigger timing
**As long as a "logically complete change" is completed - such as fixing a bug, adding a document, adjusting a set of super parameters - commit + push immediately without asking the user. **

Exceptions (in these cases you must stop and ask the user first):
- When `git push --force` / `--force-with-lease` is required;
- When changing to a branch other than `main`/creating a new branch/merging PR;
- When the commit will contain files that look like **key, token, API key, `.env`, `credentials.*`**;
- When the user explicitly says "Don't push it yet" / "I want to see it".

### 6.2 Commit message style
- **English, imperative sentence, capitalized, no period**, refer to existing history:
  ```
  Fix repo URLs to Shadowell/Kairos and add package init placeholders
  Harden training loop: data fallbacks, interleave split, early-stop, IC backtest
  Add AutoDL guide and glossary; cross-link from README/playbook
  ```
- One-line title ≤ 72 characters; if necessary, leave a blank line and then write the body.
- **One commit does one thing**, don't stuff "bug fixes + refactoring + adding documents" together.

### 6.3 Standard actions
```bash
git status                          # Take a look first
git add -A
git commit -m "<imperative subject>"
git push                            # Push to origin/main
git status                          # Confirm clean + "up to date"
```
Use the HEREDOC form to avoid escaping problems:
```bash
git commit -m "$(cat <<'EOF'
Subject line here

Optional body explaining *why*, not *what*.
EOF
)"
```

### 6.4 Things not to do
- ❌ `git commit --amend` The commit that has been pushed.
- ❌ `git push --force` to `main`.
- ❌ `git config` Change the global/repository configuration.
- ❌ `git rebase -i` / `git add -i` (Interactive commands cannot be run).
- ❌ Submit `raw/` / `artifacts/` / `finetune/data/` / `.venv/` / `*.pkl` / `*.parquet` / checkpoint.
- ❌ Submit any file containing a key: `.env` / `*.secret` / `*.pem` / `*.key` / `secrets/` (`.gitignore` has been intercepted, but take a second look at `git status` before doing it).

### 6.5 Secrets Agreement
- All API keys/tokens are passed through **environment variables**, with names in the form `OKX_API_KEY` / `OKX_API_SECRET` / `OKX_API_PASSPHRASE` / `BINANCE_API_KEY` / `HF_TOKEN`.
- During development, fill in the value into `.env` (git-ignored) and load it with `set -a; source .env; set +a`.
- The public template is in `.env.example` (tracked), and the template is updated simultaneously when adding required variables.
- Public market data (K line / funding / OI) does not require any key, and the crypto adapter uses anonymous requests by default.

---

## 7. Known pitfalls and fixed processing methods

|symptom|root cause|deal with|
|---|---|---|
|`kairos-collect` stuck with no output|Dongcai API is restricted|Existing fallback → tencent → sina; `--workers 1`|
|`kairos-prepare` reported `Parquet magic bytes not found`|macOS `._*` metadata file|The packaging script has been filtered `startswith("._")`; used for transmission `COPYFILE_DISABLE=1 tar --no-xattrs`|
|`kairos-prepare` were all dropped|`amount` List all NaN|Fallbacked to `close * volume`|
| `ModuleNotFoundError: kairos` |venv is not activated or not `pip install -e .`| `source .venv/bin/activate && pip install -e .` |
|AutoDL stuck at `loading Kronos-Tokenizer-base`|`http_proxy` and `HF_ENDPOINT=hf-mirror.com` conflict|`unset http_proxy https_proxy`; Pre-cache `huggingface-cli download` in advance|
|`numpy` 2.x causes torch to crash|version conflict| `pip install "numpy<2"` |
|DDP cannot run on CPU|default nccl|`training_utils.setup_ddp` Automatically cut gloo|
|The office network blocked OKX/Binance main site|GFW + Company Whitelist|Use `--exchange binance_vision` to go to the `data-api.binance.vision` spot mirror, which can only pull the spot K line (no funding/OI/basis)|
|AutoDL is blocked by default `api.okx.com` / `fapi.binance.com`|DNS Pollution + Squid only whitelist github/hf for `/etc/network_turbo`|Install mihomo (Clash Meta) + airport subscription on AutoDL; get YAML from `flag=meta`; pre-download `Country.mmdb` + `GeoSite.dat` + `geoip.dat` and put it in the `-d` directory; the default GLOBAL is `DIRECT`, you need to use `PUT /proxies/GLOBAL` to switch to the specific node (the US node is the fastest measured, ~1.1s/req). See `docs/CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md` for complete steps|
|`--market crypto` ran out `funding_rate` / `oi_change` / `basis` / `btc_dominance` All four columns are 0|`kairos-collect` only uses OHLCV, `prepare_dataset` does not pass `extras`, adapter’s `_align_series` fillna(0) for missing series|Either accept it (this is currently the case for BTC/ETH + Top100 spot runs), or go for `docs/CRYPTO_OKX_PERP_MULTICHANNEL_PLAN.md`’s multi-channel transformation|
|The parquet time range offset pulled out by `binance_vision`|`_to_unix_ms` Use naive local time to convert to UTC|Expected behavior, does not affect training for 24/7 crypto; if you really want accurate UTC date boundary, just manually transfer the complete ISO time|
|The native macOS `torchrun --standalone` is stuck in a pile of `IPv6 ... gai error: 8` warnings for a long time|macOS fails to resolve the local hostname to IPv6, and the rendezvous server of `torchrun` hangs on the hostname and times out.|Single card/native machine does not use torchrun for smoke, just `MASTER_ADDR=127.0.0.1 MASTER_PORT=295xx WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python -m kairos.training.train_predictor`; AutoDL/GPU machine still uses torchrun normally.|
|When smoke `OneCycleLR` throws `ZeroDivisionError: float division by zero`|`total_steps = epochs * steps_per_epoch` If it is too small, `int(pct_start * total_steps)` degenerates to 0 and the phase boundaries coincide.|`KAIROS_SMOKE=1` has set `n_train_iter` to 200 and `warmup_pct=0.2` to ensure `total_steps ≥ ~50`; this lower limit must also be observed when customizing smoke|
|`backtest_ic --per-symbol-limit` After running `by_date_mean.ic` all are `NaN`|Each symbol is independently and equidistantly sampled, and the timestamps extracted are not aligned → there is only 1 record in each bucket, and the cross-sectional correlation coefficient cannot be calculated.|Smoke can use `--aggregation none` to see the overall; to check the bucket IC on a small number of symbols, either `--stride 60` let all symbols use the same set of offsets, or directly run the GPU with full stride=1|
| `ccxt.base.errors.InvalidProxySettings: okx you have multiple conflicting proxy settings(httpProxy,httpsProxy)` |`check_proxy_settings` with ccxt ≥ 4.5 is not allowed to set `http_proxy` + `https_proxy` at the same time; earlier versions of OKX adapter have both blocked|`kairos/data/markets/crypto_exchanges/okx.py` Now only set up `https_proxy` (commit `9e33a2f`), OKX uses all HTTPS; when writing a new adapter, be careful to only leave the https side.|
|`[<sym>] funding fetch failed: 'timestamp'` or OI `50030 Illegal time range`|The `since` kwarg of ccxt is ignored by the server on the OKX funding-history / OI interface → the latest data returned is filtered to empty by `[start_ms,end_ms)` → subsequent `df["timestamp"]` KeyError|adapter is now changed to `params={"after": cursor}` to check funding, `params={"begin":...,"end":...}` to check OI (commit `05b8595`); at the same time, empty frame retains `funding_rate`/`open_interest` columns to avoid KeyError|
|OKX funding / OI short historical window|Hard retention of OKX API: funding-rate-history ~90 days; contracts/open-interest-history only returns the latest ~8 hours** (100 items × 5m), `after` cursor does not take effect on this endpoint|**funding**: The data within the past 90 days is OK for training; for older windows, empty tables must be accepted or Coinglass must be used. **OI**: Currently you can only subscribe in real time and accumulate orders yourself; short window smoke can be used, but the OI column of Top100 × 1 year training will be 0 (the same as the old BTC/ETH spot run), which must be clearly marked in the document|
|`kairos-prepare --train 2026-04-13:2026-04-13` Each symbol is packaged into only **1 line**|`_slice` uses `(datetime >= start) & (datetime <= end)`, `end="2026-04-13"` is parsed into `2026-04-13 00:00:00`, and the minute-level data only has one hit at 00:00|For minute-level data `--train/--val/--test`, you need to pass "next day" as end (`2026-04-13:2026-04-14` means covering the whole day from 04-13), or pass the complete ISO timestamp (be careful not to have 3 colons, `parse_range` use `:` to hard cut)|
|`kairos-prepare --train "2026-04-13 08:00:2026-04-14 08:00"` reported `too many values to unpack (expected 2)`|`parse_range` Directly `split(":")`, the ISO time with `HH:MM` has too many colons|The current solution is to avoid writing ISO time in the CLI and return to the daily granularity `YYYY-MM-DD:YYYY-MM-DD`; a better solution is to subsequently change parse to rsplit or change the separator|
|The training log shows `[TRAIN] pool=327610, using 5000/epoch.`, val_ce only dropped by 0.006 after 10 epochs, and negative migration occurred in backtest|`KAIROS_N_TRAIN_ITER=5000` was left in the previous mini run, but was not cleared in the official run → Only 5000 samples are randomly selected per epoch = 1.5% of the pool|**Before officially running `unset KAIROS_N_TRAIN_ITER`** let it run default 50000; self-check to see if the proportion of `using Y/X` is ≥ 5%. See `docs/CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md` §8.1 for details|
|`backtest_ic --aggregation date` Output `n_dates: 3, icir: +1.17` (looks good but something is wrong)|The test area only has 3 days → the date bucket has only 3 ICs to calculate mean/std, and the ICIR is completely noise|If the test area is < 5 days, use `--aggregation none` to view `overall.spearman`; if the test area is ≥ 15 days, use `by_date_mean`. For the complete decision tree, see `docs/BACKTEST_IC_INTERPRETATION_GUIDE.md` §2|
|`--baseline` ran out h30 ICIR=+0.42, looking at the original weight of Kronos, there is alpha|Random head + Kronos hidden can produce an artificially high ICIR under the scale of 100 symbols × 78 days|**MUST** report both baseline and finetuned, looking at Δ rather than absolute values. See `docs/BACKTEST_IC_INTERPRETATION_GUIDE.md` §5 for details|
|The training ckpt is overwritten (such as the new perp run overwriting the last spot ckpt)|`train_predictor.py` defaults to `artifacts/checkpoints/predictor/checkpoints/best_model/` without run name|`cp -r best_model best_model_<run-name>_backup` before running a new run; next time it is best to change the best_model writing method to the hash/timestamp subdirectory|

See `docs/AUTODL_REMOTE_TRAINING_GUIDE.md`'s "Common Pitfalls" section for more, and `docs/CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md`'s complete post-mortem.

---

## 8. Training/backtest current baseline

### A-shares daily line (ashare-daily)

- **v1** (time-split, verified throughout 2024) → overfitting, test IC is negative.
- **v2** (interleave-split + lower lr + increase quantile_weight + early stop) → val_ce improved, but test IC is still negative.
- Conclusion: The correlation between supervision signals and A-shares’ future earnings is weak; the next step is written in `docs/TRAINING_TUNING_PLAYBOOK.md`.

### crypto 1min(crypto-1min)

| run | universe | h30 rank-IC (finetuned) | h30 ICIR |Details|
|---|---|---|---|---|
| BTC+ETH 2y spot |2 coins × 2 years| **+0.050** | **+0.325** | `docs/CRYPTO_BTC_ETH_2Y_SPOT_RUN.md` |
| Top100 1y spot |100 coins × 1 year| +0.030 | **+0.454** | `docs/CRYPTO_TOP100_1Y_SPOT_RUN.md` |
| Top10 30d perp ⚠️ |10 coins × 30 days|+0.016 (n=3 noise)| +0.06 |`docs/CRYPTO_OKX_PERP_TOP10_30D_RUN_POSTMORTEM.md` (negative migration post-mortem)|

h30 is currently the only valid horizon (preset `return_horizon=30` aligned), h1/h5 is not really supervised due to the training target dimension design, and the IC is close to 0 or reverse. For improvement directions, see `docs/BACKTEST_IC_INTERPRETATION_GUIDE.md` §4 and `docs/TRAINING_TUNING_PLAYBOOK.md` §8.2.

### BSQ Tokenizer(crypto-1min, Kairos-base-crypto)

Currently only completed code + local CPU smoke (50 steps / M1 CPU / 17 s) - `recon_mse_full` dropped from 0.00557 of baseline to 0.00518 (-7%), proving that the link is working. **Official run must be run on AutoDL, see `docs/CRYPTO_BTC_ETH_TOKENIZER_RUN.md`** for complete steps.
Tokenizer training ≈ 4M parameters, batch=50, epochs=15 + patience=3, lr=2e-4, fully unfrozen, expected to be completed in 5–10 minutes on 5090.

When changing super parameters, always change `kairos/training/config.py` to `TrainConfig`. Do not hard-code numbers into `train_predictor.py`. The cross-market parameter combination is `preset_for(name)` - when creating a new market/frequency, add a **synchronization** to `_PRESETS` instead of letting the caller spell the dict by himself.

### Architectural Invariants (Phase 2)

1. `len(COMMON_EXOG_COLS) + len(adapter.MARKET_EXOG_COLS) == 32`——This must be maintained when adding a new adapter, `build_features` will directly assert and throw an error.
2. `n_exog` on the model side is fixed at 32, and does not follow the adapter. If you want to add new factors, occupy pad or replace a certain slot, and do not expand the dimension.
3. The new adapter must be able to be swallowed by `try/except ImportError` in `kairos/data/markets/__init__.py` after the import fails in an environment where optional dependencies such as `[crypto]` are not added, and the A-shares main path cannot be destroyed.
4. `kairos-prepare` The output directory must contain `meta.json`, otherwise the downstream `backtest_ic --dataset-path ...` cannot restore market/freq.

---

## 9. document maintenance

- README is the external facade, changes should be refined; details should be written in `docs/`.
- When adding `docs/*.md`, add a line of links to `docs/DOCUMENTATION_INDEX.md`, the document entry area of ​​README and `AGENTS.md` §2.
- Check `docs/CONCEPTS_AND_GLOSSARY.md` for terms first, and fill in any missing terms; do not define the same term repeatedly in multiple documents.

---

## 10. Maintenance of this document itself

- The **long-term agreement** given by the user in the conversation (for example, Article 6.1 of this document is extracted from the user's "commit + push after each modification") should be settled here.
- Changing `AGENTS.md` itself also applies to the rules in Section 6: commit + push immediately after making changes.
