# Implementation Plan: Funding and VWAP robustness

**Date**: 2026-10-04
**Spec**: [spec.md](spec.md)

## Technical Context

Python 3.10+, pandas/NumPy, existing pytest suite. Work uses a managed isolated worktree with file ownership for parallel issue fixes, protecting another chat's work on `main`. Spec Kit integration is healthy; no extension hooks are configured. Only spec/plan/tasks are retained for this small fix, per the project constitution.

## Constitution Check

- Trailing timestamp windows and prefix-invariance tests enforce causality.
- Preserve all 32 factor names, pickle layout, and `meta.json`.
- Run failing regressions before implementation, targeted tests, `pytest -x`, and a temporary synthetic packaging run.
- Use only temporary offline data; do not touch existing experiments or credentials.

## Design

1. `kairos/data/markets/crypto.py`: replace the 60-bar funding z-score with a three-day trailing window over timestamp-indexed, forward-filled rates. Retain missing observations as missing until the final neutral fill; minimum two observations and zero variance produce zero.
2. `kairos/data/common_features.py`: fill missing amount from close times volume, calculate VWAP only for finite positive price/volume/amount, and explicitly neutralize unavailable/proxy deviations. Log counts of unavailable and proxy-equivalent rows; equality alone does not identify true source provenance.
3. `tests/test_feature_robustness.py`: cover settlement persistence, elapsed-time window boundaries, missing/constant series, causality, invalid turnover, exact fallback neutrality, valid VWAP, and 32-column compatibility.
4. Before `amount_z` applies `log1p`, mask negative and nonfinite turnover as missing while retaining finite zero. Verify domain boundaries with runtime warnings treated as errors; preserve the existing trailing 60-bar statistics and shared feature cleanup.

## Verification

- Run new regressions before the fix, then feature/adapter regressions and `pytest -x`.
- Build one small synthetic symbol and sidecars in a fresh temporary directory, run the actual `kairos-prepare` entry point with explicit splits, and inspect finite factors and metadata.
- Review FR-001–FR-004 and SC-001–SC-003 during convergence; the root agent performs final integrated review and Git operations.

## Scope Boundary

No changes to the common 60-bar technical indicators, external data sources, model dimension, packaging's downstream normalization, or managed Spec Kit files. Scripts that persist shared `.specify/feature.json` are replaced by explicit feature-path checks to avoid cross-agent state collisions.
