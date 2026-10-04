# Feature Specification: Funding and VWAP robustness

**Feature Branch**: managed isolated worktree (parallel issue fixes)
**Created**: 2026-10-04
**Status**: Implemented and verified
**Input**: GitHub issue #4: preserve funding variation across settlements and avoid spurious VWAP signals on zero-volume or estimated-amount bars.

## User Scenarios & Testing

### User Story 1 - Useful funding signal (Priority: P1)

Researchers packing minute swap data need funding standardization to retain differences between successive settlements.

**Independent Test**: Synthetic eight-hour funding settlements remain informative between events, including more than 60 minutes after a rate change.

**Acceptance Scenarios**:
1. Given varying settlement rates, the signal uses the trailing three days at each bar and remains nonzero when the current rate differs from recent history.
2. Given missing, constant, or insufficient funding history, the standardized signal is zero; missing history is not interpreted as zero-rate observations.
3. Appending or changing future observations cannot change earlier features.

### User Story 2 - Neutral unavailable VWAP (Priority: P1)

Researchers need bars without a measurable VWAP to produce a neutral feature and an observable data-quality diagnostic.

**Independent Test**: Zero-volume, invalid-amount, and estimated-amount fixtures yield exactly zero VWAP deviation before clipping, while valid turnover preserves the actual deviation.

**Acceptance Scenarios**:
1. Nonpositive or nonfinite volume/amount produces zero VWAP deviation and a diagnostic count.
2. Missing amount is estimated as close times volume; amount equal to that proxy yields zero deviation and a diagnostic count without claiming the source is known.

### Edge Cases

- Empty or constant funding series; leading gaps before the first observation; gaps longer than three days; different bar frequencies.
- Missing amount column, partly missing amount, tiny positive volume, and invalid volume/amount.

## Requirements

### Functional Requirements

- **FR-001**: Standardize funding using only observations in the trailing three-day interval ending at the current bar, with zero for insufficient/constant history.
- **FR-002**: Preserve the latest available funding rate without introducing artificial zero observations before the first rate.
- **FR-003**: Return zero VWAP deviation when VWAP is unavailable or amount equals close times volume; report aggregate diagnostic counts.
- **FR-004**: Preserve valid VWAP calculations, chronological causality, and the existing 24+8=32 factor names/order.
- **FR-005**: Exclude negative or nonfinite amount observations before `log1p` standardization without suppressing runtime warnings; a finite zero amount remains a valid observation. Missing amount retains the existing close-times-volume fallback.

### Key Entities

- Funding observations: timestamp and rate, forward-filled only from historical observations.
- OHLCV bars: timestamp, prices, volume, and optional quote amount.

## Success Criteria

- **SC-001**: All defect regressions fail before the fix and pass afterward; future-prefix comparisons agree.
- **SC-002**: Offline minimum dataset packaging succeeds with finite 32-factor outputs and unchanged metadata layout.
- **SC-003**: Existing feature/adapter tests and the repository test suite pass.

## Assumptions

- Three days is the issue's accepted example window; statistics are bar-weighted, not settlement-event-weighted.
- Existing factor clipping and packaging normalization remain unchanged; existing datasets must be rebuilt to adopt the correction.
- No remote collection, training, API, model, or artifact changes are in scope. Clarification review found no blocking ambiguity.
