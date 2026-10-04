# Tasks: Funding and VWAP robustness

## Phase 1: Setup and specification

- [x] T001 Verify repository conventions, constitution, integration, and existing features; write `specs/002-feature-robustness/spec.md` and `plan.md` and resolve clarification/analysis.

## Phase 2: Funding signal (US1)

- [x] T002 [US1] Add failing funding persistence, elapsed-time, missing-history, and causality regressions in `tests/test_feature_robustness.py` (FR-001, FR-002, FR-004).
- [x] T003 [US1] Implement trailing three-day funding normalization in `kairos/data/markets/crypto.py`.

## Phase 3: VWAP signal (US2)

- [x] T004 [US2] Add failing invalid-volume/amount, proxy neutrality, and diagnostics regressions in `tests/test_feature_robustness.py` (FR-003, FR-004).
- [x] T005 [US2] Implement guarded VWAP deviation and diagnostics in `kairos/data/common_features.py`.

## Phase 4: Verification and convergence

- [x] T006 Verify `tests/test_feature_robustness.py`, existing tests, and temporary synthetic `kairos-prepare` output (SC-001–SC-003).
- [x] T007 Converge code against `specs/002-feature-robustness/spec.md` and hand off exact validation results for root review.
- [x] T008 During v2 integration, reproduce the invalid-amount warning and negative fractional amount signal, mask invalid `amount_z` inputs, and verify warning-free feature tests plus actual schema-version-2 packaging (FR-005).

Dependencies: T001 → T002 → T003; T001 → T004 → T005; both stories → T006 → T007. The two implementation files are independent, but tests are owned by one agent to avoid conflicts. Deliver both corrections together.
