# Sprint 17 — AGATE Resurrection Plan (Shadow-First Protocol)

**Status:** STAGED (pre-Apr-25 deploy). Pre-staging on local machine only. No VM touches until Apr 25.
**Created:** 2026-04-24
**Owner:** P. Jenkins (+ Claude pair)
**Source of design:** User-articulated plan — codified here per explicit authorization 2026-04-24.

---

## Epistemic context

AGATE was suspended **2026-04-20 (Sprint 16)** after 11 closed trades produced 2W/9L / −$463.82 realized.

The stated kill reason was *"Gaussian emission misspecification. Crypto has kurtosis 8-15× equities — GaussianHMM's emission model is structurally wrong."*

**This diagnosis was an educated guess, not empirically falsified.** Jumping directly to a custom Student's t-HMM (custom EM algorithm, ~2-3 weeks of engineering) without proving the failure mode would be a colossal misallocation of bandwidth. Sprint 17 replaces the guess with a deductive, empirically falsifiable roadmap.

**Core principle:** prove the pathology before engineering the cure; cheap fix first, expensive fix only if necessary.

---

## 4-Phase Decision Tree

### Phase 0 — Pre-stage (2026-04-24, TODAY)

Goal: prepare the instruments without touching the patient.

- ✅ Sprint 17a: add `--shadow-only` flag to `trading-core/live_trading.py`
  - Generates signals + logs posteriors + writes to `agate_journal.db`
  - Bypasses `LiveBroker` execution entirely (no `_enter_position` / `_exit_position`)
  - Respects existing `--test` / `--live` semantics
- ✅ Codify this plan in `memory/sprint17_agate_resurrection_plan.md` (local + Claude memory)
- ⏸ No VM touches: `agate-trader` service stays `disabled` per Sprint 16
- ⏸ No model changes: still GaussianHMM + ensemble, same config

### Phase 1 — Shadow observation (2026-04-25 deploy → Late May)

Goal: collect forward-observation data on the current (guilty-looking) model before judging it.

- Deploy `--shadow-only` flag to VM's systemd unit (one-line ExecStart change) on **2026-04-25**
- Service resumes execution as `agate-trader-shadow` but with zero broker action
- 2-4 weeks of forward-observation on 14 crypto tickers at 4h bars
- All signals + confidence posteriors + 4h/24h forward returns persisted to `agate_journal.db`
- **NO live execution.** Broker/Position ledger writes physically bypassed.
- Daily sanity check: shadow journal row count growing at expected rate (~84 rows/day/ticker × 14 = ~1,176/day)

### Phase 2 — Brier Score Decomposition (Late May 2026)

Goal: **prove or falsify the Gaussian misspecification hypothesis** with a calibration-based test, not volatility plots.

**The Test:**
- Group Phase-1 shadow predictions into confidence deciles (e.g., `[0.85-0.90)`, `[0.90-0.95)`, `[0.95-1.00]`)
- For each decile, compute the *empirical* frequency that the asset realized a positive return over the next 4h / 24h
- Plot: nominal confidence (x-axis) vs realized frequency (y-axis). A well-calibrated model tracks y=x.

**The Proof / Falsification thresholds:**
| Decile | Nominal conf | Realized freq needed to PASS |
|---|---|---|
| `[0.90, 0.95)` | 92.5% | ≥ 85% realized |
| `[0.95, 1.00]` | 97.5% | ≥ 90% realized |

- **PASS (Gaussian OK):** top deciles realized ≈ nominal. The kill was wrong; AGATE is resurrectable with current emission model.
- **FAIL (Gaussian broken):** top decile realized < 55% of nominal → model is hallucinating certainty on the tails. Proceed to Phase 3.

**Artifacts:**
- `reports/2026_MM_DD_agate_gaussian_calibration_test.html` (milestone report — standard neon palette, decile table, calibration plot)
- Memory handoff: `memory/phase2_brier_decomposition_result.md`

### Phase 3 — GMMHMM intermediate (contingent on Phase 2 FAIL)

Goal: cheapest possible fix. A 5-line configuration change, not a custom algorithm.

- Swap `hmmlearn.GaussianHMM` → `hmmlearn.GMMHMM` (native support, no custom EM)
- Each HMM state gains a mixture of `n_mix` Gaussians (start with `n_mix=3` per state)
- One Gaussian captures normal-volatility regime
- One captures low-volatility
- One explicitly captures fat-tail outliers (the hypothesized Gaussian failure mode)

**Implementation:** `src/hmm_model.py` — feature-flag `HMM_EMISSION_MODEL` with values `{"gaussian", "gmm"}`. Default stays `gaussian` until Phase 3 validates.

**Re-validate with Phase-2 Brier decomposition protocol.** If GMM calibrates correctly on top deciles → GMM becomes AGATE's production emission model. **RESURRECTION COMPLETE.**

**Artifacts:**
- `reports/2026_MM_DD_agate_gmm_resurrection.html`
- Memory handoff

### Phase 4 — Student's t-HMM (contingent on Phase 3 FAIL)

Goal: custom emission model only if GMM also fails.

- Write custom Expectation-Maximization for Student's t-HMM (no hmmlearn native support)
- Estimated eng bandwidth: 2-3 weeks
- Re-validate with same Brier decomposition protocol

### Exit criteria

- **RESURRECT** at Phase 2 pass (Gaussian was OK — the kill was premature) OR Phase 3 pass (GMM fixes it)
- **RETIRE** if Phase 4 also fails — accept that AGATE's signal is structurally weak on crypto, shut down permanently, redirect bandwidth
- **NEVER** deploy AGATE live without passing the Brier calibration test first

---

## 72-Hour Hands-Off Protocol (active until 2026-04-25)

- `agate-trader` systemd unit stays `stopped` and `disabled` on OCI VM
- No config changes to `agate_per_ticker_configs.json`
- No model retraining
- No dashboard changes (AGATE panel on :8060 continues to show last-known state)
- DIAMOND patience window is UNAFFECTED (Sprint 14 work continues to print data toward N=500)
- Pre-staging happens on Mac only (`/Users/perryjenkins/Documents/quant/trading-core/`)
- Apr 25 deploy: single atomic commit + `systemctl restart agate-trader` with updated ExecStart

---

## Code patch applied 2026-04-24 (Sprint 17a)

**File:** `trading-core/live_trading.py`

Three changes:

1. **Argparse** — new `--shadow-only` flag, documented intent, does not require `--test` or `--live`
2. **`LiveTradingEngine.__init__`** — new `shadow_mode: bool = False` parameter, stored as `self.shadow_mode`, logged at startup
3. **Main loop at line ~717** — gate `self.process_signals(signals)` behind `if not self.shadow_mode`. Shadow mode emits per-ticker "would-enter" log lines and increments signal journal, but skips broker calls.

See `git show <commit>` for exact diff once committed.

---

## Reference artifacts

- Sprint 16 suspension postmortem: `trading-core/CLAUDE.md` § AGATE, lines near "SUSPENDED (Sprint 16, 2026-04-20)"
- Current AGATE config: `agate_per_ticker_configs.json` (14 tickers, `extended_v2` feature set)
- Shadow-mode flag implementation: `live_trading.py:~745` (Sprint 17a patch)
- Journal DB (pre-existing, will receive shadow rows): `agate_journal.db`
- Brier decomposition reference implementation (reuseable): analogous to `backtest_kelly.py` Phase B σ_max bucketing pattern
- hmmlearn GMMHMM docs: https://hmmlearn.readthedocs.io/en/latest/api.html#gmmhmm

## Dependencies on other sprints

- **DIAMOND N=500 patience window (active):** independent. Sprint 17 work happens in parallel, no shared code paths. DIAMOND does not block on AGATE resurrection.
- **BERYL + CITRINE (live):** independent. AGATE shares `src/hmm_model.py` with them — Phase 3 GMMHMM swap uses a feature flag so BERYL/CITRINE keep using GaussianHMM by default unless explicitly migrated.

---

## Next Session Handoff (for a fresh Claude invocation)

If you are a future Claude session picking up this plan:

1. Read this file first. The plan is deductive and sequential; do not skip phases.
2. Phase 2 is a *falsifiable* test. A failed top decile means GMM, not "try harder with Gaussian." A passing top decile means *resurrect now*, not "but the kill reason was real, right?"
3. Do NOT skip to Phase 4. The value of this plan is the cheap-fix-first philosophy — skipping GMM would reintroduce the original misallocation.
4. VM hands-off until Apr 25 is non-negotiable.
5. If the user asks you to derive the Brier decomposition code, the structure is: load `agate_journal.db` → bucket by `confidence` decile → compute realized next-bar positive-return rate per bucket → compare to nominal.
