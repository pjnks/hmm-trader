#!/usr/bin/env python3
"""
Fix 6: Semi-Markov Research Stub

Compares the standard HMM geometric sojourn distribution against empirical
sojourn times from Viterbi-decoded state sequences. If the empirical
distribution significantly deviates from geometric (KS test p < 0.05),
recommends investigating Semi-Markov / Explicit-Duration HMMs.

Usage:
    python research/semi_markov_analysis.py
"""

import sys
sys.path.insert(0, ".")

import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def compute_sojourn_times(state_sequence: np.ndarray) -> dict[int, list[int]]:
    """Extract sojourn durations per state from a Viterbi sequence."""
    sojourns: dict[int, list[int]] = {}
    if len(state_sequence) == 0:
        return sojourns

    current_state = state_sequence[0]
    duration = 1

    for i in range(1, len(state_sequence)):
        if state_sequence[i] == current_state:
            duration += 1
        else:
            sojourns.setdefault(current_state, []).append(duration)
            current_state = state_sequence[i]
            duration = 1
    # Don't count the last sojourn (right-censored)
    return sojourns


def geometric_cdf(k: np.ndarray, p: float) -> np.ndarray:
    """CDF of geometric distribution: P(duration <= k) = 1 - (1-p)^k."""
    return 1.0 - (1.0 - p) ** k


def analyze_ticker(ticker: str, n_states: int = 6):
    """Run sojourn analysis for one ticker."""
    from src.data_fetcher import fetch_btc_hourly, build_hmm_features, resample_ohlcv
    from src.hmm_model import HMMRegimeModel
    import config

    print(f"\n  Analyzing {ticker} (n_states={n_states})...")

    # Fetch data
    polygon_ticker = f"X:{ticker}USD" if not ticker.startswith("X:") else ticker
    df = fetch_btc_hourly(days=730, ticker=polygon_ticker)
    if df is None or len(df) < 200:
        print(f"  Insufficient data for {ticker}")
        return

    df = build_hmm_features(df)
    df = resample_ohlcv(df, "4h")

    # Fit HMM
    feature_cols = config.FEATURE_SETS.get("extended_v2",
                                            ["log_return", "price_range", "volume_change"])
    model = HMMRegimeModel(n_states=n_states, feature_cols=feature_cols)
    model.fit(df)

    if not model.converged:
        print(f"  HMM did not converge for {ticker}")
        return

    # Get Viterbi sequence
    df = model.predict(df)
    states = df["regime"].values  # Raw state assignments

    # Map states to numeric IDs
    state_map = {label: idx for idx, label in enumerate(
        sorted(set(states), key=lambda x: str(x))
    )}
    numeric_states = np.array([state_map[s] for s in states])

    # Extract sojourn times
    sojourns = compute_sojourn_times(numeric_states)

    # Get transition matrix
    A = model._hmm.transmat_

    print(f"\n  {'State':>15s}  {'Category':>8s}  {'A[k,k]':>8s}  {'Half-life':>10s}  "
          f"{'N sojourns':>10s}  {'Mean dur':>10s}  {'KS stat':>8s}  {'p-value':>8s}  {'Geometric?':>10s}")
    print("  " + "-" * 100)

    any_non_geometric = False

    for raw_state in range(n_states):
        label = model.state_label(raw_state)
        cat = model.state_category(raw_state)
        self_trans = A[raw_state, raw_state]

        half_life = -np.log(2) / np.log(self_trans) if 0 < self_trans < 1 else float("inf")

        # Find which numeric state this maps to
        matching_numeric = [k for k, v in state_map.items()
                           if k == label]
        if not matching_numeric:
            continue

        numeric_id = state_map[matching_numeric[0]]
        durations = sojourns.get(numeric_id, [])

        if len(durations) < 5:
            print(f"  {label:>15s}  {cat:>8s}  {self_trans:>8.4f}  {half_life:>9.1f}b  "
                  f"{len(durations):>10d}  {'N/A':>10s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>10s}")
            continue

        mean_dur = np.mean(durations)
        p_exit = 1.0 - self_trans  # Geometric parameter

        # KS test: empirical vs geometric CDF
        emp_sorted = np.sort(durations)
        theo_cdf = geometric_cdf(emp_sorted, p_exit)
        ks_stat, ks_p = stats.ks_1samp(durations,
                                        lambda x, p=p_exit: geometric_cdf(np.array([x]), p)[0])

        is_geometric = "YES" if ks_p >= 0.05 else "NO"
        if ks_p < 0.05:
            any_non_geometric = True

        print(f"  {label:>15s}  {cat:>8s}  {self_trans:>8.4f}  {half_life:>9.1f}b  "
              f"{len(durations):>10d}  {mean_dur:>9.1f}b  {ks_stat:>8.3f}  {ks_p:>8.3f}  {is_geometric:>10s}")

    print(f"\n  -- Verdict for {ticker} --")
    if any_non_geometric:
        print("  At least one state has NON-GEOMETRIC sojourn distribution (KS p < 0.05)")
        print("  → Semi-Markov / Explicit-Duration HMM may improve regime modeling")
        print("  → Consider: pyhsmm, hmmlearn HSMM, or explicit duration penalties")
    else:
        print("  All states have geometric sojourn distributions (p >= 0.05)")
        print("  → Standard HMM assumption is valid; Semi-Markov not needed")


def main():
    print("=" * 60)
    print("  Semi-Markov Sojourn Distribution Analysis")
    print("=" * 60)

    for ticker in ["SOL", "ETH"]:
        try:
            analyze_ticker(ticker, n_states=6)
        except Exception as e:
            print(f"  ERROR for {ticker}: {e}")

    print()


if __name__ == "__main__":
    main()
