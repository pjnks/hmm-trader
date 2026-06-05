"""diagnose_sprint18_backfill.py — Sprint 18 historical-data Cohort B verdict.

Companion harness to ``diagnose_sprint18_citrine.py`` (live shadow stream).
This script applies the live engine's measured 28 bps round-trip friction
to neutralized forward returns from the Sprint 15 ``scan_journal_backfill``
table (13,230 scan-rows × 98 NDX tickers × 135 days, Oct 2025 -> Apr 2026)
and produces a Sprint 18 verdict on the [0.80, 0.90) confidence cohort.

PROTOCOL NOTE: This is local-only research on already-cached data. Running
it does NOT break the hands-off observation window. Output is informational
until paired with the live shadow stream on 2026-05-26.

PHYSICAL FRICTION (verified via diagnose_sprint18_citrine.py 2026-05-04):
  Round-trip drag = 28 bps (8 taker fees + 20 slippage), confirmed against
  the live engine's pnl_net to < $0.01 reconciliation delta.

The two streams must AGREE before Sprint 18 deploy:
  - Backfill stream: high N (~2,590), simulated friction, no execution variance
  - Live stream:     low N (gated by future Sprint 17b), real friction
  - Convergent PASS: both streams clear gates -> deploy
  - Divergent:       investigate before deploy

Usage:
    python research/diagnose_sprint18_backfill.py
    python research/diagnose_sprint18_backfill.py --friction-bps 28
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the Apr 24 audit's data loader, feature engine, and friction model.
# This guarantees the Sprint 18 verdict computes against the SAME data and
# SAME neutralization as the +5.1 bps pocket finding that triggered the
# sprint in the first place. Drift between these two scripts would be a
# silent contamination source on May 26.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_engine_eval import load_data, add_features

# Total friction in bps (one round-trip). Live engine measurement.
TOTAL_FRICTION_BPS = 28

# Cohort definitions (long-only)
COHORT_A_BOUNDS = (0.90, 1.01)    # >= 0.90  (current entry rule)
COHORT_B_BOUNDS = (0.80, 0.90)    # [0.80, 0.90) — Sprint 18 candidate

# Diagnostic bins to display the full alpha-vs-friction curve
DIAGNOSTIC_BINS = [
    ("[0.60, 0.80)", 0.60, 0.80),
    ("[0.80, 0.90)", 0.80, 0.90),
    ("[0.90, 0.95)", 0.90, 0.95),
    ("[0.95, 1.00]", 0.95, 1.01),
]

# Horizons (Sprint 18 memo references T+1 and T+3 specifically)
HORIZONS = (1, 3)
PRIMARY_HORIZON = 1

# Sprint 18 success criteria (from sprint18_citrine_080_pocket.md)
MIN_N_COHORT_B = 30
WILSON_Z = 1.96
WILSON_LOWER_BOUND_GATE = 0.50
SHARPE_PROXY_GATE = 0.30


def wilson_ci(wins: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score 95% CI on a binomial proportion. Identical to the live
    harness implementation — reproduced here for self-containment."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))
    return (center - spread) / denom, (center + spread) / denom


def filter_to_cohort(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Slice rows whose `confidence` falls in [lower, upper)."""
    return df[(df["confidence"] >= lower) & (df["confidence"] < upper)]


def cohort_stats(
    df_cohort: pd.DataFrame, horizon: int, friction_bps: float,
) -> dict:
    """Friction-honest EV stats for one cohort × one forward horizon.

    Returns a dict with: n, gross_bps, net_bps, hit_rate, wilson_lower,
    wilson_upper, sharpe_proxy. Empty cohort returns {n: 0}.
    """
    col = f"neut_ret_{horizon}d"
    rets = df_cohort[col].dropna()
    n = len(rets)
    if n == 0:
        return {"n": 0}

    rets_bps = rets * 10000
    gross_bps = rets_bps.mean()
    net_bps = gross_bps - friction_bps

    # Per-observation hit rate: fraction where the row's net return > 0
    hits = int((rets_bps - friction_bps > 0).sum())
    hit_rate = hits / n
    wlo, whi = wilson_ci(hits, n)

    # Sharpe-like proxy in net-bps space
    if rets_bps.std() > 0:
        sharpe = net_bps / rets_bps.std()
    else:
        sharpe = 0.0

    return {
        "n": n,
        "gross_bps": float(gross_bps),
        "net_bps": float(net_bps),
        "hit_rate": float(hit_rate),
        "wilson_lower": float(wlo),
        "wilson_upper": float(whi),
        "sharpe_proxy": float(sharpe),
    }


def evaluate_pass_fail(b_stats: dict) -> dict[str, bool]:
    """Apply Sprint 18 cohort B gates. Mirrors the live harness gates,
    minus drawdown_bounded which doesn't translate from per-observation
    backfill data to per-trade closed-trade data."""
    return {
        "n_floor": b_stats.get("n", 0) >= MIN_N_COHORT_B,
        "positive_ev": b_stats.get("net_bps", 0) > 0,
        "wilson_lower": b_stats.get("wilson_lower", 0) >= WILSON_LOWER_BOUND_GATE,
        "sharpe_proxy": b_stats.get("sharpe_proxy", 0) >= SHARPE_PROXY_GATE,
    }


def format_report(
    df: pd.DataFrame, friction_bps: float, b_stats: dict, gates: dict,
) -> str:
    n_total = len(df)
    n_with_fwd = int(df[f"neut_ret_{PRIMARY_HORIZON}d"].notna().sum())

    lines = [
        "Sprint 18 — CITRINE 0.80 Pocket A/B (HISTORICAL BACKFILL)",
        "=" * 70,
        f"Source:    scan_journal_backfill ({n_total:,} scan-rows)",
        f"With fwd:  {n_with_fwd:,} rows have forward return at T+{PRIMARY_HORIZON}d",
        f"Friction:  {friction_bps:.0f} bps round-trip "
        "(8 fees + 20 slippage, verified live)",
        "",
        "Per-bin EV (neutralized forward returns, friction-adjusted):",
        f"  {'Bin':<16}{'N':>8}  "
        f"{'Gross T+1':>11}  {'Net T+1':>11}  "
        f"{'Gross T+3':>11}  {'Net T+3':>11}",
        f"  {'-' * 16}{'-' * 8}  {'-' * 11}  {'-' * 11}  "
        f"{'-' * 11}  {'-' * 11}",
    ]

    for label, lo, hi in DIAGNOSTIC_BINS:
        sub = filter_to_cohort(df, lo, hi)
        cells = [f"{label:<16}{len(sub):>8}"]
        for h in HORIZONS:
            s = cohort_stats(sub, h, friction_bps)
            if s.get("n", 0) > 0:
                cells.append(f"{s['gross_bps']:>+9.1f} bps  "
                             f"{s['net_bps']:>+9.1f} bps")
            else:
                cells.append(f"{'—':>11}  {'—':>11}")
        lines.append("  " + "  ".join(cells))

    lines.extend([
        "",
        f"Sprint 18 verdict — Cohort B [0.80, 0.90), T+{PRIMARY_HORIZON}d:",
        "  " + "-" * 64,
    ])
    if b_stats.get("n", 0) > 0:
        lines.extend([
            f"  N observations:    {b_stats['n']:,}",
            f"  Hit rate:          {b_stats['hit_rate']:.1%}  "
            f"(Wilson 95% lower: {b_stats['wilson_lower']:.1%})",
            f"  Gross bps:         {b_stats['gross_bps']:+.2f}",
            f"  Net bps:           {b_stats['net_bps']:+.2f}",
            f"  Sharpe proxy:      {b_stats['sharpe_proxy']:.4f}",
        ])
    else:
        lines.append("  N=0 — no observations in cohort.")

    lines.extend(["", "Sprint 18 gates (Cohort B must pass all):"])
    for gate, ok in gates.items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {gate}")

    if all(gates.values()):
        verdict = ("PASS — backfill stream supports lowering "
                   "CITRINE_ENTRY_CONFIDENCE to 0.80")
    else:
        verdict = ("FAIL — backfill alpha consumed by friction; "
                   "do not deploy on this stream alone")

    lines.extend([
        "",
        f"VERDICT (backfill stream): {verdict}",
        "",
        "Pair this verdict with diagnose_sprint18_citrine.py (live shadow",
        "stream) on 2026-05-26. Both must clear gates before deploy.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-pull from VM (default: use parquet cache)")
    parser.add_argument("--friction-bps", type=float, default=TOTAL_FRICTION_BPS,
                        help=f"Override round-trip friction (default: "
                             f"{TOTAL_FRICTION_BPS} bps)")
    args = parser.parse_args()

    df = load_data(refresh=args.refresh)
    df = add_features(df)

    cohort_b = filter_to_cohort(df, *COHORT_B_BOUNDS)
    b_stats = cohort_stats(cohort_b, PRIMARY_HORIZON, args.friction_bps)
    gates = evaluate_pass_fail(b_stats)

    print(format_report(df, args.friction_bps, b_stats, gates))


if __name__ == "__main__":
    main()
