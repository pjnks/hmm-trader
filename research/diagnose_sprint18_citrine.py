"""diagnose_sprint18_citrine.py — Sprint 18 cohort A/B harness.

Reads ``shadow_trades`` from ``citrine_trades.db``, pairs each EXIT row to
its preceding ENTER row by (ticker, timestamp), assigns each closed trade
to a confidence cohort using the ENTER-row confidence, and reports the
friction-honest Grinold A/B comparison the May 26 review needs.

Cohort A (current entry rule): entry confidence >= 0.90
Cohort B (proposed pocket):    entry confidence in [0.80, 0.90)

This harness is BUILT TO SHELF. Do not run before 2026-05-26 — the friction
window opened 2026-04-28 and needs ~4 weeks to accumulate. Running early
contaminates the eventual evaluation with low-N noise.

Usage (May 26):
    python research/diagnose_sprint18_citrine.py \\
        --db /home/ubuntu/HMM-Trader/citrine_trades.db
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Friction params — must match config.CITRINE_TAKER_FEE / CITRINE_SLIPPAGE_BPS
# at the time of the May 26 run. Re-verify before executing.
CITRINE_TAKER_FEE = 0.0004        # 4 bps per side  (8 bps round-trip)
CITRINE_SLIPPAGE_BPS = 10         # 10 bps per side (20 bps round-trip)

# Sprint 18 success criteria (verbatim from sprint18_citrine_080_pocket.md)
MIN_N_COHORT_B = 30
WILSON_Z = 1.96                   # 95% two-sided
WILSON_LOWER_BOUND_GATE = 0.50
SHARPE_PROXY_GATE = 0.30
MAX_DD_STREAK_GATE = 5


@dataclass
class CohortStats:
    label: str
    n: int
    n_wins: int
    win_rate: float
    wilson_lower: float
    wilson_upper: float
    mean_pnl_gross: float
    mean_pnl_net: float
    std_pnl_net: float
    sharpe_proxy: float
    max_dd_streak: int


# ---------------------------------------------------------------------------
# Data extraction — pure DB / pandas, no judgment calls
# ---------------------------------------------------------------------------

def load_paired_trades(db_path: Path) -> pd.DataFrame:
    """Pull shadow_trades, pair each EXIT to its most recent prior ENTER.

    Returns one row per closed shadow trade with columns:
        ticker, entry_ts, exit_ts, entry_confidence, exit_confidence,
        pnl_gross, pnl_net, recomputed_friction_delta
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """SELECT timestamp, ticker, action, confidence,
                      notional, price, pnl, pnl_net
               FROM shadow_trades
               WHERE action IN ('ENTER', 'EXIT')
               ORDER BY ticker, timestamp""",
            conn,
            parse_dates=["timestamp"],
        )

    # Pair ENTER -> EXIT per ticker via groupby state machine.
    paired = []
    for ticker, group in df.groupby("ticker"):
        open_entry = None
        for row in group.itertuples(index=False):
            if row.action == "ENTER":
                open_entry = row
            elif row.action == "EXIT" and open_entry is not None:
                paired.append({
                    "ticker": ticker,
                    "entry_ts": open_entry.timestamp,
                    "exit_ts": row.timestamp,
                    "entry_confidence": open_entry.confidence,
                    "exit_confidence": row.confidence,
                    "entry_price": open_entry.price,
                    "exit_price": row.price,
                    # NOTE: pull entry_notional from EXIT row, not ENTER row.
                    # The engine writes a misleading global-cap value to the
                    # ENTER row's notional column, but the actual capital
                    # deployed (used by the live friction calc) is the
                    # position-dict notional, which gets stamped to the EXIT
                    # row. Verified via friction sanity check 2026-05-04.
                    "entry_notional": row.notional,
                    "pnl_gross": row.pnl,
                    "pnl_net": row.pnl_net,
                })
                open_entry = None  # Closed; await next ENTER
    return pd.DataFrame(paired)


def recompute_friction(row: pd.Series) -> float:
    """Recompute pnl_net from first principles as a sanity check.

    Mirrors ``ShadowTracker.run_shadow_cycle`` lines 331-341 exactly.
    Returns the friction delta (pnl_gross - pnl_net_recomputed); should be
    near-zero versus the stored pnl_net on rows where pnl_net is populated.
    """
    if pd.isna(row["pnl_gross"]) or row["entry_price"] <= 0:
        return float("nan")
    entry_notional = row["entry_notional"] or 0
    if entry_notional == 0:
        return float("nan")
    exit_notional = abs(row["exit_price"] * (entry_notional / row["entry_price"]))
    fee = (entry_notional + exit_notional) * CITRINE_TAKER_FEE
    slip = (entry_notional + exit_notional) * (CITRINE_SLIPPAGE_BPS / 10000.0)
    return fee + slip


# ---------------------------------------------------------------------------
# Statistical primitives — TWO of these need your input (see TODO blocks)
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """
    Computes the Wilson score interval for a binomial proportion.
    Using z=1.96 yields a 95% confidence interval.
    """
    if n == 0:
        return 0.0, 0.0

    p = wins / n
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))

    lower_bound = (center - spread) / denominator
    upper_bound = (center + spread) / denominator

    return lower_bound, upper_bound


def max_consecutive_loss_streak(pnl_series: list[float]) -> int:
    """
    Calculates the maximum consecutive sequence of negative P&L trades.
    Strictly ignores exact 0.0 artifacts to prevent them from artificially
    breaking a genuine losing streak.
    """
    max_streak = 0
    current_streak = 0

    for pnl in pnl_series:
        if pnl < 0:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        elif pnl > 0:
            current_streak = 0
        # If pnl == 0.0, do nothing. Do not increment, do not reset.

    return max_streak


# ---------------------------------------------------------------------------
# Cohort summary — fully built once the two TODOs above are filled
# ---------------------------------------------------------------------------

def summarize_cohort(df: pd.DataFrame, label: str) -> CohortStats:
    closed = df.dropna(subset=["pnl_net"])
    n = len(closed)
    if n == 0:
        return CohortStats(label, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    n_wins = int((closed["pnl_net"] > 0).sum())
    win_rate = n_wins / n
    wlo, whi = wilson_ci(n_wins, n)

    mean_net = closed["pnl_net"].mean()
    std_net = closed["pnl_net"].std(ddof=1) if n > 1 else 0.0
    sharpe_proxy = mean_net / std_net if std_net > 0 else 0.0

    return CohortStats(
        label=label,
        n=n,
        n_wins=n_wins,
        win_rate=win_rate,
        wilson_lower=wlo,
        wilson_upper=whi,
        mean_pnl_gross=closed["pnl_gross"].mean(),
        mean_pnl_net=mean_net,
        std_pnl_net=std_net,
        sharpe_proxy=sharpe_proxy,
        max_dd_streak=max_consecutive_loss_streak(closed["pnl_net"]),
    )


def evaluate_pass_fail(cohort_b: CohortStats) -> dict[str, bool]:
    """Apply the Sprint 18 success criteria to cohort B.

    PASS requires ALL of:
      - n >= MIN_N_COHORT_B (statistical floor)
      - mean_pnl_net > 0 (clears friction)
      - wilson_lower >= WILSON_LOWER_BOUND_GATE
      - sharpe_proxy >= SHARPE_PROXY_GATE
      - max_dd_streak <= MAX_DD_STREAK_GATE
    """
    return {
        "n_floor": cohort_b.n >= MIN_N_COHORT_B,
        "positive_ev": cohort_b.mean_pnl_net > 0,
        "wilson_lower": cohort_b.wilson_lower >= WILSON_LOWER_BOUND_GATE,
        "sharpe_proxy": cohort_b.sharpe_proxy >= SHARPE_PROXY_GATE,
        "drawdown_bounded": cohort_b.max_dd_streak <= MAX_DD_STREAK_GATE,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(
    paired: pd.DataFrame, a: CohortStats, b: CohortStats,
    gates: dict[str, bool],
) -> str:
    closed = paired["pnl_net"].notna().sum()
    orphans = len(paired) - closed
    lines = [
        "Sprint 18 — CITRINE 0.80 Pocket A/B (friction-honest)",
        "=" * 62,
        f"Total paired trades:     {len(paired)}",
        f"  with pnl_net populated: {closed}  (post-Sprint-17 cohort)",
        f"  pre-Sprint-17 orphans:  {orphans}  (excluded from A/B)",
        "",
        f"{'':<20}{'Cohort A (>=0.90)':<22}{'Cohort B [0.80,0.90)':<22}",
        f"{'-' * 64}",
        f"{'N':<20}{a.n:<22}{b.n:<22}",
        f"{'Win rate':<20}{a.win_rate:<22.1%}{b.win_rate:<22.1%}",
        f"{'Wilson 95% lower':<20}{a.wilson_lower:<22.1%}{b.wilson_lower:<22.1%}",
        f"{'Mean P&L gross':<20}${a.mean_pnl_gross:<21.2f}${b.mean_pnl_gross:<21.2f}",
        f"{'Mean P&L net':<20}${a.mean_pnl_net:<21.2f}${b.mean_pnl_net:<21.2f}",
        f"{'Sharpe proxy':<20}{a.sharpe_proxy:<22.3f}{b.sharpe_proxy:<22.3f}",
        f"{'Max loss streak':<20}{a.max_dd_streak:<22}{b.max_dd_streak:<22}",
        "",
        "Sprint 18 gates (Cohort B must pass all):",
    ]
    for gate, ok in gates.items():
        marker = "PASS" if ok else "FAIL"
        lines.append(f"  [{marker}] {gate}")
    verdict = "PASS — issue patch lowering CITRINE_ENTRY_CONFIDENCE to 0.80" \
        if all(gates.values()) else \
        "FAIL — keep entry at 0.90; document negative result"
    lines.extend(["", f"VERDICT: {verdict}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True,
                        help="Path to citrine_trades.db")
    parser.add_argument("--friction-check", action="store_true",
                        help="Print sanity check: stored vs recomputed pnl_net")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")

    paired = load_paired_trades(args.db)
    if paired.empty:
        raise SystemExit("No paired ENTER/EXIT shadow trades found.")

    if args.friction_check:
        paired["friction_recomputed"] = paired.apply(recompute_friction, axis=1)
        paired["pnl_net_recomputed"] = paired["pnl_gross"] - paired["friction_recomputed"]
        delta = paired["pnl_net"] - paired["pnl_net_recomputed"]
        print(f"Friction sanity check (stored vs recomputed pnl_net):")
        print(f"  N rows with both: {delta.notna().sum()}")
        print(f"  Max |delta|:      ${delta.abs().max():.4f}")
        print(f"  Mean |delta|:     ${delta.abs().mean():.4f}")
        print()

    cohort_a = paired[paired["entry_confidence"] >= 0.90]
    cohort_b = paired[(paired["entry_confidence"] >= 0.80)
                      & (paired["entry_confidence"] < 0.90)]

    a_stats = summarize_cohort(cohort_a, "A: >=0.90")
    b_stats = summarize_cohort(cohort_b, "B: [0.80, 0.90)")
    gates = evaluate_pass_fail(b_stats)

    print(format_report(paired, a_stats, b_stats, gates))


if __name__ == "__main__":
    main()
