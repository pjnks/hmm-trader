"""
beryl_scorecard.py
──────────────────
Prediction scorecard for BERYL scan journal.

Compares daily HMM regime predictions against actual next-day returns.
Answers: "Is the HMM telling us anything useful before we even trade?"

Three core metrics:
  1. Regime accuracy:  Do BULL tickers go up more than BEAR tickers?
  2. Confidence calibration:  Does higher confidence → higher accuracy?
  3. Confirmation lift:  Do BUY signals (7+/8) outperform raw BULL?

Usage
─────
  python beryl_scorecard.py                  # Score all available days
  python beryl_scorecard.py --last 5         # Score last 5 scan days
  python beryl_scorecard.py --verbose        # Show per-ticker detail
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "beryl_trades.db"


# ── Data Loading ──────────────────────────────────────────────────────────

def load_journal(db_path: Path = DB_PATH, last_n_days: int | None = None) -> pd.DataFrame:
    """Load scan journal from SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM scan_journal ORDER BY scan_date, ticker", conn)
    conn.close()
    if df.empty:
        return df
    if last_n_days:
        dates = sorted(df["scan_date"].unique())
        keep = dates[-last_n_days:]
        df = df[df["scan_date"].isin(keep)]
    return df


def fetch_next_day_prices(tickers: list[str], scan_dates: list[str]) -> dict[tuple[str, str], float]:
    """
    For each (scan_date, ticker), fetch the close price on the NEXT trading day.
    Returns dict of (scan_date, ticker) → next_day_close.
    """
    from walk_forward_ndx import fetch_equity_daily

    # Determine date range we need
    min_date = min(scan_dates)
    max_date = max(scan_dates)
    # Need a few extra days after max_date for "next day" lookup
    start = pd.Timestamp(min_date) - pd.Timedelta(days=5)

    next_day_prices = {}
    unique_tickers = sorted(set(tickers))

    print(f"Fetching price data for {len(unique_tickers)} tickers...")
    for i, ticker in enumerate(unique_tickers):
        try:
            df = fetch_equity_daily(ticker, years=1)
            if df.empty:
                continue
            # Build date→close map
            closes = df["close"].to_dict()  # index (Timestamp) → close
            trading_dates = sorted(closes.keys())

            for sd in scan_dates:
                sd_ts = pd.Timestamp(sd)
                # Find next trading day after scan_date
                future = [d for d in trading_dates if d > sd_ts]
                if future:
                    next_day_prices[(sd, ticker)] = closes[future[0]]
        except Exception:
            pass

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(unique_tickers)} tickers fetched...")

    print(f"  Price data complete: {len(next_day_prices)} data points")
    return next_day_prices


# ── Scorecard Computation ─────────────────────────────────────────────────

def compute_scorecard(journal: pd.DataFrame, next_prices: dict) -> dict:
    """
    Score regime predictions against actual next-day returns.
    Returns dict of metrics organized by category.
    """
    rows = []
    for _, row in journal.iterrows():
        key = (row["scan_date"], row["ticker"])
        if key not in next_prices or row["close_price"] <= 0:
            continue
        next_close = next_prices[key]
        ret = (next_close - row["close_price"]) / row["close_price"]
        rows.append({
            "scan_date": row["scan_date"],
            "ticker": row["ticker"],
            "regime": row["regime"],
            "confidence": row["confidence"],
            "confirmations": row["confirmations"],
            "min_confirmations": row["min_confirmations"],
            "signal": row["signal"],
            "close_price": row["close_price"],
            "next_close": next_close,
            "next_day_return": ret,
        })

    if not rows:
        return {"error": "No scorable data (need at least 2 scan days)"}

    df = pd.DataFrame(rows)

    # ── 1. Regime Accuracy ────────────────────────────────────────────
    regime_stats = {}
    for regime in ["BULL", "BEAR", "CHOP"]:
        subset = df[df["regime"] == regime]
        if len(subset) == 0:
            continue
        regime_stats[regime] = {
            "count": len(subset),
            "mean_return_bps": round(subset["next_day_return"].mean() * 10000, 1),
            "median_return_bps": round(subset["next_day_return"].median() * 10000, 1),
            "pct_positive": round((subset["next_day_return"] > 0).mean() * 100, 1),
            "std_bps": round(subset["next_day_return"].std() * 10000, 1),
        }

    # Regime separation: BULL mean - BEAR mean (should be positive)
    bull_mean = regime_stats.get("BULL", {}).get("mean_return_bps", 0)
    bear_mean = regime_stats.get("BEAR", {}).get("mean_return_bps", 0)
    regime_separation_bps = round(bull_mean - bear_mean, 1)

    # ── 2. Confidence Calibration ─────────────────────────────────────
    # Bucket confidence into bands and check if higher → better
    bulls = df[df["regime"] == "BULL"].copy()
    conf_bands = []
    for lo, hi, label in [(0.70, 0.85, "0.70-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.01, "0.95+")]:
        band = bulls[(bulls["confidence"] >= lo) & (bulls["confidence"] < hi)]
        if len(band) >= 3:
            conf_bands.append({
                "band": label,
                "count": len(band),
                "mean_return_bps": round(band["next_day_return"].mean() * 10000, 1),
                "pct_positive": round((band["next_day_return"] > 0).mean() * 100, 1),
            })

    # Is confidence monotonically improving?
    conf_monotonic = False
    if len(conf_bands) >= 2:
        returns = [b["mean_return_bps"] for b in conf_bands]
        conf_monotonic = all(returns[i] <= returns[i + 1] for i in range(len(returns) - 1))

    # ── 3. Confirmation Lift ──────────────────────────────────────────
    # Compare BUY signals vs raw BULL regime
    buys = df[df["signal"] == "BUY"]
    bulls_no_buy = bulls[bulls["signal"] != "BUY"]

    confirmation_lift = {}
    if len(buys) >= 3 and len(bulls_no_buy) >= 3:
        confirmation_lift = {
            "buy_signal_count": len(buys),
            "buy_mean_bps": round(buys["next_day_return"].mean() * 10000, 1),
            "buy_pct_positive": round((buys["next_day_return"] > 0).mean() * 100, 1),
            "bull_no_buy_count": len(bulls_no_buy),
            "bull_no_buy_mean_bps": round(bulls_no_buy["next_day_return"].mean() * 10000, 1),
            "bull_no_buy_pct_positive": round((bulls_no_buy["next_day_return"] > 0).mean() * 100, 1),
            "lift_bps": round(buys["next_day_return"].mean() * 10000 - bulls_no_buy["next_day_return"].mean() * 10000, 1),
        }

    # ── 4. Per-Day Summary ────────────────────────────────────────────
    daily = []
    for date, grp in df.groupby("scan_date"):
        bulls_d = grp[grp["regime"] == "BULL"]
        bears_d = grp[grp["regime"] == "BEAR"]
        daily.append({
            "date": date,
            "tickers_scored": len(grp),
            "bull_count": len(bulls_d),
            "bear_count": len(bears_d),
            "bull_mean_bps": round(bulls_d["next_day_return"].mean() * 10000, 1) if len(bulls_d) else 0,
            "bear_mean_bps": round(bears_d["next_day_return"].mean() * 10000, 1) if len(bears_d) else 0,
            "bull_beat_bear": (bulls_d["next_day_return"].mean() > bears_d["next_day_return"].mean())
            if len(bulls_d) > 0 and len(bears_d) > 0 else None,
        })

    return {
        "total_predictions": len(df),
        "scan_days": len(df["scan_date"].unique()),
        "regime_stats": regime_stats,
        "regime_separation_bps": regime_separation_bps,
        "confidence_calibration": conf_bands,
        "confidence_monotonic": conf_monotonic,
        "confirmation_lift": confirmation_lift,
        "daily_summary": daily,
    }


# ── Display ───────────────────────────────────────────────────────────────

def print_scorecard(sc: dict, verbose: bool = False) -> None:
    """Pretty-print the scorecard results."""
    if "error" in sc:
        print(f"\n  ⚠  {sc['error']}")
        print("  Journal needs at least 2 consecutive scan days to score predictions.")
        print("  Data will start accumulating after the next weekday scan cycle.\n")
        return

    print(f"\n{'='*64}")
    print(f"  BERYL PREDICTION SCORECARD")
    print(f"  {sc['total_predictions']} predictions across {sc['scan_days']} scan days")
    print(f"{'='*64}")

    # ── 1. Regime Accuracy
    print(f"\n  1. REGIME ACCURACY (next-day returns by regime)")
    print(f"  {'Regime':<8} {'N':>5} {'Mean':>8} {'Median':>8} {'%Pos':>7} {'StdDev':>8}")
    print(f"  {'─'*48}")
    for regime in ["BULL", "BEAR", "CHOP"]:
        s = sc["regime_stats"].get(regime)
        if s:
            print(f"  {regime:<8} {s['count']:>5} {s['mean_return_bps']:>+7.1f} "
                  f"{s['median_return_bps']:>+7.1f} {s['pct_positive']:>6.1f}% "
                  f"{s['std_bps']:>7.1f}")

    sep = sc["regime_separation_bps"]
    verdict = "PASS" if sep > 0 else "FAIL"
    print(f"\n  Regime separation (BULL - BEAR): {sep:+.1f} bps  [{verdict}]")
    if sep > 0:
        print(f"  → HMM regime labels have predictive content")
    else:
        print(f"  → BEAR tickers outperformed BULL — regime labels may be inverted or noisy")

    # ── 2. Confidence Calibration
    if sc["confidence_calibration"]:
        print(f"\n  2. CONFIDENCE CALIBRATION (BULL tickers only)")
        print(f"  {'Band':<12} {'N':>5} {'Mean':>8} {'%Pos':>7}")
        print(f"  {'─'*36}")
        for b in sc["confidence_calibration"]:
            print(f"  {b['band']:<12} {b['count']:>5} {b['mean_return_bps']:>+7.1f} {b['pct_positive']:>6.1f}%")
        mono = "PASS" if sc["confidence_monotonic"] else "FAIL"
        print(f"\n  Monotonic improvement: [{mono}]")
        if sc["confidence_monotonic"]:
            print(f"  → Higher confidence → better returns (confidence is informative)")
        else:
            print(f"  → Higher confidence ≠ better returns (confidence may be noise)")
    else:
        print(f"\n  2. CONFIDENCE CALIBRATION: insufficient data (need 3+ per band)")

    # ── 3. Confirmation Lift
    cl = sc["confirmation_lift"]
    if cl:
        print(f"\n  3. CONFIRMATION LIFT (BUY signals vs raw BULL)")
        print(f"  BUY signals:  N={cl['buy_signal_count']}, mean={cl['buy_mean_bps']:+.1f} bps, "
              f"{cl['buy_pct_positive']:.1f}% positive")
        print(f"  BULL (no BUY): N={cl['bull_no_buy_count']}, mean={cl['bull_no_buy_mean_bps']:+.1f} bps, "
              f"{cl['bull_no_buy_pct_positive']:.1f}% positive")
        lift = cl["lift_bps"]
        verdict = "PASS" if lift > 0 else "FAIL"
        print(f"\n  Confirmation lift: {lift:+.1f} bps  [{verdict}]")
        if lift > 0:
            print(f"  → Indicator stack adds value on top of HMM regime")
        else:
            print(f"  → BUY filter doesn't improve over raw BULL (indicators may be too strict or not useful)")
    else:
        print(f"\n  3. CONFIRMATION LIFT: insufficient data (need 3+ BUY signals)")

    # ── 4. Daily Summary
    if verbose and sc["daily_summary"]:
        print(f"\n  4. DAILY BREAKDOWN")
        print(f"  {'Date':<12} {'N':>4} {'BULL':>5} {'BEAR':>5} {'BULL ret':>9} {'BEAR ret':>9} {'BULL>BEAR':>10}")
        print(f"  {'─'*58}")
        for d in sc["daily_summary"]:
            bb = "  ✓" if d["bull_beat_bear"] else (" ✗" if d["bull_beat_bear"] is False else "  -")
            print(f"  {d['date']:<12} {d['tickers_scored']:>4} {d['bull_count']:>5} "
                  f"{d['bear_count']:>5} {d['bull_mean_bps']:>+8.1f} "
                  f"{d['bear_mean_bps']:>+8.1f} {bb:>10}")

        days_bull_wins = sum(1 for d in sc["daily_summary"] if d["bull_beat_bear"] is True)
        days_total = sum(1 for d in sc["daily_summary"] if d["bull_beat_bear"] is not None)
        if days_total > 0:
            print(f"\n  BULL beat BEAR: {days_bull_wins}/{days_total} days "
                  f"({days_bull_wins/days_total*100:.0f}%)")

    # ── Overall Verdict
    print(f"\n{'─'*64}")
    passes = 0
    total = 0
    if sep != 0:
        total += 1
        passes += 1 if sep > 0 else 0
    if sc["confidence_calibration"]:
        total += 1
        passes += 1 if sc["confidence_monotonic"] else 0
    if cl:
        total += 1
        passes += 1 if cl["lift_bps"] > 0 else 0

    if total == 0:
        print(f"  VERDICT: Insufficient data — keep accumulating scans")
    else:
        print(f"  VERDICT: {passes}/{total} checks passing")
        if passes == total:
            print(f"  → Signal quality looks good — keep accumulating trades")
        elif passes >= total / 2:
            print(f"  → Mixed signals — worth monitoring, may need tuning")
        else:
            print(f"  → Weak signal — investigate before scaling up")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BERYL prediction scorecard")
    parser.add_argument("--last", type=int, default=None,
                        help="Score only the last N scan days")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-day breakdown")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to beryl_trades.db (default: local)")
    args = parser.parse_args()

    db = Path(args.db) if args.db else DB_PATH
    if not db.exists():
        print(f"Database not found: {db}")
        sys.exit(1)

    # Check if scan_journal table exists
    conn = sqlite3.connect(db)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    if "scan_journal" not in tables:
        print("No scan_journal table found. Deploy updated live_trading_beryl.py first.")
        sys.exit(1)

    journal = load_journal(db, last_n_days=args.last)
    if journal.empty:
        print("scan_journal is empty. Scorecard will populate after the first daily scan.")
        sys.exit(0)

    scan_dates = sorted(journal["scan_date"].unique())
    print(f"Journal has {len(journal)} entries across {len(scan_dates)} scan days: "
          f"{scan_dates[0]} → {scan_dates[-1]}")

    # Fetch next-day prices for scoring
    next_prices = fetch_next_day_prices(
        journal["ticker"].tolist(),
        scan_dates,
    )

    scorecard = compute_scorecard(journal, next_prices)
    print_scorecard(scorecard, verbose=args.verbose)


if __name__ == "__main__":
    main()
