#!/usr/bin/env python3
"""
evaluate_model.py — Pure Model (Tier 1) Signal Quality Evaluation

Separates MODEL evaluation from STRATEGY evaluation by measuring the
raw predictive power of HMM regime signals against fixed-horizon forward
returns, with ZERO execution logic (no stops, no fees, no position sizing).

This answers the question: "Does the HMM actually predict future returns?"
independently of whether the trading plumbing captures that alpha.

Metrics computed:
  1. N-bar Forward Return by Regime (T+1, T+3, T+5 daily)
  2. Information Coefficient (IC): Spearman rank(confidence, T+3 return)
  3. Hit Rate: % of BULL signals followed by positive N-day return
  4. Calibration: when model says 90% confident, does regime persist 90%?

Data sources:
  - BERYL scan_journal (97 tickers × daily scans)
  - CITRINE scan via portfolio_snapshots (regime + confidence at scan time)
  - Polygon daily prices for forward return calculation

Usage:
  python evaluate_model.py                    # Full analysis from scan_journal
  python evaluate_model.py --horizon 3        # T+3 forward returns only
  python evaluate_model.py --min-confidence 0.80  # Filter low-conf signals
  python evaluate_model.py --ticker NVDA      # Single-ticker analysis
"""

import argparse
import sqlite3
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH_BERYL = Path(__file__).parent / "beryl_trades.db"
DB_PATH_CITRINE = Path(__file__).parent / "citrine_trades.db"

# Forward return horizons (trading days)
DEFAULT_HORIZONS = [1, 3, 5]


def fetch_forward_prices(ticker: str, start_date: str, days_forward: int = 10) -> pd.DataFrame:
    """Fetch daily closing prices from Polygon for forward return calc.

    Returns DataFrame indexed by date with 'close' column.
    """
    from walk_forward_ndx import fetch_equity_daily
    df = fetch_equity_daily(ticker, years=1)
    if df is None or df.empty:
        return pd.DataFrame()
    # Ensure date index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df[["close"]].sort_index()


def load_scan_journal(db_path: Path, min_date: str = None) -> pd.DataFrame:
    """Load BERYL scan_journal entries."""
    con = sqlite3.connect(str(db_path))
    query = """
        SELECT scan_date, ticker, regime, confidence, confirmations, signal, close_price
        FROM scan_journal
    """
    if min_date:
        query += f" WHERE scan_date >= '{min_date}'"
    query += " ORDER BY scan_date, ticker"
    df = pd.read_sql_query(query, con)
    con.close()
    return df


def compute_forward_returns(scans: pd.DataFrame, horizons: list[int],
                            rate_limit_sec: float = 0.5) -> pd.DataFrame:
    """Join scan signals with N-day forward returns from Polygon.

    For each scan entry (ticker, date, regime, confidence), fetches
    the closing price on date+N and computes the return.
    """
    # Get unique tickers
    tickers = scans["ticker"].unique()
    print(f"  Fetching prices for {len(tickers)} tickers...")

    # Cache prices per ticker (one API call each)
    price_cache: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            print(f"    ... {i}/{len(tickers)} tickers fetched")
        try:
            df = fetch_forward_prices(ticker, scans["scan_date"].min())
            if not df.empty:
                price_cache[ticker] = df
        except Exception as e:
            print(f"    [WARN] Failed to fetch {ticker}: {e}")
        time.sleep(rate_limit_sec)  # Polygon rate limit

    print(f"  Prices loaded for {len(price_cache)}/{len(tickers)} tickers")

    # Compute forward returns
    results = []
    for _, row in scans.iterrows():
        ticker = row["ticker"]
        scan_date = pd.Timestamp(row["scan_date"])

        if ticker not in price_cache:
            continue

        prices = price_cache[ticker]

        # Find the closing price on scan date (or nearest trading day after)
        mask = prices.index >= scan_date
        if mask.sum() == 0:
            continue
        entry_idx = prices.index[mask][0]
        entry_price = prices.loc[entry_idx, "close"]

        # Compute forward returns at each horizon
        entry_pos = prices.index.get_loc(entry_idx)
        forward_returns = {}
        for h in horizons:
            fwd_pos = entry_pos + h
            if fwd_pos < len(prices):
                fwd_price = prices.iloc[fwd_pos]["close"]
                forward_returns[f"fwd_ret_T{h}"] = (fwd_price - entry_price) / entry_price
            else:
                forward_returns[f"fwd_ret_T{h}"] = np.nan

        results.append({
            "scan_date": row["scan_date"],
            "ticker": ticker,
            "regime": row["regime"],
            "confidence": row["confidence"],
            "confirmations": row.get("confirmations", np.nan),
            "signal": row.get("signal", ""),
            "entry_price": entry_price,
            **forward_returns,
        })

    return pd.DataFrame(results)


def compute_information_coefficient(df: pd.DataFrame, horizon: int) -> dict:
    """Compute Spearman IC between confidence and forward return."""
    col = f"fwd_ret_T{horizon}"
    valid = df.dropna(subset=[col, "confidence"])
    if len(valid) < 10:
        return {"ic": np.nan, "p_value": np.nan, "n": len(valid)}

    ic, p_value = stats.spearmanr(valid["confidence"], valid[col])
    return {"ic": ic, "p_value": p_value, "n": len(valid)}


def compute_hit_rate(df: pd.DataFrame, horizon: int, regime: str = "BULL") -> dict:
    """Compute hit rate: % of regime signals followed by positive return."""
    col = f"fwd_ret_T{horizon}"
    subset = df[(df["regime"] == regime) & df[col].notna()]
    if len(subset) == 0:
        return {"hit_rate": np.nan, "n": 0, "avg_return": np.nan}

    hits = (subset[col] > 0).sum()
    return {
        "hit_rate": hits / len(subset),
        "n": len(subset),
        "avg_return": subset[col].mean(),
        "median_return": subset[col].median(),
        "std_return": subset[col].std(),
    }


def compute_calibration(df: pd.DataFrame, horizon: int,
                        confidence_bins: list[float] = None) -> pd.DataFrame:
    """Check calibration: does 90% confidence → 90% regime persistence?

    Groups signals by confidence bucket and measures actual hit rate.
    """
    if confidence_bins is None:
        confidence_bins = [0.70, 0.80, 0.90, 0.95, 1.01]

    col = f"fwd_ret_T{horizon}"
    bull = df[(df["regime"] == "BULL") & df[col].notna()].copy()
    if bull.empty:
        return pd.DataFrame()

    bull["conf_bucket"] = pd.cut(bull["confidence"], bins=[0] + confidence_bins,
                                  labels=[f"<{confidence_bins[0]:.0%}"] +
                                         [f"{confidence_bins[i]:.0%}-{confidence_bins[i+1]:.0%}"
                                          for i in range(len(confidence_bins) - 1)])

    grouped = bull.groupby("conf_bucket", observed=True).agg(
        n=("confidence", "count"),
        avg_confidence=("confidence", "mean"),
        hit_rate=(col, lambda x: (x > 0).mean()),
        avg_return=(col, "mean"),
    ).reset_index()

    return grouped


def main():
    parser = argparse.ArgumentParser(description="Pure Model Signal Quality Evaluation")
    parser.add_argument("--horizon", type=int, nargs="+", default=DEFAULT_HORIZONS,
                        help="Forward return horizons (trading days)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence to include")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker to analyze")
    parser.add_argument("--min-date", type=str, default=None,
                        help="Earliest scan date (YYYY-MM-DD)")
    parser.add_argument("--db", type=str, default=str(DB_PATH_BERYL),
                        help="Path to scan_journal database")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip Polygon API calls (use cached data)")
    args = parser.parse_args()

    print("=" * 72)
    print("  MODEL EVALUATION — Pure Signal Quality (Tier 1)")
    print("  No stops, no fees, no position sizing. Just: does the HMM predict?")
    print("=" * 72)
    print()

    # Load scan journal
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    scans = load_scan_journal(db_path, min_date=args.min_date)
    print(f"Loaded {len(scans)} scan entries from {db_path.name}")
    print(f"  Date range: {scans['scan_date'].min()} → {scans['scan_date'].max()}")
    print(f"  Tickers: {scans['ticker'].nunique()}")
    print(f"  Regime distribution: "
          f"BULL={len(scans[scans['regime']=='BULL'])}, "
          f"BEAR={len(scans[scans['regime']=='BEAR'])}, "
          f"CHOP={len(scans[scans['regime']=='CHOP'])}")

    # Filter
    if args.min_confidence > 0:
        scans = scans[scans["confidence"] >= args.min_confidence]
        print(f"  After confidence filter (>= {args.min_confidence}): {len(scans)} entries")
    if args.ticker:
        scans = scans[scans["ticker"] == args.ticker]
        print(f"  Filtered to ticker={args.ticker}: {len(scans)} entries")

    if scans.empty:
        print("No scan entries match filters. Exiting.")
        sys.exit(0)

    # Compute forward returns
    print()
    print("─" * 72)
    print("  STEP 1: Computing forward returns (Polygon API)")
    print("─" * 72)

    if args.skip_fetch:
        print("  [--skip-fetch] Skipping API calls. Load from cache not implemented yet.")
        sys.exit(0)

    df = compute_forward_returns(scans, args.horizon)
    print(f"  Computed returns for {len(df)} signal/date/ticker rows")

    if df.empty:
        print("  No data — exiting.")
        sys.exit(0)

    # ── RESULTS ──────────────────────────────────────────────────────────────
    print()
    print("═" * 72)
    print("  RESULTS")
    print("═" * 72)

    # 1. Forward returns by regime
    print()
    print("  1. AVERAGE FORWARD RETURNS BY REGIME")
    print("  " + "─" * 60)
    for h in args.horizon:
        col = f"fwd_ret_T{h}"
        print(f"\n  T+{h} Forward Return:")
        for regime in ["BULL", "BEAR", "CHOP"]:
            subset = df[(df["regime"] == regime) & df[col].notna()]
            if len(subset) > 0:
                avg = subset[col].mean() * 100
                med = subset[col].median() * 100
                std = subset[col].std() * 100
                n = len(subset)
                hit = (subset[col] > 0).mean() * 100
                print(f"    {regime:5s}: avg={avg:+.3f}%  med={med:+.3f}%  "
                      f"std={std:.3f}%  hit={hit:.1f}%  N={n}")

    # 2. Information Coefficient
    print()
    print("  2. INFORMATION COEFFICIENT (Spearman: confidence → forward return)")
    print("  " + "─" * 60)
    for h in args.horizon:
        # IC for BULL signals only (our entry condition)
        bull_df = df[df["regime"] == "BULL"]
        ic_result = compute_information_coefficient(bull_df, h)
        sig = "***" if ic_result["p_value"] < 0.01 else "**" if ic_result["p_value"] < 0.05 else "*" if ic_result["p_value"] < 0.10 else ""
        print(f"    T+{h} BULL IC: {ic_result['ic']:+.4f}  "
              f"(p={ic_result['p_value']:.4f}{sig}, N={ic_result['n']})")

        # IC for all regimes
        ic_all = compute_information_coefficient(df, h)
        sig = "***" if ic_all["p_value"] < 0.01 else "**" if ic_all["p_value"] < 0.05 else "*" if ic_all["p_value"] < 0.10 else ""
        print(f"    T+{h}  ALL IC: {ic_all['ic']:+.4f}  "
              f"(p={ic_all['p_value']:.4f}{sig}, N={ic_all['n']})")

    # 3. Hit rates
    print()
    print("  3. HIT RATE BY REGIME (% positive forward return)")
    print("  " + "─" * 60)
    for h in args.horizon:
        for regime in ["BULL", "BEAR", "CHOP"]:
            hr = compute_hit_rate(df, h, regime)
            if hr["n"] > 0:
                print(f"    T+{h} {regime:5s}: hit={hr['hit_rate']:.1%}  "
                      f"avg={hr['avg_return']*100:+.3f}%  N={hr['n']}")

    # 4. Calibration
    print()
    print("  4. CALIBRATION (confidence bucket → actual hit rate)")
    print("  " + "─" * 60)
    for h in args.horizon:
        cal = compute_calibration(df, h)
        if not cal.empty:
            print(f"\n    T+{h}:")
            for _, row in cal.iterrows():
                print(f"      {row['conf_bucket']:20s}  "
                      f"hit={row['hit_rate']:.1%}  "
                      f"avg_ret={row['avg_return']*100:+.3f}%  "
                      f"N={row['n']:.0f}")

    # 5. Summary verdict
    print()
    print("═" * 72)
    print("  VERDICT")
    print("═" * 72)
    bull_t3 = compute_hit_rate(df, 3, "BULL")
    bear_t3 = compute_hit_rate(df, 3, "BEAR")
    ic_t3 = compute_information_coefficient(df[df["regime"] == "BULL"], 3)

    model_works = (bull_t3.get("hit_rate", 0) > 0.50 and
                   bull_t3.get("avg_return", 0) > 0 and
                   bear_t3.get("hit_rate", 0) < 0.55)

    if model_works:
        print("  ✅ MODEL HAS PREDICTIVE EDGE")
        print(f"     BULL T+3 hit rate: {bull_t3['hit_rate']:.1%} (>{50}% required)")
        print(f"     BULL T+3 avg return: {bull_t3['avg_return']*100:+.3f}%")
        print(f"     BULL IC: {ic_t3['ic']:+.4f}")
        print()
        print("  → If live trading loses money despite positive model edge,")
        print("    the EXECUTION LAYER (stops, sizing, timing) is the problem.")
    else:
        print("  ⚠️  MODEL EDGE UNCLEAR OR ABSENT")
        print(f"     BULL T+3 hit rate: {bull_t3.get('hit_rate', 0):.1%}")
        print(f"     BULL T+3 avg return: {bull_t3.get('avg_return', 0)*100:+.3f}%")
        print()
        print("  → Strategy fixes won't help if the model lacks predictive power.")
        print("    Re-evaluate HMM parameters, feature set, or training regime.")

    print()


if __name__ == "__main__":
    main()
