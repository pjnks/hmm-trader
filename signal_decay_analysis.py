#!/usr/bin/env python3
"""
Fix 5: Signal Decay Analysis for AGATE

Measures forward returns at t+1, t+2, t+4 bars after each BUY signal
to determine if edge exists immediately or requires regime settling.

Usage:
    python signal_decay_analysis.py
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_buy_signals() -> list[dict]:
    """Extract BUY signals from AGATE journal DB or scan logs."""
    # Try journal DB first
    journal_path = Path("agate_journal.db")
    if journal_path.exists():
        conn = sqlite3.connect(journal_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM journal WHERE signal='BUY' ORDER BY timestamp ASC"
        ).fetchall()
        conn.close()
        if rows:
            return [dict(r) for r in rows]

    # Fallback: parse status JSON files for historical signals
    status_path = Path("agate_status.json")
    if status_path.exists():
        with open(status_path) as f:
            data = json.load(f)
        log.info(f"Loaded status JSON (latest only — need journal DB for history)")
        return []

    log.warning("No signal data found. Need agate_journal.db or scan logs.")
    return []


def fetch_forward_prices(ticker: str, signal_time: str, bars: list[int] = [1, 2, 4]):
    """Fetch prices at t+N bars (4h each) after signal time."""
    from src.data_fetcher import fetch_btc_hourly
    import config

    polygon_ticker = ticker if ticker.startswith("X:") else f"X:{ticker}"
    try:
        df = fetch_btc_hourly(days=30, ticker=polygon_ticker)
        if df is None or len(df) == 0:
            return {}

        # Find the bar closest to signal time
        signal_dt = pd.Timestamp(signal_time, tz="UTC")
        idx = df.index.get_indexer([signal_dt], method="nearest")[0]

        prices = {"t0": float(df["Close"].iloc[idx])}
        for n in bars:
            target_idx = idx + n
            if target_idx < len(df):
                prices[f"t+{n}"] = float(df["Close"].iloc[target_idx])
        return prices
    except Exception as e:
        log.warning(f"Failed to fetch forward prices for {ticker}: {e}")
        return {}


def analyze_decay(signals: list[dict]):
    """Compute forward returns for each signal."""
    results = []
    for sig in signals:
        ticker = sig.get("ticker", "")
        timestamp = sig.get("timestamp", "")
        price_at_signal = sig.get("current_price", 0)

        if not ticker or price_at_signal <= 0:
            continue

        prices = fetch_forward_prices(ticker, timestamp)
        if not prices or "t0" not in prices:
            continue

        t0 = prices["t0"]
        row = {
            "ticker": ticker,
            "timestamp": timestamp,
            "price_t0": t0,
            "confidence": sig.get("confidence", 0),
            "confirmations": sig.get("confirmations", 0),
        }
        for key in ["t+1", "t+2", "t+4"]:
            if key in prices:
                ret = (prices[key] - t0) / t0 * 100
                row[f"return_{key}"] = ret

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("  AGATE Signal Decay Analysis")
    print("=" * 60)

    signals = load_buy_signals()
    if not signals:
        print("\n  No BUY signals found in journal DB.")
        print("  Run AGATE for a few scan cycles to accumulate data.")
        print("  The journal DB (agate_journal.db) stores all scan decisions.")
        return

    print(f"\n  Found {len(signals)} BUY signals")

    df = analyze_decay(signals)
    if df.empty:
        print("  No forward price data available.")
        return

    # Summary by ticker
    print("\n  -- Forward Returns by Ticker (%) --")
    print(f"  {'Ticker':>10s}  {'N':>4s}  {'t+1 (4h)':>10s}  {'t+2 (8h)':>10s}  {'t+4 (16h)':>10s}")
    for ticker, group in df.groupby("ticker"):
        n = len(group)
        t1 = group["return_t+1"].mean() if "return_t+1" in group else float("nan")
        t2 = group["return_t+2"].mean() if "return_t+2" in group else float("nan")
        t4 = group["return_t+4"].mean() if "return_t+4" in group else float("nan")
        print(f"  {ticker:>10s}  {n:>4d}  {t1:>+9.2f}%  {t2:>+9.2f}%  {t4:>+9.2f}%")

    # Overall
    print(f"\n  -- Overall --")
    for col in ["return_t+1", "return_t+2", "return_t+4"]:
        if col in df:
            mean = df[col].mean()
            std = df[col].std()
            pct_pos = (df[col] > 0).mean() * 100
            print(f"  {col}: mean={mean:+.3f}%, std={std:.3f}%, "
                  f"positive={pct_pos:.0f}%")

    # Verdict
    print(f"\n  -- Verdict --")
    if "return_t+1" in df and df["return_t+1"].mean() > 0.1:
        print("  Edge appears IMMEDIATE — trade on first BUY signal")
    elif "return_t+4" in df and df["return_t+4"].mean() > 0.1:
        print("  Edge appears DELAYED — wait for regime to settle (~16h)")
    else:
        print("  No clear edge detected — insufficient data or no timing alpha")


if __name__ == "__main__":
    main()
