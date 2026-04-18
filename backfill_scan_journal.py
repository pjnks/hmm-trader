"""
Sprint 15 Backfill — scan_journal_backfill table

Strategy: Option C — Quarterly expanding-window ensemble HMM refits.
Architecture: Ensemble (matches production exactly).

For each of the 98 NDX100 tickers:
  - Fetch 3 years of daily prices (enough for 365d lookback + decode window).
  - Per-ticker HMM config loaded from beryl_per_ticker_configs.json (production-faithful).
  - Refit schedule (expanding window):
      Fit 1: train on data up to Q_start_1, decode Q1
      Fit 2: train on data up to Q_start_2, decode Q2
      Fit 3: train on data up to Q_start_3, decode Q3
  - Ensemble with fallback retry (single HMM n_states=4 if ensemble fails).
  - Extract (scan_date, regime, confidence, confirmations, signal, close_price)
    for each day in each decode window.

Output: `scan_journal_backfill` table in /home/ubuntu/HMM-Trader/beryl_trades.db
Target: ~12,600 rows (98 tickers × 126 trading days)

Usage:
    python backfill_scan_journal.py [--tickers NVDA,TSLA] [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data_fetcher import build_hmm_features
from src.ensemble import EnsembleHMM
from src.hmm_model import HMMRegimeModel
from src.indicators import attach_all
from src.strategy import build_signal_series
from walk_forward_ndx import fetch_equity_daily


# ── Configuration ────────────────────────────────────────────────────────
DB_PATH = Path("/home/ubuntu/HMM-Trader/beryl_trades.db")
CONFIG_PATH = Path("/home/ubuntu/HMM-Trader/beryl_per_ticker_configs.json")
UNIVERSE_PATH = Path("/home/ubuntu/HMM-Trader/citrine_per_ticker_configs.json")

HISTORY_YEARS = 3  # Fetch window; enough for 1yr lookback + 6mo decode
TRAINING_LOOKBACK_DAYS = 365  # Match production (Sprint 5)
MIN_TRAINING_OBS = 200  # HMM needs at least this many daily bars

DEFAULT_CONFIG = {
    "n_states": 4,
    "feature_set": "base",
    "confirmations": 5,
    "cov_type": "diag",
}


# ── Quarterly window planning ────────────────────────────────────────────
def build_quarterly_windows(end_date: datetime, n_quarters: int = 3) -> list[tuple[datetime, datetime]]:
    """
    Return list of (decode_start, decode_end) tuples for n_quarters ending
    at end_date. Each decode window is ~63 trading days (one calendar quarter).
    Training = all data strictly before decode_start.
    """
    windows = []
    # Align to quarter boundary just before end_date
    cursor = end_date
    for _ in range(n_quarters):
        # Find the start of the quarter containing cursor
        q_month = ((cursor.month - 1) // 3) * 3 + 1
        q_start = datetime(cursor.year, q_month, 1)
        q_end = cursor
        windows.append((q_start, q_end))
        # Step back to previous quarter's end
        cursor = q_start - timedelta(days=1)
    windows.reverse()  # chronological order
    return windows


# ── Per-ticker backfill ──────────────────────────────────────────────────
def load_per_ticker_config(ticker: str, configs: dict) -> dict:
    cfg = configs.get(ticker, {})
    return {**DEFAULT_CONFIG, **cfg}


def fit_ensemble_with_fallback(train_df: pd.DataFrame, feature_cols: list[str],
                               n_states: int, cov_type: str) -> tuple[object, bool]:
    """
    Try ensemble fit; on failure, fall back to single HMM n_states=4 diag.
    Returns (model, used_fallback_bool).
    """
    try:
        n_list = [max(2, n_states - 1), n_states, n_states + 1]
        ens = EnsembleHMM(n_states_list=n_list, cov_type=cov_type,
                          feature_cols=feature_cols)
        ens.fit(train_df)
        if hasattr(ens, "converged") and not ens.converged:
            raise RuntimeError("ensemble did not converge")
        return ens, False
    except Exception as e:
        # Fallback: single HMM, 4 states, diag
        fallback = HMMRegimeModel(n_states=4, cov_type="diag",
                                  feature_cols=["log_return", "price_range", "volume_change"])
        fallback.fit(train_df)
        return fallback, True


def backfill_ticker(ticker: str, configs: dict,
                    windows: list[tuple[datetime, datetime]],
                    prices_cache: dict) -> list[dict]:
    """Run expanding-window fits over all quarters for one ticker. Return rows."""
    cfg = load_per_ticker_config(ticker, configs)
    feature_set = cfg.get("feature_set", "base")
    feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])
    n_states = int(cfg.get("n_states", 4))
    cov_type = cfg.get("cov_type", "diag")
    confirmations_required = int(cfg.get("confirmations", 5))

    # Fetch full history once
    df_full = prices_cache.get(ticker)
    if df_full is None:
        return []
    # Keep OHLCV capitalized to match build_hmm_features / attach_all expectations
    df_full = df_full.copy()

    rows = []
    for q_start, q_end in windows:
        # Training = strict <q_start (prevents look-ahead)
        train = df_full[df_full.index < q_start].copy()
        if len(train) < MIN_TRAINING_OBS:
            continue
        # Cap training window at TRAINING_LOOKBACK_DAYS from q_start (match production)
        cutoff = q_start - timedelta(days=TRAINING_LOOKBACK_DAYS)
        train = train[train.index >= cutoff]
        if len(train) < MIN_TRAINING_OBS:
            continue

        # Build features on training slice
        try:
            train_feat = build_hmm_features(train)
            train_feat = attach_all(train_feat)
            # Drop NaN rows from rolling-window features (realized_vol_ratio,
            # return_autocorr, realized_kurtosis use ~20-day windows)
            train_feat = train_feat.dropna(subset=feature_cols)
            if len(train_feat) < MIN_TRAINING_OBS:
                continue
        except Exception as e:
            print(f"    [{ticker} {q_start:%Y-%m-%d}] feature build failed: {type(e).__name__}: {e}")
            continue

        # Fit ensemble (fallback to single)
        try:
            model, used_fallback = fit_ensemble_with_fallback(
                train_feat, feature_cols, n_states, cov_type
            )
        except Exception as e:
            print(f"    [{ticker} {q_start:%Y-%m-%d}] fit failed: {type(e).__name__}: {e}")
            continue

        # Decode window: train + decode slice so Viterbi has history
        # Simpler: all data up to q_end
        full_slice = df_full[df_full.index <= q_end].copy()
        try:
            full_feat = build_hmm_features(full_slice)
            full_feat = attach_all(full_feat)
            full_feat = full_feat.dropna(subset=feature_cols)
            predictions = model.predict(full_feat)
        except Exception as e:
            print(f"    [{ticker} {q_start:%Y-%m-%d}] predict failed: {type(e).__name__}: {e}")
            continue

        # Also compute signal series for confirmations
        try:
            signals = build_signal_series(predictions, use_regime_mapper=False)
        except Exception as e:
            print(f"    [{ticker} {q_start:%Y-%m-%d}] signal build failed: {type(e).__name__}: {e}")
            signals = predictions  # fallback; confirmations will be NaN

        # Extract decode-window rows only
        decode_mask = (signals.index >= q_start) & (signals.index <= q_end)
        decode_rows = signals[decode_mask]

        for idx, row in decode_rows.iterrows():
            regime = row.get("regime_cat", "UNKNOWN")
            if isinstance(regime, (int, float)):
                regime = {0: "BEAR", 1: "CHOP", 2: "BULL"}.get(int(regime), "UNKNOWN")
            conf = float(row.get("confidence", np.nan))
            conf_count = row.get("confirmation_count", np.nan)
            if pd.isna(conf_count):
                conf_count = 0
            close = float(row.get("close", row.get("Close", np.nan)))
            # Signal decision (production logic): BUY if BULL + confirmations hit + raw_long_signal
            raw_long = bool(row.get("raw_long_signal", False))
            if regime == "BULL" and int(conf_count) >= confirmations_required and raw_long:
                sig = "BUY"
            elif regime == "BEAR":
                sig = "SELL"
            else:
                sig = "HOLD"

            rows.append({
                "scan_date": idx.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "regime": regime,
                "confidence": conf,
                "confirmations": int(conf_count),
                "signal": sig,
                "close_price": close,
                "used_fallback": int(used_fallback),
                "quarter_start": q_start.strftime("%Y-%m-%d"),
            })

        gc.collect()

    return rows


# ── Storage ──────────────────────────────────────────────────────────────
def init_backfill_table(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS scan_journal_backfill (
            scan_date TEXT,
            ticker TEXT,
            regime TEXT,
            confidence REAL,
            confirmations INTEGER,
            signal TEXT,
            close_price REAL,
            used_fallback INTEGER,
            quarter_start TEXT,
            PRIMARY KEY (scan_date, ticker, quarter_start)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_bf_date ON scan_journal_backfill(scan_date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_bf_ticker ON scan_journal_backfill(ticker)")
    con.commit()


def insert_rows(con: sqlite3.Connection, rows: list[dict]):
    if not rows:
        return
    con.executemany(
        """INSERT OR REPLACE INTO scan_journal_backfill
           (scan_date, ticker, regime, confidence, confirmations, signal,
            close_price, used_fallback, quarter_start)
           VALUES (:scan_date, :ticker, :regime, :confidence, :confirmations,
                   :signal, :close_price, :used_fallback, :quarter_start)""",
        rows,
    )
    con.commit()


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated ticker subset (default: full NDX100 from citrine config)")
    ap.add_argument("--n-quarters", type=int, default=3,
                    help="Number of past quarters to backfill (default 3 = ~6mo)")
    ap.add_argument("--dry-run", action="store_true", help="Plan only, don't fit")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip tickers already in scan_journal_backfill")
    args = ap.parse_args()

    # Load universe
    if args.tickers:
        universe = args.tickers.split(",")
    else:
        if not UNIVERSE_PATH.exists():
            print(f"❌ Universe config not found: {UNIVERSE_PATH}")
            return
        universe = sorted(json.loads(UNIVERSE_PATH.read_text()).keys())
    print(f"Universe: {len(universe)} tickers")

    # Load per-ticker HMM configs
    if CONFIG_PATH.exists():
        configs = json.loads(CONFIG_PATH.read_text())
        print(f"Loaded HMM configs for {len(configs)} tickers "
              f"({sum(1 for t in universe if t in configs)}/{len(universe)} coverage)")
    else:
        configs = {}
        print("⚠ No per-ticker configs — all tickers use DEFAULT_CONFIG")

    # Plan quarterly windows
    end_date = datetime.now()
    windows = build_quarterly_windows(end_date, args.n_quarters)
    print(f"\nQuarterly decode windows (Option C — expanding training):")
    for q_start, q_end in windows:
        print(f"  Train < {q_start:%Y-%m-%d}  →  Decode {q_start:%Y-%m-%d} to {q_end:%Y-%m-%d}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without fitting.")
        return

    # DB setup
    con = sqlite3.connect(str(DB_PATH))
    init_backfill_table(con)

    if args.skip_existing:
        existing = pd.read_sql("SELECT DISTINCT ticker FROM scan_journal_backfill",
                               con)["ticker"].tolist()
        universe = [t for t in universe if t not in existing]
        print(f"Skipping {len(existing)} tickers already backfilled. "
              f"Remaining: {len(universe)}")

    # Pre-fetch all prices (with Polygon rate limit)
    print(f"\n── Pre-fetching {HISTORY_YEARS}yr prices for {len(universe)} tickers ──")
    prices_cache = {}
    for i, ticker in enumerate(universe, 1):
        try:
            df = fetch_equity_daily(ticker, years=HISTORY_YEARS)
            if df is None or df.empty:
                print(f"  {ticker}: no data")
                continue
            # fetch_equity_daily returns capitalized OHLCV columns — do NOT lowercase
            # (build_hmm_features + attach_all require 'Close', 'High', etc.)
            if "date" in df.columns:
                df = df.set_index(pd.to_datetime(df["date"]))
            elif "Date" in df.columns:
                df = df.set_index(pd.to_datetime(df["Date"]))
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            prices_cache[ticker] = df
        except Exception as e:
            print(f"  {ticker}: fetch error: {e}")
        if i % 20 == 0:
            print(f"  {i}/{len(universe)} fetched ({len(prices_cache)} cached)")
        time.sleep(0.05)  # Light rate limit

    print(f"\n── Starting ensemble fits ({len(prices_cache)} tickers × "
          f"{len(windows)} quarters = {len(prices_cache)*len(windows)} fits) ──")
    start = time.time()
    total_rows = 0
    fallbacks = 0
    failures = 0

    for i, ticker in enumerate(prices_cache.keys(), 1):
        t0 = time.time()
        try:
            rows = backfill_ticker(ticker, configs, windows, prices_cache)
            insert_rows(con, rows)
            total_rows += len(rows)
            fallbacks += sum(r["used_fallback"] for r in rows)
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(prices_cache)}] {ticker:<6} "
                  f"{len(rows):>4d} rows  {elapsed:5.1f}s "
                  f"(total {total_rows} rows, ETA "
                  f"{(time.time()-start)/i*(len(prices_cache)-i)/60:.0f}min)")
        except Exception as e:
            failures += 1
            print(f"  [{i:3d}/{len(prices_cache)}] {ticker}: FAILED: {e}")
            traceback.print_exc()
        gc.collect()

    elapsed_total = time.time() - start
    print(f"\n{'=' * 72}")
    print(f"  BACKFILL COMPLETE")
    print(f"  Rows written: {total_rows}")
    print(f"  Fallback rows: {fallbacks} ({fallbacks/max(1,total_rows):.1%})")
    print(f"  Ticker failures: {failures}")
    print(f"  Elapsed: {elapsed_total/60:.1f} min")
    print(f"  Table: scan_journal_backfill in {DB_PATH}")
    print(f"{'=' * 72}")
    con.close()


if __name__ == "__main__":
    main()
