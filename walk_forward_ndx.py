"""
walk_forward_ndx.py
────────────────────
Walk-forward analysis for NDX100 equities (AAPL, MSFT, TSLA, NVDA, GOOGL).

Fetches daily OHLCV via yfinance and runs 6m train / 3m test rolling windows.
Reuses HMM regime detection + 8-indicator strategy from crypto, but on 1d/5d timeframes.

Usage
─────
  python walk_forward_ndx.py                  # all 5 tickers, 6m/3m windows
  python walk_forward_ndx.py --ticker AAPL    # single ticker
  python walk_forward_ndx.py --train-months 6 --test-months 1
  python walk_forward_ndx.py --feature-set full

Output
──────
  walk_forward_ndx_results.csv   — per-window metrics table (all tickers)
  walk_forward_ndx_{TICKER}.html — per-ticker equity + metrics chart
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import NamedTuple
from datetime import datetime, timedelta, timezone
import os
import time

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load .env for Polygon API key
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import build_hmm_features, resample_ohlcv
from src.hmm_model import HMMRegimeModel
from src.indicators import (
    attach_all,
    compute_realized_vol_ratio,
    compute_return_autocorr,
    compute_vol_price_diverge,
    compute_candle_body_ratio,
    compute_bb_width,
    compute_realized_kurtosis,
    compute_volume_return_intensity,
    compute_return_momentum_ratio,
)
from src.strategy import build_signal_series
from src.backtester import Backtester

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("walk_forward_ndx")

# ── Output paths ───────────────────────────────────────────────────────────────
RESULTS_CSV = ROOT / "walk_forward_ndx_results.csv"

# NDX100 top 5 constituents (by market cap, highest volume)
NDX_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

# Backtester warmup: bars to prepend to each test window
_WARMUP = max(
    config.TREND_MA_PERIOD,
    config.MACD_SLOW + config.MACD_SIGNAL,
    config.ADX_PERIOD * 3,
    config.VOLATILITY_PERIOD * 5,
)


# ─────────────────────────────────────────────────────────────────────────────

class WindowResult(NamedTuple):
    ticker:            str
    window:            int
    train_start:       str
    train_end:         str
    test_start:        str
    test_end:          str
    return_pct:        float
    bh_return_pct:     float
    alpha_pct:         float
    max_drawdown_pct:  float
    sharpe_ratio:      float
    n_trades:          int
    win_rate_pct:      float
    start_equity:      float
    end_equity:        float
    hmm_converged:     bool


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_polygon_bars(ticker: str, multiplier: int, timespan: str, years: int = 20) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Polygon.io with pagination.
    """
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key or api_key == "your_polygon_api_key_here":
        log.error("POLYGON_API_KEY not set in .env")
        return pd.DataFrame()

    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=365 * years)
    to_ms = int(end_date.timestamp() * 1000)
    cursor_ms = int(start_date.timestamp() * 1000)

    log.info(f"Fetching {ticker} {multiplier}{timespan} data via Polygon")

    max_retries = 5
    retry_wait_s = 60
    all_bars = []

    while cursor_ms < to_ms:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{cursor_ms}/{to_ms}"
        for attempt in range(1, max_retries + 2):
            try:
                params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    if attempt > max_retries:
                        log.error(f"{ticker}: Polygon rate limit persisted")
                        break
                    log.warning(f"{ticker}: Polygon 429 — waiting {retry_wait_s}s")
                    time.sleep(retry_wait_s)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") not in ("OK", "DELAYED"):
                    log.error(f"{ticker}: Polygon returned {data.get('status')}")
                    break
                results = data.get("results", [])
                if not results:
                    cursor_ms = to_ms
                    break
                all_bars.extend(results)
                last_ts = results[-1]["t"]
                if last_ts <= cursor_ms:
                    cursor_ms = to_ms
                    break
                cursor_ms = last_ts + 1
                break
            except Exception as e:
                log.error(f"{ticker}: {e}")
                cursor_ms = to_ms
                break

    if not all_bars:
        log.warning(f"{ticker}: No data returned from Polygon")
        return pd.DataFrame()

    rows = []
    for bar in all_bars:
        rows.append({
            "Open": bar.get("o", 0),
            "High": bar.get("h", 0),
            "Low": bar.get("l", 0),
            "Close": bar.get("c", 0),
            "Volume": bar.get("v", 0),
            "Datetime": pd.Timestamp(bar.get("t"), unit="ms", tz="UTC"),
        })

    df = pd.DataFrame(rows)
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    log.info(f"✅ {ticker}: Fetched {len(df)} {multiplier}{timespan} bars")
    return df


def fetch_equity_daily(ticker: str, years: int = 20) -> pd.DataFrame:
    """Fetch daily OHLCV data for an equity ticker from Polygon.io."""
    return _fetch_polygon_bars(ticker, multiplier=1, timespan="day", years=years)


def fetch_equity_hourly(ticker: str, years: int = 2) -> pd.DataFrame:
    """Fetch hourly OHLCV data for an equity ticker from Polygon.io (with pagination)."""
    return _fetch_polygon_bars(ticker, multiplier=1, timespan="hour", years=years)


def resample_equity_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample hourly equity bars to 2h or 4h."""
    if timeframe == "1h":
        return df
    tf_map = {"2h": "2h", "4h": "4h", "1d": "1D"}
    rule = tf_map.get(timeframe, timeframe)
    resampled = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return resampled


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def _attach_hmm_features(df_raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """
    Compute base HMM features + extended features for equities.
    Uses config.FEATURE_SETS to determine which columns are needed, then computes
    only the required indicators. This ensures new feature sets (extended_v2, etc.)
    work automatically without hardcoded names.
    """
    df = build_hmm_features(df_raw)

    # Determine which features the set requires
    needed = set(config.FEATURE_SETS.get(feature_set, []))

    # Extended features — compute only what's needed
    if "realized_vol_ratio" in needed:
        df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
    if "return_autocorr" in needed:
        df["return_autocorr"]   = compute_return_autocorr(df)
    if "vol_price_diverge" in needed:
        df["vol_price_diverge"] = compute_vol_price_diverge(df)
    if "candle_body_ratio" in needed:
        df["candle_body_ratio"] = compute_candle_body_ratio(df)
    if "bb_width" in needed:
        df["bb_width"]          = compute_bb_width(df)
    # Sprint 3.1: new continuous features
    if "realized_kurtosis" in needed:
        df["realized_kurtosis"] = compute_realized_kurtosis(df)
    if "volume_return_intensity" in needed:
        df["volume_return_intensity"] = compute_volume_return_intensity(df)
    if "return_momentum_ratio" in needed:
        df["return_momentum_ratio"] = compute_return_momentum_ratio(df)

    # Drop NaN rows from extended feature warmup periods (rolling windows need 24+ bars)
    feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])
    cols_present = [c for c in feature_cols if c in df.columns]
    if cols_present:
        df.dropna(subset=cols_present, inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-window analysis
# ─────────────────────────────────────────────────────────────────────────────

def _run_window(
    full_df:            pd.DataFrame,
    ticker:             str,
    feature_cols:       list[str],
    train_start:        pd.Timestamp,
    test_start:         pd.Timestamp,
    test_end:           pd.Timestamp,
    chain_equity:       float,
    window_idx:         int,
    use_ensemble:       bool = False,
    bars_per_year:      int = 252,
) -> tuple[WindowResult | None, pd.Series, pd.Series, pd.DataFrame]:
    """
    Run one walk-forward window for equity data.
    Supports both single-model and ensemble HMM.
    """
    empty = pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

    # ── Training slice ─────────────────────────────────────────────────────
    df_train = full_df[(full_df.index >= train_start) & (full_df.index < test_start)]
    if len(df_train) < 50:  # Fewer warmup bars needed for daily data
        log.warning("Window %d: only %d train bars — skipping", window_idx + 1, len(df_train))
        return None, *empty

    # ── Fit a fresh HMM on training data only ─────────────────────────────
    if use_ensemble:
        from src.ensemble import EnsembleHMM
        model = EnsembleHMM(
            cov_type     = config.COV_TYPE,
            feature_cols = feature_cols,
        )
    else:
        model = HMMRegimeModel(
            n_states     = config.N_STATES,
            cov_type     = config.COV_TYPE,
            feature_cols = feature_cols,
        )
    model.fit(df_train)
    if not model.converged:
        log.warning("Window %d: HMM did not converge — skipping", window_idx + 1)
        return None, *empty

    # ── Build buffer + test slice ─────────────────────────────────────────
    buf_iloc  = max(0, full_df.index.searchsorted(test_start) - _WARMUP)
    end_iloc  = full_df.index.searchsorted(test_end, side="right")
    df_bt_raw = full_df.iloc[buf_iloc:end_iloc].copy()

    if len(df_bt_raw) < _WARMUP + 10:
        log.warning("Window %d: too few buffer+test bars (%d) — skipping",
                    window_idx + 1, len(df_bt_raw))
        return None, *empty

    # Predict regimes on buffer+test using the train-fitted model
    df_bt = model.predict(df_bt_raw)

    # Strategy indicators and signal masks
    df_bt = attach_all(df_bt)
    df_bt = build_signal_series(df_bt, use_regime_mapper=False)

    # ── Run backtester with chain equity as starting capital ───────────────
    saved_cap = config.INITIAL_CAPITAL
    config.INITIAL_CAPITAL = chain_equity
    try:
        res = Backtester(use_regime_mapper=False).run(df_bt)
    finally:
        config.INITIAL_CAPITAL = saved_cap

    if res.equity_curve.empty:
        log.warning("Window %d: empty equity curve — skipping", window_idx + 1)
        return None, *empty

    # ── Extract test-period equity (filter out the buffer) ─────────────────
    def _tz(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts

    ts_start = _tz(test_start)
    ts_end   = _tz(test_end)

    eq_oos = res.equity_curve[
        (res.equity_curve.index >= ts_start) &
        (res.equity_curve.index <= ts_end)
    ]
    if eq_oos.empty:
        eq_oos = res.equity_curve

    # ── Test-period trades ─────────────────────────────────────────────────
    trades_oos = res.trades.copy()
    if not trades_oos.empty and "entry_time" in trades_oos.columns:
        entry_ts = pd.to_datetime(trades_oos["entry_time"])
        if entry_ts.dt.tz is None:
            entry_ts = entry_ts.dt.tz_localize("UTC")
        trades_oos = trades_oos[(entry_ts >= ts_start) & (entry_ts <= ts_end)]

    # ── Buy-and-hold equity for this test window ───────────────────────────
    df_test_slice = full_df[
        (full_df.index >= ts_start) & (full_df.index <= ts_end)
    ]
    if df_test_slice.empty:
        return None, *empty

    bh_start = float(df_test_slice["Close"].iloc[0])
    bh_end   = float(df_test_slice["Close"].iloc[-1])
    bh_eq    = (df_test_slice["Close"] / bh_start) * chain_equity

    # ── Per-window metrics ─────────────────────────────────────────────────
    w_initial = float(eq_oos.iloc[0])
    w_final   = float(eq_oos.iloc[-1])

    ret_pct   = (w_final / w_initial - 1) * 100
    bh_ret    = (bh_end  / bh_start  - 1) * 100

    rolling_max = eq_oos.cummax()
    max_dd      = ((eq_oos - rolling_max) / rolling_max * 100).min()

    # ── Sharpe: annualize based on bar frequency ──────────────────────────
    hr = eq_oos.pct_change().dropna()
    sharpe = (hr.mean() / hr.std() * np.sqrt(bars_per_year)
              if len(hr) > 1 and hr.std() > 0 else 0.0)

    n_tr    = len(trades_oos)
    wr      = ((trades_oos["pnl"] > 0).sum() / n_tr * 100) if n_tr > 0 else 0.0

    wr = WindowResult(
        ticker           = ticker,
        window           = window_idx + 1,
        train_start      = str(train_start.date()),
        train_end        = str(test_start.date()),
        test_start       = str(test_start.date()),
        test_end         = str(test_end.date()),
        return_pct       = round(ret_pct,  2),
        bh_return_pct    = round(bh_ret,   2),
        alpha_pct        = round(ret_pct - bh_ret, 2),
        max_drawdown_pct = round(float(max_dd), 2),
        sharpe_ratio     = round(float(sharpe), 3),
        n_trades         = n_tr,
        win_rate_pct     = round(wr, 1),
        start_equity     = round(w_initial, 2),
        end_equity       = round(w_final,   2),
        hmm_converged    = model.converged,
    )
    return wr, eq_oos, bh_eq, trades_oos


# ─────────────────────────────────────────────────────────────────────────────
# Main walk-forward loop
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward_ndx(
    tickers:         list[str] | None = None,
    train_months:    int = 6,
    test_months:     int = 3,
    feature_set:     str = "base",
    quiet:           bool = False,
    use_ensemble:    bool = False,
    timeframe:       str = "1d",
) -> tuple[list[WindowResult], dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """
    Run walk-forward analysis on NDX100 equities.
    Returns (all_results, equity_curves_by_ticker, trades_by_ticker).
    """
    if tickers is None:
        tickers = NDX_TICKERS

    feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])
    all_results = []
    equity_curves = {}
    all_trades = {}

    # Calculate bars per year for Sharpe annualization
    bars_per_year_map = {"1h": 252 * 7, "2h": 252 * 3.5, "4h": 252 * 1.75, "1d": 252}
    bpy = int(bars_per_year_map.get(timeframe, 252))

    for ticker in tickers:
        if not quiet:
            print(f"\n{'='*70}")
            print(f"Processing {ticker} ({timeframe})")
            print(f"{'='*70}")

        # Fetch data based on timeframe
        if timeframe == "1d":
            df_raw = fetch_equity_daily(ticker)
        else:
            df_raw = fetch_equity_hourly(ticker)
            if not df_raw.empty:
                df_raw = resample_equity_ohlcv(df_raw, timeframe)
        if df_raw.empty:
            log.warning(f"Skipping {ticker}: no data available")
            continue

        # Attach HMM features
        df = _attach_hmm_features(df_raw, feature_set)

        # Generate walk-forward windows
        # For daily data: 6 months ≈ 120 bars, 3 months ≈ 60 bars
        train_days = train_months * 21
        test_days = test_months * 21

        windows = []
        idx = len(df) - 1
        while idx > train_days + test_days:
            test_end = df.index[idx]
            test_start = test_end - pd.Timedelta(days=test_days)
            train_end = test_start
            train_start = train_end - pd.Timedelta(days=train_days)

            windows.append((train_start, test_start, test_end))
            idx -= test_days

        windows.reverse()

        # Run windows
        chain_eq = config.INITIAL_CAPITAL
        ticker_results = []
        ticker_equity = pd.Series(dtype=float)
        ticker_trades = pd.DataFrame()

        for window_idx, (ts_start, ts_test_start, ts_test_end) in enumerate(windows):
            wr, eq_oos, bh_eq, trades_oos = _run_window(
                df, ticker, feature_cols,
                ts_start, ts_test_start, ts_test_end,
                chain_eq, window_idx,
                use_ensemble=use_ensemble,
                bars_per_year=bpy,
            )
            if wr is None:
                continue

            ticker_results.append(wr)
            all_results.append(wr)
            chain_eq = wr.end_equity
            ticker_equity = pd.concat([ticker_equity, eq_oos])
            ticker_trades = pd.concat([ticker_trades, trades_oos], ignore_index=True)

        equity_curves[ticker] = ticker_equity
        all_trades[ticker] = ticker_trades

        if not quiet and ticker_results:
            df_summary = pd.DataFrame(ticker_results)
            print(f"\n{ticker} Results ({len(ticker_results)} windows):")
            print(df_summary.to_string(index=False))

    return all_results, equity_curves, all_trades


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NDX100 walk-forward analysis")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker to analyze (default: all NDX tickers)")
    parser.add_argument("--train-months", type=int, default=6,
                        help="Training window in months (default: 6)")
    parser.add_argument("--test-months", type=int, default=3,
                        help="Test window in months (default: 3)")
    parser.add_argument("--feature-set", type=str, default="base",
                        choices=["base", "extended", "full", "volume_focused"],
                        help="HMM feature set (default: base)")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else NDX_TICKERS

    results, eq_curves, trades = run_walk_forward_ndx(
        tickers=tickers,
        train_months=args.train_months,
        test_months=args.test_months,
        feature_set=args.feature_set,
        quiet=False,
    )

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(RESULTS_CSV, index=False)
        print(f"\n✅ Saved {len(results)} window results to {RESULTS_CSV}")

        # Summary by ticker
        print("\n" + "="*70)
        print("SUMMARY BY TICKER")
        print("="*70)
        for ticker in tickers:
            ticker_results = [r for r in results if r.ticker == ticker]
            if ticker_results:
                df_ticker = pd.DataFrame(ticker_results)
                mean_sharpe = df_ticker["sharpe_ratio"].mean()
                mean_return = df_ticker["return_pct"].mean()
                mean_dd = df_ticker["max_drawdown_pct"].min()
                print(f"{ticker:6s} | Sharpe: {mean_sharpe:6.3f} | Return: {mean_return:7.2f}% | Max DD: {mean_dd:7.2f}%")
    else:
        print("❌ No valid windows found")


if __name__ == "__main__":
    main()
