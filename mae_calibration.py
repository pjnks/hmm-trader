"""
mae_calibration.py
──────────────────
Offline MAE/MFE calibration for CITRINE's Chandelier exit multiplier.

Runs walk-forward HMM backtests on NDX100 tickers using HMM-only exits
(no trailing stops), and measures per-trade Maximum Adverse Excursion (MAE)
and Maximum Favorable Excursion (MFE) in ATR units.

Purpose: Empirically determine the optimal ATR multiplier for the Chandelier
exit by analyzing the excursion distributions of winning vs losing trades.

Statistical traps addressed:
  1. Zero-MAE mass:  reports percentiles on non-zero MAE winners separately
  2. Time-in-trade:  logs the day within the trade when MAE occurred
  3. Regime coverage: uses 3 years of data to include both bull and bear markets

Usage
─────
  python mae_calibration.py                   # full analysis, 20 tickers
  python mae_calibration.py --tickers 10      # quick mode, 10 tickers
  python mae_calibration.py --tickers 40      # thorough mode

Output
──────
  Terminal report with MAE/MFE distributions and recommended multiplier.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import build_hmm_features
from src.hmm_model import HMMRegimeModel
from src.indicators import attach_all, compute_atr
from src.strategy import build_signal_series
from walk_forward_ndx import fetch_equity_daily, _attach_hmm_features

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mae_calibration")


# ── Trade record with excursion tracking ────────────────────────────────────

@dataclass
class ExcursionTrade:
    """A single trade with MAE/MFE tracked in both % and ATR units."""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    entry_atr: float          # ATR at entry (14-period)
    pnl_pct: float            # realized return %
    # Excursion tracking
    mae_pct: float = 0.0      # max adverse excursion as % of entry price (negative)
    mfe_pct: float = 0.0      # max favorable excursion as % of entry price (positive)
    mae_atr: float = 0.0      # MAE in ATR units (negative)
    mfe_atr: float = 0.0      # MFE in ATR units (positive)
    mae_day: int = 0          # day within trade when MAE occurred
    mfe_day: int = 0          # day within trade when MFE occurred
    hold_days: int = 0        # total days held
    is_winner: bool = False


# ── Per-ticker configs ──────────────────────────────────────────────────────

def _load_per_ticker_configs() -> dict:
    """Load per-ticker HMM configs from citrine_per_ticker_configs.json."""
    cfg_path = ROOT / "citrine_per_ticker_configs.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


# ── Core analysis ───────────────────────────────────────────────────────────

def _run_ticker_analysis(
    ticker: str,
    df: pd.DataFrame,
    ticker_cfg: dict,
    train_months: int = 6,
    test_months: int = 3,
) -> list[ExcursionTrade]:
    """
    Run walk-forward HMM on a single ticker with HMM-only exits.
    Track MAE/MFE for every trade in ATR units.
    """
    trades: list[ExcursionTrade] = []

    n_states = ticker_cfg.get("n_states", 4)
    feature_set = ticker_cfg.get("feature_set", "base")
    min_cf = ticker_cfg.get("confirmations", 5)
    cov_type = ticker_cfg.get("cov_type", "diag")

    feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])

    # Prepare features — deduplicate index first (Polygon sometimes returns dupes)
    df = df[~df.index.duplicated(keep='first')]
    df_feat = _attach_hmm_features(df.copy(), feature_set)
    df_feat = attach_all(df_feat)
    df_feat = df_feat[~df_feat.index.duplicated(keep='first')]

    # Compute ATR on original OHLCV (before any feature dropna)
    atr_series = compute_atr(df)
    # Align ATR to feature df index
    atr_series = atr_series.reindex(df_feat.index)

    if len(df_feat) < 250:
        return trades  # need at least ~1 year

    # Build walk-forward windows
    dates = df_feat.index
    total_days = (dates[-1] - dates[0]).days
    window_size = timedelta(days=train_months * 30 + test_months * 30)

    cursor = dates[0]
    while True:
        train_start = cursor
        train_end = train_start + timedelta(days=train_months * 30)
        test_start = train_end
        test_end = test_start + timedelta(days=test_months * 30)

        if test_end > dates[-1]:
            break

        # Training slice — drop NaN on feature cols to avoid HMM errors
        train_mask = (df_feat.index >= train_start) & (df_feat.index < train_end)
        train_df = df_feat[train_mask].dropna(subset=feature_cols)

        # Test slice
        test_mask = (df_feat.index >= test_start) & (df_feat.index < test_end)
        test_df = df_feat[test_mask].dropna(subset=feature_cols)

        if len(train_df) < 60 or len(test_df) < 20:
            cursor += timedelta(days=test_months * 30)
            continue

        # Fit HMM
        model = HMMRegimeModel(n_states=n_states, cov_type=cov_type,
                               feature_cols=feature_cols)
        model.fit(train_df)
        if not model.converged:
            cursor += timedelta(days=test_months * 30)
            continue

        # Predict on test data (use train+test for HMM context)
        combined = pd.concat([train_df, test_df]).dropna(subset=feature_cols)
        predicted = model.predict(combined)

        # Extract test portion
        test_predicted = predicted[predicted.index >= test_start]
        if len(test_predicted) < 10:
            cursor += timedelta(days=test_months * 30)
            continue

        # Build signals using the strategy engine
        config.MIN_CONFIRMATIONS = min_cf
        signals = build_signal_series(test_predicted)

        # Simulate HMM-only trades (LONG only, HMM exit on regime flip)
        in_trade = False
        entry_price = 0.0
        entry_atr_val = 0.0
        entry_date = None
        high_wm = 0.0
        low_wm = float('inf')
        mae_day_idx = 0
        mfe_day_idx = 0
        bars_held = 0

        for i in range(len(signals)):
            row = signals.iloc[i]
            date = signals.index[i]
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            regime = row.get("regime_cat", "CHOP")
            raw_buy = row.get("raw_long_signal", False)

            current_atr = atr_series.get(date, np.nan)
            if pd.isna(current_atr) or current_atr <= 0:
                continue

            if not in_trade:
                # Entry: BULL regime + raw_long_signal
                if regime == "BULL" and raw_buy:
                    in_trade = True
                    entry_price = close
                    entry_atr_val = current_atr
                    entry_date = date
                    high_wm = high
                    low_wm = low
                    mae_day_idx = 0
                    mfe_day_idx = 0
                    bars_held = 0
            else:
                bars_held += 1

                # Update watermarks using intraday high/low
                if high > high_wm:
                    high_wm = high
                    mfe_day_idx = bars_held
                if low < low_wm:
                    low_wm = low
                    mae_day_idx = bars_held

                # HMM-only exit: regime flips away from BULL
                if regime != "BULL":
                    exit_price = close
                    pnl_pct = (exit_price - entry_price) / entry_price

                    # Compute excursions
                    mae_pct = (low_wm - entry_price) / entry_price   # negative
                    mfe_pct = (high_wm - entry_price) / entry_price  # positive

                    # In ATR units (using entry_atr)
                    mae_atr = (low_wm - entry_price) / entry_atr_val
                    mfe_atr = (high_wm - entry_price) / entry_atr_val

                    trade = ExcursionTrade(
                        ticker=ticker,
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_atr=entry_atr_val,
                        pnl_pct=pnl_pct,
                        mae_pct=mae_pct,
                        mfe_pct=mfe_pct,
                        mae_atr=mae_atr,
                        mfe_atr=mfe_atr,
                        mae_day=mae_day_idx,
                        mfe_day=mfe_day_idx,
                        hold_days=bars_held,
                        is_winner=(pnl_pct > 0),
                    )
                    trades.append(trade)

                    in_trade = False
                    high_wm = 0.0
                    low_wm = float('inf')

        cursor += timedelta(days=test_months * 30)

    return trades


def run_calibration(n_tickers: int = 20) -> None:
    """Run the full MAE calibration analysis."""

    per_ticker_cfgs = _load_per_ticker_configs()

    # Use CITRINE universe — tickers with optimized configs
    all_tickers = list(per_ticker_cfgs.keys())
    if not all_tickers:
        # Fallback to a representative set
        all_tickers = [
            "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META",
            "AVGO", "COST", "NFLX", "ADBE", "AMD", "QCOM", "INTU",
            "AMGN", "PEP", "CSCO", "WMT", "KDP", "PYPL",
        ]

    # Take a representative sample
    tickers = all_tickers[:n_tickers]
    print(f"\n{'='*70}")
    print(f"  MAE/MFE CALIBRATION — {len(tickers)} NDX100 tickers, 3-year window")
    print(f"  HMM-only exits (no trailing stops) — measuring excursions in ATR units")
    print(f"{'='*70}\n")

    all_trades: list[ExcursionTrade] = []
    import time as _time

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker} ... ", end="", flush=True)
        t0 = _time.time()

        try:
            df = fetch_equity_daily(ticker, years=3)
            if df is None or len(df) < 300:
                print(f"skipped (insufficient data: {len(df) if df is not None else 0} bars)")
                continue

            cfg = per_ticker_cfgs.get(ticker, {
                "n_states": 4, "feature_set": "base",
                "confirmations": 5, "cov_type": "diag"
            })

            ticker_trades = _run_ticker_analysis(ticker, df, cfg)
            all_trades.extend(ticker_trades)

            wins = sum(1 for t in ticker_trades if t.is_winner)
            elapsed = _time.time() - t0
            print(f"{len(ticker_trades)} trades ({wins}W/{len(ticker_trades)-wins}L) "
                  f"[{elapsed:.1f}s]")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Rate limit
        _time.sleep(0.5)

    if not all_trades:
        print("\nNo trades generated. Check data availability.")
        return

    # ── Analysis ────────────────────────────────────────────────────────────
    winners = [t for t in all_trades if t.is_winner]
    losers = [t for t in all_trades if not t.is_winner]

    print(f"\n{'='*70}")
    print(f"  RESULTS — {len(all_trades)} total trades "
          f"({len(winners)}W / {len(losers)}L, "
          f"{100*len(winners)/len(all_trades):.1f}% WR)")
    print(f"{'='*70}")

    # ── MAE Distribution (in ATR units) ─────────────────────────────────────
    winner_maes = [t.mae_atr for t in winners]  # negative values
    loser_maes = [t.mae_atr for t in losers]

    # Non-zero MAE winners (trap #1: zero-MAE mass)
    nonzero_winner_maes = [m for m in winner_maes if m < -0.01]

    print(f"\n  ── MAE Distribution (ATR units, negative = adverse) ──")
    print(f"  {'':30s} {'Winners':>10s}  {'Losers':>10s}")
    print(f"  {'Count':30s} {len(winner_maes):10d}  {len(loser_maes):10d}")
    if winner_maes:
        print(f"  {'Mean MAE (ATR)':30s} {np.mean(winner_maes):10.3f}  "
              f"{np.mean(loser_maes):10.3f}")
        print(f"  {'Median MAE (ATR)':30s} {np.median(winner_maes):10.3f}  "
              f"{np.median(loser_maes):10.3f}")
        for pct in [75, 90, 95, 99]:
            w_val = np.percentile(winner_maes, 100 - pct) if winner_maes else 0
            l_val = np.percentile(loser_maes, 100 - pct) if loser_maes else 0
            print(f"  {f'{pct}th pctile MAE (ATR)':30s} {w_val:10.3f}  {l_val:10.3f}")

    # Non-zero MAE analysis (trap #1)
    if nonzero_winner_maes:
        print(f"\n  ── Non-Zero MAE Winners (n={len(nonzero_winner_maes)}/{len(winners)}) ──")
        print(f"  {'Zero-MAE winners':30s} {len(winners)-len(nonzero_winner_maes):10d}  "
              f"({100*(len(winners)-len(nonzero_winner_maes))/max(len(winners),1):.0f}%)")
        for pct in [75, 90, 95, 99]:
            val = np.percentile(nonzero_winner_maes, 100 - pct)
            print(f"  {f'{pct}th pctile non-zero MAE':30s} {val:10.3f}")

    # ── MFE Distribution ────────────────────────────────────────────────────
    winner_mfes = [t.mfe_atr for t in winners]
    loser_mfes = [t.mfe_atr for t in losers]

    print(f"\n  ── MFE Distribution (ATR units, positive = favorable) ──")
    print(f"  {'':30s} {'Winners':>10s}  {'Losers':>10s}")
    if winner_mfes:
        print(f"  {'Mean MFE (ATR)':30s} {np.mean(winner_mfes):10.3f}  "
              f"{np.mean(loser_mfes):10.3f}")
        print(f"  {'Median MFE (ATR)':30s} {np.median(winner_mfes):10.3f}  "
              f"{np.median(loser_mfes):10.3f}")

    # ── Hold Duration (trap #2) ─────────────────────────────────────────────
    winner_days = [t.hold_days for t in winners]
    loser_days = [t.hold_days for t in losers]
    winner_mae_days = [t.mae_day for t in winners if t.mae_atr < -0.01]
    loser_mae_days = [t.mae_day for t in losers if t.mae_atr < -0.01]

    print(f"\n  ── Hold Duration & MAE Timing (trap #2: time normalization) ──")
    if winner_days:
        print(f"  {'Avg hold (days) — winners':30s} {np.mean(winner_days):10.1f}")
        print(f"  {'Avg hold (days) — losers':30s} {np.mean(loser_days):10.1f}")
    if winner_mae_days:
        print(f"  {'Avg MAE day — winners':30s} {np.mean(winner_mae_days):10.1f}")
    if loser_mae_days:
        print(f"  {'Avg MAE day — losers':30s} {np.mean(loser_mae_days):10.1f}")
    if winner_mae_days:
        early = sum(1 for d in winner_mae_days if d <= 2)
        late = sum(1 for d in winner_mae_days if d > 5)
        print(f"  {'Winner MAE in first 2 days':30s} {early:10d}  "
              f"({100*early/len(winner_mae_days):.0f}%)")
        print(f"  {'Winner MAE after day 5':30s} {late:10d}  "
              f"({100*late/len(winner_mae_days):.0f}%)")

    # ── Regime Coverage (trap #3) ───────────────────────────────────────────
    # Check date range of trades
    if all_trades:
        dates = sorted(t.entry_date for t in all_trades)
        print(f"\n  ── Sample Coverage (trap #3: regime representation) ──")
        print(f"  {'Date range':30s} {str(dates[0].date()):>10s}  to {str(dates[-1].date())}")
        # Bucket by year
        by_year = {}
        for t in all_trades:
            y = t.entry_date.year
            by_year[y] = by_year.get(y, 0) + 1
        for y in sorted(by_year):
            print(f"  {f'Trades in {y}':30s} {by_year[y]:10d}")

    # ── P&L Stats ───────────────────────────────────────────────────────────
    winner_pnls = [t.pnl_pct * 100 for t in winners]
    loser_pnls = [t.pnl_pct * 100 for t in losers]
    print(f"\n  ── Return Distribution (%) ──")
    if winner_pnls:
        print(f"  {'Avg winner return':30s} {np.mean(winner_pnls):+10.2f}%")
    if loser_pnls:
        print(f"  {'Avg loser return':30s} {np.mean(loser_pnls):+10.2f}%")
    if winner_pnls and loser_pnls:
        exp = (len(winners)/len(all_trades) * np.mean(winner_pnls)
               + len(losers)/len(all_trades) * np.mean(loser_pnls))
        print(f"  {'Expectancy per trade':30s} {exp:+10.3f}%")

    # ── RECOMMENDATION ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CHANDELIER MULTIPLIER RECOMMENDATION")
    print(f"{'='*70}")

    if nonzero_winner_maes:
        p95 = np.percentile(nonzero_winner_maes, 5)  # 5th pctile of negative values = 95th of magnitude
        p99 = np.percentile(nonzero_winner_maes, 1)
        recommended = round(abs(p95) * 1.1, 1)  # 10% buffer above 95th

        print(f"\n  95th pctile of non-zero winner MAE: {p95:.3f} ATR")
        print(f"  99th pctile of non-zero winner MAE: {p99:.3f} ATR")
        print(f"\n  → Recommended multiplier: {recommended:.1f} ATR")
        print(f"    (95th pctile magnitude × 1.1 safety buffer)")

        if recommended < 2.0:
            print(f"    ⚠️  Empirical says {recommended:.1f}, but floor at 2.0 for safety")
            recommended = 2.0
        elif recommended > 4.0:
            print(f"    ⚠️  Empirical says {recommended:.1f}, capping at 4.0")
            recommended = 4.0

        print(f"\n  FINAL: Use {recommended:.1f} ATR for Chandelier exit")
        print(f"  (Prior expectation was 3.0 — empirical {'confirms' if 2.5 <= recommended <= 3.5 else 'disagrees'})")

        # Would-have analysis
        for mult in [2.0, 2.5, 3.0, 3.5]:
            # How many winners would have been stopped out?
            stopped_winners = sum(1 for m in winner_maes if m < -mult)
            stopped_losers = sum(1 for m in loser_maes if m < -mult)
            saved_losers = len(loser_maes) - stopped_losers  # these wouldn't be caught
            print(f"\n  At {mult:.1f} ATR stop: "
                  f"would stop {stopped_winners}/{len(winners)} winners "
                  f"({100*stopped_winners/max(len(winners),1):.0f}%), "
                  f"{stopped_losers}/{len(losers)} losers "
                  f"({100*stopped_losers/max(len(losers),1):.0f}%)")
    else:
        print("\n  Insufficient non-zero MAE data for recommendation.")
        print("  Defaulting to 3.0 ATR (theoretical prior).")

    print(f"\n{'='*70}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE/MFE Calibration for CITRINE")
    parser.add_argument("--tickers", type=int, default=20,
                        help="Number of tickers to analyze (default: 20)")
    args = parser.parse_args()

    run_calibration(n_tickers=args.tickers)
