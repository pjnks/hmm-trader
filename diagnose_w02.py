"""
diagnose_w02.py
───────────────
Deep-dive into Walk-Forward Window 2 (Jun-Sep 2025) to understand losses.

Reproduces exactly the same data slice, HMM fit, and backtest as walk_forward.py
but prints:
  - HMM state table for the train period
  - Regime distribution during the test period
  - Per-hour regime + signal breakdown for the test period
  - Per-trade detail (entry reason, exit reason, P&L)
  - Confirmation count distribution (how close were near-misses?)

Usage:
    python diagnose_w02.py
    python diagnose_w02.py --feature-set full
    python diagnose_w02.py --window 1          # re-run W01 instead
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import fetch_btc_hourly, build_hmm_features
from src.hmm_model import HMMRegimeModel
from src.indicators import (
    attach_all,
    compute_realized_vol_ratio,
    compute_return_autocorr,
    compute_vol_price_diverge,
    compute_candle_body_ratio,
    compute_bb_width,
)
from src.strategy import build_signal_series
from src.backtester import Backtester

# Walk-forward window definitions (same logic as walk_forward.py)
_WINDOWS = {
    1: ("2024-03-12", "2025-03-12", "2025-06-12"),  # train_start, test_start, test_end
    2: ("2024-06-12", "2025-06-12", "2025-09-12"),
    3: ("2024-09-12", "2025-09-12", "2025-12-12"),
}

_WARMUP = max(
    config.TREND_MA_PERIOD,
    config.MACD_SLOW + config.MACD_SIGNAL,
    config.ADX_PERIOD * 3,
    config.VOLATILITY_PERIOD * 5,
)


def _attach_hmm_features(df_raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    df = build_hmm_features(df_raw)
    if feature_set in ("extended", "full", "volume_focused"):
        df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
    if feature_set in ("extended", "full"):
        df["return_autocorr"]   = compute_return_autocorr(df)
        df["vol_price_diverge"] = compute_vol_price_diverge(df)
    if feature_set == "full":
        df["candle_body_ratio"] = compute_candle_body_ratio(df)
        df["bb_width"]          = compute_bb_width(df)
    return df


def _tz(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def run_diagnosis(window: int = 2, feature_set: str = "volume_focused") -> None:
    train_start_str, test_start_str, test_end_str = _WINDOWS[window]

    train_start = _tz(train_start_str)
    test_start  = _tz(test_start_str)
    test_end    = _tz(test_end_str)

    feature_cols = config.FEATURE_SETS[feature_set]

    print(f"\n{'═'*70}")
    print(f"  DIAGNOSIS — Window {window}  |  feature_set={feature_set}")
    print(f"  Train : {train_start.date()} → {test_start.date()}")
    print(f"  Test  : {test_start.date()} → {test_end.date()}")
    print(f"{'═'*70}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading 730-day OHLCV …")
    raw_ohlcv = fetch_btc_hourly(days=730, ticker=config.TICKER)
    full_df   = _attach_hmm_features(raw_ohlcv, feature_set)
    full_df   = full_df.dropna(subset=feature_cols)
    print(f"  {len(full_df)} bars  ({full_df.index[0].date()} → {full_df.index[-1].date()})\n")

    # ── Fit HMM on training slice ──────────────────────────────────────────
    df_train = full_df[(full_df.index >= train_start) & (full_df.index < test_start)]
    print(f"Training HMM on {len(df_train)} bars ({train_start.date()} → {test_start.date()}) …")

    model = HMMRegimeModel(
        n_states     = config.N_STATES,
        cov_type     = config.COV_TYPE,
        feature_cols = feature_cols,
    )
    model.fit(df_train)
    print(f"  Converged: {model.converged}\n")

    # ── State table ────────────────────────────────────────────────────────
    print("── HMM State Table (trained on train period) ──────────────────────")
    stats = model.get_state_stats()
    print(stats.to_string())
    print()

    # ── Build buffer+test slice ───────────────────────────────────────────
    buf_iloc = max(0, full_df.index.searchsorted(test_start) - _WARMUP)
    end_iloc = full_df.index.searchsorted(test_end, side="right")
    df_bt_raw = full_df.iloc[buf_iloc:end_iloc].copy()

    print(f"Buffer+test slice: {len(df_bt_raw)} bars  "
          f"({df_bt_raw.index[0].date()} → {df_bt_raw.index[-1].date()})")
    print(f"  (_WARMUP={_WARMUP} bar buffer prepended)\n")

    # ── Predict + signal pipeline ─────────────────────────────────────────
    df_bt = model.predict(df_bt_raw)
    df_bt = attach_all(df_bt)
    df_bt = build_signal_series(df_bt)

    # Filter to test period only
    df_test = df_bt[(df_bt.index >= test_start) & (df_bt.index <= test_end)].copy()

    # ── Regime distribution in test period ────────────────────────────────
    print("── Regime Distribution (test period) ──────────────────────────────")
    regime_counts = df_test["regime_cat"].value_counts()
    total_bars    = len(df_test)
    for cat, cnt in regime_counts.items():
        bar = "█" * int(cnt / total_bars * 40)
        print(f"  {cat:<10}  {cnt:>5} bars  ({cnt/total_bars*100:>5.1f}%)  {bar}")

    bull_bars = df_test[df_test["regime_cat"] == "BULL"]
    print(f"\n  BULL bars with confidence ≥ {config.REGIME_CONFIDENCE_MIN}:  "
          f"{(bull_bars['confidence'] >= config.REGIME_CONFIDENCE_MIN).sum()}")
    print(f"  Mean confidence on BULL bars:  "
          f"{bull_bars['confidence'].mean():.3f}" if len(bull_bars) > 0 else "  (no BULL bars)")
    print()

    # ── Confirmation count distribution (BULL + confident bars only) ──────
    bull_conf = df_test[
        (df_test["regime_cat"] == "BULL") &
        (df_test["confidence"] >= config.REGIME_CONFIDENCE_MIN)
    ]
    if len(bull_conf) > 0:
        print("── Confirmation Counts (BULL + confident bars) ─────────────────────")
        cc_dist = bull_conf["confirmation_count"].value_counts().sort_index()
        for cc, cnt in cc_dist.items():
            bar = "█" * int(cnt / len(bull_conf) * 30)
            needed = "✓ ENTRY" if cc >= config.MIN_CONFIRMATIONS else f"  need {config.MIN_CONFIRMATIONS - cc} more"
            print(f"  confirmations={cc}  {cnt:>4} bars  {needed}  {bar}")
        print()

        # Per-indicator miss breakdown on BULL+conf bars that didn't trigger
        near_miss = bull_conf[bull_conf["confirmation_count"] >= config.MIN_CONFIRMATIONS - 2]
        if len(near_miss) > 0:
            indicator_cols = [
                "rsi", "momentum", "volatility", "vol_median",
                "volume_ratio", "adx", "price_trend_pct", "sma_50",
                "macd", "macd_signal", "stoch_k",
            ]
            checks = {
                "RSI 30–70":      lambda d: (d["rsi"] >= 30) & (d["rsi"] <= 70),
                "Momentum>0":     lambda d: d["momentum"] > 0,
                "Vol<2×med":      lambda d: d["volatility"] < 2 * d["vol_median"],
                "Volume>1.1×MA":  lambda d: d["volume_ratio"] > 1.1,
                "ADX>25":         lambda d: d["adx"] > 25,
                "Close>SMA50":    lambda d: d["Close"] > d["sma_50"],
                "MACD>Signal":    lambda d: d["macd"] > d["macd_signal"],
                "StochK<80":      lambda d: d["stoch_k"] < 80,
            }
            print("── Indicator Pass Rate on near-miss BULL+conf bars ─────────────────")
            for name, fn in checks.items():
                try:
                    pass_rate = fn(near_miss).mean() * 100
                    bar = "█" * int(pass_rate / 100 * 30)
                    print(f"  {name:<16}  {pass_rate:>5.1f}%  {bar}")
                except Exception:
                    print(f"  {name:<16}  (error)")
            print()

    # ── Run backtester ─────────────────────────────────────────────────────
    res = Backtester().run(df_bt)
    trades = res.trades.copy()

    # Filter to test period
    if not trades.empty and "entry_time" in trades.columns:
        et = pd.to_datetime(trades["entry_time"])
        if et.dt.tz is None:
            et = et.dt.tz_localize("UTC")
        trades = trades[(et >= test_start) & (et <= test_end)].copy()

    print(f"── Trades in Test Period ({len(trades)} total) ─────────────────────────")
    if trades.empty:
        print("  No trades.\n")
    else:
        # Enrich with duration
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            et = pd.to_datetime(trades["entry_time"]).dt.tz_localize("UTC") \
                 if pd.to_datetime(trades["entry_time"]).dt.tz is None \
                 else pd.to_datetime(trades["entry_time"])
            xt = pd.to_datetime(trades["exit_time"])
            if xt.dt.tz is None:
                xt = xt.dt.tz_localize("UTC")
            trades = trades.copy()
            trades["duration_h"] = ((xt - et).dt.total_seconds() / 3600).round(1)

        display_cols = [c for c in [
            "entry_time", "exit_time", "duration_h",
            "entry_price", "exit_price", "pnl", "pnl_pct",
            "exit_reason",
        ] if c in trades.columns]

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(trades[display_cols].to_string(index=False))

        wins   = (trades["pnl"] > 0).sum()
        losses = (trades["pnl"] <= 0).sum()
        avg_win  = trades.loc[trades["pnl"] > 0,  "pnl"].mean() if wins   > 0 else 0
        avg_loss = trades.loc[trades["pnl"] <= 0, "pnl"].mean() if losses > 0 else 0

        print(f"\n  Wins:  {wins}  avg +${avg_win:,.2f}")
        print(f"  Losses:{losses}  avg -${abs(avg_loss):,.2f}")
        if losses > 0 and avg_loss != 0:
            print(f"  Payoff ratio: {abs(avg_win/avg_loss):.2f}×")

        if "exit_reason" in trades.columns:
            print("\n  Exit reasons:")
            for reason, cnt in trades["exit_reason"].value_counts().items():
                print(f"    {reason:<30}  {cnt}")

        # BTC price context for each trade
        print("\n── BTC price context per trade ──────────────────────────────────────")
        for _, tr in trades.iterrows():
            try:
                e_t = pd.Timestamp(tr["entry_time"]).tz_localize("UTC") \
                      if pd.Timestamp(tr["entry_time"]).tzinfo is None \
                      else pd.Timestamp(tr["entry_time"])
                x_t = pd.Timestamp(tr["exit_time"]).tz_localize("UTC") \
                      if pd.Timestamp(tr["exit_time"]).tzinfo is None \
                      else pd.Timestamp(tr["exit_time"])
                # Regime at entry
                entry_bar = df_test[df_test.index <= e_t]
                if entry_bar.empty:
                    entry_bar = df_bt[df_bt.index <= e_t]
                regime_at_entry    = entry_bar["regime_cat"].iloc[-1] if not entry_bar.empty else "?"
                conf_at_entry      = entry_bar["confidence"].iloc[-1]  if not entry_bar.empty else 0
                confirms_at_entry  = entry_bar["confirmation_count"].iloc[-1] if not entry_bar.empty else 0
                # Regime at exit
                exit_bar = df_test[df_test.index <= x_t]
                if exit_bar.empty:
                    exit_bar = df_bt[df_bt.index <= x_t]
                regime_at_exit = exit_bar["regime_cat"].iloc[-1] if not exit_bar.empty else "?"
                pnl_str = f"+${tr['pnl']:,.2f}" if tr["pnl"] > 0 else f"-${abs(tr['pnl']):,.2f}"
                print(f"  {str(e_t.date()):<12} entry  regime={regime_at_entry:<5}  conf={conf_at_entry:.2f}  "
                      f"confirms={confirms_at_entry:.0f}  →  "
                      f"exit={str(x_t.date()):<12} regime={regime_at_exit:<5}  "
                      f"pnl={pnl_str:<12} ({tr.get('exit_reason','?')})")
            except Exception as exc:
                print(f"  (error enriching trade: {exc})")

    # ── Regime timeline (weekly buckets) ──────────────────────────────────
    print("\n── Weekly Regime Distribution (test period) ────────────────────────")
    df_test_copy = df_test.copy()
    df_test_copy["week"] = df_test_copy.index.to_period("W")
    weekly = df_test_copy.groupby(["week", "regime_cat"]).size().unstack(fill_value=0)
    # Percentage
    weekly_pct = (weekly.div(weekly.sum(axis=1), axis=0) * 100).round(1)
    print(weekly_pct.to_string())
    print()

    # ── Confidence distribution on BULL bars ──────────────────────────────
    if len(bull_bars) > 0:
        print("── Confidence Distribution on BULL bars ────────────────────────────")
        bins = [0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
        labels = ["<50%", "50-60%", "60-70%", "70-75%", "75-80%", "80-85%", "85-90%", "≥90%"]
        bull_bars_copy = bull_bars.copy()
        bull_bars_copy["conf_bin"] = pd.cut(bull_bars_copy["confidence"], bins=bins, labels=labels)
        dist = bull_bars_copy["conf_bin"].value_counts().reindex(labels, fill_value=0)
        for label, cnt in dist.items():
            bar = "█" * int(cnt / max(dist.max(), 1) * 30)
            mark = " ← threshold" if label == "70-75%" else ""
            print(f"  {label:<10}  {cnt:>4} bars  {bar}{mark}")
        print()

    print("═" * 70)
    print("  Done.\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose walk-forward window losses")
    parser.add_argument("--window", type=int, default=2, choices=[1, 2, 3],
                        help="Which window to diagnose (default: 2)")
    parser.add_argument("--feature-set", default="volume_focused",
                        choices=list(config.FEATURE_SETS.keys()),
                        help="Feature set to use (default: volume_focused)")
    args = parser.parse_args()
    run_diagnosis(window=args.window, feature_set=args.feature_set)


if __name__ == "__main__":
    main()
