"""
walk_forward.py
───────────────
Walk-forward (out-of-sample) analysis for the HMM Regime Trader.

For each rolling window:
  1. Slice *train_months* of data ending at test_start.
  2. Fit a fresh HMM on that training slice.
  3. Run the full strategy pipeline on a buffer + test slice so that
     indicator warmup is satisfied and trading starts at test_start.
  4. Record per-window metrics and stitch equity curves together.

The result is a single continuous out-of-sample equity curve covering
the entire test period — no look-ahead, no data reuse.

Usage
─────
  python walk_forward.py                      # defaults (12m train / 3m test, BTC)
  python walk_forward.py --train-months 6     # shorter training window
  python walk_forward.py --test-months 1      # monthly test periods
  python walk_forward.py --ticker X:ETHUSD
  python walk_forward.py --feature-set full

Output
──────
  walk_forward_results.csv   — per-window metrics table
  walk_forward_chart.html    — interactive equity + metrics chart
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import fetch_btc_hourly, build_hmm_features, resample_ohlcv
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
log = logging.getLogger("walk_forward")

# ── Output paths ───────────────────────────────────────────────────────────────
RESULTS_CSV = ROOT / "walk_forward_results.csv"
CHART_HTML  = ROOT / "walk_forward_chart.html"

# Backtester warmup: bars to prepend to each test window so that the
# backtester's internal warmup skip eats exactly the buffer and trading
# starts at the first bar of the test period.
_WARMUP = max(
    config.TREND_MA_PERIOD,
    config.MACD_SLOW + config.MACD_SIGNAL,
    config.ADX_PERIOD * 3,
    config.VOLATILITY_PERIOD * 5,
)


# ─────────────────────────────────────────────────────────────────────────────

class WindowResult(NamedTuple):
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
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def _attach_hmm_features(df_raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """
    Compute base HMM features + whichever extended features the feature_set needs.
    Always adds the base 3 columns even if not in feature_set (log_return is
    used by _build_label_map and the backtester BH benchmark).

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

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-window analysis
# ─────────────────────────────────────────────────────────────────────────────

def _run_window(
    full_df:            pd.DataFrame,
    feature_cols:       list[str],
    train_start:        pd.Timestamp,
    test_start:         pd.Timestamp,
    test_end:           pd.Timestamp,
    chain_equity:       float,
    window_idx:         int,
    use_regime_mapper:  bool = False,
    use_ensemble:       bool = False,
) -> tuple[WindowResult | None, pd.Series, pd.Series, pd.DataFrame]:
    """
    Run one walk-forward window.  Returns:
      (WindowResult | None, oos_equity_curve, bh_equity_curve, trades_df)
    """
    empty = pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

    # ── Training slice ─────────────────────────────────────────────────────
    df_train = full_df[(full_df.index >= train_start) & (full_df.index < test_start)]
    if len(df_train) < 300:
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
    # Prepend _WARMUP bars from just before test_start so the backtester's
    # internal warmup skip eats exactly the buffer, leaving trading to start
    # at (approximately) test_start.
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
    df_bt = build_signal_series(df_bt, use_regime_mapper=use_regime_mapper)

    # ── Run backtester with chain equity as starting capital ───────────────
    saved_cap = config.INITIAL_CAPITAL
    config.INITIAL_CAPITAL = chain_equity
    try:
        res = Backtester(use_regime_mapper=use_regime_mapper).run(df_bt)
    finally:
        config.INITIAL_CAPITAL = saved_cap

    if res.equity_curve.empty:
        log.warning("Window %d: empty equity curve — skipping", window_idx + 1)
        return None, *empty

    # ── Extract test-period equity (filter out the buffer) ─────────────────
    # Make tz-aware for comparison
    def _tz(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts

    ts_start = _tz(test_start)
    ts_end   = _tz(test_end)

    eq_oos = res.equity_curve[
        (res.equity_curve.index >= ts_start) &
        (res.equity_curve.index <= ts_end)
    ]
    if eq_oos.empty:
        # Timestamp mismatch — use full backtester curve as fallback
        eq_oos = res.equity_curve

    # ── Test-period trades ─────────────────────────────────────────────────
    trades_oos = res.trades.copy()
    if not trades_oos.empty and "entry_time" in trades_oos.columns:
        entry_ts = pd.to_datetime(trades_oos["entry_time"])
        if entry_ts.dt.tz is None:
            entry_ts = entry_ts.dt.tz_localize("UTC")
        trades_oos = trades_oos[(entry_ts >= ts_start) & (entry_ts <= ts_end)]

    # ── BTC buy-and-hold equity for this test window ───────────────────────
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

    hr = eq_oos.pct_change().dropna()
    sharpe = (hr.mean() / hr.std() * np.sqrt(24 * 365)
              if len(hr) > 1 and hr.std() > 0 else 0.0)

    n_tr    = len(trades_oos)
    wr      = ((trades_oos["pnl"] > 0).sum() / n_tr * 100) if n_tr > 0 else 0.0

    wr = WindowResult(
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

def run_walk_forward(
    train_months:        int = 12,
    test_months:         int = 3,
    ticker:              str = config.TICKER,
    feature_set:         str = config.FEATURE_SET,
    confirmations:       int | None = None,
    confirmations_short: int | None = None,
    timeframe:           str = "1h",
    quiet:               bool = False,
    use_regime_mapper:   bool = False,
    use_ensemble:        bool = False,
) -> tuple[list[WindowResult], pd.Series, pd.Series, pd.DataFrame]:
    """
    Returns (window_results, combined_oos_equity, combined_bh_equity, all_trades).
    """
    feature_cols = config.FEATURE_SETS[feature_set]

    # Temporarily override confirmations if caller requested specific values
    _saved_confirms       = config.MIN_CONFIRMATIONS
    _saved_confirms_short = config.MIN_CONFIRMATIONS_SHORT
    _saved_taker_fee      = config.TAKER_FEE
    if confirmations is not None:
        config.MIN_CONFIRMATIONS = confirmations
    if confirmations_short is not None:
        config.MIN_CONFIRMATIONS_SHORT = confirmations_short

    # Apply tiered slippage for crypto tickers (Sprint 9)
    ticker_fee = config.get_ticker_fee(ticker)
    config.TAKER_FEE = ticker_fee

    if not quiet:
        print(f"\n{'═'*60}")
        print(f"  Walk-Forward Analysis")
        print(f"{'═'*60}")
        print(f"  Ticker       : {ticker}")
        print(f"  Timeframe    : {timeframe}")
        print(f"  Feature set  : {feature_set}  →  {feature_cols}")
        print(f"  N_STATES     : {config.N_STATES}    COV_TYPE : {config.COV_TYPE}")
        print(f"  Train window : {train_months} months")
        print(f"  Test window  : {test_months} months")
        if use_regime_mapper:
            print(f"  Confirmations: {config.MIN_CONFIRMATIONS} (LONG)  "
                  f"{config.MIN_CONFIRMATIONS_SHORT} (SHORT)   "
                  f"Leverage: {config.LEVERAGE}×")
            print(f"  Cooldown     : {config.COOLDOWN_HOURS}h (LONG)  "
                  f"{config.COOLDOWN_HOURS_SHORT}h (SHORT)   "
                  f"ADX min : {config.ADX_MIN} / {config.ADX_MIN_SHORT}")
        else:
            print(f"  Confirmations: {config.MIN_CONFIRMATIONS}   Leverage: {config.LEVERAGE}×")
            print(f"  Cooldown     : {config.COOLDOWN_HOURS}h")
        print()

    # ── Load full OHLCV + features once ────────────────────────────────────
    days_needed = max(730, (train_months + test_months + 2) * 31)
    if not quiet:
        print(f"Loading {days_needed}-day OHLCV …")
    raw_ohlcv = fetch_btc_hourly(days=days_needed, ticker=ticker)

    # Resample to target timeframe if not already 1h
    if timeframe != "1h":
        raw_ohlcv = resample_ohlcv(raw_ohlcv, timeframe)

    full_df   = _attach_hmm_features(raw_ohlcv, feature_set)
    full_df   = full_df.dropna(subset=feature_cols)

    data_start = full_df.index[0]
    data_end   = full_df.index[-1]
    if not quiet:
        print(f"  {len(full_df)} bars  ({data_start.date()} → {data_end.date()})")

    # ── Build rolling window list ──────────────────────────────────────────
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    test_start = data_start + pd.DateOffset(months=train_months)
    while True:
        test_end    = test_start + pd.DateOffset(months=test_months)
        train_start = test_start - pd.DateOffset(months=train_months)
        if test_end > data_end:
            break
        windows.append((train_start, test_start, test_end))
        test_start = test_end

    if not windows:
        avail = round((data_end - data_start).days / 30)
        if not quiet:
            print(f"\nERROR: Need {train_months + test_months} months of data; "
                  f"only {avail} available.")
        return [], pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

    if not quiet:
        print(f"  Windows: {len(windows)}  "
              f"(covering {windows[0][1].date()} → {windows[-1][2].date()})\n")

    # ── Run windows sequentially ───────────────────────────────────────────
    all_results:  list[WindowResult]   = []
    combined_eq:  pd.Series            = pd.Series(dtype=float)
    combined_bh:  pd.Series            = pd.Series(dtype=float)
    all_trades:   list[pd.DataFrame]   = []
    chain_equity: float                = config.INITIAL_CAPITAL

    for i, (train_start, test_start, test_end) in enumerate(windows):
        if not quiet:
            label = (f"  W{i+1:02d}  train [{train_start.date()}→{test_start.date()}]"
                     f"  test [{test_start.date()}→{test_end.date()}]")
            print(label, end=" … ", flush=True)

        result, eq_oos, bh_eq, trades_w = _run_window(
            full_df, feature_cols,
            train_start, test_start, test_end,
            chain_equity, i,
            use_regime_mapper=use_regime_mapper,
            use_ensemble=use_ensemble,
        )

        if result is None:
            if not quiet:
                print("skipped")
            continue

        all_results.append(result)
        all_trades.append(trades_w)
        combined_eq = pd.concat([combined_eq, eq_oos])
        combined_bh = pd.concat([combined_bh, bh_eq])
        chain_equity = result.end_equity

        if not quiet:
            print(
                f"return={result.return_pct:+.1f}%  "
                f"α={result.alpha_pct:+.1f}%  "
                f"sharpe={result.sharpe_ratio:.2f}  "
                f"DD={result.max_drawdown_pct:.1f}%  "
                f"trades={result.n_trades}"
            )

    config.MIN_CONFIRMATIONS       = _saved_confirms
    config.MIN_CONFIRMATIONS_SHORT = _saved_confirms_short
    config.TAKER_FEE               = _saved_taker_fee
    return all_results, combined_eq, combined_bh, pd.concat(all_trades) if all_trades else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: list[WindowResult], combined_eq: pd.Series) -> None:
    if not results:
        print("\nNo completed windows to summarise.")
        return

    sep = "─" * 88
    print(f"\n{'═'*88}")
    print(f"  WALK-FORWARD SUMMARY  ({len(results)} windows)")
    print(f"{'═'*88}")
    header = (f"  {'Win':>3}  {'Test period':>22}  "
              f"{'Return':>8}  {'B&H':>7}  {'Alpha':>7}  "
              f"{'MaxDD':>7}  {'Sharpe':>7}  {'Trades':>6}  {'WR':>5}")
    print(header)
    print(sep)

    for r in results:
        ret_color   = "+" if r.return_pct >= 0 else ""
        alpha_color = "+" if r.alpha_pct  >= 0 else ""
        print(
            f"  {r.window:>3}  "
            f"{r.test_start} → {r.test_end}  "
            f"{ret_color}{r.return_pct:>6.1f}%  "
            f"{r.bh_return_pct:>+6.1f}%  "
            f"{alpha_color}{r.alpha_pct:>5.1f}%  "
            f"{r.max_drawdown_pct:>6.1f}%  "
            f"{r.sharpe_ratio:>7.3f}  "
            f"{r.n_trades:>6}  "
            f"{r.win_rate_pct:>4.0f}%"
        )

    print(sep)

    # Aggregate OOS stats
    if not combined_eq.empty and len(combined_eq) > 1:
        oos_initial = combined_eq.iloc[0]
        oos_final   = combined_eq.iloc[-1]
        oos_ret     = (oos_final / oos_initial - 1) * 100
        roll_max    = combined_eq.cummax()
        oos_dd      = ((combined_eq - roll_max) / roll_max * 100).min()
        hr          = combined_eq.pct_change().dropna()
        oos_sharpe  = (hr.mean() / hr.std() * np.sqrt(24 * 365)
                       if hr.std() > 0 else 0.0)
        oos_wins    = sum(r.return_pct > 0 for r in results)
        total_tr    = sum(r.n_trades for r in results)

        print(f"\n  Combined OOS equity : ${oos_initial:,.0f} → ${oos_final:,.0f}")
        print(f"  Combined OOS return : {oos_ret:+.2f}%")
        print(f"  Combined OOS Sharpe : {oos_sharpe:.3f}")
        print(f"  Combined OOS max DD : {oos_dd:.2f}%")
        print(f"  Positive windows    : {oos_wins}/{len(results)}")
        print(f"  Total OOS trades    : {total_tr}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Chart generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_chart(
    results:     list[WindowResult],
    combined_eq: pd.Series,
    combined_bh: pd.Series,
    output:      Path,
) -> None:
    if not results or combined_eq.empty:
        print("No data to chart.")
        return

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.50, 0.25, 0.25],
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=(
            "Combined OOS Equity Curve",
            "Per-Window Return % vs Buy-and-Hold",
            "Per-Window Sharpe Ratio & Max Drawdown",
        ),
    )

    # ── Row 1: equity curve + BH benchmark ──────────────────────────────────
    fig.add_trace(go.Scatter(
        x=combined_eq.index, y=combined_eq.values,
        name="OOS Strategy",
        line=dict(color="#00ff88", width=1.8),
        hovertemplate="%{x}<br>Equity: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    if not combined_bh.empty:
        fig.add_trace(go.Scatter(
            x=combined_bh.index, y=combined_bh.values,
            name="BTC Buy-and-Hold",
            line=dict(color="#ffcc00", width=1.2, dash="dot"),
            hovertemplate="%{x}<br>B&H: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

    # Vertical window separators + shading
    colors = ["rgba(0,255,136,0.04)", "rgba(100,120,255,0.04)"]
    for i, r in enumerate(results):
        x0 = pd.Timestamp(r.test_start, tz="UTC")
        x1 = pd.Timestamp(r.test_end,   tz="UTC")
        clr = "#00ff88" if r.return_pct >= 0 else "#ff4455"
        # Thin vertical lines at window boundaries
        fig.add_vline(x=x0.timestamp() * 1000,
                      line=dict(color="#2a3040", width=1, dash="dot"),
                      row=1, col=1)
        # Window label annotation
        mid = x0 + (x1 - x0) / 2
        fig.add_annotation(
            x=mid, y=0.97,
            xref="x", yref="paper",
            text=f"W{r.window}",
            showarrow=False,
            font=dict(color="#6b7394", size=9, family="monospace"),
        )

    # ── Row 2: per-window return bars ────────────────────────────────────────
    x_labels = [f"W{r.window}<br>{r.test_start}" for r in results]

    fig.add_trace(go.Bar(
        x=x_labels,
        y=[r.return_pct for r in results],
        name="Strategy Return %",
        marker_color=[("#00ff88" if r.return_pct >= 0 else "#ff4455")
                      for r in results],
        hovertemplate="W%{x}<br>Return: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x_labels,
        y=[r.bh_return_pct for r in results],
        name="B&H Return %",
        marker_color="rgba(255,204,0,0.45)",
        hovertemplate="W%{x}<br>B&H: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # ── Row 3: Sharpe + drawdown bars ────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=x_labels,
        y=[r.sharpe_ratio for r in results],
        name="Sharpe Ratio",
        marker_color=[("#00aaff" if r.sharpe_ratio >= 0 else "#ff9900")
                      for r in results],
        hovertemplate="W%{x}<br>Sharpe: %{y:.3f}<extra></extra>",
        yaxis="y3",
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=x_labels,
        y=[r.max_drawdown_pct for r in results],
        name="Max Drawdown %",
        marker_color="rgba(255,68,85,0.5)",
        hovertemplate="W%{x}<br>Max DD: %{y:.1f}%<extra></extra>",
    ), row=3, col=1)

    # ── Summary annotation ────────────────────────────────────────────────────
    oos_ret  = (combined_eq.iloc[-1] / combined_eq.iloc[0] - 1) * 100
    hr       = combined_eq.pct_change().dropna()
    oos_sh   = (hr.mean() / hr.std() * np.sqrt(24 * 365)
                if hr.std() > 0 else 0.0)
    roll_max = combined_eq.cummax()
    oos_dd   = ((combined_eq - roll_max) / roll_max * 100).min()
    pos_wins = sum(r.return_pct > 0 for r in results)
    total_tr = sum(r.n_trades for r in results)

    summary_text = (
        f"OOS Return: {oos_ret:+.1f}%  |  "
        f"Sharpe: {oos_sh:.2f}  |  "
        f"Max DD: {oos_dd:.1f}%  |  "
        f"Positive windows: {pos_wins}/{len(results)}  |  "
        f"Total trades: {total_tr}"
    )

    fig.update_layout(
        height=820,
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141820",
        font=dict(family="monospace", color="#e0e4f0", size=11),
        legend=dict(
            bgcolor="#141820", bordercolor="#1e2330",
            x=0.01, y=0.99,
            font=dict(size=10),
        ),
        barmode="group",
        margin=dict(l=60, r=30, t=60, b=40),
        title=dict(
            text=f"Walk-Forward OOS Analysis — {summary_text}",
            font=dict(size=11, color="#a0c8ff"),
            x=0.01,
        ),
        xaxis=dict(gridcolor="#1e2330", showgrid=True),
        xaxis2=dict(gridcolor="#1e2330"),
        xaxis3=dict(gridcolor="#1e2330"),
        yaxis=dict(gridcolor="#1e2330", title="Equity ($)"),
        yaxis2=dict(gridcolor="#1e2330", title="Return (%)"),
        yaxis3=dict(gridcolor="#1e2330", title="%"),
    )

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font.size   = 11
        ann.font.color  = "#a0c8ff"
        ann.font.family = "monospace"

    fig.write_html(str(output), include_plotlyjs="cdn")
    print(f"Chart written → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def _save_csv(results: list[WindowResult], path: Path) -> None:
    if not results:
        return
    df = pd.DataFrame([r._asdict() for r in results])
    df.to_csv(path, index=False)
    print(f"Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM Trader — Walk-Forward Out-of-Sample Analysis"
    )
    parser.add_argument(
        "--train-months", type=int, default=12,
        help="Training window length in months (default: 12)",
    )
    parser.add_argument(
        "--test-months", type=int, default=3,
        help="Test window length in months (default: 3)",
    )
    parser.add_argument(
        "--ticker", default=config.TICKER,
        help=f"Polygon ticker (default: {config.TICKER})",
    )
    parser.add_argument(
        "--feature-set", default=config.FEATURE_SET,
        choices=list(config.FEATURE_SETS.keys()),
        help=f"HMM feature set (default: {config.FEATURE_SET})",
    )
    parser.add_argument(
        "--confirmations", type=int, default=None,
        help=f"Override MIN_CONFIRMATIONS for LONG (default: {config.MIN_CONFIRMATIONS} from config)",
    )
    parser.add_argument(
        "--confirmations-short", type=int, default=None,
        help=f"Override MIN_CONFIRMATIONS_SHORT (default: {config.MIN_CONFIRMATIONS_SHORT} from config)",
    )
    parser.add_argument(
        "--timeframe", default="1h",
        help="Candle timeframe: 1h, 2h, 3h, 4h (default: 1h)",
    )
    parser.add_argument(
        "--regime-mapper", action="store_true",
        help="Enable multi-direction trading (LONG/SHORT/FLAT)",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use 3-model HMM ensemble (n_states=[5,6,7])",
    )
    args = parser.parse_args()

    results, combined_eq, combined_bh, all_trades = run_walk_forward(
        train_months        = args.train_months,
        test_months         = args.test_months,
        ticker              = args.ticker,
        feature_set         = args.feature_set,
        confirmations       = args.confirmations,
        confirmations_short = args.confirmations_short,
        timeframe           = args.timeframe,
        use_regime_mapper   = args.regime_mapper,
        use_ensemble        = args.ensemble,
    )

    _print_summary(results, combined_eq)
    _save_csv(results, RESULTS_CSV)
    _build_chart(results, combined_eq, combined_bh, CHART_HTML)


if __name__ == "__main__":
    main()
