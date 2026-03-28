"""
optimize_allocator.py
─────────────────────
CITRINE allocator parameter optimizer. Grid-searches entry/exit confidence,
persistence, cash bands, scaling, and max positions by replaying daily
rebalancing with pre-computed HMM regime data.

Strategy: Pre-compute HMMs and daily scans ONCE (~15 min), then replay
the daily allocation loop for each parameter combo (~0.2s per trial).

Usage
─────
  python optimize_allocator.py                     # exhaustive grid (~3,500 combos)
  python optimize_allocator.py --resume            # resume from checkpoint
  python optimize_allocator.py --heatmap-only      # regenerate heatmap from CSV
  python optimize_allocator.py --tickers AAPL,TSLA # custom ticker subset
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.citrine_scanner import TickerScan
from src.citrine_allocator import CitrineAllocator, PortfolioWeight
from citrine_backtest import (
    CitrineBacktester,
    _WARMUP, _TAKER_FEE, _SLIPPAGE,
)
from walk_forward_ndx import fetch_equity_daily

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimize_allocator")

# ── Output paths ─────────────────────────────────────────────────────────────
RESULTS_CSV  = ROOT / "allocator_optimization_results.csv"
HEATMAP_HTML = ROOT / "allocator_optimization_heatmap.html"

# ── Default tickers (39 positive-Sharpe from CITRINE winner) ─────────────────
DEFAULT_TICKERS = [
    "CTAS", "CRWD", "MPWR", "TXN", "MAR", "ASML", "PANW", "INSM", "ROST",
    "NVDA", "EXC", "AXON", "MSFT", "QCOM", "TSLA", "MU", "CEG", "TMUS",
    "GEHC", "MCHP", "GOOGL", "FER", "COST", "INTC", "AMZN", "PLTR", "ADSK",
    "PCAR", "CPRT", "WDC", "BKR", "APP", "CCEP", "ROP", "HON", "WMT",
    "FAST", "CSCO", "XEL",
]

# ── Parameter Grid ───────────────────────────────────────────────────────────
ENTRY_CONFS  = [0.70, 0.75, 0.80, 0.85, 0.90]
EXIT_CONFS   = [0.50, 0.55, 0.60, 0.65, 0.70]
PERSIST_DAYS = [1, 2, 3, 5]
MAX_POS      = [8, 10, 15, 20]

CASH_PROFILES = {
    "defensive":  [(15, 999, 0.15), (8, 14, 0.20), (3, 7, 0.30), (0, 2, 0.70)],
    "balanced":   [(15, 999, 0.10), (8, 14, 0.15), (3, 7, 0.25), (0, 2, 0.50)],
    "aggressive": [(15, 999, 0.05), (8, 14, 0.10), (3, 7, 0.15), (0, 2, 0.30)],
}

SCALE_PROFILES = {
    "slow":    {1: 0.25, 3: 0.50, 5: 1.00},
    "default": {1: 0.33, 2: 0.66, 4: 1.00},
    "fast":    {1: 0.50, 2: 1.00},
}


def build_grid() -> list[dict]:
    """Build exhaustive parameter grid (filtering entry <= exit)."""
    combos = []
    for entry_c, exit_c, persist, max_p, cash_name, scale_name in itertools.product(
        ENTRY_CONFS, EXIT_CONFS, PERSIST_DAYS, MAX_POS,
        CASH_PROFILES.keys(), SCALE_PROFILES.keys(),
    ):
        if entry_c <= exit_c:
            continue
        combos.append({
            "entry_confidence": entry_c,
            "exit_confidence": exit_c,
            "persistence_days": persist,
            "max_positions": max_p,
            "cash_profile": cash_name,
            "scale_profile": scale_name,
        })
    return combos


# ── Pre-computation ──────────────────────────────────────────────────────────

@dataclass
class WindowCache:
    """Pre-computed data for one walk-forward window."""
    window_idx:     int
    test_start:     pd.Timestamp
    test_end:       pd.Timestamp
    trading_days:   list
    daily_scans:    dict          # {Timestamp: list[TickerScan]}
    test_data:      dict          # {ticker: pd.DataFrame}
    qqq_return:     float         # QQQ buy-and-hold %


def precompute(
    tickers: list[str],
    train_months: int = 6,
    test_months: int = 3,
) -> list[WindowCache]:
    """
    One-time expensive computation.
    Fetches data, fits HMMs, generates all daily scans.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: Pre-computing HMMs and daily scans (one-time)")
    print("=" * 70)

    bt = CitrineBacktester(tickers=tickers, long_only=True, quiet=False)

    # Pre-cache all data
    all_data = bt._precache_data()
    if not all_data:
        print("ERROR: No data available")
        return []

    print("  Fetching QQQ benchmark ...", end=" ", flush=True)
    qqq_data = fetch_equity_daily("QQQ", years=20)
    print(f"{len(qqq_data)} bars")

    # Build windows
    start_dates, end_dates = [], []
    for ticker, df in all_data.items():
        if len(df) > _WARMUP + 50:
            start_dates.append(df.index[0])
            end_dates.append(df.index[-1])

    if not start_dates:
        print("ERROR: Insufficient data")
        return []

    data_start = max(start_dates)
    data_end = min(end_dates)
    print(f"\n  Data range: {data_start.date()} → {data_end.date()}")

    windows = CitrineBacktester._build_windows(
        data_start, data_end, train_months, test_months
    )
    print(f"  Windows: {len(windows)}")

    # For each window: fit HMMs, prepare test data, pre-generate all scans
    caches: list[WindowCache] = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n  -- Window {i+1}/{len(windows)} --")
        print(f"    Train: {train_start.date()} -> {train_end.date()}")
        print(f"    Test:  {test_start.date()} -> {test_end.date()}")

        models = bt._fit_models(all_data, train_start, train_end)
        if not models:
            print("    SKIP: no converged models")
            continue

        test_data = bt._prepare_test_data(
            all_data, models, train_start, test_start, test_end
        )

        # Trading days
        all_dates: set[pd.Timestamp] = set()
        for ticker, df in test_data.items():
            mask = (df.index >= test_start) & (df.index < test_end)
            all_dates.update(df.index[mask])
        trading_days = sorted(all_dates)

        if not trading_days:
            print("    SKIP: no trading days")
            continue

        # Pre-generate scans for every day
        daily_scans: dict = {}
        for day in trading_days:
            daily_scans[day] = bt._generate_scans_for_day(test_data, models, day)

        # QQQ benchmark
        qqq_mask = (qqq_data.index >= test_start) & (qqq_data.index < test_end)
        qqq_test = qqq_data[qqq_mask]
        qqq_ret = 0.0
        if len(qqq_test) >= 2:
            qqq_ret = (
                float(qqq_test["Close"].iloc[-1])
                / float(qqq_test["Close"].iloc[0])
                - 1
            ) * 100

        caches.append(WindowCache(
            window_idx=i + 1,
            test_start=test_start,
            test_end=test_end,
            trading_days=trading_days,
            daily_scans=daily_scans,
            test_data=test_data,
            qqq_return=qqq_ret,
        ))

        print(
            f"    Trading days: {len(trading_days)}, "
            f"Tickers with data: {len(test_data)}"
        )

    print(f"\n  Pre-computation complete: {len(caches)} windows ready")
    return caches


# ── Trial Execution ──────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Results for one allocator optimization trial."""
    trial_id:            int
    entry_confidence:    float
    exit_confidence:     float
    persistence_days:    int
    max_positions:       int
    cash_profile:        str
    scale_profile:       str
    total_return_pct:    float
    mean_sharpe:         float
    std_sharpe:          float
    mean_alpha_pct:      float
    positive_windows:    int
    total_windows:       int
    max_drawdown_pct:    float
    total_trades:        int
    avg_positions:       float
    avg_cash_pct:        float
    window_sharpes:      str
    window_returns:      str
    window_alphas:       str


def _patch_config(params: dict) -> dict:
    """Monkey-patch config with trial params. Returns originals for restore."""
    originals = {}

    simple = {
        "entry_confidence": "CITRINE_ENTRY_CONFIDENCE",
        "exit_confidence":  "CITRINE_EXIT_CONFIDENCE",
        "persistence_days": "CITRINE_PERSISTENCE_DAYS",
        "max_positions":    "CITRINE_MAX_POSITIONS",
    }
    for key, attr in simple.items():
        originals[attr] = getattr(config, attr)
        setattr(config, attr, params[key])

    originals["CITRINE_CASH_BANDS"] = config.CITRINE_CASH_BANDS
    config.CITRINE_CASH_BANDS = CASH_PROFILES[params["cash_profile"]]

    originals["CITRINE_SCALE_SCHEDULE"] = config.CITRINE_SCALE_SCHEDULE
    config.CITRINE_SCALE_SCHEDULE = SCALE_PROFILES[params["scale_profile"]]

    return originals


def _restore_config(originals: dict) -> None:
    """Restore config to original values."""
    for attr, value in originals.items():
        setattr(config, attr, value)


def run_trial(
    caches: list[WindowCache],
    params: dict,
    trial_id: int,
) -> TrialResult:
    """
    Run one trial with given allocator params.
    Replays daily rebalancing using pre-computed scans.
    """
    originals = _patch_config(params)

    try:
        window_sharpes: list[float] = []
        window_returns: list[float] = []
        window_alphas: list[float] = []
        window_drawdowns: list[float] = []
        total_trades = 0
        all_position_counts: list[int] = []
        all_cash_pcts: list[float] = []

        chain_equity = config.CITRINE_INITIAL_CAPITAL

        for cache in caches:
            equity = chain_equity
            cash = chain_equity
            positions: dict[str, dict] = {}
            daily_returns: list[float] = []
            equity_values: list[float] = []
            position_counts: list[int] = []
            cash_pcts: list[float] = []
            prev_equity = equity
            window_trades = 0

            allocator = CitrineAllocator(
                capital=chain_equity,
                long_only=True,
                cooldown_mode="none",
            )

            for day in cache.trading_days:
                # Mark-to-market
                mtm = cash
                for ticker, pos in positions.items():
                    if ticker in cache.test_data and day in cache.test_data[ticker].index:
                        price = float(cache.test_data[ticker].loc[day, "Close"])
                        if pos["direction"] == "LONG":
                            mtm += pos["shares"] * price
                        else:
                            mtm += pos["notional"] + (pos["entry_price"] - price) * pos["shares"]
                equity = mtm

                # Get pre-computed scans
                scans = cache.daily_scans.get(day, [])

                # Allocate
                weights, _ = allocator.allocate(scans)

                # Execute rebalance (inlined for speed)
                new_positions: dict[str, dict] = {}
                new_cash = cash
                trades = 0

                # Exits first
                for w in weights:
                    if w.action == "EXIT" and w.ticker in positions:
                        pos = positions[w.ticker]
                        if w.ticker in cache.test_data and day in cache.test_data[w.ticker].index:
                            exit_price = float(cache.test_data[w.ticker].loc[day, "Close"])
                            if pos["direction"] == "LONG":
                                exit_price *= (1 - _SLIPPAGE / 10000)
                            else:
                                exit_price *= (1 + _SLIPPAGE / 10000)
                            if pos["direction"] == "LONG":
                                pnl = pos["shares"] * (exit_price - pos["entry_price"])
                            else:
                                pnl = pos["shares"] * (pos["entry_price"] - exit_price)
                            fee = pos["notional"] * _TAKER_FEE
                            pnl -= fee
                            new_cash += pos["notional"] + pnl
                            trades += 1

                # Entries / holds
                for w in weights:
                    if w.action in ("ENTER", "SCALE_UP"):
                        if w.ticker in cache.test_data and day in cache.test_data[w.ticker].index:
                            entry_price = float(cache.test_data[w.ticker].loc[day, "Close"])
                            if w.direction == "LONG":
                                entry_price *= (1 + _SLIPPAGE / 10000)
                            else:
                                entry_price *= (1 - _SLIPPAGE / 10000)
                            notional = w.notional_usd

                            # Close old position if scaling up
                            if w.ticker in positions:
                                old = positions[w.ticker]
                                old_price = float(cache.test_data[w.ticker].loc[day, "Close"])
                                if old["direction"] == "LONG":
                                    new_cash += old["shares"] * old_price
                                else:
                                    new_cash += old["notional"] + (old["entry_price"] - old_price) * old["shares"]

                            fee = notional * _TAKER_FEE
                            if new_cash >= notional + fee:
                                shares = notional / entry_price
                                new_cash -= (notional + fee)
                                new_positions[w.ticker] = {
                                    "shares": shares,
                                    "entry_price": entry_price,
                                    "direction": w.direction,
                                    "notional": notional,
                                }
                                if w.action == "ENTER":
                                    trades += 1

                    elif w.action == "HOLD":
                        if w.ticker in positions:
                            new_positions[w.ticker] = positions[w.ticker]

                window_trades += trades
                positions = new_positions
                cash = new_cash

                # Final MTM
                final_eq = cash
                for ticker, pos in positions.items():
                    if ticker in cache.test_data and day in cache.test_data[ticker].index:
                        price = float(cache.test_data[ticker].loc[day, "Close"])
                        if pos["direction"] == "LONG":
                            final_eq += pos["shares"] * price
                        else:
                            final_eq += pos["notional"] + (pos["entry_price"] - price) * pos["shares"]
                equity = final_eq
                equity_values.append(equity)

                if prev_equity > 0:
                    daily_returns.append((equity - prev_equity) / prev_equity)
                prev_equity = equity

                position_counts.append(len(positions))
                cash_pcts.append(cash / equity if equity > 0 else 1.0)

                allocator.update_holdings(weights)

            # ── Window metrics ────────────────────────────────────────────
            ret = (equity / chain_equity - 1) * 100

            if len(daily_returns) > 1:
                mean_r = np.mean(daily_returns)
                std_r = np.std(daily_returns, ddof=1)
                sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
            else:
                sharpe = 0.0

            alpha = ret - cache.qqq_return

            if equity_values:
                eq_arr = np.array(equity_values)
                running_max = np.maximum.accumulate(eq_arr)
                dd = (eq_arr - running_max) / running_max * 100
                max_dd = float(dd.min())
            else:
                max_dd = 0.0

            window_sharpes.append(sharpe)
            window_returns.append(ret)
            window_alphas.append(alpha)
            window_drawdowns.append(max_dd)
            total_trades += window_trades
            all_position_counts.extend(position_counts)
            all_cash_pcts.extend(cash_pcts)

            chain_equity = equity

        # ── Aggregate metrics ─────────────────────────────────────────────
        total_return = (chain_equity / config.CITRINE_INITIAL_CAPITAL - 1) * 100
        mean_sharpe = float(np.mean(window_sharpes)) if window_sharpes else 0.0
        std_sharpe = float(np.std(window_sharpes)) if len(window_sharpes) > 1 else 0.0
        mean_alpha = float(np.mean(window_alphas)) if window_alphas else 0.0
        pos_windows = sum(1 for r in window_returns if r > 0)
        max_dd = float(min(window_drawdowns)) if window_drawdowns else 0.0

        return TrialResult(
            trial_id=trial_id,
            entry_confidence=params["entry_confidence"],
            exit_confidence=params["exit_confidence"],
            persistence_days=params["persistence_days"],
            max_positions=params["max_positions"],
            cash_profile=params["cash_profile"],
            scale_profile=params["scale_profile"],
            total_return_pct=round(total_return, 2),
            mean_sharpe=round(mean_sharpe, 3),
            std_sharpe=round(std_sharpe, 3),
            mean_alpha_pct=round(mean_alpha, 2),
            positive_windows=pos_windows,
            total_windows=len(caches),
            max_drawdown_pct=round(max_dd, 2),
            total_trades=total_trades,
            avg_positions=round(float(np.mean(all_position_counts)), 1) if all_position_counts else 0.0,
            avg_cash_pct=round(float(np.mean(all_cash_pcts)) * 100, 1) if all_cash_pcts else 100.0,
            window_sharpes=",".join(f"{s:.3f}" for s in window_sharpes),
            window_returns=",".join(f"{r:.2f}" for r in window_returns),
            window_alphas=",".join(f"{a:.2f}" for a in window_alphas),
        )

    finally:
        _restore_config(originals)


# ── Heatmap ──────────────────────────────────────────────────────────────────

def generate_heatmap(results_csv: Path, output_html: Path) -> None:
    """Generate interactive heatmaps from optimization results."""
    df = pd.read_csv(results_csv)

    if df.empty:
        print("  No results to plot")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Entry x Exit Confidence (mean Sharpe)",
            "Persistence x Max Positions (mean Sharpe)",
            "Cash x Scale Profile (mean Sharpe)",
            "Return vs Sharpe (all trials)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # 1. Entry x Exit
    p1 = df.groupby(["entry_confidence", "exit_confidence"])["mean_sharpe"].mean().reset_index()
    hm1 = p1.pivot(index="exit_confidence", columns="entry_confidence", values="mean_sharpe")
    fig.add_trace(
        go.Heatmap(
            z=hm1.values,
            x=[str(c) for c in hm1.columns],
            y=[str(r) for r in hm1.index],
            colorscale="RdYlGn",
            text=np.round(hm1.values, 3),
            texttemplate="%{text:.3f}",
            showscale=False,
        ),
        row=1, col=1,
    )

    # 2. Persistence x Max Positions
    p2 = df.groupby(["persistence_days", "max_positions"])["mean_sharpe"].mean().reset_index()
    hm2 = p2.pivot(index="persistence_days", columns="max_positions", values="mean_sharpe")
    fig.add_trace(
        go.Heatmap(
            z=hm2.values,
            x=[str(c) for c in hm2.columns],
            y=[str(r) for r in hm2.index],
            colorscale="RdYlGn",
            text=np.round(hm2.values, 3),
            texttemplate="%{text:.3f}",
            showscale=False,
        ),
        row=1, col=2,
    )

    # 3. Cash x Scale
    p3 = df.groupby(["cash_profile", "scale_profile"])["mean_sharpe"].mean().reset_index()
    hm3 = p3.pivot(index="cash_profile", columns="scale_profile", values="mean_sharpe")
    fig.add_trace(
        go.Heatmap(
            z=hm3.values,
            x=[str(c) for c in hm3.columns],
            y=[str(r) for r in hm3.index],
            colorscale="RdYlGn",
            text=np.round(hm3.values, 3),
            texttemplate="%{text:.3f}",
            colorbar=dict(title="Sharpe", len=0.4, y=0.2),
        ),
        row=2, col=1,
    )

    # 4. Scatter: Return vs Sharpe
    fig.add_trace(
        go.Scatter(
            x=df["mean_sharpe"],
            y=df["total_return_pct"],
            mode="markers",
            marker=dict(
                size=4,
                color=df["positive_windows"] / df["total_windows"],
                colorscale="RdYlGn",
                opacity=0.6,
            ),
            text=[
                f"entry={r['entry_confidence']:.2f} exit={r['exit_confidence']:.2f}<br>"
                f"persist={r['persistence_days']} maxpos={r['max_positions']}<br>"
                f"cash={r['cash_profile']} scale={r['scale_profile']}"
                for _, r in df.iterrows()
            ],
            hovertemplate="%{text}<br>Sharpe=%{x:.3f}<br>Return=%{y:.1f}%<extra></extra>",
        ),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Entry Confidence", row=1, col=1)
    fig.update_yaxes(title_text="Exit Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Max Positions", row=1, col=2)
    fig.update_yaxes(title_text="Persistence Days", row=1, col=2)
    fig.update_xaxes(title_text="Scale Profile", row=2, col=1)
    fig.update_yaxes(title_text="Cash Profile", row=2, col=1)
    fig.update_xaxes(title_text="Mean Sharpe", row=2, col=2)
    fig.update_yaxes(title_text="Total Return %", row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        width=1200,
        title="CITRINE Allocator Optimization — Parameter Heatmaps",
        showlegend=False,
    )

    fig.write_html(str(output_html))
    print(f"  Heatmap: {output_html}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CITRINE Allocator Parameter Optimizer"
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated tickers (default: 39 positive-Sharpe)",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint CSV")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate heatmap from existing CSV")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    args = parser.parse_args()

    if args.heatmap_only:
        generate_heatmap(RESULTS_CSV, HEATMAP_HTML)
        return

    tickers = args.tickers.split(",") if args.tickers else DEFAULT_TICKERS

    # Build grid
    grid = build_grid()
    print(f"\n  Parameter grid: {len(grid)} valid combinations")

    # Check for resume
    completed: set = set()
    if args.resume and RESULTS_CSV.exists():
        df_done = pd.read_csv(RESULTS_CSV)
        for _, row in df_done.iterrows():
            key = (
                row["entry_confidence"], row["exit_confidence"],
                row["persistence_days"], row["max_positions"],
                row["cash_profile"], row["scale_profile"],
            )
            completed.add(key)
        print(f"  Resuming: {len(completed)} trials already complete")

    remaining = []
    for params in grid:
        key = (
            params["entry_confidence"], params["exit_confidence"],
            params["persistence_days"], params["max_positions"],
            params["cash_profile"], params["scale_profile"],
        )
        if key not in completed:
            remaining.append(params)

    print(f"  Remaining: {len(remaining)} trials")

    if not remaining:
        print("  All trials complete! Generating heatmap...")
        generate_heatmap(RESULTS_CSV, HEATMAP_HTML)
        return

    # Pre-compute (one-time, expensive)
    caches = precompute(tickers, args.train_months, args.test_months)
    if not caches:
        print("ERROR: Pre-computation failed")
        return

    # Run grid search
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: Grid Search ({len(remaining)} trials)")
    print(f"{'=' * 70}")

    results: list[TrialResult] = []
    start_time = time.time()

    write_header = not RESULTS_CSV.exists() or not args.resume
    csvfile = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
    writer = None

    try:
        for i, params in enumerate(remaining):
            trial_id = len(completed) + i + 1

            result = run_trial(caches, params, trial_id)
            results.append(result)

            # Write to CSV incrementally
            row = asdict(result)
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                    write_header = False
            writer.writerow(row)

            # Progress
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0

            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"  [{trial_id:4d}/{len(grid)}] "
                    f"entry={params['entry_confidence']:.2f} "
                    f"exit={params['exit_confidence']:.2f} "
                    f"persist={params['persistence_days']} "
                    f"maxpos={params['max_positions']} "
                    f"cash={params['cash_profile']:10s} "
                    f"scale={params['scale_profile']:7s} "
                    f"-> Sharpe {result.mean_sharpe:+.3f}  "
                    f"Return {result.total_return_pct:+.1f}%  "
                    f"({rate:.1f}/s, ETA {eta:.0f}s)"
                )

            # Flush periodically
            if (i + 1) % 100 == 0:
                csvfile.flush()

    finally:
        csvfile.close()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Trials: {len(results)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    if elapsed > 0:
        print(f"  Rate: {len(results) / elapsed:.1f} trials/s")

    # Load all results
    df_all = pd.read_csv(RESULTS_CSV)

    # Top 10
    top = df_all.nlargest(10, "mean_sharpe")
    print(f"\n  {'─' * 60}")
    print(f"  TOP 10 CONFIGS BY MEAN SHARPE")
    print(f"  {'─' * 60}")
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        print(
            f"  #{rank:2d}  entry={r['entry_confidence']:.2f} "
            f"exit={r['exit_confidence']:.2f} "
            f"persist={int(r['persistence_days'])} "
            f"maxpos={int(r['max_positions'])} "
            f"cash={r['cash_profile']:10s} "
            f"scale={r['scale_profile']:7s}"
        )
        print(
            f"       Sharpe {r['mean_sharpe']:+.3f}  "
            f"Return {r['total_return_pct']:+.1f}%  "
            f"Alpha {r['mean_alpha_pct']:+.1f}%  "
            f"+Win {int(r['positive_windows'])}/{int(r['total_windows'])}  "
            f"DD {r['max_drawdown_pct']:.1f}%  "
            f"Trades {int(r['total_trades'])}"
        )

    # Best all-windows-positive
    consistent = df_all[df_all["positive_windows"] == df_all["total_windows"]]
    if not consistent.empty:
        best_c = consistent.nlargest(5, "mean_sharpe")
        print(f"\n  {'─' * 60}")
        print(f"  TOP 5 ALL-WINDOWS-POSITIVE")
        print(f"  {'─' * 60}")
        for rank, (_, r) in enumerate(best_c.iterrows(), 1):
            print(
                f"  #{rank:2d}  entry={r['entry_confidence']:.2f} "
                f"exit={r['exit_confidence']:.2f} "
                f"persist={int(r['persistence_days'])} "
                f"maxpos={int(r['max_positions'])} "
                f"cash={r['cash_profile']:10s} "
                f"scale={r['scale_profile']:7s}"
            )
            print(
                f"       Sharpe {r['mean_sharpe']:+.3f}  "
                f"Return {r['total_return_pct']:+.1f}%  "
                f"Alpha {r['mean_alpha_pct']:+.1f}%  "
                f"DD {r['max_drawdown_pct']:.1f}%"
            )
    else:
        print("\n  No configs with all windows positive")

    # Stats
    positive = df_all[df_all["mean_sharpe"] > 0]
    print(
        f"\n  Positive Sharpe: {len(positive)}/{len(df_all)} "
        f"({100 * len(positive) / len(df_all):.0f}%)"
    )
    print(f"  Mean Sharpe: {df_all['mean_sharpe'].mean():.3f}")
    print(f"  Max Sharpe:  {df_all['mean_sharpe'].max():.3f}")
    print(f"  Saved: {RESULTS_CSV}")

    # Generate heatmap
    generate_heatmap(RESULTS_CSV, HEATMAP_HTML)

    # Send notification
    try:
        from src.notifier import Notifier
        n = Notifier()
        best = top.iloc[0]
        n.send_notification(
            "CITRINE Allocator Optimization Complete",
            f"Best Sharpe {best['mean_sharpe']:.3f} | "
            f"Return {best['total_return_pct']:+.1f}% | "
            f"entry={best['entry_confidence']:.2f} "
            f"exit={best['exit_confidence']:.2f}",
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
