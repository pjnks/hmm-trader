"""
optimize_beryl_p3.py
────────────────────
BERYL Phase 3: Expanded NDX100 walk-forward optimizer.

Addresses gaps from Phase 2:
  1. Restores 'full' feature set for intraday timeframes
  2. Restores 8-confirmations gate
  3. Adds AAPL, MSFT, GOOGL (abandoned after Phase 1)
  4. Single-model only (ensemble hurts equities per Phase 2 analysis)
  5. Restores n_states=7,8 for intraday

Grid: 5 tickers x 5 n_states x 3 features x 3 confirms x 2 leverage x
      2 cooldown x 2 cov x 3 timeframe = 10,800 combos
Random sample: --runs N (default 200)

Usage
─────
  python optimize_beryl_p3.py --runs 200          # 200 random trials
  python optimize_beryl_p3.py --runs 200 --resume # resume
  python optimize_beryl_p3.py --heatmap-only      # regen heatmap
  python optimize_beryl_p3.py --workers 4         # 4 parallel workers
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import warnings
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from walk_forward_ndx import (
    fetch_equity_daily,
    fetch_equity_hourly,
    run_walk_forward_ndx,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer_beryl_p3")

# ── Output paths ─────────────────────────────────────────────────────────────
RESULTS_CSV  = ROOT / "beryl_p3_results.csv"
HEATMAP_HTML = ROOT / "beryl_p3_heatmap.html"

# ── Parameter grid (expanded from Phase 2) ───────────────────────────────────
GRID = {
    "ticker":          ["TSLA", "NVDA", "AAPL", "MSFT", "GOOGL"],
    "n_states":        [4, 5, 6, 7, 8],
    "feature_set":     ["base", "extended", "extended_v2", "full"],
    "confirmations":   [6, 7, 8],
    "leverage":        [1.0, 1.5],
    "cooldown_hours":  [48, 72],
    "covariance_type": ["full", "diag"],
    "timeframe":       ["1d", "2h", "4h"],
}


def _grid_size() -> int:
    size = 1
    for v in GRID.values():
        size *= len(v)
    return size


def _sample_combinations(n: int, seed: int = 42) -> list[dict]:
    """Sample n random combinations from the grid."""
    keys = list(GRID.keys())
    all_combos = list(product(*GRID.values()))
    rng = random.Random(seed)
    k = min(n, len(all_combos))
    return [dict(zip(keys, combo)) for combo in rng.sample(all_combos, k)]


def _run_trial(params: dict) -> dict | None:
    """
    Run one walk-forward trial. Returns results dict or None.

    Safe for multiprocessing: patches config in the worker's own memory space.
    Accepts a single dict (required for Pool.imap_unordered pickling).
    """
    ticker = params["ticker"]
    timeframe = params["timeframe"]

    # Patch config in this worker process (safe — separate memory space)
    config.N_STATES = params["n_states"]
    config.MIN_CONFIRMATIONS = params["confirmations"]
    config.LEVERAGE = params["leverage"]
    config.COOLDOWN_HOURS = params["cooldown_hours"]
    config.COV_TYPE = params["covariance_type"]
    config.FEATURE_SET = params["feature_set"]

    try:
        results, eq_curves, trades = run_walk_forward_ndx(
            tickers=[ticker],
            train_months=6,
            test_months=3,
            feature_set=params["feature_set"],
            quiet=True,
            use_ensemble=False,  # Single-model only for equities
            timeframe=timeframe,
        )

        if not results:
            return None

        total_return = (results[-1].end_equity / results[0].start_equity - 1) * 100
        mean_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        std_sharpe = (
            float(np.std([r.sharpe_ratio for r in results]))
            if len(results) > 1 else 0.0
        )
        positive_windows = sum(1 for r in results if r.sharpe_ratio > 0)
        total_trades = sum(r.n_trades for r in results)
        max_dd = min(r.max_drawdown_pct for r in results)

        return {
            **params,
            "wf_sharpe": round(mean_sharpe, 3),
            "wf_return": round(total_return, 2),
            "wf_drawdown": round(max_dd, 2),
            "wf_trades": total_trades,
            "wf_pos_windows": positive_windows,
            "wf_n_windows": len(results),
            "wf_std_sharpe": round(std_sharpe, 3),
        }

    except Exception as e:
        log.warning(f"Trial failed: {params} -- {e}")
        return None


def _generate_heatmap(results: list[dict]) -> None:
    """Generate interactive Plotly heatmap."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.DataFrame(results)
    if df.empty:
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Ticker x Timeframe (mean Sharpe)",
            "N_States x Feature Set (mean Sharpe)",
            "Confirmations x Leverage (mean Sharpe)",
            "Cooldown x Covariance (mean Sharpe)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    pairs = [
        ("ticker", "timeframe"),
        ("n_states", "feature_set"),
        ("confirmations", "leverage"),
        ("cooldown_hours", "covariance_type"),
    ]

    for idx, (y_col, x_col) in enumerate(pairs):
        r, c = divmod(idx, 2)
        p = df.groupby([y_col, x_col])["wf_sharpe"].mean().reset_index()
        hm = p.pivot(index=y_col, columns=x_col, values="wf_sharpe")

        fig.add_trace(
            go.Heatmap(
                z=hm.values,
                x=[str(v) for v in hm.columns],
                y=[str(v) for v in hm.index],
                colorscale="RdYlGn",
                text=np.round(hm.values, 3),
                texttemplate="%{text:.3f}",
                showscale=(idx == 3),
                colorbar=dict(title="Sharpe") if idx == 3 else None,
            ),
            row=r + 1, col=c + 1,
        )
        fig.update_xaxes(title_text=x_col, row=r + 1, col=c + 1)
        fig.update_yaxes(title_text=y_col, row=r + 1, col=c + 1)

    fig.update_layout(
        template="plotly_dark",
        height=800,
        width=1100,
        title="BERYL Phase 3 — Expanded NDX100 Walk-Forward Optimization",
    )
    fig.write_html(str(HEATMAP_HTML))
    print(f"  Heatmap: {HEATMAP_HTML}")


def main():
    parser = argparse.ArgumentParser(
        description="BERYL Phase 3: Expanded NDX100 Optimizer"
    )
    parser.add_argument("--runs", type=int, default=200,
                        help="Number of random trials (default: 200)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint CSV")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate heatmap from existing CSV")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help=f"Number of parallel workers (default: {max(1, cpu_count() - 2)})")
    args = parser.parse_args()

    total_grid = _grid_size()
    print(f"\n  BERYL Phase 3 Optimizer")
    print(f"  Grid: {total_grid} total combinations")
    print(f"  Sampling: {args.runs} trials ({100*args.runs/total_grid:.1f}% coverage)")
    print(f"  Workers: {args.workers}")

    if args.heatmap_only:
        if RESULTS_CSV.exists():
            results = pd.read_csv(RESULTS_CSV).to_dict("records")
            _generate_heatmap(results)
        return

    combos = _sample_combinations(args.runs)
    print(f"  Tickers: {GRID['ticker']}")
    print(f"  Features: {GRID['feature_set']}")
    print(f"  Confirmations: {GRID['confirmations']}")
    print(f"  Timeframes: {GRID['timeframe']}")

    # Check for resume
    existing: list[dict] = []
    completed_keys: set = set()
    if args.resume and RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV).to_dict("records")
        for row in existing:
            key = tuple(str(row.get(k, "")) for k in GRID.keys())
            completed_keys.add(key)
        print(f"  Resuming: {len(existing)} trials done")

    remaining = []
    for combo in combos:
        key = tuple(str(combo.get(k, "")) for k in GRID.keys())
        if key not in completed_keys:
            remaining.append(combo)

    print(f"  Remaining: {len(remaining)} trials\n")

    if not remaining:
        print("  All trials complete!")
        _generate_heatmap(existing)
        return

    # Run trials
    results = list(existing)
    valid = 0
    skipped = 0
    unsaved_results: list[dict] = []

    write_header = not RESULTS_CSV.exists() or not args.resume

    if args.workers > 1:
        # ── Parallel execution ────────────────────────────────────────────
        print(f"  Starting {args.workers} parallel workers...\n")

        # Open CSV for batch writing
        csvfile = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
        writer = None

        try:
            with Pool(args.workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_run_trial, remaining),
                    total=len(remaining),
                    desc="BERYL P3",
                ):
                    if result is not None:
                        results.append(result)
                        valid += 1

                        if writer is None:
                            writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                            if write_header:
                                writer.writeheader()
                        writer.writerow(result)

                        if valid % 5 == 0:
                            csvfile.flush()
                    else:
                        skipped += 1
        finally:
            csvfile.close()

    else:
        # ── Sequential execution (--workers 1) ────────────────────────────
        csvfile = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
        writer = None

        try:
            for i, combo in enumerate(tqdm(remaining, desc="BERYL P3")):
                result = _run_trial(combo)

                if result is not None:
                    results.append(result)
                    valid += 1

                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                        if write_header:
                            writer.writeheader()
                            write_header = False
                    writer.writerow(result)

                    if valid % 5 == 0:
                        csvfile.flush()
                else:
                    skipped += 1

        finally:
            csvfile.close()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  BERYL PHASE 3 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Valid: {valid}, Skipped: {skipped}, Workers: {args.workers}")

    if results:
        df = pd.DataFrame(results)
        top = df.nlargest(10, "wf_sharpe")
        print(f"\n  TOP 10:")
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            print(
                f"  #{rank:2d}  {r['ticker']:5s} {r['timeframe']:3s} "
                f"n={int(r['n_states'])} {r['feature_set']:8s} "
                f"cf={int(r['confirmations'])} lev={r['leverage']:.1f} "
                f"cd={int(r['cooldown_hours'])} {r['covariance_type']:4s} "
                f"-> Sharpe {r['wf_sharpe']:+.3f}  "
                f"Return {r['wf_return']:+.1f}%"
            )

        # Per-ticker best
        print(f"\n  BEST PER TICKER:")
        for ticker in GRID["ticker"]:
            ticker_df = df[df["ticker"] == ticker]
            if not ticker_df.empty:
                best = ticker_df.nlargest(1, "wf_sharpe").iloc[0]
                print(
                    f"  {ticker:5s}: Sharpe {best['wf_sharpe']:+.3f}  "
                    f"Return {best['wf_return']:+.1f}%  "
                    f"({best['timeframe']}/{int(best['n_states'])}st/"
                    f"{best['feature_set']}/{int(best['confirmations'])}cf)"
                )

        _generate_heatmap(results)

    # Notification
    try:
        from src.notifier import Notifier
        n = Notifier()
        if results:
            best = max(results, key=lambda r: r.get("wf_sharpe", -999))
            n.send_notification(
                "BERYL Phase 3 Complete",
                f"Best: {best['ticker']} Sharpe {best['wf_sharpe']:.3f} | {args.workers} workers",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
