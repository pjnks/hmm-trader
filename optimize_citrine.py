"""
optimize_citrine.py
───────────────────
Per-ticker HMM parameter optimization for CITRINE portfolio rotation.

Tests each ticker across a grid of HMM configurations using walk-forward
analysis, then saves the best config per ticker to citrine_per_ticker_configs.json.
The CITRINE scanner loads this file to use optimized params instead of defaults.

Grid per ticker:
  n_states(3) × feature_set(2) × confirmations(3) × cov_type(2) = 36 combos
  With 100 tickers: 3,600 total (random sample via --runs)

Also tests portfolio-level params:
  cooldown_mode(3) × long_only(2) = 6 portfolio configs
  Each run as a full portfolio backtest with per-ticker optimized HMM params.

Usage
─────
  python optimize_citrine.py --runs 200           # random sample
  python optimize_citrine.py --resume             # resume from checkpoint
  python optimize_citrine.py --ticker AAPL        # single ticker
  python optimize_citrine.py --heatmap-only       # regenerate heatmap
  python optimize_citrine.py --workers 4          # use 4 parallel workers
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.notifier import _macos_notify, _pushover_notify

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("citrine_opt")


# ── Grid Definition ────────────────────────────────────────────────────────

TICKER_GRID = {
    "n_states":      [4, 5, 6],
    "feature_set":   ["base", "extended", "extended_v2"],
    "confirmations": [6, 7, 8],
    "cov_type":      ["diag", "full"],
}

# Total combos per ticker: 3 × 3 × 3 × 2 = 54
_COMBOS_PER_TICKER = 1
for v in TICKER_GRID.values():
    _COMBOS_PER_TICKER *= len(v)

PORTFOLIO_GRID = {
    "cooldown_mode": ["none", "time", "threshold"],
    "long_only":     [False, True],
}


# ── Results file ───────────────────────────────────────────────────────────
RESULTS_CSV = ROOT / "citrine_optimization_results.csv"
PER_TICKER_JSON = ROOT / "citrine_per_ticker_configs.json"


def _build_trial_combos(tickers: list[str]) -> list[dict]:
    """Build all possible trial combinations (ticker × grid)."""
    combos = []
    keys = list(TICKER_GRID.keys())
    for ticker in tickers:
        for values in itertools.product(*TICKER_GRID.values()):
            trial = {"ticker": ticker}
            for k, v in zip(keys, values):
                trial[k] = v
            combos.append(trial)
    return combos


def _run_single_trial(trial: dict) -> dict | None:
    """
    Run walk-forward analysis for a single ticker with given HMM params.

    Safe for multiprocessing: patches config in the worker's own memory space.
    Returns dict with metrics or None if the trial fails.
    """
    ticker = trial["ticker"]
    n_states = trial["n_states"]
    feature_set = trial["feature_set"]
    confirmations = trial["confirmations"]
    cov_type = trial["cov_type"]

    try:
        from walk_forward_ndx import (
            fetch_equity_daily,
            _attach_hmm_features,
            run_walk_forward_ndx,
        )

        # Patch config in this worker process (safe — separate memory space)
        config.N_STATES = n_states
        config.FEATURE_SET = feature_set
        config.MIN_CONFIRMATIONS = confirmations
        config.COV_TYPE = cov_type

        results, equity_curves, trades_by_ticker = run_walk_forward_ndx(
            tickers=[ticker],
            train_months=6,
            test_months=3,
            feature_set=feature_set,
            quiet=True,
            use_ensemble=False,
            timeframe="1d",
        )

        if not results:
            return None

        # Aggregate metrics across windows
        sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None]
        returns = [r.return_pct for r in results if r.return_pct is not None]
        drawdowns = [r.max_drawdown_pct for r in results if r.max_drawdown_pct is not None]
        trades_list = [r.n_trades for r in results]
        converged = [r.hmm_converged for r in results]

        if not sharpes:
            return None

        wf_sharpe = float(np.mean(sharpes))
        wf_return = float(np.mean(returns)) if returns else 0.0
        wf_drawdown = float(np.min(drawdowns)) if drawdowns else 0.0
        total_trades = sum(trades_list)
        n_windows = len(results)
        positive_windows = sum(1 for s in sharpes if s > 0)
        std_sharpe = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0
        all_converged = all(converged)

        return {
            "ticker": ticker,
            "n_states": n_states,
            "feature_set": feature_set,
            "confirmations": confirmations,
            "cov_type": cov_type,
            "wf_sharpe": round(wf_sharpe, 4),
            "wf_return": round(wf_return, 2),
            "wf_drawdown": round(wf_drawdown, 2),
            "total_trades": total_trades,
            "n_windows": n_windows,
            "positive_windows": positive_windows,
            "std_sharpe": round(std_sharpe, 4),
            "all_converged": all_converged,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        # Use print instead of log for multiprocessing worker visibility
        print(f"  Trial failed ({ticker} {n_states}st/{feature_set}/{confirmations}cf/{cov_type}): {e}")
        return None


def _load_existing_results() -> pd.DataFrame:
    """Load existing results CSV for resume."""
    if RESULTS_CSV.exists():
        try:
            return pd.read_csv(RESULTS_CSV)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_results_batch(results: list[dict]):
    """Write a batch of results to CSV (append mode, thread-safe from main process)."""
    if not results:
        return
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(result)


def _save_per_ticker_configs(results_df: pd.DataFrame):
    """
    Extract best config per ticker and save to JSON.
    The CITRINE scanner loads this to override default HMM params.
    """
    if results_df.empty:
        return

    # Filter to converged, positive Sharpe configs with enough trades
    valid = results_df[
        (results_df["all_converged"] == True) &
        (results_df["total_trades"] >= 5)
    ].copy()

    if valid.empty:
        # Fall back to any config with trades
        valid = results_df[results_df["total_trades"] >= 1].copy()

    if valid.empty:
        log.warning("No valid results to save per-ticker configs")
        return

    # Best config per ticker (highest WF Sharpe)
    best_configs = {}
    for ticker in valid["ticker"].unique():
        ticker_df = valid[valid["ticker"] == ticker]
        best_idx = ticker_df["wf_sharpe"].idxmax()
        best = ticker_df.loc[best_idx]

        best_configs[ticker] = {
            "n_states": int(best["n_states"]),
            "feature_set": str(best["feature_set"]),
            "cov_type": str(best["cov_type"]),
            "confirmations": int(best["confirmations"]),
            "wf_sharpe": float(best["wf_sharpe"]),
        }

    with open(PER_TICKER_JSON, "w") as f:
        json.dump(best_configs, f, indent=2)

    log.info(f"Saved per-ticker configs for {len(best_configs)} tickers -> {PER_TICKER_JSON}")


def _generate_heatmap(results_df: pd.DataFrame):
    """Generate interactive Plotly heatmap of optimization results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if results_df.empty:
        log.warning("No results to generate heatmap")
        return

    # Pivot: ticker × config → Sharpe
    results_df["config"] = (
        results_df["n_states"].astype(str) + "st/" +
        results_df["feature_set"] + "/" +
        results_df["confirmations"].astype(str) + "cf/" +
        results_df["cov_type"]
    )

    pivot = results_df.pivot_table(
        values="wf_sharpe",
        index="ticker",
        columns="config",
        aggfunc="mean",
    )

    # Sort by mean Sharpe per ticker
    ticker_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[ticker_order]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#ff1744"],
            [0.3, "#ff6d00"],
            [0.5, "#ffea00"],
            [0.7, "#66bb6a"],
            [1.0, "#00e676"],
        ],
        zmid=0,
        text=np.round(pivot.values, 3).astype(str),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="WF Sharpe"),
    ))

    fig.update_layout(
        title="CITRINE Optimization — WF Sharpe by Ticker x Config",
        template="plotly_dark",
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141820",
        font=dict(family="monospace", color="#e0e4f0", size=10),
        height=max(400, len(pivot) * 25 + 200),
        width=1200,
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(l=80, r=40, t=60, b=120),
    )

    output_path = ROOT / "citrine_optimization_heatmap.html"
    fig.write_html(str(output_path))
    log.info(f"Heatmap saved -> {output_path}")


def run_optimization(
    tickers: list[str] | None = None,
    max_runs: int = 200,
    resume: bool = False,
    single_ticker: str | None = None,
    n_workers: int = 1,
):
    """Run CITRINE per-ticker HMM parameter optimization."""

    if single_ticker:
        tickers = [single_ticker]
    elif tickers is None:
        tickers = config.CITRINE_UNIVERSE

    all_combos = _build_trial_combos(tickers)
    total_possible = len(all_combos)

    log.info(f"\n{'='*70}")
    log.info(f"  CITRINE OPTIMIZER — Per-Ticker HMM Params")
    log.info(f"{'='*70}")
    log.info(f"  Tickers: {len(tickers)}")
    log.info(f"  Combos per ticker: {_COMBOS_PER_TICKER}")
    log.info(f"  Total possible: {total_possible}")
    log.info(f"  Max runs: {max_runs}")
    log.info(f"  Resume: {resume}")
    log.info(f"  Workers: {n_workers}")

    # Load existing results for resume
    existing_df = pd.DataFrame()
    completed_keys = set()
    if resume:
        existing_df = _load_existing_results()
        if not existing_df.empty:
            for _, row in existing_df.iterrows():
                key = (row["ticker"], row["n_states"], row["feature_set"],
                       row["confirmations"], row["cov_type"])
                completed_keys.add(key)
            log.info(f"  Resuming from {len(completed_keys)} completed trials")

    # Filter out already-completed combos
    remaining = []
    for combo in all_combos:
        key = (combo["ticker"], combo["n_states"], combo["feature_set"],
               combo["confirmations"], combo["cov_type"])
        if key not in completed_keys:
            remaining.append(combo)

    log.info(f"  Remaining: {len(remaining)} trials")

    # Random sample if more remaining than max_runs
    if len(remaining) > max_runs:
        random.seed(42 + len(completed_keys))
        remaining = random.sample(remaining, max_runs)

    runs_to_do = len(remaining)
    log.info(f"  Running: {runs_to_do} trials\n")

    if runs_to_do == 0:
        log.info("No trials to run. Use --runs to increase sample size.")
        return

    # Run trials
    completed = 0
    skipped = 0
    positive = 0
    start_time = time.time()
    unsaved_results: list[dict] = []

    if n_workers > 1:
        # ── Parallel execution ────────────────────────────────────────────
        log.info(f"  Starting {n_workers} parallel workers...\n")
        with Pool(n_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_run_single_trial, remaining),
                total=runs_to_do,
                desc="CITRINE",
            ):
                if result is None:
                    skipped += 1
                    continue

                unsaved_results.append(result)
                completed += 1
                sharpe = result["wf_sharpe"]
                if sharpe > 0:
                    positive += 1

                # Checkpoint every 10 completed results
                if len(unsaved_results) >= 10:
                    _save_results_batch(unsaved_results)
                    unsaved_results = []

        # Save any remaining unsaved results
        if unsaved_results:
            _save_results_batch(unsaved_results)
            unsaved_results = []

    else:
        # ── Sequential execution (--workers 1) ────────────────────────────
        for i, trial in enumerate(remaining):
            ticker = trial["ticker"]
            label = (f"{trial['n_states']}st/{trial['feature_set']}/"
                     f"{trial['confirmations']}cf/{trial['cov_type']}")

            log.info(f"[{i+1}/{runs_to_do}] {ticker} {label} ...")

            result = _run_single_trial(trial)

            if result is None:
                skipped += 1
                log.info(f"  SKIPPED")
                continue

            unsaved_results.append(result)
            completed += 1

            sharpe = result["wf_sharpe"]
            trades = result["total_trades"]
            if sharpe > 0:
                positive += 1

            marker = "+" if sharpe > 0 else "-"
            log.info(f"  [{marker}] Sharpe={sharpe:.3f}  Return={result['wf_return']:.1f}%  "
                     f"Trades={trades}  Windows={result['n_windows']}")

            # Checkpoint every 10 completed results
            if len(unsaved_results) >= 10:
                _save_results_batch(unsaved_results)
                unsaved_results = []

            # Progress update every 20 trials
            if completed > 0 and completed % 20 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta_s = (runs_to_do - i - 1) / rate if rate > 0 else 0
                eta_m = eta_s / 60

                log.info(f"\n  -- Progress: {completed}/{runs_to_do} done, "
                         f"{skipped} skipped, {positive} positive "
                         f"(ETA: {eta_m:.0f}m) --\n")

        # Save any remaining unsaved results
        if unsaved_results:
            _save_results_batch(unsaved_results)
            unsaved_results = []

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    log.info(f"\n{'='*70}")
    log.info(f"  OPTIMIZATION COMPLETE")
    log.info(f"  Completed: {completed}/{runs_to_do} ({skipped} skipped)")
    log.info(f"  Positive Sharpe: {positive}/{completed} ({positive/max(completed,1)*100:.0f}%)")
    log.info(f"  Time: {elapsed/60:.1f} minutes")
    log.info(f"  Workers: {n_workers}")
    log.info(f"{'='*70}")

    # Load all results and save best per-ticker configs
    all_results = _load_existing_results()
    if not all_results.empty:
        _save_per_ticker_configs(all_results)
        _generate_heatmap(all_results)

        # Print top 10
        top = all_results.nlargest(10, "wf_sharpe")
        log.info(f"\n  TOP 10 CONFIGS:")
        for _, row in top.iterrows():
            log.info(f"    {row['ticker']:6s} {row['n_states']}st/{row['feature_set']}/"
                     f"{int(row['confirmations'])}cf/{row['cov_type']}  "
                     f"Sharpe={row['wf_sharpe']:.3f}  "
                     f"Return={row['wf_return']:.1f}%  "
                     f"Trades={int(row['total_trades'])}")

    # Notify completion
    _macos_notify(
        "CITRINE Optimizer Done",
        f"{completed} trials | {positive} positive | {elapsed/60:.0f}m | {n_workers} workers"
    )
    _pushover_notify(
        "CITRINE Optimizer Done",
        f"{completed}/{runs_to_do} trials | {positive} positive Sharpe | "
        f"Time: {elapsed/60:.0f}m | Workers: {n_workers}",
        priority=0,
    )


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CITRINE: Per-ticker HMM parameter optimization"
    )
    parser.add_argument("--runs", type=int, default=200,
                        help="Number of random trials to run (default: 200)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results CSV")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Optimize a single ticker only")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate heatmap from existing results only")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help=f"Number of parallel workers (default: {max(1, cpu_count() - 2)})")
    args = parser.parse_args()

    if args.heatmap_only:
        df = _load_existing_results()
        if df.empty:
            log.error("No existing results found")
            return
        _generate_heatmap(df)
        _save_per_ticker_configs(df)
        return

    tickers = args.tickers.split(",") if args.tickers else None

    run_optimization(
        tickers=tickers,
        max_runs=args.runs,
        resume=args.resume,
        single_ticker=args.ticker,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
