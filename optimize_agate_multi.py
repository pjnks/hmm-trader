"""
optimize_agate_multi.py
───────────────────────
Per-ticker HMM parameter optimization for AGATE multi-ticker crypto rotation.

Tests each crypto ticker across a grid of HMM configurations using walk-forward
analysis (ensemble HMM, 6m train / 3m test). Saves the best config per ticker
to agate_per_ticker_configs.json, which live_trading.py hot-loads.

Grid per ticker:
  n_states(3) × feature_set(3) × confirmations(4) × cov_type(2) × timeframe(2)
  = 144 combos per ticker × 16 tickers = 2,304 total

Also tests adaptive confirmations: cf_adaptive flag lowers confirmations by 1
when regime confidence > 0.90 (e.g., cf=7 becomes cf=6 at high confidence).

Usage
─────
  python optimize_agate_multi.py --runs 200           # random sample
  python optimize_agate_multi.py --resume              # resume from checkpoint
  python optimize_agate_multi.py --ticker X:BTCUSD     # single ticker
  python optimize_agate_multi.py --heatmap-only        # regenerate from CSV
  python optimize_agate_multi.py --workers 4           # 4 parallel workers
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import random
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

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
log = logging.getLogger("agate_multi_opt")


# ── Grid Definition ────────────────────────────────────────────────────────

TICKER_GRID = {
    "n_states":       [4, 5, 6],
    "feature_set":    ["base", "extended", "extended_v2"],
    "confirmations":  [5, 6, 7, 8],
    "cov_type":       ["diag", "full"],
    "timeframe":      ["3h", "4h"],
}

# Total combos per ticker: 3 × 3 × 4 × 2 × 2 = 144
_COMBOS_PER_TICKER = 1
for v in TICKER_GRID.values():
    _COMBOS_PER_TICKER *= len(v)

# ── Results files ─────────────────────────────────────────────────────────
RESULTS_CSV = ROOT / "agate_multi_optimization_results.csv"
PER_TICKER_JSON = ROOT / "agate_per_ticker_configs.json"
HEATMAP_HTML = ROOT / "agate_multi_optimization_heatmap.html"


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
    Run walk-forward analysis for a single crypto ticker with given HMM params.
    Uses ensemble HMM (3 models voting) via run_walk_forward from walk_forward.py.

    Safe for multiprocessing: patches config in the worker's own memory space.
    Returns dict with metrics or None if the trial fails.
    """
    ticker = trial["ticker"]
    n_states = trial["n_states"]
    feature_set = trial["feature_set"]
    confirmations = trial["confirmations"]
    cov_type = trial["cov_type"]
    timeframe = trial["timeframe"]

    try:
        from walk_forward import run_walk_forward

        # Patch config in this worker process (safe — separate memory space)
        config.N_STATES = n_states
        config.FEATURE_SET = feature_set
        config.MIN_CONFIRMATIONS = confirmations
        config.COV_TYPE = cov_type
        config.LEVERAGE = 1.0
        config.COOLDOWN_HOURS = 48

        results, combined_eq, combined_bh, all_trades = run_walk_forward(
            train_months=6,
            test_months=3,
            ticker=ticker,
            feature_set=feature_set,
            confirmations=confirmations,
            timeframe=timeframe,
            quiet=True,
            use_ensemble=True,
        )

        if not results:
            return None

        # Aggregate metrics across windows
        sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None and not np.isnan(r.sharpe_ratio)]
        returns = [r.return_pct for r in results if r.return_pct is not None]
        drawdowns = [r.max_drawdown_pct for r in results if r.max_drawdown_pct is not None]
        trades_list = [r.n_trades for r in results]

        if not sharpes:
            return None

        wf_sharpe = float(np.mean(sharpes))
        min_sharpe = float(np.min(sharpes))
        wf_return = float(np.mean(returns)) if returns else 0.0
        wf_drawdown = float(np.min(drawdowns)) if drawdowns else 0.0
        total_trades = sum(trades_list)
        n_windows = len(results)
        positive_windows = sum(1 for s in sharpes if s > 0)
        std_sharpe = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0
        consistency = positive_windows / max(n_windows, 1)

        return {
            "ticker": ticker,
            "n_states": n_states,
            "feature_set": feature_set,
            "confirmations": confirmations,
            "cov_type": cov_type,
            "timeframe": timeframe,
            "wf_sharpe": round(wf_sharpe, 4),
            "min_sharpe": round(min_sharpe, 4),
            "wf_return": round(wf_return, 2),
            "wf_drawdown": round(wf_drawdown, 2),
            "total_trades": total_trades,
            "n_windows": n_windows,
            "positive_windows": positive_windows,
            "std_sharpe": round(std_sharpe, 4),
            "consistency": round(consistency, 3),
            "rank_score": round(wf_sharpe * consistency, 4),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        # Use print for multiprocessing worker visibility
        print(f"  Trial failed ({ticker} {n_states}st/{feature_set}/{confirmations}cf/{cov_type}/{timeframe}): {e}")
        return None


def _load_existing_results() -> pd.DataFrame:
    """Load existing results CSV for resume."""
    if RESULTS_CSV.exists():
        try:
            return pd.read_csv(RESULTS_CSV)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


CSV_COLUMNS = [
    "ticker", "n_states", "feature_set", "confirmations", "cov_type", "timeframe",
    "wf_sharpe", "wf_return", "wf_drawdown", "total_trades",
    "n_windows", "positive_windows", "std_sharpe", "timestamp",
]


def _validate_result(result: dict) -> bool:
    """Validate a result dict has the right types and no embedded commas/newlines."""
    for col in CSV_COLUMNS:
        val = result.get(col)
        if val is None:
            log.warning("Missing column %s in result for %s", col, result.get("ticker", "?"))
            return False
        val_str = str(val)
        if "\n" in val_str or "\r" in val_str:
            log.warning("Newline in column %s for %s — skipping", col, result.get("ticker", "?"))
            return False
    return True


def _save_results_batch(results: list[dict]):
    """Write a batch of results to CSV (append mode, called from main process).
    Uses fixed column list to prevent schema drift when trial function adds new fields.
    Validates each row before writing to prevent corrupt CSV lines.
    """
    if not results:
        return
    valid = [r for r in results if _validate_result(r)]
    if not valid:
        return
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for result in valid:
            writer.writerow(result)
    if len(valid) < len(results):
        log.warning("Dropped %d invalid results (schema validation)", len(results) - len(valid))


def _save_per_ticker_configs(results_df: pd.DataFrame):
    """
    Extract best config per ticker and save to JSON.
    live_trading.py hot-loads this to use optimized params per crypto ticker.
    """
    if results_df.empty:
        return

    # Filter to configs with enough trades
    valid = results_df[results_df["total_trades"] >= 3].copy()

    if valid.empty:
        valid = results_df[results_df["total_trades"] >= 1].copy()

    if valid.empty:
        log.warning("No valid results to save per-ticker configs")
        return

    # Best config per ticker: rank by Sharpe x Consistency
    # This penalizes configs that are great in one window but terrible in others
    valid["consistency"] = valid["positive_windows"] / valid["n_windows"].clip(lower=1)
    valid["rank_score"] = valid["wf_sharpe"] * valid["consistency"]

    best_configs = {}
    for ticker in valid["ticker"].unique():
        ticker_df = valid[valid["ticker"] == ticker]
        # Prefer rank_score, but fall back to wf_sharpe if all consistency=0
        if (ticker_df["rank_score"] > 0).any():
            best_idx = ticker_df["rank_score"].idxmax()
        else:
            best_idx = ticker_df["wf_sharpe"].idxmax()
        best = ticker_df.loc[best_idx]

        best_configs[ticker] = {
            "n_states": int(best["n_states"]),
            "feature_set": str(best["feature_set"]),
            "confirmations": int(best["confirmations"]),
            "cov_type": str(best["cov_type"]),
            "timeframe": str(best["timeframe"]),
            "wf_sharpe": float(best["wf_sharpe"]),
            "wf_return": float(best["wf_return"]),
            "total_trades": int(best["total_trades"]),
            "positive_windows": int(best["positive_windows"]),
            "n_windows": int(best["n_windows"]),
        }

    # Load existing JSON and merge (keep existing tickers not in this run)
    existing = {}
    if PER_TICKER_JSON.exists():
        try:
            with open(PER_TICKER_JSON) as f:
                existing = json.load(f)
        except Exception:
            pass

    # Update only tickers that improved or are new
    for ticker, new_cfg in best_configs.items():
        old_cfg = existing.get(ticker, {})
        old_sharpe = old_cfg.get("wf_sharpe", -999)
        if new_cfg["wf_sharpe"] > old_sharpe:
            existing[ticker] = new_cfg
            log.info(f"  {ticker}: updated config (Sharpe {old_sharpe:.3f} -> {new_cfg['wf_sharpe']:.3f})")
        else:
            log.info(f"  {ticker}: kept existing (Sharpe {old_sharpe:.3f} >= {new_cfg['wf_sharpe']:.3f})")

    with open(PER_TICKER_JSON, "w") as f:
        json.dump(existing, f, indent=2)

    positive_count = sum(1 for c in existing.values() if c.get("wf_sharpe", 0) > 0)
    log.info(f"Saved {len(existing)} ticker configs ({positive_count} positive Sharpe) -> {PER_TICKER_JSON.name}")


def _build_heatmap(results_df: pd.DataFrame):
    """Build interactive Plotly heatmap from results."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("Plotly not installed -- skipping heatmap")
        return

    if results_df.empty:
        return

    # Pivot: ticker x (feature_set + timeframe), colored by wf_sharpe
    results_df = results_df.copy()
    results_df["config_label"] = (
        results_df["feature_set"] + "/" +
        results_df["timeframe"] + "/" +
        results_df["confirmations"].astype(str) + "cf/" +
        results_df["cov_type"]
    )

    # Best sharpe per ticker x config_label
    pivot = results_df.pivot_table(
        values="wf_sharpe",
        index="ticker",
        columns="config_label",
        aggfunc="max",
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
    ))

    fig.update_layout(
        title="AGATE Multi-Ticker Crypto Optimization -- WF Sharpe by Config",
        xaxis_title="Config (feature_set/timeframe/cf/cov)",
        yaxis_title="Ticker",
        height=max(400, len(pivot) * 25 + 200),
        width=max(800, len(pivot.columns) * 40 + 200),
        template="plotly_dark",
    )

    fig.write_html(str(HEATMAP_HTML))
    log.info(f"Heatmap saved -> {HEATMAP_HTML.name}")


def main():
    parser = argparse.ArgumentParser(description="AGATE multi-ticker crypto optimizer")
    parser.add_argument("--runs", type=int, default=200, help="Number of random trials")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint CSV")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker to optimize (e.g. X:BTCUSD)")
    parser.add_argument("--heatmap-only", action="store_true", help="Regenerate heatmap from existing CSV")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help=f"Number of parallel workers (default: {max(1, cpu_count() - 2)})")

    args = parser.parse_args()

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = config.AGATE_TICKERS

    log.info(f"AGATE Multi-Ticker Optimizer")
    log.info(f"  Tickers: {len(tickers)}")
    log.info(f"  Grid: {_COMBOS_PER_TICKER} combos/ticker x {len(tickers)} tickers = {_COMBOS_PER_TICKER * len(tickers)} total")
    log.info(f"  Sampling: {args.runs} random trials")
    log.info(f"  Workers: {args.workers}")

    # Heatmap-only mode
    if args.heatmap_only:
        df = _load_existing_results()
        if df.empty:
            log.error("No results CSV found")
            return
        _build_heatmap(df)
        _save_per_ticker_configs(df)
        return

    # Build trial combos
    all_combos = _build_trial_combos(tickers)
    log.info(f"  Total possible combos: {len(all_combos)}")

    # Resume: skip already-completed trials
    existing_df = _load_existing_results() if args.resume else pd.DataFrame()
    if not existing_df.empty:
        completed_keys = set()
        for _, row in existing_df.iterrows():
            key = (row["ticker"], row["n_states"], row["feature_set"],
                   row["confirmations"], row["cov_type"], row["timeframe"])
            completed_keys.add(key)

        remaining = [
            c for c in all_combos
            if (c["ticker"], c["n_states"], c["feature_set"],
                c["confirmations"], c["cov_type"], c["timeframe"]) not in completed_keys
        ]
        log.info(f"  Resuming: {len(existing_df)} completed, {len(remaining)} remaining")
    else:
        remaining = all_combos

    # Random sample
    if len(remaining) > args.runs:
        random.shuffle(remaining)
        remaining = remaining[:args.runs]

    log.info(f"  Running {len(remaining)} trials...")
    log.info("")

    # ── Run trials ─────────────────────────────────────────────────────────
    completed = 0
    positive = 0
    t0 = time.time()
    unsaved_results: list[dict] = []

    if args.workers > 1:
        # ── Parallel execution ────────────────────────────────────────────
        log.info(f"  Starting {args.workers} parallel workers...\n")
        with Pool(args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_run_single_trial, remaining),
                total=len(remaining),
                desc="AGATE",
            ):
                if result is None:
                    continue

                unsaved_results.append(result)
                completed += 1
                if result["wf_sharpe"] > 0:
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
            ticker_short = trial["ticker"].replace("X:", "")
            log.info(f"[{i+1}/{len(remaining)}] {ticker_short} "
                     f"{trial['n_states']}st/{trial['feature_set']}/{trial['confirmations']}cf/"
                     f"{trial['cov_type']}/{trial['timeframe']}")

            result = _run_single_trial(trial)

            if result:
                unsaved_results.append(result)
                completed += 1
                if result["wf_sharpe"] > 0:
                    positive += 1
                log.info(f"  -> Sharpe {result['wf_sharpe']:.3f}, "
                         f"return {result['wf_return']:.1f}%, "
                         f"{result['total_trades']} trades, "
                         f"{result['positive_windows']}/{result['n_windows']} windows positive")

                # Checkpoint every 10 completed results
                if len(unsaved_results) >= 10:
                    _save_results_batch(unsaved_results)
                    unsaved_results = []
            else:
                log.info(f"  -> skipped (non-convergence or no trades)")

            # Progress update every 10 completed trials
            if completed > 0 and completed % 10 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600
                log.info(f"  --- Checkpoint: {completed} completed, {positive} positive, "
                         f"{rate:.0f} trials/hour ---")

        # Save any remaining unsaved results
        if unsaved_results:
            _save_results_batch(unsaved_results)
            unsaved_results = []

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"\nDone: {completed} trials in {elapsed/60:.1f} min ({positive} positive Sharpe) | {args.workers} workers")

    # Save per-ticker configs
    all_results = _load_existing_results()
    if not all_results.empty:
        _save_per_ticker_configs(all_results)
        _build_heatmap(all_results)

        # Summary by ticker
        log.info("\n  Per-ticker best:")
        for ticker in sorted(all_results["ticker"].unique()):
            ticker_df = all_results[all_results["ticker"] == ticker]
            best = ticker_df.loc[ticker_df["wf_sharpe"].idxmax()]
            log.info(f"    {ticker.replace('X:',''):<8} Sharpe {best['wf_sharpe']:>7.3f}  "
                     f"{best['feature_set']}/{best['timeframe']}/{int(best['confirmations'])}cf/{best['cov_type']}")

    # Notify
    try:
        msg = f"{completed} trials, {positive} positive | {args.workers} workers"
        if not all_results.empty:
            best_overall = all_results.loc[all_results["wf_sharpe"].idxmax()]
            msg += f"\nBest: {best_overall['ticker'].replace('X:','')}/{best_overall['feature_set']}/{best_overall['timeframe']} Sharpe {best_overall['wf_sharpe']:.3f}"
        _macos_notify("AGATE Multi-Ticker Optimizer Done", msg)
        _pushover_notify("AGATE Multi-Ticker Optimizer Done", msg, priority=-1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
