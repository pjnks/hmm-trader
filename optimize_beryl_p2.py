"""
optimize_beryl_p2.py
────────────────────
BERYL Phase 2: Focused NDX100 walk-forward optimizer for TSLA and NVDA.

Builds on Phase 1 results — restricts to the two best-performing tickers
and adds intraday timeframes (2h, 4h) alongside daily.

Parameter grid (tighter, informed by Phase 1 winners):
  - ticker: TSLA, NVDA
  - n_states: 4, 5, 6
  - feature_set: base, extended
  - confirmations: 6, 7
  - leverage: 1.0, 1.5
  - cooldown_hours: 48, 72
  - covariance_type: full, diag
  - use_ensemble: True, False
  - timeframe: 1d, 2h, 4h

Grid size: 2 x 3 x 2 x 2 x 2 x 2 x 2 x 2 x 3 = 1,152 combinations
Walk-forward: 6m train / 3m test rolling windows.

Usage
-----
  python optimize_beryl_p2.py                    # 200 random trials
  python optimize_beryl_p2.py --runs 100         # fewer trials
  python optimize_beryl_p2.py --resume           # resume from checkpoint

Output
------
  beryl_p2_results.csv   -- ranked by WF Sharpe
  beryl_p2_heatmap.html  -- interactive Plotly heatmap
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import warnings
from itertools import product
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
    resample_equity_ohlcv,
    _attach_hmm_features,
    run_walk_forward_ndx,
)
from src.notifier import _macos_notify, _pushover_notify, _terminal_bell

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer_beryl_p2")

# -- Output paths -------------------------------------------------------------
RESULTS_CSV = ROOT / "beryl_p2_results.csv"
HEATMAP_HTML = ROOT / "beryl_p2_heatmap.html"

# -- Tickers (Phase 1 winners) ------------------------------------------------
P2_TICKERS = ["TSLA", "NVDA"]

# -- Parameter grid (tighter, informed by Phase 1) ----------------------------
GRID = {
    "ticker":          P2_TICKERS,
    "n_states":        [4, 5, 6],
    "feature_set":     ["base", "extended"],
    "confirmations":   [6, 7],
    "leverage":        [1.0, 1.5],
    "cooldown_hours":  [48, 72],
    "covariance_type": ["full", "diag"],
    "use_ensemble":    [True, False],
    "timeframe":       ["1d", "2h", "4h"],
}


def _grid_size() -> int:
    size = 1
    for v in GRID.values():
        size *= len(v)
    return size


def _sample_combinations(n: int) -> list[dict]:
    """Sample n random combinations from the grid."""
    keys = list(GRID.keys())
    all_combos = list(product(*GRID.values()))
    random.seed(42)
    if n >= len(all_combos):
        sample = all_combos
    else:
        sample = random.sample(all_combos, n)
    return [dict(zip(keys, combo)) for combo in sample]


def _patch_config(params: dict) -> dict:
    """Monkey-patch config module with trial parameters. Returns saved values."""
    saved = {}
    mappings = {
        "n_states":        "N_STATES",
        "confirmations":   "MIN_CONFIRMATIONS",
        "leverage":        "LEVERAGE",
        "cooldown_hours":  "COOLDOWN_HOURS",
        "covariance_type": "COV_TYPE",
        "feature_set":     "FEATURE_SET",
    }
    for param_key, config_key in mappings.items():
        if param_key in params:
            saved[config_key] = getattr(config, config_key)
            setattr(config, config_key, params[param_key])
    return saved


def _restore_config(saved: dict) -> None:
    """Restore config module to original values."""
    for key, value in saved.items():
        setattr(config, key, value)


# -- Pre-cache data -----------------------------------------------------------
# Keyed by "{ticker}_{timeframe_base}" where timeframe_base is "daily" or "hourly"
_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _precache_tickers(combos: list[dict]) -> None:
    """Download and cache equity data for all needed ticker+timeframe combos."""
    # Determine which (ticker, timeframe_base) pairs we need
    needed: set[tuple[str, str]] = set()
    for combo in combos:
        ticker = combo["ticker"]
        tf = combo["timeframe"]
        tf_base = "daily" if tf == "1d" else "hourly"
        needed.add((ticker, tf_base))

    for ticker, tf_base in sorted(needed):
        cache_key = f"{ticker}_{tf_base}"
        if cache_key not in _DATA_CACHE:
            print(f"  Pre-caching {ticker} ({tf_base}) ...", end=" ", flush=True)
            if tf_base == "daily":
                df = fetch_equity_daily(ticker, years=20)
            else:
                df = fetch_equity_hourly(ticker, years=2)
            if not df.empty:
                _DATA_CACHE[cache_key] = df
                print(f"{len(df)} bars")
            else:
                print(f"no data")


def _run_trial(params: dict, train_months: int = 6, test_months: int = 3) -> dict | None:
    """
    Run one walk-forward trial with given parameters.
    Returns results dict or None if trial failed.
    """
    ticker = params["ticker"]
    timeframe = params.get("timeframe", "1d")
    use_ensemble = params.get("use_ensemble", False)

    saved = _patch_config(params)

    try:
        results, eq_curves, trades = run_walk_forward_ndx(
            tickers=[ticker],
            train_months=train_months,
            test_months=test_months,
            feature_set=params["feature_set"],
            quiet=True,
            use_ensemble=use_ensemble,
            timeframe=timeframe,
        )

        if not results:
            return None

        # Calculate combined metrics
        total_return = (results[-1].end_equity / results[0].start_equity - 1) * 100
        mean_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        std_sharpe = np.std([r.sharpe_ratio for r in results]) if len(results) > 1 else 0.0
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
            "wf_mean_sharpe": round(mean_sharpe, 3),
            "wf_std_sharpe": round(std_sharpe, 3),
        }

    except Exception as e:
        log.warning(f"Trial failed: {params} -- {e}")
        return None

    finally:
        _restore_config(saved)


def _save_results(results: list[dict]) -> None:
    """Save results to CSV, sorted by Sharpe."""
    if not results:
        return

    df = pd.DataFrame(results)
    df = df.sort_values("wf_sharpe", ascending=False)
    df.to_csv(RESULTS_CSV, index=False)


def _build_heatmap(results: list[dict]) -> None:
    """Generate interactive Plotly heatmap with multiple views."""
    if not results:
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.DataFrame(results)

    # Heatmap 1: ticker x timeframe, colored by mean Sharpe
    pivot_tf = df.pivot_table(
        values="wf_sharpe",
        index="ticker",
        columns="timeframe",
        aggfunc="mean",
    )

    # Heatmap 2: ticker x feature_set, colored by mean Sharpe
    pivot_fs = df.pivot_table(
        values="wf_sharpe",
        index="ticker",
        columns="feature_set",
        aggfunc="mean",
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Ticker x Timeframe", "Ticker x Feature Set"],
        horizontal_spacing=0.12,
    )

    # Left: ticker x timeframe
    fig.add_trace(
        go.Heatmap(
            z=pivot_tf.values,
            x=pivot_tf.columns.tolist(),
            y=pivot_tf.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot_tf.values, 2),
            texttemplate="%{text}",
            showscale=False,
        ),
        row=1, col=1,
    )

    # Right: ticker x feature_set
    fig.add_trace(
        go.Heatmap(
            z=pivot_fs.values,
            x=pivot_fs.columns.tolist(),
            y=pivot_fs.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot_fs.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="WF Sharpe"),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title="BERYL Phase 2 -- Walk-Forward Sharpe (TSLA & NVDA, Multi-Timeframe)",
        template="plotly_dark",
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141820",
        font=dict(family="Consolas, monospace"),
        height=400,
    )

    fig.write_html(HEATMAP_HTML)
    print(f"Heatmap written -> {HEATMAP_HTML}")


def _notify_completion(successful: int, skipped: int, results: list[dict]) -> None:
    """Send Pushover + macOS notification when optimization completes."""
    best_sharpe = 0.0
    best_config = ""
    if results:
        df = pd.DataFrame(results).sort_values("wf_sharpe", ascending=False)
        top = df.iloc[0]
        best_sharpe = top["wf_sharpe"]
        best_config = (
            f"{top['ticker']}/{top['timeframe']}/{top['n_states']}st/"
            f"{top['feature_set']}/{top['confirmations']}cf/"
            f"{top['leverage']}x/{top['cooldown_hours']}h/"
            f"{top['covariance_type']}"
            f"{'/ens' if top.get('use_ensemble') else ''}"
        )

    title = "BERYL P2 Optimization Complete"
    message = (
        f"{successful} successful, {skipped} skipped\n"
        f"Best Sharpe: {best_sharpe:.3f}\n"
        f"Config: {best_config}"
    )

    _macos_notify(title, message)
    _pushover_notify(title, message, priority=0)
    _terminal_bell()


def main():
    parser = argparse.ArgumentParser(
        description="BERYL Phase 2: Focused NDX100 walk-forward optimizer (TSLA & NVDA)"
    )
    parser.add_argument("--runs", type=int, default=200,
                        help="Number of random trials (default: 200)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    args = parser.parse_args()

    total_grid = _grid_size()
    print(f"\nBERYL Phase 2 -- Focused Walk-Forward Optimizer")
    print(f"  Tickers: {', '.join(P2_TICKERS)}")
    print(f"  Timeframes: 1d, 2h, 4h")
    print(f"  Total grid size: {total_grid:,}")
    print(f"  Trials to run: {args.runs}")

    # Sample combinations
    combos = _sample_combinations(args.runs)

    # Load existing results if resuming
    existing_results = []
    existing_keys = set()
    if args.resume and RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        existing_results = existing_df.to_dict("records")
        # Build set of already-completed param combos for dedup
        key_cols = [k for k in GRID.keys()]
        for rec in existing_results:
            key = tuple(str(rec.get(k, "")) for k in key_cols)
            existing_keys.add(key)
        print(f"  Resuming from {len(existing_results)} existing results")

    # Filter out already-completed combos when resuming
    if existing_keys:
        key_cols = list(GRID.keys())
        combos = [
            c for c in combos
            if tuple(str(c.get(k, "")) for k in key_cols) not in existing_keys
        ]
        print(f"  Remaining trials after dedup: {len(combos)}")

    # Pre-cache ticker data
    print(f"\nPre-caching ticker data:")
    _precache_tickers(combos)

    # Run trials
    results = list(existing_results)
    successful = 0
    skipped = 0

    for combo in tqdm(combos, desc="BERYL-P2-Optimizing"):
        result = _run_trial(combo)

        if result:
            results.append(result)
            successful += 1
        else:
            skipped += 1

        # Checkpoint every 5 trials
        if successful > 0 and successful % 5 == 0:
            _save_results(results)

    # Final save
    _save_results(results)

    print(f"\nDone -- {successful} successful, {skipped} skipped")
    print(f"Results saved -> {RESULTS_CSV}")

    # Print top 10
    if results:
        df = pd.DataFrame(results).sort_values("wf_sharpe", ascending=False)
        print(f"\n-- TOP 10 by Walk-Forward OOS Sharpe [BERYL P2] --------")
        cols = ["ticker", "timeframe", "n_states", "feature_set", "confirmations",
                "leverage", "cooldown_hours", "covariance_type", "use_ensemble",
                "wf_sharpe", "wf_return", "wf_drawdown", "wf_trades",
                "wf_pos_windows", "wf_n_windows"]
        print(df[cols].head(10).to_string(index=False))

    # Build heatmap
    _build_heatmap(results)

    # Send completion notification
    _notify_completion(successful, skipped, results)


if __name__ == "__main__":
    main()
