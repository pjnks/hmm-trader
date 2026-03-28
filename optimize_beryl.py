"""
optimize_beryl.py
─────────────────
BERYL: NDX100 walk-forward optimizer.

Random-search over parameter grid for equity tickers:
  • ticker: AAPL, MSFT, TSLA, NVDA, GOOGL
  • n_states: 4, 5, 6, 7, 8
  • feature_set: base, extended, full
  • confirmations: 6, 7, 8
  • leverage: 1.0, 1.5, 2.0
  • cooldown_hours: 48, 72
  • cov_type: full, diag
  • use_ensemble: True, False

Walk-forward: 6m train / 3m test rolling windows.
Uses Polygon.io for equity data (not yfinance).

Usage
─────
  python optimize_beryl.py                    # 200 random trials
  python optimize_beryl.py --runs 50          # fewer trials
  python optimize_beryl.py --resume           # resume from checkpoint
  python optimize_beryl.py --ensemble-only    # only test ensemble configs

Output
──────
  beryl_optimization_results.csv   — ranked by WF Sharpe
  beryl_optimization_heatmap.html  — interactive Plotly heatmap
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
    _attach_hmm_features,
    run_walk_forward_ndx,
    NDX_TICKERS,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer_beryl")

# ── Output paths ───────────────────────────────────────────────────────────
RESULTS_CSV = ROOT / "beryl_optimization_results.csv"
HEATMAP_HTML = ROOT / "beryl_optimization_heatmap.html"

# ── Parameter grid ─────────────────────────────────────────────────────────
GRID = {
    "ticker":          NDX_TICKERS,
    "n_states":        [4, 5, 6, 7, 8],
    "feature_set":     ["base", "extended", "full"],
    "confirmations":   [6, 7, 8],
    "leverage":        [1.0, 1.5, 2.0],
    "cooldown_hours":  [48, 72],
    "covariance_type": ["full", "diag"],
    "use_ensemble":    [True, False],
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


# ── Pre-cache data ─────────────────────────────────────────────────────────
_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _precache_tickers(tickers: list[str]) -> None:
    """Download and cache equity data for all tickers."""
    for ticker in tickers:
        if ticker not in _DATA_CACHE:
            print(f"  Pre-caching {ticker} ...", end=" ", flush=True)
            df = fetch_equity_daily(ticker, years=20)
            if not df.empty:
                _DATA_CACHE[ticker] = df
                print(f"✅ {len(df)} bars")
            else:
                print(f"❌ no data")


def _run_trial(params: dict, train_months: int = 6, test_months: int = 3) -> dict | None:
    """
    Run one walk-forward trial with given parameters.
    Returns results dict or None if trial failed.
    """
    ticker = params["ticker"]
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
        log.warning(f"Trial failed: {params} — {e}")
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
    """Generate interactive Plotly heatmap."""
    if not results:
        return

    import plotly.graph_objects as go

    df = pd.DataFrame(results)

    # Simple heatmap: ticker × feature_set, colored by mean Sharpe
    pivot = df.pivot_table(
        values="wf_sharpe",
        index="ticker",
        columns="feature_set",
        aggfunc="mean",
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
    ))

    fig.update_layout(
        title="BERYL — NDX100 Walk-Forward Sharpe (Ticker × Feature Set)",
        xaxis_title="Feature Set",
        yaxis_title="Ticker",
        template="plotly_dark",
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141820",
        font=dict(family="Consolas, monospace"),
    )

    fig.write_html(HEATMAP_HTML)
    print(f"Heatmap written → {HEATMAP_HTML}")


def main():
    parser = argparse.ArgumentParser(description="BERYL: NDX100 walk-forward optimizer")
    parser.add_argument("--runs", type=int, default=200,
                        help="Number of random trials (default: 200)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--ensemble-only", action="store_true",
                        help="Only test ensemble configurations")
    args = parser.parse_args()

    total_grid = _grid_size()
    print(f"\nBERYL — NDX100 Walk-Forward Optimizer")
    print(f"  Total grid size: {total_grid:,}")
    print(f"  Trials to run: {args.runs}")

    # Sample combinations
    combos = _sample_combinations(args.runs)

    if args.ensemble_only:
        combos = [c for c in combos if c.get("use_ensemble", False)]
        print(f"  Ensemble-only: {len(combos)} trials")

    # Load existing results if resuming
    existing_results = []
    if args.resume and RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        existing_results = existing_df.to_dict("records")
        print(f"  Resuming from {len(existing_results)} existing results")

    # Pre-cache ticker data
    print(f"\nPre-caching ticker data:")
    tickers_needed = list(set(c["ticker"] for c in combos))
    _precache_tickers(tickers_needed)

    # Run trials
    results = list(existing_results)
    successful = 0
    skipped = 0

    for trial_result in tqdm(
        (combo for combo in combos),
        total=len(combos),
        desc="BERYL-Optimizing",
    ):
        combo = trial_result
        result = _run_trial(combo)

        if result:
            results.append(result)
            successful += 1
        else:
            skipped += 1

        # Checkpoint every 10 trials
        if successful > 0 and successful % 10 == 0:
            _save_results(results)

    # Final save
    _save_results(results)

    print(f"\nDone — {successful} successful, {skipped} skipped")
    print(f"Results saved → {RESULTS_CSV}")

    # Print top 10
    if results:
        df = pd.DataFrame(results).sort_values("wf_sharpe", ascending=False)
        print(f"\n── TOP 10 by Walk-Forward OOS Sharpe [BERYL] ────────")
        cols = ["ticker", "n_states", "feature_set", "confirmations", "leverage",
                "cooldown_hours", "covariance_type", "use_ensemble",
                "wf_sharpe", "wf_return", "wf_drawdown", "wf_trades",
                "wf_pos_windows", "wf_n_windows"]
        print(df[cols].head(10).to_string(index=False))

    # Build heatmap
    _build_heatmap(results)


if __name__ == "__main__":
    main()
