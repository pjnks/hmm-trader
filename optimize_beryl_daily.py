"""
optimize_beryl_daily.py
───────────────────────
BERYL Daily: Full NDX100 walk-forward optimizer (daily bars only, ensemble HMM).

Scans all 98 tickers from citrine_per_ticker_configs.json using daily bars
with ensemble HMM (3 models, n_states=[N-1, N, N+1]). Outputs per-ticker
best configs to beryl_per_ticker_configs.json for live_trading_beryl.py.

Grid: 98 tickers x 4 n_states x 3 features x 4 confirms x 1 cov x 1 lev
      x 1 cooldown x 1 timeframe = 4,704 combos
Random sample: --runs N (default 200)

Usage
─────
  python optimize_beryl_daily.py --runs 200          # 200 random trials
  python optimize_beryl_daily.py --runs 200 --resume # resume
  python optimize_beryl_daily.py --heatmap-only      # regen heatmap
  python optimize_beryl_daily.py --workers 4         # 4 parallel workers
"""

from __future__ import annotations

import argparse
import csv
import json
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
    run_walk_forward_ndx,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer_beryl_daily")

# ── Output paths ─────────────────────────────────────────────────────────────
RESULTS_CSV       = ROOT / "beryl_daily_results.csv"
HEATMAP_HTML      = ROOT / "beryl_daily_heatmap.html"
PER_TICKER_JSON   = ROOT / "beryl_per_ticker_configs.json"
CITRINE_CONFIGS   = ROOT / "citrine_per_ticker_configs.json"

# ── Default fallback tickers ─────────────────────────────────────────────────
FALLBACK_TICKERS = ["TSLA", "NVDA", "AAPL", "MSFT", "GOOGL"]


def _load_tickers() -> list[str]:
    """Load ticker list from citrine_per_ticker_configs.json, fall back to 5 defaults."""
    if CITRINE_CONFIGS.exists():
        try:
            with open(CITRINE_CONFIGS) as f:
                data = json.load(f)
            tickers = sorted(data.keys())
            if tickers:
                return tickers
        except Exception as e:
            log.warning(f"Failed to load {CITRINE_CONFIGS}: {e}")
    log.warning(f"Using fallback tickers: {FALLBACK_TICKERS}")
    return FALLBACK_TICKERS


# ── Parameter grid (daily bars only, ensemble, diag cov) ─────────────────────
TICKERS = _load_tickers()

GRID = {
    "ticker":          TICKERS,
    "n_states":        [3, 4, 5, 6],
    "feature_set":     ["base", "extended", "extended_v2"],
    "confirmations":   [4, 5, 6, 7],
    "covariance_type": ["diag"],
    "leverage":        [1.0],
    "cooldown_hours":  [48],
    "timeframe":       ["1d"],
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
    Run one walk-forward trial with ensemble HMM. Returns results dict or None.

    Safe for multiprocessing: patches config in the worker's own memory space.
    Accepts a single dict (required for Pool.imap_unordered pickling).
    """
    ticker = params["ticker"]

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
            use_ensemble=True,   # Ensemble HMM (3 models, n_states=[N-1, N, N+1])
            timeframe="1d",
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
            "N_States x Feature Set (mean Sharpe)",
            "Confirmations x Feature Set (mean Sharpe)",
            "Ticker x N_States (mean Sharpe, top 20 tickers)",
            "Ticker x Confirmations (mean Sharpe, top 20 tickers)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Panel 1: n_states x feature_set
    p = df.groupby(["n_states", "feature_set"])["wf_sharpe"].mean().reset_index()
    hm = p.pivot(index="n_states", columns="feature_set", values="wf_sharpe")
    fig.add_trace(
        go.Heatmap(
            z=hm.values,
            x=[str(v) for v in hm.columns],
            y=[str(v) for v in hm.index],
            colorscale="RdYlGn",
            text=np.round(hm.values, 3),
            texttemplate="%{text:.3f}",
            showscale=False,
        ),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="feature_set", row=1, col=1)
    fig.update_yaxes(title_text="n_states", row=1, col=1)

    # Panel 2: confirmations x feature_set
    p = df.groupby(["confirmations", "feature_set"])["wf_sharpe"].mean().reset_index()
    hm = p.pivot(index="confirmations", columns="feature_set", values="wf_sharpe")
    fig.add_trace(
        go.Heatmap(
            z=hm.values,
            x=[str(v) for v in hm.columns],
            y=[str(v) for v in hm.index],
            colorscale="RdYlGn",
            text=np.round(hm.values, 3),
            texttemplate="%{text:.3f}",
            showscale=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="feature_set", row=1, col=2)
    fig.update_yaxes(title_text="confirmations", row=1, col=2)

    # Panels 3-4: top 20 tickers by mean Sharpe for readability
    ticker_means = df.groupby("ticker")["wf_sharpe"].mean()
    top_tickers = ticker_means.nlargest(20).index.tolist()
    df_top = df[df["ticker"].isin(top_tickers)]

    # Panel 3: ticker x n_states (top 20)
    p = df_top.groupby(["ticker", "n_states"])["wf_sharpe"].mean().reset_index()
    hm = p.pivot(index="ticker", columns="n_states", values="wf_sharpe")
    fig.add_trace(
        go.Heatmap(
            z=hm.values,
            x=[str(v) for v in hm.columns],
            y=[str(v) for v in hm.index],
            colorscale="RdYlGn",
            text=np.round(hm.values, 3),
            texttemplate="%{text:.3f}",
            showscale=False,
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="n_states", row=2, col=1)
    fig.update_yaxes(title_text="ticker", row=2, col=1)

    # Panel 4: ticker x confirmations (top 20)
    p = df_top.groupby(["ticker", "confirmations"])["wf_sharpe"].mean().reset_index()
    hm = p.pivot(index="ticker", columns="confirmations", values="wf_sharpe")
    fig.add_trace(
        go.Heatmap(
            z=hm.values,
            x=[str(v) for v in hm.columns],
            y=[str(v) for v in hm.index],
            colorscale="RdYlGn",
            text=np.round(hm.values, 3),
            texttemplate="%{text:.3f}",
            showscale=True,
            colorbar=dict(title="Sharpe"),
        ),
        row=2, col=2,
    )
    fig.update_xaxes(title_text="confirmations", row=2, col=2)
    fig.update_yaxes(title_text="ticker", row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=1000,
        width=1100,
        title="BERYL Daily — Full NDX100 Walk-Forward Optimization (Ensemble, Daily Bars)",
    )
    fig.write_html(str(HEATMAP_HTML))
    print(f"  Heatmap: {HEATMAP_HTML}")


def _write_per_ticker_configs(results: list[dict]) -> None:
    """Write best config per ticker to beryl_per_ticker_configs.json."""
    df = pd.DataFrame(results)
    if df.empty:
        return

    configs = {}
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker]
        best = ticker_df.nlargest(1, "wf_sharpe").iloc[0]
        configs[ticker] = {
            "n_states": int(best["n_states"]),
            "feature_set": best["feature_set"],
            "cov_type": best["covariance_type"],
            "confirmations": int(best["confirmations"]),
            "wf_sharpe": round(float(best["wf_sharpe"]), 4),
        }

    with open(PER_TICKER_JSON, "w") as f:
        json.dump(configs, f, indent=2)

    positive = sum(1 for v in configs.values() if v["wf_sharpe"] > 0)
    print(f"  Per-ticker configs: {PER_TICKER_JSON}")
    print(f"  {positive}/{len(configs)} tickers with positive Sharpe")


def main():
    parser = argparse.ArgumentParser(
        description="BERYL Daily: Full NDX100 Optimizer (ensemble, daily bars)"
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
    print(f"\n  BERYL Daily Optimizer (Ensemble, Daily Bars)")
    print(f"  Grid: {total_grid} total combinations")
    print(f"  Tickers: {len(TICKERS)} (from citrine_per_ticker_configs.json)")
    print(f"  Sampling: {args.runs} trials ({100*args.runs/total_grid:.1f}% coverage)")
    print(f"  Workers: {args.workers}")

    if args.heatmap_only:
        if RESULTS_CSV.exists():
            results = pd.read_csv(RESULTS_CSV).to_dict("records")
            _generate_heatmap(results)
            _write_per_ticker_configs(results)
        return

    combos = _sample_combinations(args.runs)
    print(f"  N_States: {GRID['n_states']}")
    print(f"  Features: {GRID['feature_set']}")
    print(f"  Confirmations: {GRID['confirmations']}")
    print(f"  Covariance: {GRID['covariance_type']}")
    print(f"  Leverage: {GRID['leverage'][0]}, Cooldown: {GRID['cooldown_hours'][0]}h, Timeframe: 1d")

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
        _write_per_ticker_configs(existing)
        return

    # Run trials
    results = list(existing)
    valid = 0
    skipped = 0

    write_header = not RESULTS_CSV.exists() or not args.resume
    csvfile = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
    writer = None

    try:
        if args.workers > 1:
            # ── Parallel execution ────────────────────────────────────────
            print(f"  Starting {args.workers} parallel workers...\n")
            with Pool(args.workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_run_trial, remaining),
                    total=len(remaining),
                    desc="BERYL Daily",
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

        else:
            # ── Sequential execution (--workers 1) ────────────────────────
            for i, combo in enumerate(tqdm(remaining, desc="BERYL Daily")):
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
    print(f"  BERYL DAILY OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Valid: {valid}, Skipped: {skipped}, Workers: {args.workers}")

    if results:
        df = pd.DataFrame(results)
        top = df.nlargest(10, "wf_sharpe")
        print(f"\n  TOP 10:")
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            print(
                f"  #{rank:2d}  {r['ticker']:5s} "
                f"n={int(r['n_states'])} {r['feature_set']:12s} "
                f"cf={int(r['confirmations'])} "
                f"-> Sharpe {r['wf_sharpe']:+.3f}  "
                f"Return {r['wf_return']:+.1f}%"
            )

        # Per-ticker best (show top 20 by Sharpe for readability)
        print(f"\n  BEST PER TICKER (top 20):")
        ticker_bests = []
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker]
            if not ticker_df.empty:
                best = ticker_df.nlargest(1, "wf_sharpe").iloc[0]
                ticker_bests.append(best)
        ticker_bests.sort(key=lambda x: x["wf_sharpe"], reverse=True)
        for best in ticker_bests[:20]:
            print(
                f"  {best['ticker']:5s}: Sharpe {best['wf_sharpe']:+.3f}  "
                f"Return {best['wf_return']:+.1f}%  "
                f"({int(best['n_states'])}st/"
                f"{best['feature_set']}/{int(best['confirmations'])}cf)"
            )

        _generate_heatmap(results)
        _write_per_ticker_configs(results)

    # Notification
    try:
        from src.notifier import Notifier
        n = Notifier()
        if results:
            best = max(results, key=lambda r: r.get("wf_sharpe", -999))
            positive = sum(1 for r in results if r.get("wf_sharpe", -999) > 0)
            n.send_notification(
                "BERYL Daily Complete",
                f"Best: {best['ticker']} Sharpe {best['wf_sharpe']:.3f} | "
                f"{valid} valid, {positive} positive | {args.workers} workers",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
