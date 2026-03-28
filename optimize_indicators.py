"""
optimize_indicators.py
──────────────────────
Focused indicator threshold optimizer for AGATE.

Fixes the winning HMM config (SOL/4h/full/8cf/1.5x/72h/diag) and varies
the 4 most impactful indicator thresholds to find optimal entry filters.

Grid: 81 combos (3^4: volatility_mult x volume_mult x adx_min x stoch_upper)
With ensemble: same 81 combos using 3-model ensemble instead of single HMM.

Usage
─────
  python optimize_indicators.py                  # single-model (81 trials)
  python optimize_indicators.py --ensemble       # ensemble mode (81 trials)
  python optimize_indicators.py --resume         # resume from checkpoint
  python optimize_indicators.py --heatmap-only   # regenerate heatmap
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from walk_forward import run_walk_forward

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimize_indicators")

# ── Output paths ─────────────────────────────────────────────────────────────
RESULTS_CSV  = ROOT / "indicator_optimization_results.csv"
HEATMAP_HTML = ROOT / "indicator_optimization_heatmap.html"

# ── Fixed winner config (only indicator thresholds vary) ─────────────────────
FIXED_CONFIG = {
    "ticker":          "X:SOLUSD",
    "timeframe":       "4h",
    "n_states":        6,
    "feature_set":     "full",
    "confirmations":   8,
    "leverage":        1.5,
    "cooldown_hours":  72,
    "covariance_type": "diag",
}

# ── Indicator threshold grid ────────────────────────────────────────────────
INDICATOR_GRID = {
    "volatility_mult": [1.5, 2.0, 2.5],
    "volume_mult":     [1.0, 1.1, 1.2],
    "adx_min":         [20, 25, 30],
    "stoch_upper":     [70, 80, 90],
}

# Walk-forward parameters
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS  = 3
MIN_WINDOWS     = 2


def build_grid() -> list[dict]:
    """Build exhaustive grid of indicator threshold combos."""
    combos = []
    for vals in itertools.product(*INDICATOR_GRID.values()):
        params = dict(zip(INDICATOR_GRID.keys(), vals))
        combos.append(params)
    return combos


def _patch_indicators(params: dict) -> dict:
    """Monkey-patch indicator thresholds in config. Returns originals."""
    saved = {}
    patch_map = {
        "volatility_mult": "VOLATILITY_MULT",
        "volume_mult":     "VOLUME_MULT",
        "adx_min":         "ADX_MIN",
        "stoch_upper":     "STOCH_UPPER",
    }
    for param_key, cfg_attr in patch_map.items():
        saved[cfg_attr] = getattr(config, cfg_attr)
        setattr(config, cfg_attr, params[param_key])

    # Also patch fixed config
    fixed_map = {
        "n_states":        "N_STATES",
        "covariance_type": "COV_TYPE",
        "leverage":        "LEVERAGE",
        "cooldown_hours":  "COOLDOWN_HOURS",
    }
    for param_key, cfg_attr in fixed_map.items():
        saved[cfg_attr] = getattr(config, cfg_attr)
        setattr(config, cfg_attr, FIXED_CONFIG[param_key])

    return saved


def _restore_config(saved: dict) -> None:
    for attr, val in saved.items():
        setattr(config, attr, val)


def run_trial(params: dict, trial_id: int, use_ensemble: bool = False) -> dict | None:
    """Run one walk-forward trial with given indicator thresholds."""
    saved = _patch_indicators(params)
    try:
        results, combined_eq, combined_bh, all_trades = run_walk_forward(
            train_months=WF_TRAIN_MONTHS,
            test_months=WF_TEST_MONTHS,
            ticker=FIXED_CONFIG["ticker"],
            feature_set=FIXED_CONFIG["feature_set"],
            confirmations=FIXED_CONFIG["confirmations"],
            timeframe=FIXED_CONFIG["timeframe"],
            quiet=True,
            use_regime_mapper=False,
            use_ensemble=use_ensemble,
        )

        if len(results) < MIN_WINDOWS:
            return None
        if combined_eq.empty or len(combined_eq) < 10:
            return None

        # Metrics
        oos_initial = float(combined_eq.iloc[0])
        oos_final = float(combined_eq.iloc[-1])
        oos_ret = (oos_final / oos_initial - 1) * 100

        roll_max = combined_eq.cummax()
        oos_dd = float(((combined_eq - roll_max) / roll_max * 100).min())

        hr = combined_eq.pct_change().dropna()
        oos_sharpe = (
            float(hr.mean() / hr.std() * np.sqrt(24 * 365))
            if len(hr) > 1 and hr.std() > 0 else 0.0
        )

        window_sharpes = [r.sharpe_ratio for r in results]
        window_returns = [r.return_pct for r in results]
        n_windows = len(results)
        pos_windows = sum(1 for r in results if r.return_pct > 0)
        total_trades = sum(r.n_trades for r in results)

        if not all_trades.empty and "pnl" in all_trades.columns:
            win_rate = float((all_trades["pnl"] > 0).sum() / len(all_trades) * 100)
        else:
            win_rate = 0.0

        row = {
            "trial_id": trial_id,
            **params,
            "ensemble": use_ensemble,
            "wf_sharpe": round(oos_sharpe, 3),
            "wf_return": round(oos_ret, 2),
            "wf_drawdown": round(oos_dd, 2),
            "wf_n_windows": n_windows,
            "wf_pos_windows": pos_windows,
            "wf_trades": total_trades,
            "wf_win_rate": round(win_rate, 1),
            "wf_mean_sharpe": round(float(np.mean(window_sharpes)), 3),
            "wf_std_sharpe": round(float(np.std(window_sharpes)), 3),
            "wf_mean_return": round(float(np.mean(window_returns)), 2),
            "window_sharpes": ",".join(f"{s:.3f}" for s in window_sharpes),
            "window_returns": ",".join(f"{r:.2f}" for r in window_returns),
        }
        return row

    except Exception as exc:
        log.warning("Trial %d failed: %s", trial_id, exc)
        return None
    finally:
        _restore_config(saved)


def generate_heatmap(results_csv: Path, output_html: Path) -> None:
    """Generate indicator threshold heatmaps."""
    df = pd.read_csv(results_csv)
    if df.empty:
        print("  No results to plot")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Volatility Mult x Volume Mult (mean Sharpe)",
            "ADX Min x Stochastic Upper (mean Sharpe)",
            "Volatility Mult x ADX Min (mean Sharpe)",
            "Volume Mult x Stochastic Upper (mean Sharpe)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    pairs = [
        ("volatility_mult", "volume_mult"),
        ("adx_min", "stoch_upper"),
        ("volatility_mult", "adx_min"),
        ("volume_mult", "stoch_upper"),
    ]

    for idx, (x_col, y_col) in enumerate(pairs):
        r, c = divmod(idx, 2)
        p = df.groupby([y_col, x_col])["wf_sharpe"].mean().reset_index()
        hm = p.pivot(index=y_col, columns=x_col, values="wf_sharpe")

        show_colorbar = (idx == 3)
        fig.add_trace(
            go.Heatmap(
                z=hm.values,
                x=[str(v) for v in hm.columns],
                y=[str(v) for v in hm.index],
                colorscale="RdYlGn",
                text=np.round(hm.values, 3),
                texttemplate="%{text:.3f}",
                showscale=show_colorbar,
                colorbar=dict(title="Sharpe") if show_colorbar else None,
            ),
            row=r + 1, col=c + 1,
        )
        fig.update_xaxes(title_text=x_col, row=r + 1, col=c + 1)
        fig.update_yaxes(title_text=y_col, row=r + 1, col=c + 1)

    fig.update_layout(
        template="plotly_dark",
        height=800,
        width=1100,
        title="AGATE Indicator Threshold Optimization — Walk-Forward Sharpe",
    )

    fig.write_html(str(output_html))
    print(f"  Heatmap: {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="AGATE Indicator Threshold Optimizer"
    )
    parser.add_argument("--ensemble", action="store_true",
                        help="Use 3-model ensemble (n_states=[5,6,7])")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint CSV")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate heatmap from existing CSV")
    args = parser.parse_args()

    if args.heatmap_only:
        generate_heatmap(RESULTS_CSV, HEATMAP_HTML)
        return

    grid = build_grid()
    print(f"\n  Indicator threshold grid: {len(grid)} combos")
    print(f"  Mode: {'ensemble' if args.ensemble else 'single-model'}")
    print(f"  Fixed config: {FIXED_CONFIG}")

    # Check for resume
    completed: set = set()
    if args.resume and RESULTS_CSV.exists():
        df_done = pd.read_csv(RESULTS_CSV)
        for _, row in df_done.iterrows():
            key = (row["volatility_mult"], row["volume_mult"],
                   row["adx_min"], row["stoch_upper"])
            completed.add(key)
        print(f"  Resuming: {len(completed)} trials done")

    remaining = []
    for params in grid:
        key = (params["volatility_mult"], params["volume_mult"],
               params["adx_min"], params["stoch_upper"])
        if key not in completed:
            remaining.append(params)

    print(f"  Remaining: {len(remaining)} trials")

    if not remaining:
        print("  All trials complete!")
        generate_heatmap(RESULTS_CSV, HEATMAP_HTML)
        return

    # Run trials
    print(f"\n{'=' * 70}")
    print(f"  INDICATOR THRESHOLD OPTIMIZATION ({len(remaining)} trials)")
    print(f"{'=' * 70}")

    all_rows: list[dict] = []
    start_time = time.time()

    write_header = not RESULTS_CSV.exists() or not args.resume
    csvfile = open(RESULTS_CSV, "a" if args.resume else "w", newline="")
    writer = None

    try:
        for i, params in enumerate(remaining):
            trial_id = len(completed) + i + 1
            t0 = time.time()

            result = run_trial(params, trial_id, use_ensemble=args.ensemble)

            dt = time.time() - t0
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0

            if result is not None:
                all_rows.append(result)

                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                    if write_header:
                        writer.writeheader()
                        write_header = False
                writer.writerow(result)
                csvfile.flush()

                print(
                    f"  [{trial_id:3d}/{len(grid)}] "
                    f"vol_mult={params['volatility_mult']:.1f} "
                    f"vol_req={params['volume_mult']:.1f} "
                    f"adx={params['adx_min']:2d} "
                    f"stoch={params['stoch_upper']:2d} "
                    f"-> Sharpe {result['wf_sharpe']:+.3f}  "
                    f"Return {result['wf_return']:+.1f}%  "
                    f"Trades {result['wf_trades']}  "
                    f"({dt:.0f}s, ETA {eta / 60:.0f}min)"
                )
            else:
                print(
                    f"  [{trial_id:3d}/{len(grid)}] "
                    f"vol_mult={params['volatility_mult']:.1f} "
                    f"vol_req={params['volume_mult']:.1f} "
                    f"adx={params['adx_min']:2d} "
                    f"stoch={params['stoch_upper']:2d} "
                    f"-> SKIP ({dt:.0f}s)"
                )

    finally:
        csvfile.close()

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Valid trials: {len(all_rows)}/{len(remaining)}")
    print(f"  Time: {elapsed / 60:.1f}min")

    if RESULTS_CSV.exists():
        df_all = pd.read_csv(RESULTS_CSV)
        top = df_all.nlargest(10, "wf_sharpe")
        print(f"\n  {'─' * 55}")
        print(f"  TOP 10 INDICATOR CONFIGS")
        print(f"  {'─' * 55}")

        # Current baseline
        baseline = df_all[
            (df_all["volatility_mult"] == 2.0) &
            (df_all["volume_mult"] == 1.1) &
            (df_all["adx_min"] == 25) &
            (df_all["stoch_upper"] == 80)
        ]
        if not baseline.empty:
            b = baseline.iloc[0]
            print(
                f"  BASELINE: Sharpe {b['wf_sharpe']:+.3f}  "
                f"Return {b['wf_return']:+.1f}%  "
                f"(vol=2.0 vol_req=1.1 adx=25 stoch=80)"
            )
            print()

        for rank, (_, r) in enumerate(top.iterrows(), 1):
            print(
                f"  #{rank:2d}  vol_mult={r['volatility_mult']:.1f} "
                f"vol_req={r['volume_mult']:.1f} "
                f"adx={int(r['adx_min']):2d} "
                f"stoch={int(r['stoch_upper']):2d} "
                f"-> Sharpe {r['wf_sharpe']:+.3f}  "
                f"Return {r['wf_return']:+.1f}%  "
                f"DD {r['wf_drawdown']:.1f}%  "
                f"Trades {int(r['wf_trades'])}  "
                f"WR {r['wf_win_rate']:.0f}%"
            )

        # Per-param analysis
        print(f"\n  {'─' * 55}")
        print(f"  PARAMETER SENSITIVITY")
        print(f"  {'─' * 55}")
        for param in INDICATOR_GRID:
            group = df_all.groupby(param)["wf_sharpe"].agg(["mean", "std", "count"])
            print(f"\n  {param}:")
            for val, row in group.iterrows():
                print(f"    {val:>5}: mean Sharpe {row['mean']:+.3f} (std {row['std']:.3f}, n={int(row['count'])})")

        print(f"\n  Saved: {RESULTS_CSV}")
        generate_heatmap(RESULTS_CSV, HEATMAP_HTML)

    # Notification
    try:
        from src.notifier import Notifier
        n = Notifier()
        if all_rows:
            best = max(all_rows, key=lambda r: r["wf_sharpe"])
            n.send_notification(
                "AGATE Indicator Optimization Complete",
                f"Best Sharpe {best['wf_sharpe']:.3f} | "
                f"vol={best['volatility_mult']} adx={best['adx_min']} "
                f"stoch={best['stoch_upper']}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
