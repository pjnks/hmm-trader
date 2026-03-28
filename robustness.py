"""
robustness.py
─────────────
Robustness testing suite for the HMM Regime Trader.

Stress-tests the best walk-forward validated config with 4 tests:
  1. Monte Carlo trade bootstrap  — resample trades 10K times
  2. Walk-forward window variation — 5 different train/test splits
  3. Parameter sensitivity         — vary each param +-1 step (tornado chart)
  4. Ticker transferability        — run exact config on BTC/ETH/XRP/SOL

Designed around a StrategyProtocol interface so the same tests can be reused
for future strategies (e.g. NDX100) without modifying this file.

Usage
─────
  python robustness.py                    # run all 4 tests
  python robustness.py --test bootstrap   # single test
  python robustness.py --test sensitivity
  python robustness.py --test windows
  python robustness.py --test transfer

Output
──────
  robustness_results.json   — structured pass/fail + metrics
  robustness_report.html    — interactive Plotly report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import fetch_btc_hourly
from src.strategy_protocol import HMMCryptoStrategy, WFResult

log = logging.getLogger("robustness")

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_JSON = ROOT / "robustness_results.json"
REPORT_HTML  = ROOT / "robustness_report.html"

# ── Test names ────────────────────────────────────────────────────────────────
TEST_NAMES = ["bootstrap", "windows", "sensitivity", "transfer"]

# ── Window variation splits ───────────────────────────────────────────────────
WINDOW_SPLITS = [
    (3,  3),   # 3m train / 3m test
    (6,  3),   # 6m train / 3m test  (base)
    (9,  3),   # 9m train / 3m test
    (6,  6),   # 6m train / 6m test
    (12, 3),   # 12m train / 3m test
]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-cache data
# ─────────────────────────────────────────────────────────────────────────────

def _pre_cache_tickers(tickers: list) -> None:
    """Pre-fetch 1h OHLCV for each ticker so walk-forward doesn't re-download."""
    print("Pre-caching ticker data:")
    for ticker in tickers:
        print(f"  {ticker} …", end=" ", flush=True)
        try:
            df = fetch_btc_hourly(days=760, ticker=ticker)
            print(f"{len(df)} bars")
        except Exception as e:
            print(f"FAILED: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Monte Carlo Trade Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap(strategy, base_wf: WFResult,
                  n_iterations: int = 10_000, seed: int = 42) -> dict:
    """Resample trades with replacement to build Sharpe/return/DD distributions.

    Pass criterion: 5th-percentile Sharpe > 0 (95% confidence the edge is real).
    """
    trades = base_wf.trades_df
    pnls = trades["pnl"].values
    n = len(pnls)

    if n < 5:
        print(f"  Only {n} trades — too few for bootstrap")
        return {"test": "bootstrap", "passed": False, "error": "too_few_trades",
                "n_trades": n}

    print(f"  Bootstrapping {n} trades × {n_iterations:,} iterations …",
          end=" ", flush=True)

    rng = np.random.RandomState(seed)
    initial_capital = config.INITIAL_CAPITAL

    sharpes   = np.empty(n_iterations)
    returns   = np.empty(n_iterations)
    drawdowns = np.empty(n_iterations)

    for i in range(n_iterations):
        # Resample trades WITH replacement
        sample_pnls = pnls[rng.randint(0, n, size=n)]

        # Reconstruct equity curve
        equity = np.empty(n + 1)
        equity[0] = initial_capital
        np.cumsum(sample_pnls, out=equity[1:])
        equity[1:] += initial_capital

        # Total return %
        returns[i] = (equity[-1] / initial_capital - 1) * 100

        # Max drawdown %
        running_max = np.maximum.accumulate(equity)
        drawdowns[i] = ((equity - running_max) / running_max * 100).min()

        # Trade-level Sharpe: mean(pnl)/std(pnl) * sqrt(n_trades)
        std = sample_pnls.std()
        sharpes[i] = (sample_pnls.mean() / std * np.sqrt(n)) if std > 0 else 0.0

    p5_sharpe = float(np.percentile(sharpes, 5))
    passed = p5_sharpe > 0

    print(f"p5_sharpe={p5_sharpe:.3f}  {'PASS' if passed else 'FAIL'}")

    return {
        "test":         "bootstrap",
        "passed":       passed,
        "n_iterations": n_iterations,
        "n_trades":     n,
        "p5_sharpe":    round(p5_sharpe, 3),
        "p50_sharpe":   round(float(np.percentile(sharpes, 50)), 3),
        "p95_sharpe":   round(float(np.percentile(sharpes, 95)), 3),
        "p5_return":    round(float(np.percentile(returns, 5)), 2),
        "p50_return":   round(float(np.percentile(returns, 50)), 2),
        "p95_return":   round(float(np.percentile(returns, 95)), 2),
        "p5_drawdown":  round(float(np.percentile(drawdowns, 5)), 2),
        "p50_drawdown": round(float(np.percentile(drawdowns, 50)), 2),
        "p95_drawdown": round(float(np.percentile(drawdowns, 95)), 2),
        # Keep distributions in memory for the report (not saved to JSON)
        "sharpe_dist":   sharpes.tolist(),
        "return_dist":   returns.tolist(),
        "drawdown_dist": drawdowns.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Walk-Forward Window Variation
# ─────────────────────────────────────────────────────────────────────────────

def run_window_variation(strategy) -> dict:
    """Run the same config with different train/test window sizes.

    Pass criterion: >= 3 of 5 splits have positive OOS Sharpe.
    """
    base_params = strategy.get_base_params()
    results = []

    for train_m, test_m in WINDOW_SPLITS:
        label = f"{train_m}m/{test_m}m"
        print(f"  Window {label} …", end=" ", flush=True)

        params = {**base_params, "_train_months": train_m, "_test_months": test_m}
        wf = strategy.run_walk_forward(params)

        if wf is None:
            print("skipped (insufficient data)")
            results.append({
                "split": label, "sharpe": None, "return_pct": None,
                "max_dd_pct": None, "n_trades": 0, "n_windows": 0,
                "passed": False,
            })
        else:
            passed = wf.sharpe > 0
            print(f"sharpe={wf.sharpe:.3f}  return={wf.return_pct:+.1f}%  "
                  f"trades={wf.n_trades}  {'PASS' if passed else 'FAIL'}")
            results.append({
                "split": label, "sharpe": wf.sharpe,
                "return_pct": wf.return_pct, "max_dd_pct": wf.max_dd_pct,
                "n_trades": wf.n_trades, "n_windows": wf.n_windows,
                "passed": passed,
            })

    n_passed = sum(1 for r in results if r["passed"])
    overall_passed = n_passed >= 3

    return {
        "test":     "window_variation",
        "passed":   overall_passed,
        "n_passed": n_passed,
        "n_total":  len(WINDOW_SPLITS),
        "splits":   results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Parameter Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(strategy, base_wf: WFResult = None) -> dict:
    """Vary each parameter +-1 step from best, measure Sharpe impact.

    Pass criterion: ALL +-1-step variants have positive WF Sharpe.
    """
    base_params = strategy.get_base_params()
    grid = strategy.get_param_sensitivity_grid()

    # Use cached base result if available
    if base_wf is not None:
        base_sharpe = base_wf.sharpe
        print(f"  Base config (cached): sharpe={base_sharpe:.3f}")
    else:
        print(f"  Base config …", end=" ", flush=True)
        base_wf = strategy.run_walk_forward(base_params)
        if base_wf is None:
            print("FAILED")
            return {"test": "sensitivity", "passed": False,
                    "error": "base_config_failed"}
        base_sharpe = base_wf.sharpe
        print(f"sharpe={base_sharpe:.3f}")

    results = []
    all_positive = True

    for param_name, values in grid.items():
        base_val = base_params[param_name]

        for val in values:
            if val == base_val:
                continue

            variant_params = {**base_params, param_name: val}
            label = f"{param_name}={val}"
            print(f"  {label} …", end=" ", flush=True)

            wf = strategy.run_walk_forward(variant_params)

            if wf is None:
                print("skipped (failed)")
                results.append({
                    "param": param_name, "value": val, "base_value": base_val,
                    "sharpe": None, "delta_sharpe": None,
                    "return_pct": None, "passed": False,
                })
                all_positive = False
            else:
                delta = wf.sharpe - base_sharpe
                passed = wf.sharpe > 0
                if not passed:
                    all_positive = False
                print(f"sharpe={wf.sharpe:.3f}  delta={delta:+.3f}  "
                      f"{'PASS' if passed else 'FAIL'}")
                results.append({
                    "param": param_name, "value": val, "base_value": base_val,
                    "sharpe": wf.sharpe, "delta_sharpe": round(delta, 3),
                    "return_pct": wf.return_pct, "passed": passed,
                })

    return {
        "test":         "sensitivity",
        "passed":       all_positive,
        "base_sharpe":  base_sharpe,
        "variants":     results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Ticker Transferability
# ─────────────────────────────────────────────────────────────────────────────

def run_transferability(strategy) -> dict:
    """Run exact config on all available tickers.

    Pass criterion: >= 2 of 4 tickers have positive WF Sharpe.
    """
    base_params = strategy.get_base_params()
    tickers = strategy.get_tickers()
    ticker_key = strategy.get_ticker_param_name()

    results = []

    for ticker in tickers:
        variant_params = {**base_params, ticker_key: ticker}
        print(f"  {ticker} …", end=" ", flush=True)

        wf = strategy.run_walk_forward(variant_params)

        if wf is None:
            print("skipped (insufficient data or failed)")
            results.append({
                "ticker": ticker, "sharpe": None, "return_pct": None,
                "max_dd_pct": None, "n_trades": 0, "n_windows": 0,
                "passed": False,
            })
        else:
            passed = wf.sharpe > 0
            print(f"sharpe={wf.sharpe:.3f}  return={wf.return_pct:+.1f}%  "
                  f"trades={wf.n_trades}  {'PASS' if passed else 'FAIL'}")
            results.append({
                "ticker": ticker, "sharpe": wf.sharpe,
                "return_pct": wf.return_pct, "max_dd_pct": wf.max_dd_pct,
                "n_trades": wf.n_trades, "n_windows": wf.n_windows,
                "passed": passed,
            })

    n_passed = sum(1 for r in results if r["passed"])
    overall_passed = n_passed >= 2

    return {
        "test":     "transferability",
        "passed":   overall_passed,
        "n_passed": n_passed,
        "n_total":  len(tickers),
        "tickers":  results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save results (JSON)
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(all_results: dict, path: Path) -> None:
    """Save results to JSON, excluding large distribution arrays."""
    clean = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            val = {k: v for k, v in val.items() if not k.endswith("_dist")}
        clean[key] = val

    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Build Plotly HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(all_results: dict, output: Path) -> None:
    """Generate an interactive Plotly HTML report for all completed tests."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Count how many test sections we have
    sections = []
    if "bootstrap" in all_results:
        sections.append("bootstrap")
    if "windows" in all_results:
        sections.append("windows")
    if "sensitivity" in all_results:
        sections.append("sensitivity")
    if "transfer" in all_results:
        sections.append("transfer")

    if not sections:
        return

    # Build subplot grid: 2 columns per section
    n_rows = len(sections)
    titles = []
    for s in sections:
        if s == "bootstrap":
            titles += ["Sharpe Distribution (10K bootstrap)",
                        "Return Distribution (10K bootstrap)"]
        elif s == "windows":
            titles += ["OOS Sharpe by Window Split",
                        "OOS Return by Window Split"]
        elif s == "sensitivity":
            titles += ["Sharpe Delta from Base (Tornado)",
                        "Absolute Sharpe by Variant"]
        elif s == "transfer":
            titles += ["OOS Sharpe by Ticker",
                        "OOS Return by Ticker"]

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.12 if n_rows <= 2 else 0.08,
        horizontal_spacing=0.10,
    )

    row_idx = 0

    # ── Bootstrap ──────────────────────────────────────────────────────────
    if "bootstrap" in all_results:
        row_idx += 1
        bs = all_results["bootstrap"]

        if "sharpe_dist" in bs:
            sharpes = bs["sharpe_dist"]
            fig.add_trace(go.Histogram(
                x=sharpes, nbinsx=80, name="Sharpe",
                marker_color="#00ff88", opacity=0.75,
                showlegend=False,
            ), row=row_idx, col=1)
            # Percentile lines
            for pct, val in [("p5", bs["p5_sharpe"]),
                             ("p50", bs["p50_sharpe"])]:
                fig.add_vline(x=val, line_dash="dash", line_color="#ffcc00",
                              annotation_text=f"{pct}={val:.2f}",
                              annotation_font_color="#ffcc00",
                              row=row_idx, col=1)
            # Zero line
            fig.add_vline(x=0, line_color="#ff2244", line_width=2,
                          row=row_idx, col=1)

        if "return_dist" in bs:
            rets = bs["return_dist"]
            fig.add_trace(go.Histogram(
                x=rets, nbinsx=80, name="Return %",
                marker_color="#66bbff", opacity=0.75,
                showlegend=False,
            ), row=row_idx, col=2)
            for pct, val in [("p5", bs["p5_return"]),
                             ("p50", bs["p50_return"])]:
                fig.add_vline(x=val, line_dash="dash", line_color="#ffcc00",
                              annotation_text=f"{pct}={val:.1f}%",
                              annotation_font_color="#ffcc00",
                              row=row_idx, col=2)
            fig.add_vline(x=0, line_color="#ff2244", line_width=2,
                          row=row_idx, col=2)

    # ── Window Variation ───────────────────────────────────────────────────
    if "windows" in all_results:
        row_idx += 1
        wv = all_results["windows"]
        splits = wv["splits"]
        labels  = [s["split"] for s in splits]
        sharpes = [s["sharpe"] if s["sharpe"] is not None else 0 for s in splits]
        rets    = [s["return_pct"] if s["return_pct"] is not None else 0
                   for s in splits]
        colors  = ["#00ff88" if s["passed"] else "#ff2244" for s in splits]

        fig.add_trace(go.Bar(
            x=labels, y=sharpes, name="Sharpe",
            marker_color=colors, showlegend=False,
        ), row=row_idx, col=1)
        fig.add_hline(y=0, line_color="#ff2244", line_width=1,
                      row=row_idx, col=1)

        fig.add_trace(go.Bar(
            x=labels, y=rets, name="Return %",
            marker_color=colors, showlegend=False,
        ), row=row_idx, col=2)
        fig.add_hline(y=0, line_color="#ff2244", line_width=1,
                      row=row_idx, col=2)

    # ── Sensitivity (Tornado) ──────────────────────────────────────────────
    if "sensitivity" in all_results:
        row_idx += 1
        sens = all_results["sensitivity"]

        if "variants" in sens:
            variants = [v for v in sens["variants"] if v["delta_sharpe"] is not None]
            # Sort by absolute delta for tornado chart
            variants.sort(key=lambda v: abs(v["delta_sharpe"]), reverse=True)

            labels  = [f"{v['param']}={v['value']}" for v in variants]
            deltas  = [v["delta_sharpe"] for v in variants]
            abs_s   = [v["sharpe"] for v in variants]
            colors_d = ["#00ff88" if d >= 0 else "#ff2244" for d in deltas]
            colors_a = ["#00ff88" if s > 0 else "#ff2244" for s in abs_s]

            fig.add_trace(go.Bar(
                y=labels, x=deltas, orientation="h", name="Delta",
                marker_color=colors_d, showlegend=False,
            ), row=row_idx, col=1)
            fig.add_vline(x=0, line_color="#e0e4f0", line_width=1,
                          row=row_idx, col=1)

            # Sort absolute Sharpe chart by value for readability
            abs_sorted = sorted(zip(labels, abs_s, colors_a),
                                key=lambda x: x[1], reverse=True)
            fig.add_trace(go.Bar(
                x=[a[0] for a in abs_sorted],
                y=[a[1] for a in abs_sorted],
                name="Sharpe", marker_color=[a[2] for a in abs_sorted],
                showlegend=False,
            ), row=row_idx, col=2)
            # Base Sharpe reference line
            fig.add_hline(y=sens["base_sharpe"], line_dash="dash",
                          line_color="#ffcc00",
                          annotation_text=f"base={sens['base_sharpe']:.3f}",
                          annotation_font_color="#ffcc00",
                          row=row_idx, col=2)
            fig.add_hline(y=0, line_color="#ff2244", line_width=1,
                          row=row_idx, col=2)

    # ── Transferability ────────────────────────────────────────────────────
    if "transfer" in all_results:
        row_idx += 1
        tr = all_results["transfer"]
        tickers_data = tr["tickers"]
        labels  = [t["ticker"].replace("X:", "") for t in tickers_data]
        sharpes = [t["sharpe"] if t["sharpe"] is not None else 0
                   for t in tickers_data]
        rets    = [t["return_pct"] if t["return_pct"] is not None else 0
                   for t in tickers_data]
        colors  = ["#00ff88" if t["passed"] else "#ff2244"
                   for t in tickers_data]

        fig.add_trace(go.Bar(
            x=labels, y=sharpes, name="Sharpe",
            marker_color=colors, showlegend=False,
        ), row=row_idx, col=1)
        fig.add_hline(y=0, line_color="#ff2244", line_width=1,
                      row=row_idx, col=1)

        fig.add_trace(go.Bar(
            x=labels, y=rets, name="Return %",
            marker_color=colors, showlegend=False,
        ), row=row_idx, col=2)
        fig.add_hline(y=0, line_color="#ff2244", line_width=1,
                      row=row_idx, col=2)

    # ── Layout ─────────────────────────────────────────────────────────────
    # Build title with pass/fail summary
    test_summary = []
    for s in sections:
        res = all_results[s]
        status = "PASS" if res.get("passed") else "FAIL"
        test_summary.append(f"{s}: {status}")
    title_text = "Robustness Report  —  " + "  |  ".join(test_summary)

    fig.update_layout(
        height=400 * n_rows,
        title=dict(text=title_text, font=dict(size=14)),
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141820",
        font=dict(family="monospace", color="#e0e4f0", size=11),
    )
    fig.update_xaxes(gridcolor="#1e2330")
    fig.update_yaxes(gridcolor="#1e2330")

    # Write HTML
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    # Inject dark body background
    html = html.replace("<body>",
        '<body style="background:#0d0f14; color:#e0e4f0; '
        'font-family:monospace; margin:0;">')

    with open(output, "w") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM Regime Trader — Robustness Testing Suite",
    )
    parser.add_argument(
        "--test", choices=TEST_NAMES, default=None,
        help="Run a single test (default: run all 4)",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use 3-model HMM ensemble (n_states=[5,6,7])",
    )
    args = parser.parse_args()

    strategy = HMMCryptoStrategy(use_ensemble=args.ensemble)
    base_params = strategy.get_base_params()

    print(f"\n{'=' * 60}")
    print(f"  Robustness Testing Suite")
    print(f"{'=' * 60}")
    print(f"  Base config:")
    for k, v in base_params.items():
        print(f"    {k:20s}: {v}")
    print()

    tests_to_run = [args.test] if args.test else TEST_NAMES

    all_results = {
        "run_date":    datetime.now().isoformat(),
        "base_config": base_params,
    }

    # ── Run base WF once (shared by bootstrap + sensitivity) ──────────────
    base_wf = None
    if "bootstrap" in tests_to_run or "sensitivity" in tests_to_run:
        print("Running base config walk-forward …", end=" ", flush=True)
        base_wf = strategy.run_walk_forward(base_params)
        if base_wf is None:
            print("FAILED — cannot proceed")
            sys.exit(1)
        print(f"sharpe={base_wf.sharpe:.3f}  return={base_wf.return_pct:+.1f}%  "
              f"trades={base_wf.n_trades}\n")

    # ── Pre-cache tickers for transferability ─────────────────────────────
    if "transfer" in tests_to_run:
        _pre_cache_tickers(strategy.get_tickers())
        print()

    # ── Run tests ─────────────────────────────────────────────────────────
    t0_total = time.time()

    for test_name in tests_to_run:
        print(f"{'─' * 60}")
        print(f"  TEST: {test_name.upper()}")
        print(f"{'─' * 60}")

        t0 = time.time()

        if test_name == "bootstrap":
            result = run_bootstrap(strategy, base_wf)
        elif test_name == "windows":
            result = run_window_variation(strategy)
        elif test_name == "sensitivity":
            result = run_sensitivity(strategy, base_wf=base_wf)
        elif test_name == "transfer":
            result = run_transferability(strategy)
        else:
            continue

        elapsed = time.time() - t0
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n  Result: {status}  ({elapsed:.0f}s)\n")

        all_results[test_name] = result

    total_elapsed = time.time() - t0_total

    # ── Summary ───────────────────────────────────────────────────────────
    test_results = {k: v for k, v in all_results.items()
                    if isinstance(v, dict) and "passed" in v}
    n_passed = sum(1 for v in test_results.values() if v["passed"])
    n_total  = len(test_results)
    all_results["tests_passed"]   = n_passed
    all_results["tests_total"]    = n_total
    all_results["overall_passed"] = (n_passed == n_total)

    print(f"{'=' * 60}")
    print(f"  ROBUSTNESS SUMMARY: {n_passed}/{n_total} tests passed  "
          f"({total_elapsed:.0f}s)")
    for name, res in test_results.items():
        status = "PASS" if res["passed"] else "FAIL"
        print(f"    {name:20s}: {status}")
    print(f"{'=' * 60}\n")

    # ── Save outputs ──────────────────────────────────────────────────────
    _save_results(all_results, RESULTS_JSON)
    print(f"Results saved  -> {RESULTS_JSON}")

    _build_report(all_results, REPORT_HTML)
    print(f"Report saved   -> {REPORT_HTML}")


if __name__ == "__main__":
    main()
