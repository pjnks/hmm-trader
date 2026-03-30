#!/usr/bin/env python3
"""
global_config_test.py — Overfitting Diagnostic (Sprint 9)
─────────────────────────────────────────────────────────
Tests whether AGATE's HMM edge is real or just per-ticker curve-fitting.

Runs a SINGLE unified config across ALL AGATE tickers via walk-forward.
If the aggregate portfolio Sharpe is positive → the system finds real structure.
If it collapses → the per-ticker optimizer was memorizing noise.

Two configs tested:
  A) Most common winner:  4h / diag / 4st / base / 5cf   (simplest model)
  B) A/B validated:       4h / diag / 5st / extended_v2 / 6cf  (Sprint 3 winner)

Usage:
  python global_config_test.py              # Run both configs
  python global_config_test.py --quick      # Fewer tickers (BTC, ETH, SOL, BCH, LINK only)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

import config


# ── Unified configs to test ──────────────────────────────────────────────────

CONFIGS = {
    "A_simple": {
        "label": "A) Simple (base/4st/5cf/diag/4h)",
        "n_states": 4,
        "feature_set": "base",
        "confirmations": 5,
        "cov_type": "diag",
        "timeframe": "4h",
    },
    "B_validated": {
        "label": "B) A/B Validated (extended_v2/5st/6cf/diag/4h)",
        "n_states": 5,
        "feature_set": "extended_v2",
        "confirmations": 6,
        "cov_type": "diag",
        "timeframe": "4h",
    },
}

QUICK_TICKERS = [
    "X:BTCUSD", "X:ETHUSD", "X:SOLUSD", "X:BCHUSD", "X:LINKUSD",
]


@dataclass
class TickerResult:
    ticker: str
    wf_sharpe: float
    wf_return: float
    n_trades: int
    n_windows: int
    positive_windows: int
    converged: bool


def run_global_test(cfg: dict, tickers: list[str], quiet: bool = False) -> list[TickerResult]:
    """Run a single unified config across all tickers."""
    from walk_forward import run_walk_forward

    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        if not quiet:
            print(f"\n  [{i}/{total}] {ticker} ...")

        # Patch config for this trial
        config.N_STATES = cfg["n_states"]
        config.FEATURE_SET = cfg["feature_set"]
        config.COV_TYPE = cfg["cov_type"]
        config.LEVERAGE = 1.0
        config.COOLDOWN_HOURS = 48

        try:
            wf_results, combined_eq, combined_bh, all_trades = run_walk_forward(
                train_months=6,
                test_months=3,
                ticker=ticker,
                feature_set=cfg["feature_set"],
                confirmations=cfg["confirmations"],
                timeframe=cfg["timeframe"],
                quiet=True,
                use_ensemble=True,
            )

            if not wf_results:
                if not quiet:
                    print(f"    → No results (insufficient data or all windows failed)")
                results.append(TickerResult(
                    ticker=ticker, wf_sharpe=0.0, wf_return=0.0,
                    n_trades=0, n_windows=0, positive_windows=0, converged=False,
                ))
                continue

            sharpes = [r.sharpe_ratio for r in wf_results
                       if r.sharpe_ratio is not None and not np.isnan(r.sharpe_ratio)]
            returns = [r.return_pct for r in wf_results if r.return_pct is not None]
            trades = sum(r.n_trades for r in wf_results)
            pos_windows = sum(1 for s in sharpes if s > 0)

            mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
            mean_return = float(np.mean(returns)) if returns else 0.0

            results.append(TickerResult(
                ticker=ticker, wf_sharpe=mean_sharpe, wf_return=mean_return,
                n_trades=trades, n_windows=len(wf_results),
                positive_windows=pos_windows, converged=True,
            ))

            if not quiet:
                status = "+" if mean_sharpe > 0 else "-"
                print(f"    → Sharpe {mean_sharpe:+.3f}  Return {mean_return:+.1f}%  "
                      f"Trades {trades}  Windows {pos_windows}/{len(wf_results)} [{status}]")

        except Exception as e:
            if not quiet:
                print(f"    → ERROR: {e}")
            results.append(TickerResult(
                ticker=ticker, wf_sharpe=0.0, wf_return=0.0,
                n_trades=0, n_windows=0, positive_windows=0, converged=False,
            ))

    return results


def print_summary(label: str, results: list[TickerResult]):
    """Print aggregate summary for one config."""
    converged = [r for r in results if r.converged]
    positive = [r for r in converged if r.wf_sharpe > 0]

    if not converged:
        print(f"\n  {label}: NO CONVERGED RESULTS")
        return

    sharpes = [r.wf_sharpe for r in converged]
    returns = [r.wf_return for r in converged]
    total_trades = sum(r.n_trades for r in converged)
    all_positive = sum(r.positive_windows for r in converged)
    all_windows = sum(r.n_windows for r in converged)

    agg_sharpe = float(np.mean(sharpes))
    agg_return = float(np.mean(returns))
    pct_positive = len(positive) / len(converged) * 100

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Tickers tested:    {len(results)}")
    print(f"  Tickers converged: {len(converged)}")
    print(f"  Tickers positive:  {len(positive)} ({pct_positive:.0f}%)")
    print(f"  Aggregate Sharpe:  {agg_sharpe:+.3f}")
    print(f"  Aggregate Return:  {agg_return:+.2f}%")
    print(f"  Total trades:      {total_trades}")
    print(f"  Positive windows:  {all_positive}/{all_windows} "
          f"({all_positive/max(all_windows,1)*100:.0f}%)")
    print()

    # Per-ticker breakdown
    print(f"  {'TICKER':<12} {'SHARPE':>8} {'RETURN':>8} {'TRADES':>7} {'WINDOWS':>9}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*7} {'─'*9}")
    for r in sorted(converged, key=lambda x: x.wf_sharpe, reverse=True):
        marker = "★" if r.positive_windows == r.n_windows and r.n_windows > 0 else " "
        print(f"  {r.ticker:<12} {r.wf_sharpe:>+8.3f} {r.wf_return:>+7.1f}% "
              f"{r.n_trades:>7} {r.positive_windows}/{r.n_windows:>2} {marker}")

    # Verdict
    print()
    if agg_sharpe > 0 and pct_positive >= 50:
        print(f"  VERDICT: REAL EDGE — aggregate Sharpe positive, "
              f"{pct_positive:.0f}% of tickers positive")
    elif agg_sharpe > 0:
        print(f"  VERDICT: WEAK EDGE — aggregate Sharpe positive but only "
              f"{pct_positive:.0f}% of tickers positive")
    else:
        print(f"  VERDICT: NO EDGE — aggregate Sharpe negative. "
              f"Per-ticker optimizer was curve-fitting.")


def main():
    parser = argparse.ArgumentParser(description="Global Config Test — Overfitting Diagnostic")
    parser.add_argument("--quick", action="store_true",
                        help="Test only 5 major tickers (BTC, ETH, SOL, BCH, LINK)")
    args = parser.parse_args()

    tickers = QUICK_TICKERS if args.quick else config.AGATE_TICKERS

    print(f"\n{'═'*60}")
    print(f"  GLOBAL CONFIG TEST — Overfitting Diagnostic")
    print(f"{'═'*60}")
    print(f"  Testing {len(tickers)} tickers with unified configs")
    print(f"  Walk-forward: 6m train / 3m test, ensemble HMM")
    print(f"  Tiered slippage: ACTIVE (per-ticker fees from config)")
    print()

    all_results = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'═'*60}")
        print(f"  Running: {cfg['label']}")
        print(f"{'═'*60}")

        t0 = time.time()
        results = run_global_test(cfg, tickers)
        elapsed = time.time() - t0

        all_results[cfg_name] = results
        print_summary(cfg["label"], results)
        print(f"  Time: {elapsed/60:.1f} min")

    # Compare configs
    print(f"\n{'═'*60}")
    print(f"  COMPARISON")
    print(f"{'═'*60}")
    for cfg_name, results in all_results.items():
        converged = [r for r in results if r.converged]
        sharpes = [r.wf_sharpe for r in converged]
        agg = float(np.mean(sharpes)) if sharpes else 0.0
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  {CONFIGS[cfg_name]['label']}")
        print(f"    Aggregate Sharpe: {agg:+.3f}  "
              f"Positive: {pos}/{len(converged)}")
    print()


if __name__ == "__main__":
    main()
