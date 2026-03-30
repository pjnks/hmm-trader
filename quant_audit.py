#!/usr/bin/env python
"""
quant_audit.py — PhD-grade statistical audit of AGATE optimizer results.

Three tests:
  1. Universal Parameter Test: run mode config across all tickers
  2. Trade Frequency Penalty: re-rank by Sharpe * log(num_trades)
  3. Beta Confounding Test: compare strategy return vs buy-and-hold

Usage:
    python quant_audit.py                # Run all 3 tests
    python quant_audit.py --universal    # Universal parameter test only
    python quant_audit.py --frequency    # Trade frequency penalty only
    python quant_audit.py --beta         # Beta confounding test only
"""
from __future__ import annotations

import argparse
import json
import sys
import gc
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats as st

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
import config
from walk_forward import run_walk_forward, WindowResult

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "agate_per_ticker_configs.json"

# ── All 14 active AGATE tickers (excluding ADA — insufficient data) ─────────
AGATE_TICKERS = [
    "X:BTCUSD", "X:ETHUSD", "X:SOLUSD", "X:XRPUSD", "X:LTCUSD",
    "X:SUIUSD", "X:DOGEUSD", "X:BCHUSD", "X:LINKUSD", "X:HBARUSD",
    "X:XLMUSD", "X:AVAXUSD", "X:ENAUSD", "X:DOTUSD", "X:SHIBUSD",
]

# DSR constants
N_TRIALS = 100
EULER_GAMMA = 0.5772
E_MAX_SR = (np.sqrt(2 * np.log(N_TRIALS))
            - (EULER_GAMMA + np.log(np.pi / 2))
            / (2 * np.sqrt(2 * np.log(N_TRIALS))))


def load_configs() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def find_mode_config(configs: dict) -> dict:
    """Find the mode (most common value) across all per-ticker configs."""
    features = [c["feature_set"] for c in configs.values()]
    states = [c["n_states"] for c in configs.values()]
    confirms = [c["confirmations"] for c in configs.values()]
    covs = [c["cov_type"] for c in configs.values()]
    tfs = [c["timeframe"] for c in configs.values()]

    return {
        "feature_set": Counter(features).most_common(1)[0][0],
        "n_states": Counter(states).most_common(1)[0][0],
        "confirmations": Counter(confirms).most_common(1)[0][0],
        "cov_type": Counter(covs).most_common(1)[0][0],
        "timeframe": Counter(tfs).most_common(1)[0][0],
    }


def run_wf_for_ticker(ticker: str, feature_set: str, confirmations: int,
                      timeframe: str, n_states: int, cov_type: str,
                      use_ensemble: bool = True
                      ) -> list[WindowResult] | None:
    """Run walk-forward for one ticker, return window results or None on failure."""
    # Patch config
    saved = {
        "N_STATES": config.N_STATES,
        "COV_TYPE": config.COV_TYPE,
        "FEATURE_SET": config.FEATURE_SET,
    }
    config.N_STATES = n_states
    config.COV_TYPE = cov_type
    config.FEATURE_SET = feature_set

    try:
        result = run_walk_forward(
            train_months=6,
            test_months=3,
            ticker=ticker,
            feature_set=feature_set,
            confirmations=confirmations,
            timeframe=timeframe,
            quiet=True,
            use_ensemble=use_ensemble,
        )
        # Returns tuple: (window_results, oos_equity, bh_equity, trades_df)
        windows = result[0]
        if not windows:
            return None
        return windows
    except Exception as e:
        print(f"  ⚠ {ticker}: {e}")
        return None
    finally:
        config.N_STATES = saved["N_STATES"]
        config.COV_TYPE = saved["COV_TYPE"]
        config.FEATURE_SET = saved["FEATURE_SET"]
        gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Universal Parameter Test
# ═════════════════════════════════════════════════════════════════════════════
def test_universal(configs: dict):
    mode = find_mode_config(configs)
    print("\n" + "═" * 95)
    print("TEST 1: UNIVERSAL PARAMETER TEST")
    print("═" * 95)
    print(f"Mode config: {mode['feature_set']}/{mode['n_states']}st/"
          f"{mode['confirmations']}cf/{mode['cov_type']}/{mode['timeframe']}")
    print(f"Testing same config across all {len(AGATE_TICKERS)} tickers...")
    print(f"(If HMM captures real structure, a single config should work broadly)")
    print("-" * 95)
    print(f"{'Ticker':<14} {'WF Sharpe':>10} {'Return%':>8} {'BH Ret%':>8} "
          f"{'Alpha%':>8} {'Trades':>7} {'Win+/N':>7} {'Positive?':>10}")
    print("-" * 95)

    results = []
    for ticker in AGATE_TICKERS:
        windows = run_wf_for_ticker(
            ticker, mode["feature_set"], mode["confirmations"],
            mode["timeframe"], mode["n_states"], mode["cov_type"],
        )
        if windows:
            sharpes = [w.sharpe_ratio for w in windows]
            returns = [w.return_pct for w in windows]
            bh_rets = [w.bh_return_pct for w in windows]
            trades = sum(w.n_trades for w in windows)
            pos_w = sum(1 for s in sharpes if s > 0)
            n_w = len(sharpes)
            mean_sr = np.mean(sharpes)
            total_ret = sum(returns)
            total_bh = sum(bh_rets)
            alpha = total_ret - total_bh
            is_pos = "YES" if mean_sr > 0 else "NO"
            results.append({
                "ticker": ticker, "sharpe": mean_sr, "return": total_ret,
                "bh_return": total_bh, "alpha": alpha, "trades": trades,
                "pos_windows": pos_w, "n_windows": n_w, "positive": mean_sr > 0,
            })
            print(f"{ticker:<14} {mean_sr:>10.3f} {total_ret:>7.1f}% {total_bh:>7.1f}% "
                  f"{alpha:>+7.1f}% {trades:>7} {pos_w}/{n_w:>3}   {is_pos:>8}")
        else:
            print(f"{ticker:<14}       FAIL")
            results.append({
                "ticker": ticker, "sharpe": 0, "return": 0, "bh_return": 0,
                "alpha": 0, "trades": 0, "pos_windows": 0, "n_windows": 0,
                "positive": False,
            })

    # Summary
    valid = [r for r in results if r["trades"] > 0]
    if not valid:
        print("\n⚠ No valid results. All tickers failed.")
        return results

    positive = sum(1 for r in valid if r["positive"])
    agg_sharpe = np.mean([r["sharpe"] for r in valid])
    agg_alpha = np.mean([r["alpha"] for r in valid])

    print("-" * 95)
    print(f"Positive Sharpe:      {positive}/{len(valid)} ({100*positive/len(valid):.0f}%)")
    print(f"Aggregate mean Sharpe: {agg_sharpe:.3f}")
    print(f"Aggregate mean Alpha:  {agg_alpha:+.1f}%")
    print()
    if agg_sharpe >= 0.5:
        print("✅ PASS — Universal config Sharpe ≥ 0.5 → HMM captures real structure")
    elif agg_sharpe > 0:
        print("⚠️  MARGINAL — Universal config Sharpe > 0 but < 0.5 → some signal, weak")
    else:
        print("❌ FAIL — Universal config Sharpe ≤ 0 → per-ticker tuning is likely overfitting")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Trade Frequency Penalty
# ═════════════════════════════════════════════════════════════════════════════
def test_frequency(configs: dict):
    print("\n" + "═" * 95)
    print("TEST 2: TRADE FREQUENCY PENALTY")
    print("═" * 95)
    print("Penalizes configs that achieve high Sharpe on few trades.")
    print("Adjusted objective: Sharpe × log(num_trades)")
    print("Minimum viable trades: 30 (for statistical significance)")
    print("-" * 95)
    print(f"{'Ticker':<14} {'Raw SR':>7} {'Trades':>7} {'log(N)':>7} "
          f"{'Adj SR':>8} {'Rank Δ':>7} {'N≥30?':>6} {'DSR?':>5}")
    print("-" * 95)

    rows = []
    for ticker, cfg in configs.items():
        sr = cfg["wf_sharpe"]
        n = cfg["total_trades"]
        log_n = np.log(max(n, 1))
        adj_sr = sr * log_n
        se = np.sqrt((1 + 0.5 * sr**2) / n) if n >= 3 else 999
        dsr_stat = (sr - E_MAX_SR) / se
        survives_dsr = (1 - st.norm.cdf(dsr_stat)) < 0.05
        rows.append({
            "ticker": ticker, "raw_sr": sr, "trades": n, "log_n": log_n,
            "adj_sr": adj_sr, "adequate_n": n >= 30, "dsr": survives_dsr,
        })

    # Sort by raw and adjusted to compute rank delta
    by_raw = sorted(rows, key=lambda r: -r["raw_sr"])
    by_adj = sorted(rows, key=lambda r: -r["adj_sr"])
    raw_rank = {r["ticker"]: i + 1 for i, r in enumerate(by_raw)}
    adj_rank = {r["ticker"]: i + 1 for i, r in enumerate(by_adj)}

    for r in by_adj:
        delta = raw_rank[r["ticker"]] - adj_rank[r["ticker"]]
        delta_str = f"{delta:+d}" if delta != 0 else "—"
        n_ok = "YES" if r["adequate_n"] else "NO"
        dsr_str = "YES" if r["dsr"] else "no"
        print(f"{r['ticker']:<14} {r['raw_sr']:>7.3f} {r['trades']:>7} {r['log_n']:>7.2f} "
              f"{r['adj_sr']:>8.3f} {delta_str:>7} {n_ok:>6} {dsr_str:>5}")

    # Analysis
    print("-" * 95)
    adequate = [r for r in rows if r["adequate_n"]]
    inadequate = [r for r in rows if not r["adequate_n"]]
    print(f"Adequate trades (≥30): {len(adequate)}/{len(rows)}")
    print(f"Statistically suspect (<30 trades): "
          f"{', '.join(r['ticker'].replace('X:','').replace('USD','') for r in inadequate)}")

    # Big movers
    big_movers = [(r["ticker"], raw_rank[r["ticker"]], adj_rank[r["ticker"]])
                  for r in rows
                  if abs(raw_rank[r["ticker"]] - adj_rank[r["ticker"]]) >= 3]
    if big_movers:
        print("\nBig rank changes (≥3 positions):")
        for t, old, new in big_movers:
            short = t.replace("X:", "").replace("USD", "")
            print(f"  {short}: #{old} → #{new} ({'↑' if new < old else '↓'})")

    # Kill condition check
    if adequate:
        above_1 = sum(1 for r in adequate if r["raw_sr"] > 1.0)
        print(f"\nKill condition: if forcing >50 trades pushes all Sharpes below 1.0")
        print(f"  Tickers with ≥30 trades AND Sharpe > 1.0: {above_1}/{len(adequate)}")
        if above_1 == 0:
            print("  ❌ FAIL — no ticker has both adequate trades AND strong Sharpe")
        else:
            print(f"  ✅ PASS — {above_1} tickers have real signal with adequate sample size")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Beta Confounding Test
# ═════════════════════════════════════════════════════════════════════════════
def test_beta(configs: dict):
    print("\n" + "═" * 95)
    print("TEST 3: BETA CONFOUNDING TEST")
    print("═" * 95)
    print("Compares strategy return vs. buy-and-hold (beta) per ticker.")
    print("Alpha = Strategy Return - Buy-and-Hold Return")
    print("If alpha ≈ 0, the HMM isn't adding value beyond holding.")
    print("-" * 95)
    print(f"{'Ticker':<14} {'Strat Ret%':>10} {'BH Ret%':>9} {'Alpha%':>8} "
          f"{'Alpha > 0?':>10} {'Sharpe':>7} {'Trades':>7}")
    print("-" * 95)

    results = []
    for ticker, cfg in configs.items():
        if ticker == "X:ADAUSD":
            continue

        windows = run_wf_for_ticker(
            ticker, cfg["feature_set"], cfg["confirmations"],
            cfg["timeframe"], cfg["n_states"], cfg["cov_type"],
        )
        if windows:
            total_ret = sum(w.return_pct for w in windows)
            total_bh = sum(w.bh_return_pct for w in windows)
            alpha = total_ret - total_bh
            trades = sum(w.n_trades for w in windows)
            mean_sr = np.mean([w.sharpe_ratio for w in windows])
            alpha_pos = "YES" if alpha > 0 else "NO"
            results.append({
                "ticker": ticker, "strat_return": total_ret,
                "bh_return": total_bh, "alpha": alpha,
                "sharpe": mean_sr, "trades": trades,
                "alpha_positive": alpha > 0,
            })
            print(f"{ticker:<14} {total_ret:>+9.1f}% {total_bh:>+8.1f}% {alpha:>+7.1f}% "
                  f"{alpha_pos:>10} {mean_sr:>7.3f} {trades:>7}")
        else:
            print(f"{ticker:<14}       FAIL")

    if not results:
        print("\n⚠ No valid results.")
        return results

    # Summary
    print("-" * 95)
    alpha_pos = sum(1 for r in results if r["alpha_positive"])
    mean_alpha = np.mean([r["alpha"] for r in results])
    mean_bh = np.mean([r["bh_return"] for r in results])
    mean_strat = np.mean([r["strat_return"] for r in results])

    print(f"Alpha positive:       {alpha_pos}/{len(results)} ({100*alpha_pos/len(results):.0f}%)")
    print(f"Mean strategy return: {mean_strat:+.1f}%")
    print(f"Mean buy-and-hold:    {mean_bh:+.1f}%")
    print(f"Mean alpha:           {mean_alpha:+.1f}%")
    print()

    # Check for beta-only returns
    beta_only = [r for r in results if not r["alpha_positive"] and r["strat_return"] > 0]
    if beta_only:
        tickers = ", ".join(r["ticker"].replace("X:", "").replace("USD", "")
                           for r in beta_only)
        print(f"⚠️  Beta-only tickers (positive return but negative alpha): {tickers}")
        print("   These are riding the market, not generating independent edge.")

    # Verdict
    if mean_alpha > 5:
        print("\n✅ STRONG PASS — significant alpha above buy-and-hold")
    elif mean_alpha > 0:
        print("\n⚠️  MARGINAL — positive alpha but modest")
    else:
        print("\n❌ FAIL — strategy does not outperform buy-and-hold on average")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="PhD-grade AGATE statistical audit")
    parser.add_argument("--universal", action="store_true", help="Universal parameter test only")
    parser.add_argument("--frequency", action="store_true", help="Trade frequency penalty only")
    parser.add_argument("--beta", action="store_true", help="Beta confounding test only")
    args = parser.parse_args()

    run_all = not (args.universal or args.frequency or args.beta)
    configs = load_configs()

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     AGATE STATISTICAL AUDIT — PhD Quant Review           ║")
    print("║     Testing for overfitting, noise-mining, and beta      ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    if args.frequency or run_all:
        # Test 2 is pure computation — no API calls needed
        test_frequency(configs)

    if args.universal or run_all:
        # Test 1 needs WF runs — takes ~20 min
        test_universal(configs)

    if args.beta or run_all:
        # Test 3 needs WF runs with per-ticker configs — takes ~20 min
        test_beta(configs)

    print("\n" + "═" * 95)
    print("AUDIT COMPLETE")
    print("═" * 95)


if __name__ == "__main__":
    main()
