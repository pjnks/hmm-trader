"""
stability_test.py
─────────────────
Walk-forward stability test suite for HMM-Trader configs.

Tests top configs for robustness across 3 dimensions:
  1. Parameter sensitivity: vary each param +/-10%, check if Sharpe drops >50%
  2. Data window sensitivity: train_months=[5,6,7], test_months=[2,3,4]
  3. Seed sensitivity: 5 random seeds, check if Sharpe varies >30%

Usage:
    python stability_test.py                      # Run all tests for all projects
    python stability_test.py --project agate      # AGATE only
    python stability_test.py --project beryl      # BERYL only
    python stability_test.py --test sensitivity   # Single test type
    python stability_test.py --quick              # Fewer variations (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from walk_forward import run_walk_forward

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config definitions for top configs per project
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StabilityConfig:
    """A config to test for stability."""
    name: str
    ticker: str
    timeframe: str
    n_states: int
    feature_set: str
    confirmations: int
    leverage: float
    cooldown_hours: int
    cov_type: str
    use_ensemble: bool = False


# Top configs to test (from optimization results)
AGATE_CONFIGS = [
    StabilityConfig(
        name="AGATE-SOL-ensemble",
        ticker="X:SOLUSD", timeframe="4h", n_states=6,
        feature_set="extended", confirmations=6,
        leverage=1.0, cooldown_hours=48, cov_type="diag",
        use_ensemble=True,
    ),
    StabilityConfig(
        name="AGATE-SOL-extended_v2",
        ticker="X:SOLUSD", timeframe="4h", n_states=6,
        feature_set="extended_v2", confirmations=6,
        leverage=1.0, cooldown_hours=48, cov_type="diag",
        use_ensemble=True,
    ),
]

BERYL_CONFIGS = [
    StabilityConfig(
        name="BERYL-NVDA-p2",
        ticker="NVDA", timeframe="1d", n_states=4,
        feature_set="extended", confirmations=7,
        leverage=1.5, cooldown_hours=48, cov_type="full",
        use_ensemble=False,
    ),
]


@dataclass
class StabilityResult:
    """Result of one stability test."""
    config_name: str
    test_type: str
    variation: str
    sharpe: float
    return_pct: float
    n_trades: int
    n_windows: int
    converged: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Parameter Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_parameter_sensitivity(
    cfg: StabilityConfig,
    quick: bool = False,
) -> list[StabilityResult]:
    """Vary each numeric parameter by +/-10%, check if Sharpe drops >50%."""
    results: list[StabilityResult] = []

    # Baseline
    base_result = _run_config(cfg)
    if base_result is None:
        print(f"  SKIP {cfg.name}: baseline failed")
        return results
    results.append(StabilityResult(
        config_name=cfg.name, test_type="sensitivity",
        variation="baseline", sharpe=base_result[0],
        return_pct=base_result[1], n_trades=base_result[2],
        n_windows=base_result[3],
    ))

    # Parameters to vary
    params = [
        ("confirmations", cfg.confirmations, [max(5, cfg.confirmations - 1), min(8, cfg.confirmations + 1)]),
        ("leverage", cfg.leverage, [round(cfg.leverage * 0.9, 2), round(cfg.leverage * 1.1, 2)]),
        ("cooldown_hours", cfg.cooldown_hours, [max(24, cfg.cooldown_hours - 12), cfg.cooldown_hours + 12]),
    ]

    if not quick:
        params.append(("n_states", cfg.n_states, [max(4, cfg.n_states - 1), min(7, cfg.n_states + 1)]))

    for param_name, base_val, variations in params:
        for var_val in variations:
            # Create modified config
            mod_cfg = StabilityConfig(
                name=cfg.name,
                ticker=cfg.ticker, timeframe=cfg.timeframe,
                n_states=cfg.n_states if param_name != "n_states" else var_val,
                feature_set=cfg.feature_set,
                confirmations=cfg.confirmations if param_name != "confirmations" else var_val,
                leverage=cfg.leverage if param_name != "leverage" else var_val,
                cooldown_hours=cfg.cooldown_hours if param_name != "cooldown_hours" else var_val,
                cov_type=cfg.cov_type,
                use_ensemble=cfg.use_ensemble,
            )

            r = _run_config(mod_cfg)
            if r is None:
                results.append(StabilityResult(
                    config_name=cfg.name, test_type="sensitivity",
                    variation=f"{param_name}={var_val}", sharpe=float("nan"),
                    return_pct=0, n_trades=0, n_windows=0, converged=False,
                ))
                continue

            results.append(StabilityResult(
                config_name=cfg.name, test_type="sensitivity",
                variation=f"{param_name}={var_val}",
                sharpe=r[0], return_pct=r[1], n_trades=r[2], n_windows=r[3],
            ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Data Window Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_window_sensitivity(
    cfg: StabilityConfig,
    quick: bool = False,
) -> list[StabilityResult]:
    """Vary train/test window lengths."""
    results: list[StabilityResult] = []

    if quick:
        windows = [(5, 3), (6, 3), (7, 3)]
    else:
        windows = [(5, 2), (5, 3), (6, 2), (6, 3), (6, 4), (7, 3), (7, 4)]

    for train_m, test_m in windows:
        r = _run_config(cfg, train_months=train_m, test_months=test_m)
        if r is None:
            results.append(StabilityResult(
                config_name=cfg.name, test_type="window",
                variation=f"train={train_m}m,test={test_m}m",
                sharpe=float("nan"), return_pct=0, n_trades=0, n_windows=0,
                converged=False,
            ))
            continue

        results.append(StabilityResult(
            config_name=cfg.name, test_type="window",
            variation=f"train={train_m}m,test={test_m}m",
            sharpe=r[0], return_pct=r[1], n_trades=r[2], n_windows=r[3],
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Seed Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_sensitivity(
    cfg: StabilityConfig,
    quick: bool = False,
) -> list[StabilityResult]:
    """Run with 5 different random seeds, check variance."""
    results: list[StabilityResult] = []
    n_seeds = 3 if quick else 5
    seeds = [42, 123, 456, 789, 1024][:n_seeds]

    for seed in seeds:
        # Patch config's random state
        original_rs = config.RANDOM_STATE
        config.RANDOM_STATE = seed

        r = _run_config(cfg)
        config.RANDOM_STATE = original_rs

        if r is None:
            results.append(StabilityResult(
                config_name=cfg.name, test_type="seed",
                variation=f"seed={seed}",
                sharpe=float("nan"), return_pct=0, n_trades=0, n_windows=0,
                converged=False,
            ))
            continue

        results.append(StabilityResult(
            config_name=cfg.name, test_type="seed",
            variation=f"seed={seed}",
            sharpe=r[0], return_pct=r[1], n_trades=r[2], n_windows=r[3],
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_config(
    cfg: StabilityConfig,
    train_months: int = 6,
    test_months: int = 3,
) -> tuple[float, float, int, int] | None:
    """
    Run walk-forward for a config. Returns (mean_sharpe, total_return, n_trades, n_windows)
    or None on failure.
    """
    # Patch config
    orig = {}
    for attr in ["TICKER", "TIMEFRAME", "N_STATES", "FEATURE_SET",
                  "MIN_CONFIRMATIONS", "LEVERAGE", "COOLDOWN_HOURS", "COV_TYPE"]:
        orig[attr] = getattr(config, attr)

    config.TICKER = cfg.ticker
    config.TIMEFRAME = cfg.timeframe
    config.N_STATES = cfg.n_states
    config.FEATURE_SET = cfg.feature_set
    config.MIN_CONFIRMATIONS = cfg.confirmations
    config.LEVERAGE = cfg.leverage
    config.COOLDOWN_HOURS = cfg.cooldown_hours
    config.COV_TYPE = cfg.cov_type

    try:
        result = run_walk_forward(
            train_months=train_months,
            test_months=test_months,
            ticker=cfg.ticker,
            feature_set=cfg.feature_set,
            confirmations=cfg.confirmations,
            timeframe=cfg.timeframe,
            quiet=True,
            use_regime_mapper=False,
            use_ensemble=cfg.use_ensemble,
        )

        windows = result[0] if result else []
        trades_df = result[3] if result and len(result) > 3 else None

        if not windows:
            return None

        sharpes = [w.sharpe_ratio for w in windows]
        returns = [w.return_pct for w in windows]
        n_trades = len(trades_df) if trades_df is not None else 0
        mean_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
        total_ret = sum(returns) if returns else 0

        return (mean_sharpe, total_ret, n_trades, len(windows))

    except Exception as e:
        log.warning("Config %s failed: %s", cfg.name, e)
        return None
    finally:
        # Restore config
        for attr, val in orig.items():
            setattr(config, attr, val)


def _analyze_results(results: list[StabilityResult]) -> dict:
    """Analyze stability test results and flag fragile configs."""
    analysis = {}

    # Group by config × test type
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r.config_name, r.test_type)].append(r)

    for (cfg_name, test_type), group in groups.items():
        valid = [r for r in group if not np.isnan(r.sharpe)]
        if len(valid) < 2:
            continue

        sharpes = [r.sharpe for r in valid]
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        min_sharpe = min(sharpes)
        max_sharpe = max(sharpes)

        # Check for fragility
        fragile = False
        reason = ""

        if test_type == "sensitivity":
            baseline = [r for r in valid if r.variation == "baseline"]
            if baseline:
                base_sharpe = baseline[0].sharpe
                if base_sharpe > 0:
                    worst_drop = (base_sharpe - min_sharpe) / base_sharpe
                    if worst_drop > 0.50:
                        fragile = True
                        reason = f"Sharpe drops {worst_drop:.0%} from baseline"

        elif test_type == "window":
            if mean_sharpe != 0:
                cv = abs(std_sharpe / mean_sharpe)
                if cv > 0.50:
                    fragile = True
                    reason = f"Window CV = {cv:.2f} (>0.50)"

        elif test_type == "seed":
            if mean_sharpe != 0:
                cv = abs(std_sharpe / mean_sharpe)
                if cv > 0.30:
                    fragile = True
                    reason = f"Seed CV = {cv:.2f} (>0.30)"

        key = f"{cfg_name}|{test_type}"
        analysis[key] = {
            "config": cfg_name,
            "test": test_type,
            "mean_sharpe": round(mean_sharpe, 3),
            "std_sharpe": round(std_sharpe, 3),
            "min_sharpe": round(min_sharpe, 3),
            "max_sharpe": round(max_sharpe, 3),
            "n_valid": len(valid),
            "fragile": fragile,
            "reason": reason,
        }

    return analysis


def run_stability_suite(
    configs: list[StabilityConfig],
    tests: list[str] | None = None,
    quick: bool = False,
) -> tuple[list[StabilityResult], dict]:
    """Run full stability suite on given configs."""
    all_results: list[StabilityResult] = []
    tests = tests or ["sensitivity", "window", "seed"]

    for cfg in configs:
        print(f"\n{'─'*60}")
        print(f"  Testing: {cfg.name}")
        print(f"  {cfg.ticker} / {cfg.timeframe} / {cfg.feature_set} / "
              f"{'ensemble' if cfg.use_ensemble else 'single'}")
        print(f"{'─'*60}")

        if "sensitivity" in tests:
            print("  [1/3] Parameter sensitivity...", end=" ", flush=True)
            t0 = time.time()
            results = test_parameter_sensitivity(cfg, quick=quick)
            all_results.extend(results)
            print(f"({len(results)} trials, {time.time()-t0:.0f}s)")

        if "window" in tests:
            print("  [2/3] Window sensitivity...", end=" ", flush=True)
            t0 = time.time()
            results = test_window_sensitivity(cfg, quick=quick)
            all_results.extend(results)
            print(f"({len(results)} trials, {time.time()-t0:.0f}s)")

        if "seed" in tests:
            print("  [3/3] Seed sensitivity...", end=" ", flush=True)
            t0 = time.time()
            results = test_seed_sensitivity(cfg, quick=quick)
            all_results.extend(results)
            print(f"({len(results)} trials, {time.time()-t0:.0f}s)")

    analysis = _analyze_results(all_results)
    return all_results, analysis


def print_report(results: list[StabilityResult], analysis: dict) -> None:
    """Print stability report to terminal."""
    print(f"\n{'='*70}")
    print(f"  STABILITY TEST REPORT")
    print(f"{'='*70}\n")

    # Summary by test type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r.test_type].append(r)

    for test_type in ["sensitivity", "window", "seed"]:
        if test_type not in by_type:
            continue
        group = by_type[test_type]
        valid = [r for r in group if not np.isnan(r.sharpe)]
        print(f"  {test_type.upper()}: {len(valid)}/{len(group)} valid trials")
        if valid:
            sharpes = [r.sharpe for r in valid]
            print(f"    Sharpe range: [{min(sharpes):.3f}, {max(sharpes):.3f}]")
            print(f"    Mean: {np.mean(sharpes):.3f}, Std: {np.std(sharpes):.3f}")

    # Fragility flags
    fragile = [v for v in analysis.values() if v["fragile"]]
    if fragile:
        print(f"\n  ⚠️  FRAGILE CONFIGS ({len(fragile)}):")
        for f in fragile:
            print(f"    {f['config']} [{f['test']}]: {f['reason']}")
            print(f"      Sharpe: {f['mean_sharpe']:.3f} ± {f['std_sharpe']:.3f} "
                  f"(range [{f['min_sharpe']:.3f}, {f['max_sharpe']:.3f}])")
    else:
        print(f"\n  ✅ No fragile configs detected!")

    # Detailed results table
    print(f"\n{'─'*70}")
    print(f"  {'Config':<25s} {'Test':<12s} {'Variation':<25s} {'Sharpe':>7s} {'Return':>7s}")
    print(f"  {'─'*25} {'─'*12} {'─'*25} {'─'*7} {'─'*7}")
    for r in results:
        s = f"{r.sharpe:.3f}" if not np.isnan(r.sharpe) else "FAIL"
        ret = f"{r.return_pct:.1f}%" if r.return_pct else "–"
        print(f"  {r.config_name:<25s} {r.test_type:<12s} {r.variation:<25s} {s:>7s} {ret:>7s}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward stability tests")
    parser.add_argument("--project", choices=["agate", "beryl", "all"], default="all")
    parser.add_argument("--test", choices=["sensitivity", "window", "seed"], default=None,
                        help="Run only one test type")
    parser.add_argument("--quick", action="store_true", help="Fewer variations (faster)")
    args = parser.parse_args()

    configs = []
    if args.project in ("agate", "all"):
        configs.extend(AGATE_CONFIGS)
    if args.project in ("beryl", "all"):
        configs.extend(BERYL_CONFIGS)

    tests = [args.test] if args.test else None

    results, analysis = run_stability_suite(configs, tests=tests, quick=args.quick)

    print_report(results, analysis)

    # Save results
    output = {
        "results": [
            {
                "config": r.config_name,
                "test": r.test_type,
                "variation": r.variation,
                "sharpe": r.sharpe if not np.isnan(r.sharpe) else None,
                "return_pct": r.return_pct,
                "n_trades": r.n_trades,
            }
            for r in results
        ],
        "analysis": analysis,
    }

    out_path = ROOT / "stability_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
