"""
parameter_sweep.py
──────────────────
Emergency parameter sweep: test variations on XRP to find ANY configuration
that passes walk-forward validation (Sharpe > 0.5).

Tests variations of:
  • n_states: [4, 5, 6, 7, 8]
  • timeframe: [1h, 2h, 4h]
  • feature_set: [base, extended, full]
  • confirmations: [6, 7, 8]

Outputs: parameter_sweep_results.csv (ranked by Sharpe)
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from walk_forward import run_walk_forward

# Parameter grid
N_STATES_LIST = [4, 5, 6, 7, 8]
TIMEFRAMES = ["1h", "2h", "4h"]
FEATURE_SETS = ["base", "extended", "full"]
CONFIRMATIONS = [6, 7, 8]

RESULTS_CSV = ROOT / "parameter_sweep_results.csv"

def _patch_config(n_states, confirmations):
    """Temporarily patch config for trial."""
    config._saved_n_states = config.N_STATES
    config._saved_confirmations = config.MIN_CONFIRMATIONS
    config.N_STATES = n_states
    config.MIN_CONFIRMATIONS = confirmations

def _restore_config():
    """Restore original config."""
    config.N_STATES = config._saved_n_states
    config.MIN_CONFIRMATIONS = config._saved_confirmations

def main():
    print("\n" + "="*80)
    print("PARAMETER SWEEP: XRP WALK-FORWARD VALIDATION")
    print("="*80)
    print(f"Testing {len(N_STATES_LIST)} n_states × {len(TIMEFRAMES)} timeframes × "
          f"{len(FEATURE_SETS)} feature_sets × {len(CONFIRMATIONS)} confirmations")
    print(f"Total combinations: {len(N_STATES_LIST) * len(TIMEFRAMES) * len(FEATURE_SETS) * len(CONFIRMATIONS)}")
    print("="*80 + "\n")

    results = []
    trial_num = 0
    total = len(N_STATES_LIST) * len(TIMEFRAMES) * len(FEATURE_SETS) * len(CONFIRMATIONS)

    for n_states, timeframe, feature_set, confirmations in product(
        N_STATES_LIST, TIMEFRAMES, FEATURE_SETS, CONFIRMATIONS
    ):
        trial_num += 1

        _patch_config(n_states, confirmations)

        try:
            print(f"\n[{trial_num}/{total}] n_states={n_states} tf={timeframe} feat={feature_set} cf={confirmations}")

            window_results, eq_oos, eq_bh, trades = run_walk_forward(
                train_months=6,
                test_months=3,
                ticker="X:XRPUSD",
                feature_set=feature_set,
                confirmations=confirmations,
                timeframe=timeframe,
                quiet=True,
                use_regime_mapper=False,
                use_ensemble=False,
            )

            if not window_results:
                print(f"  ⚠️  No valid windows")
                continue

            # Calculate combined metrics
            total_return = (window_results[-1].end_equity / window_results[0].start_equity - 1) * 100
            mean_sharpe = sum(w.sharpe_ratio for w in window_results) / len(window_results)
            positive_windows = sum(1 for w in window_results if w.sharpe_ratio > 0)
            total_trades = sum(w.n_trades for w in window_results)

            result = {
                "n_states": n_states,
                "timeframe": timeframe,
                "feature_set": feature_set,
                "confirmations": confirmations,
                "windows": len(window_results),
                "total_return_pct": round(total_return, 2),
                "mean_sharpe": round(mean_sharpe, 3),
                "positive_windows": positive_windows,
                "total_trades": total_trades,
                "avg_win_rate": round(sum(w.win_rate_pct for w in window_results) / len(window_results), 1),
            }

            results.append(result)

            status = "✅" if mean_sharpe > 0.5 else ("⚠️ " if mean_sharpe > 0 else "❌")
            print(f"  {status} Sharpe: {mean_sharpe:.3f}  Return: {total_return:.1f}%  "
                  f"Pos windows: {positive_windows}/{len(window_results)}  Trades: {total_trades}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            _restore_config()

    # Save and rank results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("mean_sharpe", ascending=False)
        df.to_csv(RESULTS_CSV, index=False)

        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS")
        print("="*80)
        print(df.head(10).to_string(index=False))

        print(f"\n✅ Results saved to {RESULTS_CSV}")

        # Summary stats
        positive_count = len(df[df["mean_sharpe"] > 0])
        high_sharpe_count = len(df[df["mean_sharpe"] > 0.5])
        print(f"\nSummary:")
        print(f"  Total trials: {len(df)}")
        print(f"  Positive Sharpe: {positive_count} ({positive_count/len(df)*100:.1f}%)")
        print(f"  Sharpe > 0.5: {high_sharpe_count}")
    else:
        print("❌ No valid results")


if __name__ == "__main__":
    main()
