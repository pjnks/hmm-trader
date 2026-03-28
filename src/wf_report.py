"""
wf_report.py
────────────
Generates a summary report from walk-forward optimizer results.

Ranks all (ticker, timeframe, config) combos by combined OOS Sharpe and
reports per-window consistency metrics.

Usage
─────
  python -m src.wf_report                                    # default CSV
  python -m src.wf_report --csv optimization_wf_results.csv  # custom CSV
  python -m src.wf_report --top 20                           # show top 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run optimize_wf.py first.")
        sys.exit(1)
    return pd.read_csv(csv_path)


def print_report(df: pd.DataFrame, top_n: int = 10) -> None:
    if df.empty:
        print("No results to report.")
        return

    sep = "─" * 96

    # ── Overall statistics ───────────────────────────────────────────────
    print(f"\n{'═'*96}")
    print(f"  WALK-FORWARD OPTIMIZER REPORT  ({len(df)} completed trials)")
    print(f"{'═'*96}")

    # By ticker
    if "ticker" in df.columns:
        print(f"\n  Results by Ticker:")
        for ticker, grp in df.groupby("ticker"):
            best = grp.loc[grp["wf_sharpe"].idxmax()]
            print(f"    {ticker:12s}  trials={len(grp):3d}  "
                  f"best_sharpe={best['wf_sharpe']:+.3f}  "
                  f"best_return={best['wf_return']:+.1f}%")

    # By timeframe
    if "timeframe" in df.columns:
        print(f"\n  Results by Timeframe:")
        for tf, grp in df.groupby("timeframe"):
            best = grp.loc[grp["wf_sharpe"].idxmax()]
            print(f"    {tf:6s}  trials={len(grp):3d}  "
                  f"best_sharpe={best['wf_sharpe']:+.3f}  "
                  f"best_return={best['wf_return']:+.1f}%")

    # ── Top N by WF Sharpe ───────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  TOP {top_n} by Walk-Forward OOS Sharpe")
    print(sep)

    rank_cols = [
        "ticker", "timeframe", "n_states", "feature_set", "confirmations",
        "leverage", "cooldown_hours", "covariance_type",
        "wf_sharpe", "wf_return", "wf_drawdown",
        "wf_trades", "wf_win_rate",
        "wf_pos_windows", "wf_n_windows",
        "wf_mean_sharpe", "wf_std_sharpe",
    ]
    rank_cols = [c for c in rank_cols if c in df.columns]
    top = df.nlargest(top_n, "wf_sharpe")[rank_cols]
    print(top.to_string(index=False))

    # ── Consistency analysis ─────────────────────────────────────────────
    if "wf_std_sharpe" in df.columns and "wf_mean_sharpe" in df.columns:
        print(f"\n{sep}")
        print(f"  MOST CONSISTENT (lowest std of per-window Sharpe, ≥{top_n//2} trials)")
        print(sep)

        # Only include combos with positive mean Sharpe
        pos = df[df["wf_mean_sharpe"] > 0].copy()
        if not pos.empty:
            consistent = pos.nsmallest(top_n, "wf_std_sharpe")[rank_cols]
            print(consistent.to_string(index=False))
        else:
            print("  No trials with positive mean window Sharpe.")

    # ── Best per ticker ──────────────────────────────────────────────────
    if "ticker" in df.columns:
        print(f"\n{sep}")
        print(f"  BEST CONFIG PER TICKER")
        print(sep)
        for ticker, grp in df.groupby("ticker"):
            if grp.empty:
                continue
            best = grp.loc[grp["wf_sharpe"].idxmax()]
            print(f"\n  {ticker}:")
            for col in rank_cols:
                if col == "ticker":
                    continue
                val = best[col]
                if isinstance(val, float):
                    print(f"    {col:20s}: {val:>10.3f}")
                else:
                    print(f"    {col:20s}: {val!s:>10s}")

    print(f"\n{'═'*96}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimizer — Summary Report"
    )
    parser.add_argument(
        "--csv", type=str,
        default=str(ROOT / "optimization_wf_results.csv"),
        help="Path to results CSV",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top results to show (default: 10)",
    )
    args = parser.parse_args()

    df = load_results(Path(args.csv))
    print_report(df, top_n=args.top)


if __name__ == "__main__":
    main()
