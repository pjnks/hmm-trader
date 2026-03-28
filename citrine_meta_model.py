#!/usr/bin/env python3
"""
citrine_meta_model.py
─────────────────────
Meta-learning from 734 CITRINE optimizer trials across 100 NDX100 tickers.

Analyses:
1. Feature set impact — which feature_set produces best Sharpe by sector?
2. Parameter sensitivity — n_states, confirmations, cov_type effect sizes
3. Sector clusters — which sectors share optimal configs?
4. Cold-start recommender — predict best config for un-optimized tickers
5. Re-optimization targets — tickers most likely to improve with extended_v2

Usage:
    python citrine_meta_model.py                # Full analysis + recommendations
    python citrine_meta_model.py --recommend     # Generate new per-ticker configs
    python citrine_meta_model.py --gaps          # Show under-explored configs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CSV_PATH = Path(__file__).parent / "citrine_optimization_results.csv"
CONFIG_PATH = Path(__file__).parent / "citrine_per_ticker_configs.json"
BERYL_CSV = Path(__file__).parent / "beryl_daily_results.csv"


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load optimizer trials and per-ticker configs."""
    df = pd.read_csv(CSV_PATH)
    # Add sector
    df["sector"] = df["ticker"].map(config.CITRINE_SECTORS).fillna("Other")
    # Filter to converged trials with trades
    df = df[df["all_converged"] == True].copy()

    with open(CONFIG_PATH) as f:
        configs = json.load(f)
    return df, configs


def analyze_feature_sets(df: pd.DataFrame):
    """Which feature set produces the best Sharpe, overall and by sector?"""
    print("\n" + "=" * 70)
    print("  1. FEATURE SET IMPACT")
    print("=" * 70)

    # Overall
    fs_stats = df.groupby("feature_set")["wf_sharpe"].agg(["mean", "median", "std", "count"])
    fs_stats["positive_pct"] = df.groupby("feature_set")["wf_sharpe"].apply(lambda x: (x > 0).mean())
    fs_stats = fs_stats.sort_values("mean", ascending=False)
    print("\n  Overall (all tickers):")
    print(f"  {'Feature Set':>15s}  {'Mean':>7s}  {'Median':>7s}  {'Std':>6s}  {'N':>5s}  {'%Pos':>5s}")
    for fs, row in fs_stats.iterrows():
        print(f"  {fs:>15s}  {row['mean']:>+7.3f}  {row['median']:>+7.3f}  "
              f"{row['std']:>6.3f}  {int(row['count']):>5d}  {row['positive_pct']:>5.1%}")

    # By sector (top sectors)
    print("\n  By sector (sectors with 10+ trials):")
    sector_fs = df.groupby(["sector", "feature_set"])["wf_sharpe"].agg(["mean", "count"])
    sector_fs = sector_fs.reset_index()
    sector_fs = sector_fs[sector_fs["count"] >= 5]

    sectors = df["sector"].value_counts()
    for sector in sectors[sectors >= 10].index:
        sdf = sector_fs[sector_fs["sector"] == sector].sort_values("mean", ascending=False)
        if len(sdf) > 0:
            best = sdf.iloc[0]
            worst = sdf.iloc[-1] if len(sdf) > 1 else best
            print(f"    {sector:25s}  Best: {best['feature_set']:>12s} ({best['mean']:>+.3f})  "
                  f"Worst: {worst['feature_set']:>12s} ({worst['mean']:>+.3f})")


def analyze_parameters(df: pd.DataFrame):
    """Which n_states, confirmations, cov_type produce best Sharpe?"""
    print("\n" + "=" * 70)
    print("  2. PARAMETER SENSITIVITY")
    print("=" * 70)

    for param in ["n_states", "confirmations", "cov_type"]:
        stats = df.groupby(param)["wf_sharpe"].agg(["mean", "median", "count"])
        stats["positive_pct"] = df.groupby(param)["wf_sharpe"].apply(lambda x: (x > 0).mean())
        stats = stats.sort_values("mean", ascending=False)
        print(f"\n  {param}:")
        for val, row in stats.iterrows():
            bar = "█" * max(1, int((row["mean"] + 0.5) * 20))
            print(f"    {str(val):>12s}  mean={row['mean']:>+.3f}  "
                  f"median={row['median']:>+.3f}  %pos={row['positive_pct']:>5.1%}  "
                  f"n={int(row['count']):>3d}  {bar}")

    # Correlation matrix
    print("\n  Correlation with wf_sharpe:")
    numeric = df[["n_states", "confirmations", "wf_sharpe"]].copy()
    numeric["cov_diag"] = (df["cov_type"] == "diag").astype(int)
    numeric["fs_extended_v2"] = (df["feature_set"] == "extended_v2").astype(int)
    numeric["fs_base"] = (df["feature_set"] == "base").astype(int)
    corr = numeric.corr()["wf_sharpe"].drop("wf_sharpe").sort_values(ascending=False)
    for feat, val in corr.items():
        direction = "↑" if val > 0 else "↓"
        print(f"    {feat:>20s}  {val:>+.3f}  {direction}")


def analyze_sectors(df: pd.DataFrame, configs: dict):
    """Sector-level patterns and cluster analysis."""
    print("\n" + "=" * 70)
    print("  3. SECTOR ANALYSIS")
    print("=" * 70)

    sector_stats = df.groupby("sector")["wf_sharpe"].agg(["mean", "median", "count", "std"])
    sector_stats["positive_pct"] = df.groupby("sector")["wf_sharpe"].apply(lambda x: (x > 0).mean())
    sector_stats = sector_stats.sort_values("mean", ascending=False)

    print(f"\n  {'Sector':>25s}  {'Mean':>7s}  {'Median':>7s}  {'%Pos':>5s}  {'N':>5s}")
    for sector, row in sector_stats.iterrows():
        marker = "★" if row["mean"] > 0.3 else ("▲" if row["mean"] > 0 else "▼")
        print(f"  {sector:>25s}  {row['mean']:>+7.3f}  {row['median']:>+7.3f}  "
              f"{row['positive_pct']:>5.1%}  {int(row['count']):>5d}  {marker}")

    # Dominant config per sector
    print("\n  Dominant config per sector (most common in positive-Sharpe trials):")
    pos = df[df["wf_sharpe"] > 0]
    for sector in sector_stats.index[:10]:
        sdf = pos[pos["sector"] == sector]
        if len(sdf) >= 3:
            mode_ns = sdf["n_states"].mode().iloc[0] if len(sdf) > 0 else "?"
            mode_fs = sdf["feature_set"].mode().iloc[0] if len(sdf) > 0 else "?"
            mode_cf = sdf["confirmations"].mode().iloc[0] if len(sdf) > 0 else "?"
            print(f"    {sector:25s}  n_states={mode_ns}  feature_set={mode_fs}  "
                  f"confirmations={mode_cf}  (from {len(sdf)} positive trials)")


def find_gaps(df: pd.DataFrame, configs: dict):
    """Find under-explored configurations and tickers."""
    print("\n" + "=" * 70)
    print("  4. OPTIMIZATION GAPS — RE-OPTIMIZATION TARGETS")
    print("=" * 70)

    # Tickers using base/extended that might benefit from extended_v2
    print("\n  Tickers NOT using extended_v2 (potential upgrade targets):")
    ev2_trials = df[df["feature_set"] == "extended_v2"]
    ev2_tickers = set(ev2_trials["ticker"].unique())
    all_tickers = set(configs.keys())
    no_ev2 = all_tickers - ev2_tickers

    # Rank by current Sharpe (lower = more room to improve)
    candidates = []
    for t in sorted(no_ev2):
        if t in configs:
            current = configs[t]["wf_sharpe"]
            sector = config.CITRINE_SECTORS.get(t, "Other")
            candidates.append((t, current, sector, configs[t]["feature_set"]))

    candidates.sort(key=lambda x: x[1])
    print(f"\n  {len(no_ev2)} tickers never tested with extended_v2:")
    print(f"  {'Ticker':>8s}  {'Current Sharpe':>14s}  {'Current FS':>12s}  {'Sector':>20s}")
    for t, sharpe, sector, fs in candidates[:20]:
        marker = "◄ priority" if sharpe < 0.2 else ""
        print(f"  {t:>8s}  {sharpe:>+14.4f}  {fs:>12s}  {sector:>20s}  {marker}")
    if len(candidates) > 20:
        print(f"  ... and {len(candidates) - 20} more")

    # Tickers with negative Sharpe that might improve
    negative = [(t, c["wf_sharpe"], c["feature_set"])
                for t, c in configs.items() if c["wf_sharpe"] < 0]
    negative.sort(key=lambda x: x[1])
    print(f"\n  {len(negative)} tickers with negative Sharpe (re-optimization needed):")
    for t, sharpe, fs in negative[:15]:
        print(f"    {t:>8s}  {sharpe:>+.4f}  ({fs})")


def generate_recommendations(df: pd.DataFrame, configs: dict):
    """Generate improved per-ticker config recommendations."""
    print("\n" + "=" * 70)
    print("  5. CONFIG RECOMMENDATIONS")
    print("=" * 70)

    recommendations = {}

    for ticker in sorted(configs.keys()):
        sector = config.CITRINE_SECTORS.get(ticker, "Other")
        current = configs[ticker]

        # Get all trials for this ticker
        ticker_trials = df[df["ticker"] == ticker].sort_values("wf_sharpe", ascending=False)

        # Get sector's best trials (transfer learning)
        sector_trials = df[(df["sector"] == sector) & (df["wf_sharpe"] > 0)]

        # Recommendation logic:
        # 1. If ticker has extended_v2 trial with positive Sharpe → use it
        ev2_trials = ticker_trials[ticker_trials["feature_set"] == "extended_v2"]
        if len(ev2_trials) > 0 and ev2_trials.iloc[0]["wf_sharpe"] > 0:
            best = ev2_trials.iloc[0]
            source = "extended_v2_direct"
        # 2. If ticker has any positive trial → use best
        elif len(ticker_trials) > 0 and ticker_trials.iloc[0]["wf_sharpe"] > 0:
            best = ticker_trials.iloc[0]
            source = "best_direct"
        # 3. Transfer from sector's dominant config
        elif len(sector_trials) >= 3:
            # Use sector's most common positive config
            best_row = {
                "n_states": int(sector_trials["n_states"].mode().iloc[0]),
                "feature_set": sector_trials["feature_set"].mode().iloc[0],
                "confirmations": int(sector_trials["confirmations"].mode().iloc[0]),
                "cov_type": sector_trials["cov_type"].mode().iloc[0],
            }
            best = pd.Series(best_row)
            best["wf_sharpe"] = sector_trials["wf_sharpe"].mean()
            source = f"sector_transfer({sector})"
        else:
            # Keep current
            continue

        new_config = {
            "n_states": int(best.get("n_states", current["n_states"])),
            "feature_set": str(best.get("feature_set", current["feature_set"])),
            "cov_type": str(best.get("cov_type", current["cov_type"])),
            "confirmations": int(best.get("confirmations", current["confirmations"])),
            "wf_sharpe": float(best.get("wf_sharpe", current["wf_sharpe"])),
        }

        # Only recommend if it's different and better
        changed = (new_config["n_states"] != current["n_states"] or
                   new_config["feature_set"] != current["feature_set"] or
                   new_config["confirmations"] != current["confirmations"] or
                   new_config["cov_type"] != current["cov_type"])

        if changed and new_config["wf_sharpe"] > current["wf_sharpe"]:
            recommendations[ticker] = {
                "current": current,
                "recommended": new_config,
                "source": source,
                "sharpe_delta": new_config["wf_sharpe"] - current["wf_sharpe"],
            }

    # Print recommendations sorted by improvement
    recs = sorted(recommendations.items(), key=lambda x: x[1]["sharpe_delta"], reverse=True)

    print(f"\n  {len(recs)} tickers with recommended config changes:")
    print(f"  {'Ticker':>8s}  {'Current':>30s}  {'Sharpe':>7s}  →  "
          f"{'Recommended':>30s}  {'Sharpe':>7s}  {'Δ':>7s}  Source")

    total_delta = 0
    for ticker, rec in recs[:30]:
        cur = rec["current"]
        new = rec["recommended"]
        cur_str = f"{cur['n_states']}st/{cur['feature_set']}/{cur['confirmations']}cf/{cur['cov_type']}"
        new_str = f"{new['n_states']}st/{new['feature_set']}/{new['confirmations']}cf/{new['cov_type']}"
        delta = rec["sharpe_delta"]
        total_delta += delta
        print(f"  {ticker:>8s}  {cur_str:>30s}  {cur['wf_sharpe']:>+7.3f}  →  "
              f"{new_str:>30s}  {new['wf_sharpe']:>+7.3f}  {delta:>+7.3f}  {rec['source']}")

    if len(recs) > 30:
        for _, rec in recs[30:]:
            total_delta += rec["sharpe_delta"]
        print(f"  ... and {len(recs) - 30} more")

    print(f"\n  Total Sharpe improvement potential: {total_delta:>+.3f} "
          f"across {len(recs)} tickers")
    print(f"  Average improvement per changed ticker: {total_delta / len(recs):>+.3f}" if recs else "")

    # CRO warning: these are unvalidated hypotheses
    print(f"\n  !! WARNING: These are HYPOTHESES, not validated results.")
    print(f"  !! Each config change must be walk-forward tested on the target ticker")
    print(f"  !! before deployment. Sector transfer Sharpe estimates are extrapolated")
    print(f"  !! from other tickers — no guarantee of transferability.")
    print(f"  !! Run: python optimize_citrine.py --ticker <TICKER> --runs 20")
    print(f"  !! to validate before applying.")

    return recommendations


def write_recommendations(recommendations: dict, configs: dict):
    """Write updated configs to a new JSON file."""
    updated = dict(configs)  # Start with current
    for ticker, rec in recommendations.items():
        updated[ticker] = rec["recommended"]

    out_path = Path(__file__).parent / "citrine_per_ticker_configs_v2.json"
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"\n  Updated configs written to: {out_path}")
    print(f"  {len(recommendations)} tickers changed, {len(configs) - len(recommendations)} unchanged")


def extended_v2_rollout_plan(df: pd.DataFrame, configs: dict):
    """Plan for systematically testing extended_v2 on all tickers."""
    print("\n" + "=" * 70)
    print("  6. EXTENDED_V2 ROLLOUT PLAN")
    print("=" * 70)

    # Current extended_v2 results
    ev2 = df[df["feature_set"] == "extended_v2"]
    print(f"\n  Current extended_v2 trials: {len(ev2)} across {ev2['ticker'].nunique()} tickers")
    print(f"  Mean Sharpe: {ev2['wf_sharpe'].mean():+.3f} (vs overall {df['wf_sharpe'].mean():+.3f})")
    print(f"  Positive rate: {(ev2['wf_sharpe'] > 0).mean():.1%} (vs overall {(df['wf_sharpe'] > 0).mean():.1%})")

    # Tickers that would benefit most from extended_v2 testing
    # Priority: tickers currently using base/extended with mediocre Sharpe
    priorities = []
    for ticker, cfg in configs.items():
        if cfg["feature_set"] != "extended_v2":
            sector = config.CITRINE_SECTORS.get(ticker, "Other")
            # Check if sector has extended_v2 success
            sector_ev2 = ev2[ev2["ticker"].isin(
                df[df["sector"] == sector]["ticker"].unique()
            )]
            sector_ev2_wins = sector_ev2[sector_ev2["wf_sharpe"] > 0]
            has_sector_ev2_win = len(sector_ev2_wins) > 0
            priorities.append({
                "ticker": ticker,
                "current_sharpe": cfg["wf_sharpe"],
                "current_fs": cfg["feature_set"],
                "sector": sector,
                "sector_has_ev2_win": has_sector_ev2_win,
                "priority": "HIGH" if cfg["wf_sharpe"] < 0.3 else "MEDIUM",
            })

    priorities.sort(key=lambda x: x["current_sharpe"])

    print(f"\n  Rollout priority ({len(priorities)} tickers to test):")
    print(f"  HIGH priority (current Sharpe < 0.3): "
          f"{sum(1 for p in priorities if p['priority'] == 'HIGH')}")
    print(f"  MEDIUM priority (current Sharpe ≥ 0.3): "
          f"{sum(1 for p in priorities if p['priority'] == 'MEDIUM')}")

    # Week-by-week plan
    high = [p for p in priorities if p["priority"] == "HIGH"]
    medium = [p for p in priorities if p["priority"] == "MEDIUM"]

    print(f"\n  Week 1-2: Test {min(20, len(high))} HIGH-priority tickers")
    for p in high[:10]:
        print(f"    {p['ticker']:>8s}  current={p['current_sharpe']:>+.3f} ({p['current_fs']})  "
              f"sector={p['sector']}")
    if len(high) > 10:
        print(f"    ... +{len(high) - 10} more")

    print(f"\n  Week 3-4: Test {min(20, len(medium))} MEDIUM-priority tickers")
    for p in medium[:5]:
        print(f"    {p['ticker']:>8s}  current={p['current_sharpe']:>+.3f} ({p['current_fs']})")

    # Generate optimizer command
    print(f"\n  To run extended_v2 optimization for HIGH-priority tickers:")
    high_tickers = ",".join(p["ticker"] for p in high[:20])
    print(f"  python optimize_citrine.py --runs 100 --resume  # Add extended_v2 trials")
    print(f"  (ensure extended_v2 is in the grid for: {high_tickers[:80]}...)")


def main():
    parser = argparse.ArgumentParser(description="CITRINE Meta-Model Analysis")
    parser.add_argument("--recommend", action="store_true",
                        help="Generate and save recommended configs")
    parser.add_argument("--gaps", action="store_true",
                        help="Show optimization gaps only")
    args = parser.parse_args()

    df, configs = load_data()
    print(f"\n  Loaded {len(df)} trials across {df['ticker'].nunique()} tickers, "
          f"{len(configs)} per-ticker configs")

    if args.gaps:
        find_gaps(df, configs)
        extended_v2_rollout_plan(df, configs)
        return

    # Full analysis
    analyze_feature_sets(df)
    analyze_parameters(df)
    analyze_sectors(df, configs)
    find_gaps(df, configs)
    extended_v2_rollout_plan(df, configs)

    recs = generate_recommendations(df, configs)

    if args.recommend and recs:
        write_recommendations(recs, configs)


if __name__ == "__main__":
    main()
