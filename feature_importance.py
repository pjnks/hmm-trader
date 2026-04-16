"""
Feature Importance Gate — Sprint 15 Crucible Step 4

Random Forest Regressor on the scan_journal dataset.
Target:    relative_return_T3 (market-neutralized)
Features:  confidence, persistence, velocity, confirmations

Gate rule (per directive): drop any feature with <5% Gini importance
before wiring the ranker to live capital. Parsimony is survival.

Usage:
    python feature_importance.py --horizon 3 [--table scan_journal_backfill]
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from cross_sectional_ranker import (
    add_persistence_and_velocity,
    add_relative_returns,
    attach_forward_returns,
)


def load_from_db(db_path: Path, table: str) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql(
        f"SELECT scan_date, ticker, regime, confidence, confirmations, "
        f"signal, close_price FROM {table} ORDER BY ticker, scan_date",
        con,
    )
    con.close()
    df["scan_date"] = pd.to_datetime(df["scan_date"])
    return df


def build_feature_matrix(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Extract (X, y) — features and market-neutralized target."""
    # Restrict to BULL regime (matches what the ranker actually sees in prod)
    bull = df[df["regime"] == "BULL"].copy()

    # Velocity as a first-class feature
    bull["velocity"] = bull["confidence"] - bull["prev_conf"]
    bull["velocity"] = bull["velocity"].fillna(0.0)

    features = ["confidence", "persistence", "velocity", "confirmations"]
    target = f"rel_ret_T{horizon}"

    sub = bull.dropna(subset=features + [target])
    X = sub[features]
    y = sub[target]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--db", default="/home/ubuntu/HMM-Trader/beryl_trades.db")
    ap.add_argument("--table", default="scan_journal")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-samples-leaf", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 72)
    print("  FEATURE IMPORTANCE GATE — Random Forest")
    print(f"  Target: rel_ret_T{args.horizon} | Source: {args.table}")
    print("=" * 72)

    df = load_from_db(Path(args.db), args.table)
    print(f"\nLoaded {len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['scan_date'].nunique()} dates")

    df = add_persistence_and_velocity(df)
    print("  Fetching forward returns...")
    df = attach_forward_returns(df, args.horizon)
    df = add_relative_returns(df, args.horizon)

    X, y = build_feature_matrix(df, args.horizon)
    print(f"  Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    if len(X) < 200:
        print(f"\n  ⚠ Sample too small ({len(X)}). Run backfill first.")
        return

    # Fit random forest
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    oos_r2 = rf.oob_score_ if rf.oob_score else None
    print(f"\n  RF trained ({args.n_estimators} trees, depth={args.max_depth})")

    # 1. Gini impurity importance (built-in)
    gini = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n── Gini Impurity Importance ──")
    for feat, imp in gini.items():
        flag = " ⚠ DROP" if imp < 0.05 else ""
        print(f"  {feat:<15} {imp:.4f}{flag}")

    # 2. Permutation importance (robust — shuffles feature, measures R² drop)
    print("\n── Permutation Importance (robustness check) ──")
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=args.seed, n_jobs=-1)
    perm_ser = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    perm_std = pd.Series(perm.importances_std, index=X.columns)
    for feat, imp in perm_ser.items():
        std = perm_std[feat]
        flag = " ⚠ NOISE" if imp < std else ""
        print(f"  {feat:<15} {imp:+.4f}  ±{std:.4f}{flag}")

    # 3. Parsimony verdict
    print("\n" + "=" * 72)
    print("  PARSIMONY VERDICT")
    print("=" * 72)
    survivors = [f for f, imp in gini.items() if imp >= 0.05]
    casualties = [f for f, imp in gini.items() if imp < 0.05]
    print(f"  Keep: {survivors}")
    if casualties:
        print(f"  Drop: {casualties}")
    else:
        print(f"  All features clear the 5% threshold.")


if __name__ == "__main__":
    main()
