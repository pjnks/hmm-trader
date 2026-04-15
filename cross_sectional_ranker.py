"""
Cross-Sectional Alpha Ranker — IC + Decile Validation

Tests whether a composite score (banded confidence × sojourn decay × velocity boost)
can rank tickers cross-sectionally against forward returns.

Per Sprint 15 directive: no eviction engine code until decile spread is undeniably positive.

Usage:
    python cross_sectional_ranker.py [--horizon 3] [--optimal-conf 0.70] [--decay 0.2]
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp


DB_PATH = Path("/home/ubuntu/HMM-Trader/beryl_trades.db")


def load_scans() -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(
        """
        SELECT scan_date, ticker, regime, confidence, confirmations, signal, close_price
        FROM scan_journal
        ORDER BY ticker, scan_date
        """,
        con,
    )
    con.close()
    df["scan_date"] = pd.to_datetime(df["scan_date"])
    return df


def add_persistence_and_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Derive prev_conf and consecutive-same-regime persistence per ticker."""
    df = df.sort_values(["ticker", "scan_date"]).copy()
    df["prev_conf"] = df.groupby("ticker")["confidence"].shift(1)
    df["prev_regime"] = df.groupby("ticker")["regime"].shift(1)
    # persistence: count of consecutive rows where regime == regime
    regime_change = (df["regime"] != df["prev_regime"]).astype(int)
    df["regime_block"] = df.groupby("ticker")[regime_change.name if regime_change.name else "regime"].transform(
        lambda _: None
    )
    # Manual streak counter
    persistence = []
    last_ticker = None
    last_regime = None
    streak = 0
    for _, row in df.iterrows():
        if row["ticker"] != last_ticker:
            streak = 1
        elif row["regime"] != last_regime:
            streak = 1
        else:
            streak += 1
        persistence.append(streak)
        last_ticker = row["ticker"]
        last_regime = row["regime"]
    df["persistence"] = persistence
    return df


def compute_alpha_score(
    current_conf: float,
    prev_conf: float,
    days_in_state: int,
    regime: str,
    optimal_conf: float = 0.70,
    decay_rate: float = 0.2,
    velocity_scale: float = 0.5,
    min_conf: float = 0.60,
) -> float:
    """Composite alpha score per Sprint 15 framework."""
    if regime != "BULL":
        return 0.0
    if pd.isna(current_conf) or current_conf < min_conf:
        return 0.0

    # Banded confidence penalty — peak at optimal_conf, with explicit 90-95% penalty zone
    conf_penalty = abs(current_conf - optimal_conf)
    base_score = max(0.0, 1.0 - conf_penalty)
    # Empirical graveyard zone: crush 90-95%
    if 0.90 <= current_conf < 0.95:
        base_score *= 0.3

    # Sojourn decay
    time_multiplier = np.exp(-decay_rate * max(0, days_in_state - 1))

    # Velocity boost (positive only)
    if pd.isna(prev_conf):
        velocity = 0.0
    else:
        velocity = max(0.0, current_conf - prev_conf)
    velocity_boost = 1.0 + velocity * velocity_scale

    return base_score * time_multiplier * velocity_boost


def attach_forward_returns(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fetch Polygon daily prices and compute T+h forward returns per row."""
    from walk_forward_ndx import fetch_equity_daily

    tickers = df["ticker"].unique().tolist()
    print(f"  Fetching prices for {len(tickers)} tickers...")
    price_cache: dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers, 1):
        try:
            p = fetch_equity_daily(t, years=1)
            if p is None or p.empty:
                continue
            p.columns = [c.lower() for c in p.columns]
            if "date" in p.columns:
                p["date"] = pd.to_datetime(p["date"])
                p = p.set_index("date")
            elif not isinstance(p.index, pd.DatetimeIndex):
                p.index = pd.to_datetime(p.index)
            if p.index.tz is not None:
                p.index = p.index.tz_localize(None)
            p = p[~p.index.duplicated(keep="last")]
            price_cache[t] = p[["close"]].sort_index()
        except Exception as e:
            print(f"    {t} failed: {e}")
        if i % 20 == 0:
            print(f"    {i}/{len(tickers)}")
    print(f"  Loaded {len(price_cache)}/{len(tickers)}")

    fwd_rets = []
    for _, row in df.iterrows():
        t = row["ticker"]
        sd = row["scan_date"]
        if t not in price_cache:
            fwd_rets.append(np.nan)
            continue
        p = price_cache[t]
        mask = p.index >= sd
        if not mask.any():
            fwd_rets.append(np.nan)
            continue
        entry_idx = p.index[mask][0]
        entry_pos = p.index.get_loc(entry_idx)
        if isinstance(entry_pos, slice):
            entry_pos = entry_pos.start
        entry_price = p.iloc[entry_pos]["close"]
        fwd_pos = entry_pos + horizon
        if fwd_pos >= len(p):
            fwd_rets.append(np.nan)
            continue
        fwd_price = p.iloc[fwd_pos]["close"]
        fwd_rets.append((fwd_price - entry_price) / entry_price)
    df = df.copy()
    df[f"fwd_ret_T{horizon}"] = fwd_rets
    return df


def run_cross_sectional_ic(df: pd.DataFrame, horizon: int) -> tuple[pd.Series, float, float]:
    """Per-day Spearman IC between alpha_score and forward return."""
    ret_col = f"fwd_ret_T{horizon}"
    daily_ics = {}
    for sd, grp in df.groupby("scan_date"):
        sub = grp.dropna(subset=["alpha_score", ret_col])
        # Require at least a few non-zero scores to rank
        if (sub["alpha_score"] > 0).sum() < 5:
            continue
        rho, _p = spearmanr(sub["alpha_score"], sub[ret_col])
        if not np.isnan(rho):
            daily_ics[sd] = rho
    ics = pd.Series(daily_ics).sort_index()
    if len(ics) < 2:
        return ics, np.nan, np.nan
    mean_ic = ics.mean()
    t_stat = mean_ic / (ics.std(ddof=1) / np.sqrt(len(ics)))
    return ics, mean_ic, t_stat


def run_decile_analysis(df: pd.DataFrame, horizon: int, n_deciles: int = 10) -> pd.DataFrame:
    """Pool all (date, ticker) — bucket by score, report mean forward return per decile."""
    ret_col = f"fwd_ret_T{horizon}"
    sub = df.dropna(subset=["alpha_score", ret_col]).copy()
    # Only rank rows with nonzero score (the candidate set)
    scored = sub[sub["alpha_score"] > 0].copy()
    if len(scored) < n_deciles * 3:
        print(f"  ⚠ Only {len(scored)} scored rows — not enough for {n_deciles}-decile analysis")
        return pd.DataFrame()
    scored["decile"] = pd.qcut(scored["alpha_score"], n_deciles, labels=False, duplicates="drop")
    summary = (
        scored.groupby("decile")
        .agg(
            n=(ret_col, "size"),
            mean_ret=(ret_col, "mean"),
            median_ret=(ret_col, "median"),
            hit_rate=(ret_col, lambda x: (x > 0).mean()),
            mean_score=("alpha_score", "mean"),
        )
        .reset_index()
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--optimal-conf", type=float, default=0.65)
    ap.add_argument("--decay", type=float, default=0.2)
    ap.add_argument("--velocity-scale", type=float, default=0.5)
    ap.add_argument("--min-conf", type=float, default=0.60)
    args = ap.parse_args()

    print("=" * 72)
    print("  CROSS-SECTIONAL ALPHA RANKER — IC + Decile Validation")
    print(f"  Horizon T+{args.horizon} | optimal_conf={args.optimal_conf} | decay={args.decay}")
    print("=" * 72)

    scans = load_scans()
    print(f"\nLoaded {len(scans)} scans, {scans['ticker'].nunique()} tickers, "
          f"{scans['scan_date'].nunique()} dates")

    scans = add_persistence_and_velocity(scans)
    scans["alpha_score"] = scans.apply(
        lambda r: compute_alpha_score(
            r["confidence"], r["prev_conf"], r["persistence"], r["regime"],
            optimal_conf=args.optimal_conf, decay_rate=args.decay,
            velocity_scale=args.velocity_scale, min_conf=args.min_conf,
        ),
        axis=1,
    )
    n_scored = (scans["alpha_score"] > 0).sum()
    print(f"Rows with nonzero score: {n_scored}/{len(scans)} ({n_scored/len(scans):.1%})")

    print("\n── Computing forward returns ──")
    scans = attach_forward_returns(scans, args.horizon)

    print("\n── Cross-Sectional IC (per-day Spearman) ──")
    ics, mean_ic, t_stat = run_cross_sectional_ic(scans, args.horizon)
    print(f"  Days with computable IC: {len(ics)}")
    if len(ics) > 0:
        print(f"  Mean daily IC:  {mean_ic:+.4f}")
        print(f"  Std daily IC:   {ics.std(ddof=1):.4f}")
        print(f"  t-stat:         {t_stat:+.2f}")
        print(f"  IC by day:")
        for d, v in ics.items():
            print(f"    {d.date()}  {v:+.4f}")

    print("\n── Decile Analysis (pooled) ──")
    dec = run_decile_analysis(scans, args.horizon)
    if not dec.empty:
        print(dec.to_string(index=False))
        top = dec.iloc[-1]  # highest decile
        bot = dec.iloc[0]
        spread = top["mean_ret"] - bot["mean_ret"]
        print(f"\n  Top decile - Bottom decile mean T+{args.horizon} return: {spread:+.2%}")
        monotone_up = all(dec["mean_ret"].diff().dropna() >= -0.001)
        print(f"  Monotonically increasing (±10bps tol): {monotone_up}")

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if not dec.empty:
        spread_bps = (dec.iloc[-1]["mean_ret"] - dec.iloc[0]["mean_ret"]) * 10_000
        if spread_bps > 100 and t_stat > 1.0:
            print(f"  ✓ Ranker shows positive cross-sectional edge ({spread_bps:.0f}bps spread)")
        elif spread_bps > 50:
            print(f"  ⚠ Marginal edge ({spread_bps:.0f}bps). Low statistical power with N={len(ics)} days.")
        else:
            print(f"  ✗ No cross-sectional edge detected. Do NOT build eviction engine on this ranker.")


if __name__ == "__main__":
    main()
