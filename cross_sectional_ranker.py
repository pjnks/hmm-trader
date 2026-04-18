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


def compute_gaussian_alpha(
    current_conf: float,
    prev_conf: float,
    days_in_state: int,
    regime: str,
    optimal_conf: float = 0.65,
    decay_rate: float = 0.2,
    velocity_scale: float = 0.5,
    min_conf: float = 0.60,
) -> float:
    """Original Gaussian scoring (kept for A/B comparison)."""
    if regime != "BULL" or pd.isna(current_conf) or current_conf < min_conf:
        return 0.0
    conf_penalty = abs(current_conf - optimal_conf)
    base_score = max(0.0, 1.0 - conf_penalty)
    if 0.90 <= current_conf < 0.95:
        base_score *= 0.3
    time_multiplier = np.exp(-decay_rate * max(0, days_in_state - 1))
    velocity = 0.0 if pd.isna(prev_conf) else max(0.0, current_conf - prev_conf)
    velocity_boost = 1.0 + velocity * velocity_scale
    return base_score * time_multiplier * velocity_boost


def compute_bimodal_alpha(
    current_conf: float,
    prev_conf: float,
    days_in_state: int,
    regime: str,
    decay_rate: float = 0.15,
    velocity_scale: float = 0.5,
) -> float:
    """
    Bimodal alpha score — piecewise reflecting the three-phase curve:
      Phase 1 (<0.70): Information Advantage — HMM catches inflection early.
      Phase 2 (0.90-0.95): Valley of Death — exhaustion / exit liquidity.
      Phase 3 (>=0.95): Structural Drift — deep confirmed trend.
    """
    if regime != "BULL" or pd.isna(current_conf):
        return 0.0

    if current_conf < 0.60:
        return 0.0  # Weed out CHOP / unclear signals
    elif current_conf < 0.70:
        base_score = 1.0      # Information Advantage
    elif current_conf < 0.90:
        base_score = 0.5      # Middle ground — mild reward
    elif current_conf < 0.95:
        base_score = 0.1      # Valley of Death
    else:
        base_score = 0.8      # Structural Drift

    # Exponential sojourn decay
    decay = np.exp(-decay_rate * max(0, days_in_state - 1))

    # Velocity boost (positive delta only)
    velocity = 0.0 if pd.isna(prev_conf) else max(0.0, current_conf - prev_conf)
    velocity_boost = 1.0 + velocity * velocity_scale

    return base_score * decay * velocity_boost


def compute_alpha_score(*args, scoring: str = "bimodal", **kwargs) -> float:
    """Dispatcher — select scoring function by name."""
    if scoring == "bimodal":
        # Drop gaussian-only kwargs
        kwargs.pop("optimal_conf", None)
        kwargs.pop("min_conf", None)
        return compute_bimodal_alpha(*args, **kwargs)
    return compute_gaussian_alpha(*args, **kwargs)


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


def add_relative_returns(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Market-neutralize forward returns: for each scan_date, subtract the universe
    mean forward return from every row. Isolates alpha from market beta.

    Baseline = mean of ALL tickers scanned that day (including non-scored rows).
    This prevents the BULL-only universe from double-dipping — we want absolute
    alpha vs the full opportunity set, not just vs other BULL signals.
    """
    ret_col = f"fwd_ret_T{horizon}"
    rel_col = f"rel_ret_T{horizon}"
    df = df.copy()
    daily_mean = df.groupby("scan_date")[ret_col].transform("mean")
    df[rel_col] = df[ret_col] - daily_mean
    return df


def run_decile_analysis(
    df: pd.DataFrame,
    horizon: int,
    n_deciles: int = 10,
    use_relative: bool = True,
) -> pd.DataFrame:
    """Pool all (date, ticker) — bucket by score, report return per decile."""
    raw_col = f"fwd_ret_T{horizon}"
    rel_col = f"rel_ret_T{horizon}"
    ret_col = rel_col if use_relative else raw_col
    sub = df.dropna(subset=["alpha_score", ret_col]).copy()
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


def run_confidence_zone_diagnostic(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Trap #1 diagnostic: break out neutralized returns by confidence zone
    to check whether the >=95% bucket actually earns alpha or just rides beta.
    """
    rel_col = f"rel_ret_T{horizon}"
    sub = df[df["regime"] == "BULL"].dropna(subset=["confidence", rel_col]).copy()

    zones = [
        ("< 0.60 (noise)",       sub["confidence"] < 0.60),
        ("0.60–0.70 (Phase 1)",  (sub["confidence"] >= 0.60) & (sub["confidence"] < 0.70)),
        ("0.70–0.90 (mid)",      (sub["confidence"] >= 0.70) & (sub["confidence"] < 0.90)),
        ("0.90–0.95 (VoD)",      (sub["confidence"] >= 0.90) & (sub["confidence"] < 0.95)),
        ("≥ 0.95 (Struct Drift)", sub["confidence"] >= 0.95),
    ]

    rows = []
    for label, mask in zones:
        z = sub[mask]
        if len(z) == 0:
            rows.append({"zone": label, "n": 0, "mean_rel_ret": np.nan,
                         "hit_rate": np.nan, "mean_conf": np.nan})
            continue
        rows.append({
            "zone": label,
            "n": len(z),
            "mean_rel_ret": z[rel_col].mean(),
            "hit_rate": (z[rel_col] > 0).mean(),
            "mean_conf": z["confidence"].mean(),
        })
    return pd.DataFrame(rows)


def check_top_decile_vs_benchmark(
    decile_df: pd.DataFrame,
) -> tuple[bool, float]:
    """
    Trap #3 check: top decile must beat universe benchmark (mean_ret > 0
    in neutralized terms). A positive spread driven only by short-side alpha
    is unmonetizable in a long-only strategy.

    Returns (passes, top_decile_return).
    """
    if decile_df.empty:
        return False, np.nan
    top_decile = decile_df.iloc[-1]  # highest score bucket
    top_ret = top_decile["mean_ret"]
    # In market-neutralized terms, mean_ret > 0 means top decile beats universe
    return top_ret > 0, top_ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--optimal-conf", type=float, default=0.65)
    ap.add_argument("--decay", type=float, default=0.2)
    ap.add_argument("--velocity-scale", type=float, default=0.5)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--scoring", choices=["bimodal", "gaussian"], default="bimodal",
                    help="bimodal: piecewise (phase 1/2/3); gaussian: continuous peak at optimal_conf")
    ap.add_argument("--db", default="/home/ubuntu/HMM-Trader/beryl_trades.db",
                    help="SQLite DB path (override for backfill DB)")
    ap.add_argument("--table", default="scan_journal",
                    help="Table name (override for scan_journal_backfill)")
    args = ap.parse_args()

    print("=" * 72)
    print("  CROSS-SECTIONAL ALPHA RANKER — IC + Decile Validation")
    print(f"  Scoring: {args.scoring} | Horizon T+{args.horizon} | decay={args.decay}")
    print(f"  DB: {args.db} | Table: {args.table}")
    print("=" * 72)

    global DB_PATH
    DB_PATH = Path(args.db)
    # Patch load_scans to use custom table if needed
    if args.table != "scan_journal":
        con = sqlite3.connect(str(DB_PATH))
        scans = pd.read_sql(
            f"SELECT scan_date, ticker, regime, confidence, confirmations, "
            f"signal, close_price FROM {args.table} ORDER BY ticker, scan_date",
            con,
        )
        con.close()
        scans["scan_date"] = pd.to_datetime(scans["scan_date"])
    else:
        scans = load_scans()
    print(f"\nLoaded {len(scans)} scans, {scans['ticker'].nunique()} tickers, "
          f"{scans['scan_date'].nunique()} dates")

    scans = add_persistence_and_velocity(scans)
    scans["alpha_score"] = scans.apply(
        lambda r: compute_alpha_score(
            r["confidence"], r["prev_conf"], r["persistence"], r["regime"],
            scoring=args.scoring,
            optimal_conf=args.optimal_conf, decay_rate=args.decay,
            velocity_scale=args.velocity_scale, min_conf=args.min_conf,
        ),
        axis=1,
    )
    n_scored = (scans["alpha_score"] > 0).sum()
    print(f"Rows with nonzero score: {n_scored}/{len(scans)} ({n_scored/len(scans):.1%})")

    print("\n── Computing forward returns ──")
    scans = attach_forward_returns(scans, args.horizon)
    scans = add_relative_returns(scans, args.horizon)

    print("\n── Cross-Sectional IC (per-day Spearman) ──")
    ics, mean_ic, t_stat = run_cross_sectional_ic(scans, args.horizon)
    print(f"  Days with computable IC: {len(ics)}")
    if len(ics) > 0:
        print(f"  Mean daily IC:  {mean_ic:+.4f}")
        print(f"  Std daily IC:   {ics.std(ddof=1):.4f}")
        print(f"  t-stat:         {t_stat:+.2f}")

    print("\n── Decile Analysis (RAW returns, beta-contaminated) ──")
    dec_raw = run_decile_analysis(scans, args.horizon, use_relative=False)
    if not dec_raw.empty:
        print(dec_raw.to_string(index=False))
        raw_spread = dec_raw.iloc[-1]["mean_ret"] - dec_raw.iloc[0]["mean_ret"]
        print(f"  Top − Bottom (raw): {raw_spread:+.2%}")

    print("\n── Decile Analysis (MARKET-NEUTRAL relative returns) ──")
    dec_rel = run_decile_analysis(scans, args.horizon, use_relative=True)
    if not dec_rel.empty:
        print(dec_rel.to_string(index=False))
        rel_spread = dec_rel.iloc[-1]["mean_ret"] - dec_rel.iloc[0]["mean_ret"]
        print(f"\n  Top − Bottom (relative, α): {rel_spread:+.2%}")
        diffs = dec_rel["mean_ret"].diff().dropna()
        monotone = (diffs >= -0.001).all()
        n_inversions = (diffs < -0.001).sum()
        print(f"  Monotonically increasing: {monotone} ({n_inversions} inversions)")

    # ── Trap #1: Confidence Zone Diagnostic ──
    print("\n── Confidence Zone Diagnostic (Trap #1: Structural Drift check) ──")
    zone_diag = run_confidence_zone_diagnostic(scans, args.horizon)
    if not zone_diag.empty:
        print(zone_diag.to_string(index=False))
        # Flag the >=95% bucket specifically
        struct_drift = zone_diag[zone_diag["zone"].str.contains("Struct")]
        if len(struct_drift) > 0 and struct_drift.iloc[0]["n"] > 10:
            sd_ret = struct_drift.iloc[0]["mean_rel_ret"]
            if sd_ret <= 0:
                print(f"\n  ⚠ STRUCTURAL DRIFT TRAP: ≥95% bucket has mean relative return "
                      f"{sd_ret:+.4f} — scoring at 0.8 rewards beta, not alpha.")
                print(f"    → Flatten right tail: set ≥95% base_score to 0.0 or 0.1")
            else:
                print(f"\n  ✓ ≥95% bucket relative return {sd_ret:+.4f} — alpha confirmed.")

    # ── Trap #3: Top Decile vs Universe Benchmark ──
    top_passes = False
    top_ret = np.nan
    if not dec_rel.empty:
        top_passes, top_ret = check_top_decile_vs_benchmark(dec_rel)
        print(f"\n── Top Decile vs Universe (Trap #3: Long-Only Illusion check) ──")
        print(f"  Top decile mean neutralized return: {top_ret:+.4f}")
        if top_passes:
            print(f"  ✓ Top decile beats universe — long-only alpha confirmed.")
        else:
            print(f"  ✗ LONG-ONLY ILLUSION: Top decile does NOT beat universe mean.")
            print(f"    Spread is driven by short-side alpha — unmonetizable long-only.")

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if not dec_rel.empty:
        spread_bps = (dec_rel.iloc[-1]["mean_ret"] - dec_rel.iloc[0]["mean_ret"]) * 10_000
        # Gate 3 now requires ALL of:
        #   1. spread > 100bps
        #   2. IC t-stat > 2.0
        #   3. near-monotonic
        #   4. top decile beats universe (Trap #3 fix)
        if spread_bps > 100 and t_stat > 2.0 and monotone and top_passes:
            print(f"  ✓ Ranker shows clean cross-sectional alpha ({spread_bps:.0f}bps monotonic spread)")
            print(f"    Top decile α: {top_ret:+.4f} (beats universe)")
        elif spread_bps > 50 and t_stat > 1.5 and top_passes:
            print(f"  ⚠ Suggestive edge ({spread_bps:.0f}bps α). Needs larger sample before shipping.")
        elif spread_bps > 100 and not top_passes:
            print(f"  ✗ FAIL: {spread_bps:.0f}bps spread but top decile below universe.")
            print(f"    This is short-side alpha only. Do NOT build eviction engine.")
        else:
            print(f"  ✗ No robust edge. Do NOT build eviction engine.")


if __name__ == "__main__":
    main()
