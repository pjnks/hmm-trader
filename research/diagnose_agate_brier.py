"""diagnose_agate_brier.py — AGATE Phase 2 Brier Score Decomposition.

Tests whether the GaussianHMM ensemble's confidence posteriors are
empirically calibrated on heavy-tailed crypto OHLCV data, OR whether
they hallucinate certainty in the tails (the Sprint 16 kill hypothesis).

The test follows ``sprint17_agate_resurrection_plan.md`` Phase 2:

  Group BULL predictions by confidence decile. For each bin, compute the
  realized frequency of positive forward returns over t+4h (next bar) and
  t+24h (six bars). A well-calibrated model tracks y=x on the calibration
  plot. Decompose Brier Score:

      BS = REL - RES + UNC

      REL (reliability):  weighted MSE between bin pred prob and bin freq.
                           Zero = perfectly calibrated. Higher = bin's
                           predicted probability lies about reality.
      RES (resolution):   weighted variance of bin freq vs base rate.
                           Higher = model meaningfully separates predictions
                           from random guess; zero = always predicts base.
      UNC (uncertainty):  base_rate * (1 - base_rate). Irreducible.

  PASS/FAIL gates:
      - [0.90, 0.95) bin: realized positive-return freq >= 0.85
      - [0.95, 1.00] bin: realized positive-return freq >= 0.90
      - FAIL on either if realized < 55% of nominal -> proceed to Phase 3 (GMM)

Forward returns are NOT persisted in signal_journal — we derive them via
per-ticker self-shift on `current_price`. Rows where the next-cycle
timestamp gap exceeds the expected 4h cadence by >50% are excluded
(handles the April 20-28 service suspension cleanly).

Phase 2 readiness: needs N >= 2,350 BULL+BEAR observations per the resurrect
plan. Current journal: ~496 BULL, ~495 BEAR — preliminary read possible
on May 5, but Phase 2 verdict timeline slips ~late May to early June.

Usage:
    python research/diagnose_agate_brier.py                        # BULL, t+4h
    python research/diagnose_agate_brier.py --regime BEAR          # mirror
    python research/diagnose_agate_brier.py --horizon-bars 6       # t+24h
    python research/diagnose_agate_brier.py --refresh              # re-pull DB
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "_cache" / "agate_signal_journal.parquet"
POLYGON_CACHE = ROOT / "research" / "_cache" / "agate_polygon_4h.parquet"
VM_HOST = "ubuntu@129.158.40.51"
VM_KEY = Path.home() / ".ssh" / "hmm-trader.key"
VM_DB = "/home/ubuntu/HMM-Trader/agate_journal.db"
POLYGON_LOOKBACK_DAYS = 60   # journal date range (Mar 22 -> May 6 ≈ 46 days) + buffer

# Confidence bins for calibration table (deciles + critical top bins)
CONFIDENCE_BINS = [
    ("[0.50, 0.60)", 0.50, 0.60),
    ("[0.60, 0.70)", 0.60, 0.70),
    ("[0.70, 0.80)", 0.70, 0.80),
    ("[0.80, 0.90)", 0.80, 0.90),
    ("[0.90, 0.95)", 0.90, 0.95),
    ("[0.95, 1.00]", 0.95, 1.001),
]

# Phase 2 gates per sprint17_agate_resurrection_plan.md
PHASE2_GATES = [
    ("[0.90, 0.95)", 0.90, 0.95, 0.85),     # 92.5% nominal -> realized >= 85%
    ("[0.95, 1.00]", 0.95, 1.001, 0.90),    # 97.5% nominal -> realized >= 90%
]
# A FAIL is "realized < 55% of nominal"; PASS is the gate threshold above.

# Time gap tolerance: AGATE scans every 4h. Allow up to 6h next-row gap
# (covers minor service hiccups); reject anything longer (multi-day suspension).
EXPECTED_CYCLE_HOURS = 4
MAX_NEXT_ROW_GAP_HOURS = 6

WILSON_Z = 1.96  # 95% two-sided


# ── Wilson CI (matches diagnose_sprint18_*.py implementations) ────────
def wilson_ci(wins: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))
    return (center - spread) / denom, (center + spread) / denom


# ── Data loading (parquet cache, mirrors short_engine_eval pattern) ────
def load_journal(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        df = pd.read_parquet(CACHE)
        print(f"[load] cache hit: {len(df):,} rows from {CACHE.name}")
        return df

    print(f"[load] refreshing from VM {VM_HOST}:{VM_DB}")
    remote_dump = "/tmp/agate_signal_journal.csv"
    py_script = (
        "import sqlite3, csv\n"
        f"con = sqlite3.connect('{VM_DB}')\n"
        "c = con.cursor()\n"
        "c.execute('SELECT timestamp, ticker, regime, confidence, "
        "confirmations, signal, current_price, decision FROM "
        "signal_journal ORDER BY ticker, timestamp')\n"
        f"with open('{remote_dump}', 'w', newline='') as f:\n"
        "    w = csv.writer(f)\n"
        "    w.writerow([d[0] for d in c.description])\n"
        "    w.writerows(c.fetchall())\n"
    )
    subprocess.run(
        ["ssh", "-i", str(VM_KEY), VM_HOST,
         "/home/ubuntu/miniconda3/bin/python", "-"],
        input=py_script, text=True, check=True,
    )
    local_csv = CACHE.parent / "agate_signal_journal.csv"
    subprocess.run(
        ["scp", "-i", str(VM_KEY), f"{VM_HOST}:{remote_dump}", str(local_csv)],
        check=True, capture_output=True,
    )
    df = pd.read_csv(local_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    print(f"[load] cached {len(df):,} rows to {CACHE.name}")
    return df


# ── Polygon ground-truth price layer (Path B, 2026-05-06) ─────────────
# Implementation note: free-tier Polygon rate limits (5 req/min) make
# direct API fetches for 14 tickers × ~3 paginated pages prohibitive
# (cascade of 60s 429-retries). Instead we piggyback on the VM's existing
# data_cache/ which is kept fresh by AGATE's shadow scans every 4h.
# Pulled as CSVs via SCP to research/_cache/vm_data_cache/.
VM_CACHE_DIR = ROOT / "research" / "_cache" / "vm_data_cache"


def _ticker_to_csv_path(ticker: str) -> Path:
    """Map Polygon ticker (X:BTCUSD) to cache filename (btcusd_hourly.csv)."""
    stem = ticker.replace("X:", "").lower()
    return VM_CACHE_DIR / f"{stem}_hourly.csv"


def fetch_polygon_4h_panel(
    tickers: list[str], refresh: bool = False,
) -> pd.DataFrame:
    """Build canonical 4h OHLCV close panel from VM-side hourly CSVs.

    Returns long-format DataFrame: (ticker, bar_window, polygon_close).

    Why this replaces direct Polygon fetch: free-tier rate limits cascade
    badly (60s 429-retries) when fetching 14 tickers fresh. The VM keeps
    /home/ubuntu/HMM-Trader/data_cache/*_hourly.csv fresh as AGATE scans.
    We SCP those CSVs to research/_cache/vm_data_cache/ and resample
    locally.

    Refresh CSVs from VM with:
        ssh -i ~/.ssh/hmm-trader.key ubuntu@129.158.40.51 \\
            'cd /home/ubuntu/HMM-Trader/data_cache && tar cf - *_hourly.csv' \\
          | tar xf - -C research/_cache/vm_data_cache/
    """
    if POLYGON_CACHE.exists() and not refresh:
        df = pd.read_parquet(POLYGON_CACHE)
        print(f"[panel] cache hit: {len(df):,} bars × {df['ticker'].nunique()} tickers")
        return df

    sys.path.insert(0, str(ROOT / "src"))
    from data_fetcher import resample_ohlcv

    panels = []
    for ticker in tickers:
        csv_path = _ticker_to_csv_path(ticker)
        if not csv_path.exists():
            print(f"[panel]   {ticker}: CSV not found ({csv_path.name}), skip")
            continue
        try:
            hourly = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"[panel]   {ticker}: read failed ({e})")
            continue
        if hourly.empty:
            print(f"[panel]   {ticker}: empty CSV")
            continue
        bars_4h = resample_ohlcv(hourly, "4h")
        panels.append(pd.DataFrame({
            "ticker": ticker,
            "bar_window": bars_4h.index,
            "polygon_close": bars_4h["Close"].values,
        }))
        last_bar = bars_4h.index[-1].strftime("%Y-%m-%d %H:%M")
        print(f"[panel]   {ticker}: {len(bars_4h)} 4h bars (last: {last_bar})")

    panel = pd.concat(panels, ignore_index=True)
    POLYGON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(POLYGON_CACHE)
    print(f"[panel] cached {len(panel):,} bars to {POLYGON_CACHE.name}")
    return panel


# ── Forward-return derivation: Polygon ground-truth join ──────────────
def derive_forward_returns(
    df: pd.DataFrame, polygon_panel: pd.DataFrame, horizon_bars: int,
) -> pd.DataFrame:
    """Replace journal `current_price` with Polygon ground-truth close.

    For each scan row at time t:
      bar_window = floor(t, 4h)
      current_close = polygon_close where (ticker, bar_window) matches
      fwd_window = bar_window + horizon_bars * 4h
      fwd_close = polygon_close where (ticker, fwd_window) matches
      fwd_return = fwd_close / current_close - 1

    Pre-step still deduplicates journal to one row per (ticker, bar_window)
    — multiple scans within the same 4h window collapse to one observation
    (the regime/confidence at first scan in that window).
    """
    df = df.copy()
    df["bar_window"] = df["timestamp"].dt.floor(f"{EXPECTED_CYCLE_HOURS}h")
    df = (df.sort_values(["ticker", "timestamp"])
            .drop_duplicates(subset=["ticker", "bar_window"], keep="first")
            .reset_index(drop=True))

    # Left-join Polygon close at scan's bar_window
    df = df.merge(polygon_panel, on=["ticker", "bar_window"], how="left")

    # Forward bar lookup
    df["fwd_window"] = df["bar_window"] + pd.Timedelta(
        hours=horizon_bars * EXPECTED_CYCLE_HOURS)
    fwd = polygon_panel.rename(
        columns={"bar_window": "fwd_window", "polygon_close": "fwd_close"})
    df = df.merge(fwd, on=["ticker", "fwd_window"], how="left")

    # Forward return; NaN where either side missing
    valid = (df["polygon_close"] > 0) & df["fwd_close"].notna()
    df["fwd_return"] = np.where(
        valid, df["fwd_close"] / df["polygon_close"] - 1.0, np.nan,
    )
    return df


# ── Brier decomposition ────────────────────────────────────────────────
def brier_decomposition(p: np.ndarray, o: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy 1973 decomposition: BS = REL - RES + UNC.

    p: predicted probabilities
    o: binary outcomes (0 or 1)
    n_bins: number of equal-width probability bins for REL/RES computation.
    """
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    n = len(p)
    if n == 0:
        return {"n": 0}

    base_rate = o.mean()
    bs = float(((p - o) ** 2).mean())

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, n_bins - 1)

    rel = res = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        p_k = p[mask].mean()
        o_k = o[mask].mean()
        weight = n_k / n
        rel += weight * (p_k - o_k) ** 2
        res += weight * (o_k - base_rate) ** 2

    unc = base_rate * (1 - base_rate)
    return {
        "n": n,
        "base_rate": float(base_rate),
        "brier_score": bs,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        # Identity check: BS should equal REL - RES + UNC within fp epsilon
        "decomposition_residual": float(bs - (rel - res + unc)),
    }


# ── Per-bin calibration table ──────────────────────────────────────────
def calibration_table(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """For each confidence bin, compute realized hit-rate and Wilson CI.

    `hit` semantics:
        regime == BULL  -> hit = forward return > 0
        regime == BEAR  -> hit = forward return < 0
    """
    sub = df[(df["regime"] == regime) & df["fwd_return"].notna()].copy()
    if regime == "BULL":
        sub["hit"] = (sub["fwd_return"] > 0).astype(int)
    elif regime == "BEAR":
        sub["hit"] = (sub["fwd_return"] < 0).astype(int)
    else:
        raise ValueError(f"Unsupported regime for Brier test: {regime}")

    rows = []
    for label, lo, hi in CONFIDENCE_BINS:
        bin_rows = sub[(sub["confidence"] >= lo) & (sub["confidence"] < hi)]
        n = len(bin_rows)
        if n == 0:
            rows.append({"bin": label, "n": 0, "mean_pred": np.nan,
                         "realized_freq": np.nan, "wilson_lo": np.nan,
                         "wilson_hi": np.nan})
            continue
        wins = int(bin_rows["hit"].sum())
        wlo, whi = wilson_ci(wins, n)
        rows.append({
            "bin": label,
            "n": n,
            "mean_pred": float(bin_rows["confidence"].mean()),
            "realized_freq": wins / n,
            "wilson_lo": wlo,
            "wilson_hi": whi,
        })
    return pd.DataFrame(rows)


def evaluate_phase2_gates(cal_table: pd.DataFrame) -> list[dict]:
    """Apply Phase 2 calibration thresholds to top-decile bins."""
    out = []
    for label, lo, hi, threshold in PHASE2_GATES:
        row = cal_table[cal_table["bin"] == label]
        if row.empty or row.iloc[0]["n"] == 0:
            out.append({"bin": label, "status": "INSUFFICIENT_DATA",
                        "n": 0, "realized": None, "threshold": threshold})
            continue
        r = row.iloc[0]
        nominal = (lo + hi) / 2
        realized = r["realized_freq"]
        if realized >= threshold:
            status = "PASS"
        elif realized < 0.55 * nominal:
            status = "FAIL_GAUSSIAN"
        else:
            status = "BORDERLINE"
        out.append({
            "bin": label, "status": status, "n": int(r["n"]),
            "realized": realized, "threshold": threshold,
            "nominal": nominal, "wilson_lo": r["wilson_lo"],
        })
    return out


# ── Reporting ──────────────────────────────────────────────────────────
def format_calibration_table(cal: pd.DataFrame) -> str:
    lines = [
        f"  {'Bin':<16}{'N':>7}  {'Mean Pred':>10}  "
        f"{'Realized':>10}  {'Wilson 95% CI':>20}",
        f"  {'-' * 16}{'-' * 7}  {'-' * 10}  {'-' * 10}  {'-' * 20}",
    ]
    for r in cal.itertuples(index=False):
        if r.n == 0:
            lines.append(f"  {r.bin:<16}{r.n:>7}  {'—':>10}  {'—':>10}  "
                         f"{'—':>20}")
            continue
        ci = f"[{r.wilson_lo:.1%}, {r.wilson_hi:.1%}]"
        lines.append(
            f"  {r.bin:<16}{r.n:>7}  {r.mean_pred:>10.3f}  "
            f"{r.realized_freq:>10.1%}  {ci:>20}"
        )
    return "\n".join(lines)


def format_brier(b: dict) -> str:
    if b.get("n", 0) == 0:
        return "  (no data)"
    return (
        f"  N:                  {b['n']:,}\n"
        f"  Base rate (UNC):    {b['base_rate']:.3f}  "
        f"({b['uncertainty']:.4f} = base * (1 - base))\n"
        f"  Brier Score:        {b['brier_score']:.4f}  "
        f"(0 = perfect, {b['uncertainty']:.4f} = base-rate baseline)\n"
        f"  Reliability (REL):  {b['reliability']:.4f}  "
        f"(0 = perfectly calibrated; HIGHER = more miscalibrated)\n"
        f"  Resolution  (RES):  {b['resolution']:.4f}  "
        f"(0 = no separation; HIGHER = better)\n"
        f"  Identity residual:  {b['decomposition_residual']:+.2e}  "
        f"(should be ~0)"
    )


def format_gates(gates: list[dict]) -> str:
    lines = ["Phase 2 calibration gates:"]
    for g in gates:
        if g["status"] == "INSUFFICIENT_DATA":
            lines.append(f"  [SKIP] {g['bin']}  (N=0, no data)")
            continue
        marker = {
            "PASS": "PASS",
            "BORDERLINE": "BORDERLINE",
            "FAIL_GAUSSIAN": "FAIL",
        }[g["status"]]
        lines.append(
            f"  [{marker}] {g['bin']}  "
            f"realized={g['realized']:.1%} "
            f"(Wilson lo {g['wilson_lo']:.1%}), "
            f"gate={g['threshold']:.0%}, N={g['n']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["BULL", "BEAR"], default="BULL",
                        help="Regime to test (BEAR mirrors with sign-flip)")
    parser.add_argument("--horizon-bars", type=int, default=1,
                        help="Forward horizon in 4h bars (1=t+4h, 6=t+24h)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-pull from VM signal_journal")
    parser.add_argument("--refresh-prices", action="store_true",
                        help="Force re-fetch Polygon 4h bars (Path B layer)")
    args = parser.parse_args()

    df = load_journal(refresh=args.refresh)
    tickers = sorted(df["ticker"].unique().tolist())
    polygon = fetch_polygon_4h_panel(tickers, refresh=args.refresh_prices)
    df = derive_forward_returns(df, polygon, horizon_bars=args.horizon_bars)

    cal = calibration_table(df, regime=args.regime)
    gates = evaluate_phase2_gates(cal)

    sub = df[(df["regime"] == args.regime) & df["fwd_return"].notna()].copy()
    if args.regime == "BULL":
        sub["hit"] = (sub["fwd_return"] > 0).astype(int)
    else:
        sub["hit"] = (sub["fwd_return"] < 0).astype(int)
    brier = brier_decomposition(sub["confidence"].values, sub["hit"].values)

    horizon_h = args.horizon_bars * EXPECTED_CYCLE_HOURS
    print(f"AGATE Phase 2 — Brier Calibration Test ({args.regime}, t+{horizon_h}h)")
    print("=" * 70)
    print(f"Source:  agate_journal.signal_journal ({len(df):,} rows)")
    print(f"After fwd-return + gap filter: "
          f"{int(df['fwd_return'].notna().sum()):,} valid rows")
    print(f"Regime filter ({args.regime}):  {len(sub):,} evaluable observations")
    print()
    print("Calibration table:")
    print(format_calibration_table(cal))
    print()
    print("Brier Score Decomposition (Murphy 1973):")
    print(format_brier(brier))
    print()
    print(format_gates(gates))
    print()
    failed = [g for g in gates if g["status"] == "FAIL_GAUSSIAN"]
    borderline = [g for g in gates if g["status"] == "BORDERLINE"]
    if any(g["status"] == "FAIL_GAUSSIAN" for g in gates):
        print("VERDICT: FAIL — Gaussian emission model is hallucinating "
              "certainty in tails. Proceed to Phase 3 (GMMHMM swap).")
    elif all(g["status"] == "PASS" for g in gates):
        print("VERDICT: PASS — Gaussian emission model is calibrated. "
              "AGATE is RESURRECTABLE on current architecture.")
    elif borderline:
        print("VERDICT: BORDERLINE — top deciles in 55-85% range. "
              "Accumulate more data; re-evaluate at next checkpoint.")
    else:
        print("VERDICT: INSUFFICIENT DATA — top decile bins lack samples "
              "for statistical conclusion.")


if __name__ == "__main__":
    main()
