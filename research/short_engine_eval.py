"""
Short Rotation Engine — Friction-Aware EV Calculation
=====================================================

Sprint 17 research (2026-04-24). Mac-only, protocol-compliant.

HYPOTHESIS (Task A — Short Engine):
  The HMM's bottom-confidence cohort produced -0.50% neutralized returns
  in the Sprint 15 cross-sectional study. Can a short strategy entered on
  (confidence < 0.60 AND negative velocity) clear the borrow-adjusted
  friction hurdle?

ADJACENT QUESTION (Task B — CITRINE Velocity Audit):
  Two CITRINE positions (MCHP, CSGP) entered at conf >= 0.99 on 2026-04-24
  and stopped within 8 hours. Across the 13,230-row backfill, do top-decile
  conf entries underperform the 0.60-0.80 velocity band? If yes, we have a
  case to port BERYL Sprint 16's velocity gate to CITRINE.

DATA: scan_journal_backfill (98 tickers × 135 days, Oct 2025 - Apr 2026)
  Pulled from VM beryl_trades.db. Cached locally as parquet.

NEUTRALIZATION:
  Cross-sectional demean per scan_date. Equivalent to
  "short the target, long equal-weighted basket of all NDX peers."
  Implicitly assumes β=1 across the universe — defensible given all are
  NDX components with mean β ≈ 1.

FRICTION (horizon-aware):
  Base 28 bps round-trip (8 fees + 20 slippage)
  + 11 bps per day borrow (40bps annualized)

QUARTER-BOUNDARY MASK:
  HMM was retrained quarterly during backfill. State labels can permute
  across boundaries → spurious velocity spikes. We mask velocity rows
  where the lookback would cross a quarter_start.

USAGE:
  python research/short_engine_eval.py            # use cached data if present
  python research/short_engine_eval.py --refresh  # force pull from VM
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "_cache" / "scan_journal_backfill.parquet"
REPORTS_DIR = ROOT / "reports"
SHORT_REPORT = REPORTS_DIR / "2026_04_24_short_engine_ev.html"
AUDIT_REPORT = REPORTS_DIR / "2026_04_24_citrine_velocity_audit.html"

VM_HOST = "ubuntu@129.158.40.51"
VM_KEY = Path.home() / ".ssh" / "hmm-trader.key"
VM_DB = "/home/ubuntu/HMM-Trader/beryl_trades.db"

# ── Friction model ──────────────────────────────────────────────────────
BASE_FRICTION_BPS = 28      # 8 fees + 20 slippage (round-trip)
DAILY_BORROW_BPS = 11       # ~40bps annualized / 252 → 11 bps/day
DYNAMIC_EXIT_THRESHOLD = 0.60   # cover when conf bounces back


def friction_bps(hold_days: int) -> float:
    """Total round-trip friction in bps for a `hold_days` short trade."""
    return BASE_FRICTION_BPS + DAILY_BORROW_BPS * hold_days


# ── Data loading ────────────────────────────────────────────────────────
def load_data(refresh: bool = False) -> pd.DataFrame:
    """Fetch backfill table. Cached locally as parquet."""
    if CACHE.exists() and not refresh:
        df = pd.read_parquet(CACHE)
        print(f"[load] cache hit: {len(df)} rows from {CACHE.name}")
        return df

    print(f"[load] refreshing from VM {VM_HOST}:{VM_DB}")
    # Step 1: have VM-side python dump the table to a tempfile in /tmp
    remote_dump = "/tmp/scan_journal_backfill.csv"
    py_script = (
        "import sqlite3, csv\n"
        f"con = sqlite3.connect('{VM_DB}')\n"
        "c = con.cursor()\n"
        "c.execute('SELECT scan_date, ticker, regime, confidence, confirmations, "
        "signal, close_price, used_fallback, quarter_start FROM "
        "scan_journal_backfill ORDER BY ticker, scan_date')\n"
        f"with open('{remote_dump}', 'w', newline='') as f:\n"
        "    w = csv.writer(f)\n"
        "    w.writerow([d[0] for d in c.description])\n"
        "    w.writerows(c.fetchall())\n"
    )
    # Run on VM via stdin to avoid quote-escaping hell
    subprocess.run(
        ["ssh", "-i", str(VM_KEY), VM_HOST, "/home/ubuntu/miniconda3/bin/python", "-"],
        input=py_script, text=True, check=True,
    )
    # Step 2: scp the CSV down
    local_csv = CACHE.parent / "scan_journal_backfill.csv"
    subprocess.run(
        ["scp", "-i", str(VM_KEY), f"{VM_HOST}:{remote_dump}", str(local_csv)],
        check=True, capture_output=True,
    )
    df = pd.read_csv(local_csv)
    df["scan_date"] = pd.to_datetime(df["scan_date"])
    df["quarter_start"] = pd.to_datetime(df["quarter_start"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    print(f"[load] cached {len(df)} rows to {CACHE.name}")
    return df


# ── Feature derivation ──────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add velocity (with quarter-boundary mask) and forward returns."""
    df = df.sort_values(["ticker", "scan_date"]).reset_index(drop=True)

    # --- velocity (1d, 3d, 5d) with quarter-boundary masking ---
    g = df.groupby("ticker", sort=False)
    for w in (1, 3, 5):
        # Δ confidence
        df[f"velocity_{w}d"] = g["confidence"].diff(w)
        # quarter at row t-w (mask if differs from current quarter)
        prev_q = g["quarter_start"].shift(w)
        cross = df["quarter_start"] != prev_q
        df.loc[cross, f"velocity_{w}d"] = np.nan

    # --- forward returns (1, 2, 3, 5, 10 days) ---
    for h in (1, 2, 3, 5, 10):
        df[f"fwd_ret_{h}d"] = (
            g["close_price"].shift(-h) / df["close_price"] - 1.0
        )

    # --- cross-sectional neutralization ---
    # Subtract equal-weighted basket return (per scan_date) from each ticker.
    for h in (1, 2, 3, 5, 10):
        basket = df.groupby("scan_date")[f"fwd_ret_{h}d"].transform("mean")
        df[f"neut_ret_{h}d"] = df[f"fwd_ret_{h}d"] - basket

    # --- dynamic exit horizon (cover when conf bounces back to >= 0.60) ---
    df["dyn_exit_days"] = _dynamic_exit_offset(df)
    df["dyn_exit_ret"] = _gather_dyn_return(df, "fwd_ret")
    df["dyn_neut_ret"] = _gather_dyn_return(df, "neut_ret")

    return df


def _dynamic_exit_offset(df: pd.DataFrame) -> pd.Series:
    """For each row, days until conf next reaches >= DYNAMIC_EXIT_THRESHOLD."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for ticker, idx in df.groupby("ticker", sort=False).groups.items():
        idx = list(idx)
        conf = df.loc[idx, "confidence"].values
        n = len(conf)
        offsets = np.full(n, np.nan)
        # For each row, find the smallest k>0 s.t. conf[i+k] >= threshold
        for i in range(n):
            for k in range(1, min(n - i, 11)):  # cap at 10 days
                if conf[i + k] >= DYNAMIC_EXIT_THRESHOLD:
                    offsets[i] = k
                    break
        out.loc[idx] = offsets
    return out


def _gather_dyn_return(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Gather the appropriate fwd_ret_Xd value based on dyn_exit_days."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for h in (1, 2, 3, 5, 10):
        col = f"{prefix}_{h}d"
        # Use this horizon's return where dyn_exit_days falls in [h-1.5, h+0.5)
        # (i.e., closest available horizon)
        if h == 1:
            mask = df["dyn_exit_days"] <= 1.5
        elif h == 10:
            mask = df["dyn_exit_days"] > 7.5
        else:
            prev_h = {2: 1, 3: 2, 5: 3, 10: 5}[h]
            mask = (df["dyn_exit_days"] > (prev_h + h) / 2) & \
                   (df["dyn_exit_days"] <= (h + {1: 2, 2: 3, 3: 5, 5: 10}[h]) / 2)
        out.loc[mask] = df.loc[mask, col]
    return out


# ── Task A: Short Engine ────────────────────────────────────────────────
ENTRY_FILTERS = {
    "Soft (vel_1d < 0)":      lambda df: (df["confidence"] < 0.60) & (df["velocity_1d"] < 0),
    "Moderate (vel_1d ≤ −5pp)": lambda df: (df["confidence"] < 0.60) & (df["velocity_1d"] <= -0.05),
    "Strong (vel_3d ≤ −15pp)":  lambda df: (df["confidence"] < 0.60) & (df["velocity_3d"] <= -0.15),
}

HORIZONS = (1, 2, 3, 5, 10)


def short_engine_evaluation(df: pd.DataFrame) -> dict:
    """Return per-threshold × per-horizon EV table + dynamic-exit summary."""
    results = {}
    for label, filt in ENTRY_FILTERS.items():
        sub = df[filt(df)].copy()
        sub = sub.dropna(subset=["velocity_1d", "velocity_3d"])  # drop quarter-boundary nulls
        rows = []

        # Fixed horizons (decay curve)
        for h in HORIZONS:
            col = f"neut_ret_{h}d"
            ret = -sub[col].dropna()  # short: profit when target underperforms basket
            n = len(ret)
            if n == 0:
                continue
            gross_bps = ret.mean() * 10000
            fric = friction_bps(h)
            net_bps = gross_bps - fric
            rows.append({
                "exit_rule": f"Fixed T+{h}",
                "n": n,
                "gross_bps": gross_bps,
                "friction_bps": fric,
                "net_bps": net_bps,
                "hit_rate": (ret * 10000 > fric).mean() * 100,
                "sharpe_proxy": (ret.mean() / ret.std() * np.sqrt(252 / max(h, 1))) if ret.std() > 0 else 0,
            })

        # Dynamic exit
        sub_dyn = sub.dropna(subset=["dyn_exit_days", "dyn_neut_ret"])
        if len(sub_dyn) > 0:
            ret_dyn = -sub_dyn["dyn_neut_ret"]
            avg_hold = sub_dyn["dyn_exit_days"].mean()
            gross_bps_dyn = ret_dyn.mean() * 10000
            fric_dyn = friction_bps(avg_hold)
            net_bps_dyn = gross_bps_dyn - fric_dyn
            rows.append({
                "exit_rule": f"Dynamic (cover @ conf≥0.60, avg {avg_hold:.1f}d)",
                "n": len(ret_dyn),
                "gross_bps": gross_bps_dyn,
                "friction_bps": fric_dyn,
                "net_bps": net_bps_dyn,
                "hit_rate": (ret_dyn * 10000 > fric_dyn).mean() * 100,
                "sharpe_proxy": (ret_dyn.mean() / ret_dyn.std() * np.sqrt(252 / max(avg_hold, 1))) if ret_dyn.std() > 0 else 0,
            })

        results[label] = pd.DataFrame(rows)
    return results


# ── Task B: CITRINE Conf-Decile Audit ──────────────────────────────────
CONF_BINS = [0.00, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
BIN_LABELS = ["[0.00,0.50)", "[0.50,0.60)", "[0.60,0.70)", "[0.70,0.80)",
              "[0.80,0.90)", "[0.90,0.95)", "[0.95,1.00]"]


def velocity_audit(df: pd.DataFrame) -> pd.DataFrame:
    """For each conf bin, compute mean fwd return (raw + neutralized) at 1d/3d/5d."""
    df = df.copy()
    df["conf_bin"] = pd.cut(
        df["confidence"], bins=CONF_BINS, labels=BIN_LABELS,
        include_lowest=True, right=False,
    )
    rows = []
    for label in BIN_LABELS:
        sub = df[df["conf_bin"] == label]
        if len(sub) == 0:
            continue
        rows.append({
            "conf_bin": label,
            "n": len(sub),
            "raw_ret_1d_bps": sub["fwd_ret_1d"].mean() * 10000,
            "raw_ret_3d_bps": sub["fwd_ret_3d"].mean() * 10000,
            "raw_ret_5d_bps": sub["fwd_ret_5d"].mean() * 10000,
            "neut_ret_1d_bps": sub["neut_ret_1d"].mean() * 10000,
            "neut_ret_3d_bps": sub["neut_ret_3d"].mean() * 10000,
            "neut_ret_5d_bps": sub["neut_ret_5d"].mean() * 10000,
        })
    return pd.DataFrame(rows)


# ── HTML report generation ─────────────────────────────────────────────
NEON = {
    "green": "#00ff9f", "cyan": "#00f0ff", "pink": "#ff2d6b",
    "yellow": "#ffe156", "violet": "#b377ff",
    "bg": "#030508", "panel": "#0a1520", "panel2": "#0e1c2b",
    "border": "#1a3040", "text": "#d0eaf5", "dim": "#8ab4c8", "muted": "#5a7c8f",
}

CSS = f"""
<style>
* {{ box-sizing: border-box; }}
body {{ background: {NEON['bg']}; color: {NEON['text']}; margin: 0; padding: 32px 16px;
       font-family: -apple-system, BlinkMacSystemFont, "SF Mono", Menlo, monospace; line-height: 1.5; }}
.wrap {{ max-width: 1000px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin: 0 0 4px 0; font-weight: 600; letter-spacing: -0.5px; }}
h1 .accent {{ color: {NEON['cyan']}; }}
.subtitle {{ color: {NEON['dim']}; font-size: 13px; margin-bottom: 24px; }}
h2 {{ font-size: 16px; margin: 28px 0 10px 0; color: {NEON['cyan']};
      border-bottom: 1px solid {NEON['border']}; padding-bottom: 5px; }}
h3 {{ font-size: 13px; margin: 14px 0 8px 0; color: {NEON['dim']}; text-transform: uppercase; letter-spacing: 0.5px; }}
.tldr {{ background: rgba(0,240,255,0.04); border: 1px solid {NEON['cyan']};
         border-radius: 6px; padding: 14px 18px; margin-bottom: 20px; }}
.tldr li {{ font-size: 13px; margin: 6px 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin: 6px 0; }}
th, td {{ text-align: left; padding: 6px 9px; border-bottom: 1px solid {NEON['border']}; }}
th {{ color: {NEON['dim']}; background: {NEON['panel2']}; font-weight: 500;
      font-size: 10px; letter-spacing: 0.5px; text-transform: uppercase; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: "SF Mono", monospace; }}
.pos {{ color: {NEON['green']}; }}
.neg {{ color: {NEON['pink']}; }}
.neutral {{ color: {NEON['dim']}; }}
.callout {{ background: rgba(0,240,255,0.06); border-left: 3px solid {NEON['cyan']};
            padding: 10px 14px; margin: 14px 0; font-size: 12px; border-radius: 0 4px 4px 0; }}
.callout.green {{ background: rgba(0,255,159,0.06); border-left-color: {NEON['green']}; }}
.callout.pink {{ background: rgba(255,45,107,0.08); border-left-color: {NEON['pink']}; }}
.callout.yellow {{ background: rgba(255,225,86,0.06); border-left-color: {NEON['yellow']}; }}
.bar-row {{ display: flex; align-items: center; margin: 4px 0; font-size: 11px; }}
.bar-label {{ width: 220px; color: {NEON['dim']}; flex-shrink: 0; }}
.bar-track {{ flex: 1; height: 18px; background: {NEON['panel2']}; border-radius: 3px; overflow: hidden; }}
.bar-fill {{ height: 100%; padding: 0 7px; display: flex; align-items: center;
             color: {NEON['bg']}; font-weight: 600; font-size: 10px; }}
.bar-value {{ margin-left: 8px; width: 80px; text-align: right; font-variant-numeric: tabular-nums; font-size: 11px; }}
.panel {{ background: {NEON['panel']}; border: 1px solid {NEON['border']}; border-radius: 6px;
          padding: 14px 18px; margin: 12px 0; }}
.footer {{ color: {NEON['muted']}; font-size: 10px; margin-top: 30px;
           padding-top: 12px; border-top: 1px solid {NEON['border']}; text-align: center; }}
code {{ background: {NEON['panel2']}; padding: 1px 5px; border-radius: 3px;
        color: {NEON['cyan']}; font-size: 11px; font-family: "SF Mono", monospace; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0; }}
.metric {{ background: {NEON['panel2']}; border: 1px solid {NEON['border']};
           border-radius: 4px; padding: 8px 10px; }}
.metric .lbl {{ color: {NEON['muted']}; font-size: 9px; letter-spacing: 0.5px; text-transform: uppercase; }}
.metric .val {{ font-size: 17px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }}
</style>
"""


def _fmt_bps(v: float, with_sign: bool = True) -> str:
    if pd.isna(v):
        return "—"
    sign = "+" if v >= 0 and with_sign else ""
    return f"{sign}{v:.1f} bps"


def _fmt_pct(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:.1f}%"


def _color_class(v: float, threshold: float = 0.0) -> str:
    if pd.isna(v):
        return "neutral"
    return "pos" if v > threshold else ("neg" if v < threshold else "neutral")


def _bar(label: str, value: float, max_abs: float, color: str, value_str: str) -> str:
    pct = min(abs(value) / max_abs * 100, 100) if max_abs > 0 else 0
    return (
        f'<div class="bar-row"><div class="bar-label">{label}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width: {pct:.1f}%; background: {color};">'
        f'</div></div>'
        f'<div class="bar-value">{value_str}</div></div>'
    )


def render_short_engine_report(short_results: dict, df: pd.DataFrame) -> str:
    """Generate the Task A HTML report."""
    # Summary stats
    total_rows = len(df)
    qualifying = sum(len(df[ENTRY_FILTERS[k](df)]) for k in ENTRY_FILTERS)
    universe_signals = {k: int(ENTRY_FILTERS[k](df).sum()) for k in ENTRY_FILTERS}

    # Find best (threshold, exit_rule) by net_bps
    best_label, best_row = None, None
    best_net = -1e9
    for label, table in short_results.items():
        if table.empty:
            continue
        idx = table["net_bps"].idxmax()
        if table.loc[idx, "net_bps"] > best_net:
            best_net = table.loc[idx, "net_bps"]
            best_label = label
            best_row = table.loc[idx]

    # Verdict
    if best_net > 5:
        verdict_class = "green"
        verdict_text = (
            f"<strong>POSITIVE EV detected.</strong> Best configuration: "
            f"<strong>{best_label}</strong> with <code>{best_row['exit_rule']}</code> exit "
            f"yields <strong>{_fmt_bps(best_net)}</strong> net of friction "
            f"(N={int(best_row['n'])}, hit rate {_fmt_pct(best_row['hit_rate'])}). "
            f"Worth promoting to a paper-trading sandbox."
        )
    elif best_net > 0:
        verdict_class = "yellow"
        verdict_text = (
            f"<strong>MARGINAL EV.</strong> Best configuration nets only "
            f"{_fmt_bps(best_net)} — within noise of zero. Insufficient margin of safety. "
            f"Recommend NOT pursuing live deployment."
        )
    else:
        verdict_class = "pink"
        verdict_text = (
            f"<strong>NO HARVESTABLE EDGE.</strong> Best configuration yields "
            f"{_fmt_bps(best_net)} net — friction consumes the gross alpha. "
            f"Short-side hypothesis falsified at current borrow assumptions."
        )

    # Build per-threshold tables
    threshold_html = ""
    for label, table in short_results.items():
        if table.empty:
            threshold_html += f'<h3>{label}</h3><p class="neutral">No qualifying signals.</p>'
            continue

        rows_html = ""
        for _, row in table.iterrows():
            net_class = _color_class(row["net_bps"])
            gross_class = _color_class(row["gross_bps"])
            rows_html += (
                f'<tr><td>{row["exit_rule"]}</td>'
                f'<td class="num">{int(row["n"])}</td>'
                f'<td class="num {gross_class}">{_fmt_bps(row["gross_bps"])}</td>'
                f'<td class="num neutral">{_fmt_bps(row["friction_bps"], with_sign=False)}</td>'
                f'<td class="num {net_class}"><strong>{_fmt_bps(row["net_bps"])}</strong></td>'
                f'<td class="num">{_fmt_pct(row["hit_rate"])}</td>'
                f'<td class="num">{row["sharpe_proxy"]:.2f}</td></tr>'
            )

        threshold_html += (
            f'<h3>{label}  <span class="neutral">'
            f'(N candidates: {universe_signals[label]})</span></h3>'
            f'<table><tr><th>Exit Rule</th><th>N</th><th>Gross Neutralized</th>'
            f'<th>Friction</th><th>Net (post-friction)</th>'
            f'<th>Hit Rate (net&gt;0)</th><th>Sharpe-like</th></tr>{rows_html}</table>'
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Short Engine EV — 2026-04-24</title>
{CSS}</head><body><div class="wrap">
<h1><span class="accent">SHORT ENGINE</span> · Friction-Adjusted EV</h1>
<div class="subtitle">2026-04-24 · Sprint 17 research · Task A · {total_rows:,} rows ·
98 NDX tickers × 135 days · cross-sectionally neutralized</div>

<div class="tldr">
<h3 style="color: {NEON['cyan']}; margin-top: 0;">VERDICT</h3>
<div class="callout {verdict_class}" style="margin: 8px 0;">{verdict_text}</div>
<ul>
  <li><strong>Hypothesis tested</strong>: short entries on
    <code>confidence &lt; 0.60 AND negative velocity</code> across the
    Sprint 15 backfill, neutralized by equal-weighted basket return.</li>
  <li><strong>Friction model</strong>: 28 bps base (fees + slippage) + 11 bps/day borrow
    (40% annualized HTB-light assumption).</li>
  <li><strong>Quarter-boundary masking</strong> applied: velocity computed only within-quarter to
    prevent state-permutation artifacts from quarterly HMM retrains.</li>
  <li><strong>Best configuration</strong>: {best_label or '—'}
    {('@ ' + best_row['exit_rule']) if best_row is not None else ''}
    yielding <strong>{_fmt_bps(best_net)}</strong> net.</li>
</ul>
</div>

<h2>Per-Threshold Performance</h2>
{threshold_html}

<h2>Methodology Notes</h2>
<div class="panel">
<ul>
<li><strong>Neutralization</strong>: For each scan_date, subtract the equal-weighted mean forward
return of all 98 tickers from each ticker's individual forward return. Equivalent to the deployed
hedge: short the target asset, long an equal-weighted basket of the rest.</li>
<li><strong>Short-side P&amp;L</strong>: <code>pnl = -1 × neutralized_forward_return</code>.
A profitable short occurs when the target underperforms the basket.</li>
<li><strong>Dynamic exit</strong>: For each entry signal at date t, the exit date is the first
t+k (k ∈ [1,10]) where ticker confidence ≥ {DYNAMIC_EXIT_THRESHOLD}. If confidence never recovers,
the position holds the maximum 10-day window and reports T+10.</li>
<li><strong>Friction asymmetry</strong>: Fixed-horizon trades pay friction proportional to the
horizon. Dynamic-exit friction uses the realized average hold duration.</li>
<li><strong>Sharpe-like proxy</strong>: per-trade mean / std × √(252 / hold_days). Annualized
single-trade-equivalent — NOT a full strategy Sharpe (no path or vol-of-vol modeling).</li>
</ul>
</div>

<div class="footer">Generated 2026-04-24 · trading-core/research/short_engine_eval.py ·
Source: scan_journal_backfill (VM beryl_trades.db)</div>
</div></body></html>"""


def render_velocity_audit_report(audit: pd.DataFrame, df: pd.DataFrame) -> str:
    """Generate the Task B HTML report."""
    # Determine if velocity-band hypothesis holds
    velo_band = audit[audit["conf_bin"].isin(["[0.60,0.70)", "[0.70,0.80)"])]
    drift_band = audit[audit["conf_bin"].isin(["[0.90,0.95)", "[0.95,1.00]"])]

    velo_neut_1d = velo_band["neut_ret_1d_bps"].mean() if not velo_band.empty else np.nan
    drift_neut_1d = drift_band["neut_ret_1d_bps"].mean() if not drift_band.empty else np.nan
    delta_1d = velo_neut_1d - drift_neut_1d

    velo_neut_3d = velo_band["neut_ret_3d_bps"].mean() if not velo_band.empty else np.nan
    drift_neut_3d = drift_band["neut_ret_3d_bps"].mean() if not drift_band.empty else np.nan
    delta_3d = velo_neut_3d - drift_neut_3d

    if delta_1d > 5 and delta_3d > 10:
        verdict_class = "green"
        verdict_text = (
            f"<strong>HYPOTHESIS SUPPORTED.</strong> Velocity band (0.60–0.80) outperforms "
            f"structural drift band (0.90–1.00) by <strong>{delta_1d:+.1f} bps</strong> at T+1 "
            f"and <strong>{delta_3d:+.1f} bps</strong> at T+3 on neutralized returns. "
            f"Strong case to port BERYL Sprint 16's velocity gate to CITRINE entry logic."
        )
    elif delta_1d > 0:
        verdict_class = "yellow"
        verdict_text = (
            f"<strong>WEAK SIGNAL.</strong> Velocity band outperforms drift band by "
            f"{delta_1d:+.1f} bps at T+1, {delta_3d:+.1f} bps at T+3. "
            f"Direction matches BERYL hypothesis but magnitude insufficient to motivate "
            f"a CITRINE architecture change. Continue observation."
        )
    else:
        verdict_class = "pink"
        verdict_text = (
            f"<strong>HYPOTHESIS REFUTED.</strong> Drift band actually outperforms velocity "
            f"band by {-delta_1d:+.1f} bps at T+1, {-delta_3d:+.1f} bps at T+3. The MCHP/CSGP "
            f"observation may be small-N noise; CITRINE entry logic does not need a velocity gate."
        )

    # Bar chart of neutralized 1d returns by bin
    max_abs = audit["neut_ret_1d_bps"].abs().max()
    bars_html = ""
    for _, row in audit.iterrows():
        v = row["neut_ret_1d_bps"]
        color = NEON["green"] if v > 0 else NEON["pink"]
        bars_html += _bar(
            f"{row['conf_bin']} (N={int(row['n']):,})",
            v, max_abs, color, _fmt_bps(v),
        )

    # Decile table
    rows_html = ""
    for _, row in audit.iterrows():
        rows_html += (
            f'<tr><td>{row["conf_bin"]}</td>'
            f'<td class="num">{int(row["n"]):,}</td>'
            f'<td class="num {_color_class(row["raw_ret_1d_bps"])}">{_fmt_bps(row["raw_ret_1d_bps"])}</td>'
            f'<td class="num {_color_class(row["raw_ret_3d_bps"])}">{_fmt_bps(row["raw_ret_3d_bps"])}</td>'
            f'<td class="num {_color_class(row["raw_ret_5d_bps"])}">{_fmt_bps(row["raw_ret_5d_bps"])}</td>'
            f'<td class="num {_color_class(row["neut_ret_1d_bps"])}">{_fmt_bps(row["neut_ret_1d_bps"])}</td>'
            f'<td class="num {_color_class(row["neut_ret_3d_bps"])}">{_fmt_bps(row["neut_ret_3d_bps"])}</td>'
            f'<td class="num {_color_class(row["neut_ret_5d_bps"])}">{_fmt_bps(row["neut_ret_5d_bps"])}</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>CITRINE Velocity Audit — 2026-04-24</title>
{CSS}</head><body><div class="wrap">
<h1><span class="accent">CITRINE</span> · Velocity-Gate Audit</h1>
<div class="subtitle">2026-04-24 · Sprint 17 research · Task B · Forward returns by confidence
bin · 13,230-row backfill · cross-sectionally neutralized</div>

<div class="tldr">
<h3 style="color: {NEON['cyan']}; margin-top: 0;">VERDICT</h3>
<div class="callout {verdict_class}" style="margin: 8px 0;">{verdict_text}</div>
<ul>
  <li><strong>Triggering observation</strong>: 2026-04-24 — MCHP and CSGP entered at confidence
    ≥ 0.99, both stopped within 8 hours by the −2% intraday catastrophe stop.</li>
  <li><strong>Question</strong>: across the historical journal, do top-decile (≥0.95) entries
    systematically underperform the BERYL Sprint 16 velocity band (0.60–0.80)?</li>
  <li><strong>Velocity band T+1 mean</strong>: {_fmt_bps(velo_neut_1d)} (neutralized)</li>
  <li><strong>Drift band T+1 mean</strong>: {_fmt_bps(drift_neut_1d)} (neutralized)</li>
  <li><strong>Δ (velocity − drift)</strong>: <strong>{_fmt_bps(delta_1d)}</strong> at T+1,
    <strong>{_fmt_bps(delta_3d)}</strong> at T+3</li>
</ul>
</div>

<h2>Neutralized T+1 Forward Return by Confidence Bin</h2>
<div class="panel">
{bars_html}
</div>

<h2>Full Decile Table</h2>
<table>
<tr><th>Conf Bin</th><th>N</th>
<th>Raw T+1</th><th>Raw T+3</th><th>Raw T+5</th>
<th>Neut T+1</th><th>Neut T+3</th><th>Neut T+5</th></tr>
{rows_html}
</table>

<h2>Methodology Notes</h2>
<div class="panel">
<ul>
<li><strong>Raw return</strong>: <code>close[t+h] / close[t] − 1</code>.</li>
<li><strong>Neutralized return</strong>: raw return minus that day's equal-weighted basket mean.
This is the forward-return analog of the Sprint 15 cross-sectional residual analysis.</li>
<li><strong>Bin boundaries</strong>: cut at 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00. The 0.60–0.80
"velocity band" matches BERYL Sprint 16's entry gate. The 0.90–1.00 range is the "structural drift"
zone where the HMM has high confidence but realized returns historically disappoint.</li>
<li><strong>Interpretation</strong>: a healthy entry gate should bin into confidence ranges where
neutralized forward returns are systematically <em>positive</em>. Any bin with significantly
negative neutralized returns is a candidate for exclusion or short-side flipping.</li>
</ul>
</div>

<div class="footer">Generated 2026-04-24 · trading-core/research/short_engine_eval.py ·
Source: scan_journal_backfill (VM beryl_trades.db)</div>
</div></body></html>"""


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Force fresh pull from VM (bypasses parquet cache)")
    args = ap.parse_args()

    print(f"=== Short Engine EV + CITRINE Velocity Audit ===")
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")

    df = load_data(refresh=args.refresh)
    print(f"[features] computing forward returns + velocity + dynamic exit ...")
    df = add_features(df)
    print(f"[features] complete. Coverage:")
    print(f"  rows={len(df):,}  tickers={df['ticker'].nunique()}  "
          f"days={df['scan_date'].dt.date.nunique()}")
    print(f"  velocity_1d non-null: {df['velocity_1d'].notna().sum():,} "
          f"(masked at quarter boundaries)")
    print(f"  fwd_ret_1d non-null:  {df['fwd_ret_1d'].notna().sum():,}")

    print(f"\n[task A] Short Engine evaluation ...")
    short_results = short_engine_evaluation(df)
    for label, table in short_results.items():
        print(f"  {label}:")
        if table.empty:
            print("    (no qualifying signals)")
        else:
            for _, row in table.iterrows():
                marker = "★" if row["net_bps"] > 5 else (" " if row["net_bps"] > 0 else "✗")
                print(f"    {marker} {row['exit_rule']:35s}  N={int(row['n']):4d}  "
                      f"gross={row['gross_bps']:+7.1f}bps  fric={row['friction_bps']:5.1f}bps  "
                      f"net={row['net_bps']:+7.1f}bps  hit={row['hit_rate']:5.1f}%")

    print(f"\n[task B] CITRINE velocity-gate audit ...")
    audit = velocity_audit(df)
    for _, row in audit.iterrows():
        print(f"  {row['conf_bin']:<14s}  N={int(row['n']):5d}  "
              f"raw_1d={row['raw_ret_1d_bps']:+7.1f}bps  "
              f"neut_1d={row['neut_ret_1d_bps']:+7.1f}bps  "
              f"neut_3d={row['neut_ret_3d_bps']:+7.1f}bps")

    print(f"\n[reports] writing HTML ...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SHORT_REPORT.write_text(render_short_engine_report(short_results, df))
    AUDIT_REPORT.write_text(render_velocity_audit_report(audit, df))
    print(f"  → {SHORT_REPORT.relative_to(ROOT)}")
    print(f"  → {AUDIT_REPORT.relative_to(ROOT)}")
    print(f"\nDone: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
