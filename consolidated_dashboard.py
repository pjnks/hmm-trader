"""
consolidated_dashboard.py
─────────────────────────
Portfolio Overview — Combined view of all 5 trading sub-projects.

Retro-futuristic bento-box dashboard with industrial/utilitarian aesthetic.
Deep obsidian background, neon-cyan accents, grainy glassmorphism cards,
monospaced tabular typography, asymmetrical grid, staggered entrance
animations, spring-physics hover states, Swiss Design muted palette.

Run:
  python consolidated_dashboard.py                     # local at :8090
  python consolidated_dashboard.py --host 0.0.0.0      # VM-accessible

Port 8090 to avoid collision with existing dashboards (8050/8060/8070/8080).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

PROJECT_DBS = {
    "AGATE":   ROOT / "paper_trades.db",
    "BERYL":   ROOT / "beryl_trades.db",
    "CITRINE": ROOT / "citrine_trades.db",
}

DIAMOND_DB_CANDIDATES = [
    ROOT.parent / "kalshi-diamond" / "diamond_trades.db",
    Path("/home/ubuntu/kalshi-diamond/diamond_trades.db"),
    ROOT.parent / "diamond" / "diamond_trades.db",
    # Legacy fallbacks
    ROOT.parent / "kalshi-diamond" / "diamond.db",
    Path("/home/ubuntu/kalshi-diamond/diamond.db"),
    ROOT / "diamond.db",
]

EMERALD_DB_CANDIDATES = [
    ROOT.parent / "emerald" / "emerald.db",
    Path("/home/ubuntu/emerald/emerald.db"),
    Path("/home/ubuntu/HMM-Trader/emerald.db"),  # synced copy
    ROOT / "emerald.db",
]

STATUS_FILES = {
    "AGATE": ROOT / "agate_status.json",
    "BERYL": ROOT / "beryl_status.json",
}

EXPECTED_SHARPE = {
    "AGATE":   0.837,
    "BERYL":   0.825,
    "CITRINE": 1.665,
    "DIAMOND": 0.500,
    "EMERALD": 1.000,
}

# ── Swiss Design Palette ─────────────────────────────────────────────────────
# Obsidian base with neon-cyan accent.  Muted secondaries.
BG          = "#08090c"
BG_RAISED   = "#0e1016"
PANEL       = "rgba(14, 18, 26, 0.72)"
PANEL_SOLID = "#0e121a"
BORDER      = "rgba(0, 232, 255, 0.08)"
BORDER_HOVER = "rgba(0, 232, 255, 0.25)"

CYAN        = "#00e8ff"
CYAN_DIM    = "rgba(0, 232, 255, 0.4)"
CYAN_GLOW   = "rgba(0, 232, 255, 0.12)"

TEXT        = "#d0d4e0"
TEXT_DIM    = "#5a6078"
TEXT_MUTED  = "#3a3f52"

GREEN       = "#00d68f"
RED         = "#ff3d71"
YELLOW      = "#ffaa00"
BLUE        = "#598bff"
PURPLE      = "#c471f5"
ORANGE      = "#ff8a50"

EMERALD_GREEN = "#50fa7b"  # Bright emerald — distinct from AGATE's #00d68f

PROJECT_COLORS = {
    "AGATE":   GREEN,
    "BERYL":   BLUE,
    "CITRINE": ORANGE,
    "DIAMOND": PURPLE,
    "EMERALD": EMERALD_GREEN,
}

PROJECT_ICONS = {
    "AGATE":   "◆",
    "BERYL":   "◈",
    "CITRINE": "◇",
    "DIAMOND": "◊",
    "EMERALD": "◈",
}


# ── Custom CSS ───────────────────────────────────────────────────────────────
GOOGLE_FONT_URL = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap"

CUSTOM_CSS = """
* { box-sizing: border-box; }

body {
    background: """ + BG + """;
    margin: 0;
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
}

/* Grainy texture overlay — z-index:-1 so it never covers content */
body::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: -1;
}

/* Ensure React mount point is above overlays */
#react-entry-point {
    position: relative;
    z-index: 1;
}

/* Bento card glassmorphism */
.bento-card {
    background: """ + PANEL + """;
    backdrop-filter: blur(12px) saturate(140%);
    -webkit-backdrop-filter: blur(12px) saturate(140%);
    border: 1px solid """ + BORDER + """;
    border-radius: 6px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1),
                border-color 0.3s ease,
                box-shadow 0.3s ease;
    will-change: transform;
}

.bento-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, """ + CYAN_DIM + """, transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.bento-card:hover {
    transform: translateY(-2px) scale(1.005);
    border-color: """ + BORDER_HOVER + """;
    box-shadow: 0 8px 32px rgba(0, 232, 255, 0.06),
                0 0 0 1px """ + BORDER_HOVER + """,
                inset 0 1px 0 rgba(0, 232, 255, 0.05);
}

.bento-card:hover::before {
    opacity: 1;
}

/* Staggered entrance animations */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Note: animation fill-mode 'both' means elements start invisible (opacity:0)
   until the animation fires.  Use 'forwards' only so elements are visible by default
   and only animate when CSS class is applied fresh (e.g. page load). */
.anim-1 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.05s forwards; }
.anim-2 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.10s forwards; }
.anim-3 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.15s forwards; }
.anim-4 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.20s forwards; }
.anim-5 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.25s forwards; }
.anim-6 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.30s forwards; }
.anim-7 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.35s forwards; }
.anim-8 { animation: slideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.40s forwards; }

/* Metric value styling */
.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

/* Project filter toggles */
.project-toggle {
    display: flex;
    align-items: center;
    cursor: pointer;
    padding: 4px 6px;
    margin: 0 -6px;
    border-radius: 4px;
    border: 1px solid transparent;
    background: transparent;
    width: calc(100% + 12px);
    transition: background 0.2s ease, border-color 0.2s ease, opacity 0.3s ease;
    font-family: inherit;
    text-align: left;
}

.project-toggle:hover {
    background: rgba(0, 232, 255, 0.04);
    border-color: rgba(0, 232, 255, 0.08);
}

.project-toggle.dimmed {
    opacity: 0.25;
}

.metric-label {
    font-size: 0.6rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: """ + TEXT_DIM + """;
    margin-bottom: 6px;
}

.metric-sub {
    font-size: 0.65rem;
    color: """ + TEXT_MUTED + """;
    margin-top: 4px;
}

/* Project name with glow */
.project-name {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

/* Health badge */
.health-badge {
    display: inline-block;
    font-size: 0.55rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
    border: 1px solid;
}

/* Scan line effect on header */
@keyframes scanline {
    0% { background-position: 0 0; }
    100% { background-position: 0 100%; }
}

.header-scanline {
    position: relative;
}

.header-scanline::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 232, 255, 0.015) 2px,
        rgba(0, 232, 255, 0.015) 4px
    );
    pointer-events: none;
    animation: scanline 8s linear infinite;
}

/* Subtle grid pattern on main container */
.grid-bg {
    background-image:
        linear-gradient(""" + BORDER + """ 1px, transparent 1px),
        linear-gradient(90deg, """ + BORDER + """ 1px, transparent 1px);
    background-size: 60px 60px;
    background-position: center;
}

/* Neon glow pulse for active indicators */
@keyframes neonPulse {
    0%, 100% { box-shadow: 0 0 4px currentColor; }
    50% { box-shadow: 0 0 12px currentColor, 0 0 24px currentColor; }
}

.neon-pulse {
    animation: neonPulse 3s ease-in-out infinite;
}

/* Chart container 3D depth */
.chart-container {
    position: relative;
    border-radius: 6px;
    overflow: hidden;
}

.chart-container::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
        180deg,
        rgba(0, 232, 255, 0.02) 0%,
        transparent 30%,
        transparent 70%,
        rgba(0, 232, 255, 0.01) 100%
    );
    pointer-events: none;
    z-index: 1;
}

/* Thin separator lines */
.separator {
    height: 1px;
    background: linear-gradient(90deg, transparent, """ + BORDER_HOVER + """, transparent);
    margin: 4px 0 12px;
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* Kill-switch indicator bar */
.kill-bar {
    height: 3px;
    border-radius: 1px;
    margin-top: 8px;
    transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
}
"""


# ── Data Loaders ──────────────────────────────────────────────────────────────

def _load_trades(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp ASC", conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "pnl" in df.columns:
            df = df.dropna(subset=["pnl"])
        return df
    except Exception:
        return pd.DataFrame()


def _load_citrine_snapshots() -> pd.DataFrame:
    db_path = PROJECT_DBS["CITRINE"]
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT timestamp, total_equity, cash, invested, num_positions, "
                "bull_count, bear_count, chop_count, cash_pct "
                "FROM portfolio_snapshots ORDER BY timestamp ASC", conn,
            )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _load_diamond_trades() -> pd.DataFrame:
    """Load DIAMOND paper_trades, normalizing schema to match AGATE/BERYL/CITRINE.

    DIAMOND uses pnl_cents (integer cents) and settled_at (Unix epoch float).
    We convert to pnl (dollars) and timestamp (datetime) for chart compatibility.
    """
    for candidate in DIAMOND_DB_CANDIDATES:
        if candidate.exists():
            try:
                with sqlite3.connect(str(candidate)) as conn:
                    df = pd.read_sql_query(
                        "SELECT ticker, side, pnl_cents, settled_at "
                        "FROM paper_trades "
                        "WHERE pnl_cents IS NOT NULL AND settled_at IS NOT NULL "
                        "ORDER BY settled_at ASC", conn)
                if df.empty:
                    return pd.DataFrame()
                df["pnl"] = df["pnl_cents"] / 100.0
                df["timestamp"] = pd.to_datetime(df["settled_at"], unit="s", utc=True)
                df = df.drop(columns=["pnl_cents", "settled_at"])
                return df
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def _load_emerald_trades() -> pd.DataFrame:
    """Load EMERALD predictions, normalizing schema to match other projects.

    EMERALD uses pnl (dollars, already correct) and resolved_at (Unix epoch float)
    in its predictions table.  We filter to resolved predictions only.
    """
    for candidate in EMERALD_DB_CANDIDATES:
        if candidate.exists():
            try:
                with sqlite3.connect(str(candidate)) as conn:
                    df = pd.read_sql_query(
                        "SELECT market_type, side, pnl, resolved_at "
                        "FROM predictions "
                        "WHERE pnl IS NOT NULL AND resolved_at IS NOT NULL "
                        "ORDER BY resolved_at ASC", conn)
                if df.empty:
                    return pd.DataFrame()
                df["timestamp"] = pd.to_datetime(df["resolved_at"], unit="s", utc=True)
                df = df.drop(columns=["resolved_at"])
                # Rename market_type→ticker for consistency with other loaders
                df = df.rename(columns={"market_type": "ticker"})
                return df
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def _load_emerald_bankroll() -> pd.DataFrame:
    """Load EMERALD bankroll history for equity curve display."""
    for candidate in EMERALD_DB_CANDIDATES:
        if candidate.exists():
            try:
                with sqlite3.connect(str(candidate)) as conn:
                    df = pd.read_sql_query(
                        "SELECT balance, change, reason, ts "
                        "FROM bankroll ORDER BY ts ASC", conn)
                if df.empty:
                    return pd.DataFrame()
                df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
                df = df.drop(columns=["ts"])
                return df
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def _diamond_summary_from_trades(df: pd.DataFrame) -> dict:
    """Build DIAMOND summary dict from normalized trades DataFrame."""
    if df.empty or "pnl" not in df.columns:
        return {"status": "no_db", "trades": 0, "pnl": 0.0, "win_rate": 0.0}
    n_trades = len(df)
    total_pnl = float(df["pnl"].sum())
    wins = int((df["pnl"] > 0).sum())
    return {
        "status": "active",
        "trades": n_trades,
        "pnl": total_pnl,
        "win_rate": float(wins / max(n_trades, 1)),
    }


def _load_status(filename: str) -> dict:
    path = ROOT / filename
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _rolling_sharpe(pnls: np.ndarray, window: int = 20) -> float:
    if len(pnls) < window:
        return 0.0
    recent = pnls[-window:]
    std = np.std(recent)
    if std < 1e-9:
        return 0.0
    return float(np.mean(recent) / std)


def _compute_project_metrics(project: str, trades_df: pd.DataFrame) -> dict:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return {
            "total_pnl": 0.0, "total_trades": 0, "win_rate": 0.0,
            "sharpe_20": 0.0, "last_trade": "—",
            "degradation": "unknown", "degradation_sigma": 0.0,
        }
    pnls = trades_df["pnl"].values.astype(float)
    total_pnl = float(np.sum(pnls))
    total_trades = len(pnls)

    # For EMERALD, exclude pushes (pnl=0) from win rate calc
    non_zero = pnls[pnls != 0]
    if project == "EMERALD" and len(non_zero) > 0:
        wins = int(np.sum(non_zero > 0))
        win_rate = wins / len(non_zero)
    else:
        wins = int(np.sum(pnls > 0))
        win_rate = wins / max(total_trades, 1)
    sharpe_20 = _rolling_sharpe(pnls, window=20)

    last_ts = trades_df["timestamp"].iloc[-1] if not trades_df.empty else "—"
    if isinstance(last_ts, pd.Timestamp):
        last_trade = last_ts.strftime("%m-%d %H:%M")
    else:
        last_trade = str(last_ts)[:16]

    expected = EXPECTED_SHARPE.get(project, 1.0)
    degradation = "unknown"
    degradation_sigma = 0.0

    # EMERALD is research-only — skip degradation model
    if project == "EMERALD":
        degradation = "research"
    elif total_trades >= 10:
        live_expected = expected * 0.15
        if live_expected > 0:
            degradation_sigma = (sharpe_20 - live_expected) / max(abs(live_expected) * 0.5, 0.1)
            if degradation_sigma < -2.0:
                degradation = "critical"
            elif degradation_sigma < -1.0:
                degradation = "warning"
            else:
                degradation = "healthy"
    elif total_trades >= 5:
        degradation = "insufficient_data"
    else:
        degradation = "too_early"

    return {
        "total_pnl": total_pnl, "total_trades": total_trades, "win_rate": win_rate,
        "sharpe_20": sharpe_20, "last_trade": last_trade,
        "degradation": degradation, "degradation_sigma": degradation_sigma,
    }


def _apply_filter(fig: go.Figure, active_filter: list[str]) -> go.Figure:
    """Set trace visibility based on active project filter."""
    for trace in fig.data:
        name = trace.name or ""
        legendgroup = getattr(trace, "legendgroup", "") or ""
        if name == "COMBINED":
            trace.visible = True
        else:
            trace.visible = any(
                name == p or name.startswith(p) or legendgroup == p
                for p in active_filter
            )
    return fig


# ── Chart Builders ────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=TEXT_DIM, size=10),
    margin=dict(l=48, r=16, t=36, b=32),
    uirevision="default",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=9, color=TEXT_DIM),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        gridcolor="rgba(0, 232, 255, 0.04)", gridwidth=1,
        zerolinecolor="rgba(0, 232, 255, 0.06)",
        tickfont=dict(size=9),
    ),
    yaxis=dict(
        gridcolor="rgba(0, 232, 255, 0.04)", gridwidth=1,
        zerolinecolor="rgba(0, 232, 255, 0.06)",
        tickfont=dict(size=9),
    ),
)


STARTING_CAPITAL = {
    "AGATE":   10_000.0,
    "BERYL":   10_000.0,
    "CITRINE": 25_000.0,
    "DIAMOND":  1_000.0,
    "EMERALD":  1_000.0,
}


def _build_equity_chart(all_trades: dict[str, pd.DataFrame],
                        snapshots: pd.DataFrame,
                        emerald_bankroll: pd.DataFrame | None = None) -> go.Figure:
    """Indexed equity chart — all projects normalized to base 100."""
    fig = go.Figure()
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="INDEXED EQUITY  (BASE = 100)", font=dict(size=10, color=TEXT_DIM)),
        height=280,
        yaxis_title=dict(text="Index", font=dict(size=9)),
    )

    # Reference line at 100
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(0, 232, 255, 0.15)", line_width=1)

    indexed_series = {}

    # CITRINE from snapshots (absolute equity → indexed)
    if not snapshots.empty and "total_equity" in snapshots.columns:
        capital = STARTING_CAPITAL["CITRINE"]
        y = (snapshots["total_equity"] / capital) * 100
        fig.add_trace(go.Scatter(
            x=snapshots["timestamp"], y=y,
            name="CITRINE", line=dict(color=ORANGE, width=1.5),
        ))
        daily = pd.Series(y.values, index=snapshots["timestamp"]).resample("D").last().dropna()
        indexed_series["CITRINE"] = daily

    # EMERALD from bankroll (absolute balance → indexed)
    if emerald_bankroll is not None and not emerald_bankroll.empty and "balance" in emerald_bankroll.columns:
        capital = STARTING_CAPITAL["EMERALD"]
        y = (emerald_bankroll["balance"] / capital) * 100
        fig.add_trace(go.Scatter(
            x=emerald_bankroll["timestamp"], y=y,
            name="EMERALD", line=dict(color=EMERALD_GREEN, width=1.5),
        ))
        daily = pd.Series(y.values, index=emerald_bankroll["timestamp"]).resample("D").last().dropna()
        indexed_series["EMERALD"] = daily

    # AGATE / BERYL / DIAMOND from cumulative P&L → indexed
    for project in ["AGATE", "BERYL", "DIAMOND"]:
        df = all_trades.get(project, pd.DataFrame())
        if df.empty or "pnl" not in df.columns:
            continue
        df = df.sort_values("timestamp")
        capital = STARTING_CAPITAL.get(project, 10_000.0)
        cum_pnl = df["pnl"].cumsum()
        y = ((capital + cum_pnl) / capital) * 100
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=y,
            name=project, line=dict(color=PROJECT_COLORS[project], width=1.5),
        ))
        daily = pd.Series(y.values, index=df["timestamp"]).resample("D").last().dropna()
        indexed_series[project] = daily

    # Combined indexed (equal-weighted average of all projects)
    if indexed_series:
        combined = pd.concat(indexed_series.values(), axis=1).ffill().dropna(how="all")
        if not combined.empty:
            avg = combined.mean(axis=1)
            fig.add_trace(go.Scatter(
                x=avg.index, y=avg.values,
                name="COMBINED", line=dict(color=CYAN, width=2, dash="dot"),
            ))

    return fig


def _build_degradation_chart(all_trades: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="MODEL HEALTH  —  ROLLING 10-TRADE SHARPE", font=dict(size=10, color=TEXT_DIM)),
        height=220,
        yaxis_title=dict(text="Sharpe", font=dict(size=9)),
    )

    for project, df in all_trades.items():
        if df.empty or "pnl" not in df.columns or len(df) < 2:
            continue
        pnls = df.sort_values("timestamp")["pnl"].values.astype(float)
        window = min(10, len(pnls))
        rolling = []
        for i in range(window, len(pnls) + 1):
            chunk = pnls[i - window:i]
            std = np.std(chunk)
            rolling.append(float(np.mean(chunk) / std) if std > 1e-9 else 0.0)

        if rolling:
            label = f"{project}" if window == 10 else f"{project} ({window}t)"
            fig.add_trace(go.Scatter(
                x=list(range(window, len(pnls) + 1)), y=rolling,
                name=label, line=dict(color=PROJECT_COLORS[project], width=1.5),
            ))

    fig.add_hline(y=0, line=dict(color="rgba(255, 61, 113, 0.3)", width=1, dash="dash"))

    return fig


def _build_pnl_distribution(all_trades: dict[str, pd.DataFrame]) -> go.Figure:
    """Histogram of P&L across all projects."""
    fig = go.Figure()
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="P&L DISTRIBUTION", font=dict(size=10, color=TEXT_DIM)),
        height=200,
        barmode="overlay",
        yaxis_title=dict(text="Count", font=dict(size=9)),
        xaxis_title=dict(text="P&L ($)", font=dict(size=9)),
    )

    for project, df in all_trades.items():
        if df.empty or "pnl" not in df.columns:
            continue
        fig.add_trace(go.Histogram(
            x=df["pnl"].values,
            name=project,
            marker_color=PROJECT_COLORS[project],
            opacity=0.6,
            nbinsx=20,
        ))

    return fig


def _build_metrics_over_time(all_trades: dict[str, pd.DataFrame]) -> go.Figure:
    """2x2 subplot: cumulative P&L, cumulative trades, rolling win rate, rolling Sharpe.

    Each metric is computed per-trade and plotted over calendar time.
    Uses adaptive window: min(20, n_trades) for projects with few trades.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["CUMULATIVE P&L", "CUMULATIVE TRADES",
                        "ROLLING WIN RATE", "ROLLING SHARPE"],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=9, color=TEXT_DIM, family="JetBrains Mono, monospace")

    target_window = 20

    for project, df in all_trades.items():
        if df.empty or "pnl" not in df.columns:
            continue
        df = df.sort_values("timestamp").copy()
        pnls = df["pnl"].values.astype(float)
        ts = df["timestamp"].values
        color = PROJECT_COLORS[project]
        n = len(pnls)
        w = min(target_window, n)

        # 1) Cumulative P&L (row=1, col=1)
        cum_pnl = np.cumsum(pnls)
        fig.add_trace(go.Scatter(
            x=ts, y=cum_pnl,
            name=project, legendgroup=project,
            line=dict(color=color, width=1.5),
            showlegend=True,
        ), row=1, col=1)

        # 2) Cumulative trades (row=1, col=2)
        cum_trades = np.arange(1, n + 1)
        fig.add_trace(go.Scatter(
            x=ts, y=cum_trades,
            name=project, legendgroup=project,
            line=dict(color=color, width=1.5),
            showlegend=False,
        ), row=1, col=2)

        if n < 2:
            continue

        # 3) Rolling win rate (row=2, col=1)
        wins = (pnls > 0).astype(float)
        rolling_wr = []
        rolling_wr_ts = []
        for i in range(w, n + 1):
            rolling_wr.append(float(np.mean(wins[i - w:i])))
            rolling_wr_ts.append(ts[i - 1])

        fig.add_trace(go.Scatter(
            x=rolling_wr_ts, y=rolling_wr,
            name=project, legendgroup=project,
            line=dict(color=color, width=1.5),
            showlegend=False,
        ), row=2, col=1)

        # 4) Rolling Sharpe (row=2, col=2)
        rolling_sh = []
        rolling_sh_ts = []
        for i in range(w, n + 1):
            chunk = pnls[i - w:i]
            std = np.std(chunk)
            rolling_sh.append(float(np.mean(chunk) / std) if std > 1e-9 else 0.0)
            rolling_sh_ts.append(ts[i - 1])

        fig.add_trace(go.Scatter(
            x=rolling_sh_ts, y=rolling_sh,
            name=project, legendgroup=project,
            line=dict(color=color, width=1.5),
            showlegend=False,
        ), row=2, col=2)

    # Reference lines
    fig.add_hline(y=0.5, row=2, col=1,
                  line=dict(color="rgba(255, 170, 0, 0.2)", width=1, dash="dash"))
    fig.add_hline(y=0, row=2, col=2,
                  line=dict(color="rgba(255, 61, 113, 0.3)", width=1, dash="dash"))
    fig.add_hline(y=0, row=1, col=1,
                  line=dict(color="rgba(255, 61, 113, 0.3)", width=1, dash="dash"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color=TEXT_DIM, size=10),
        margin=dict(l=48, r=16, t=36, b=32),
        height=380,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1,
            font=dict(size=9, color=TEXT_DIM),
            bgcolor="rgba(0,0,0,0)",
            tracegroupgap=0,
        ),
        showlegend=True,
    )

    # Style all axes
    axis_style = dict(
        gridcolor="rgba(0, 232, 255, 0.04)", gridwidth=1,
        zerolinecolor="rgba(0, 232, 255, 0.06)",
        tickfont=dict(size=8),
    )
    for ax_name in ["xaxis", "xaxis2", "xaxis3", "xaxis4",
                     "yaxis", "yaxis2", "yaxis3", "yaxis4"]:
        fig.update_layout(**{ax_name: axis_style})

    # Y-axis labels
    fig.update_yaxes(title=dict(text="$", font=dict(size=8)), row=1, col=1)
    fig.update_yaxes(title=dict(text="#", font=dict(size=8)), row=1, col=2)
    fig.update_yaxes(title=dict(text="%", font=dict(size=8)), row=2, col=1)
    fig.update_yaxes(title=dict(text="σ", font=dict(size=8)), row=2, col=2)

    # Format win rate as percentage
    fig.update_yaxes(tickformat=".0%", row=2, col=1)

    return fig


# ── Dashboard Components ──────────────────────────────────────────────────────

def _metric_cell(label: str, value: str, color: str = TEXT,
                 sub: str = "", anim: int = 1) -> html.Div:
    """Single metric in a bento cell."""
    children = [
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value", style={"color": color}),
    ]
    if sub:
        children.append(html.Div(sub, className="metric-sub"))
    return html.Div(children, className=f"bento-card anim-{anim}",
                    style={"height": "100%"})


def _project_panel(project: str, metrics: dict, status: dict,
                   anim: int = 1) -> html.Div:
    """Full project panel — bento card with name, regime, metrics."""
    color = PROJECT_COLORS.get(project, TEXT)
    icon = PROJECT_ICONS.get(project, "●")
    pnl = metrics["total_pnl"]
    pnl_color = GREEN if pnl >= 0 else RED

    # Degradation
    deg = metrics["degradation"]
    deg_map = {
        "healthy": (GREEN, "NOMINAL"),
        "warning": (YELLOW, "DRIFT"),
        "critical": (RED, "DEGRADED"),
        "insufficient_data": (TEXT_DIM, "COLLECTING"),
        "too_early": (TEXT_MUTED, "PENDING"),
        "research": (CYAN_DIM, "RESEARCH"),
        "unknown": (TEXT_MUTED, "—"),
    }
    deg_color, deg_label = deg_map.get(deg, (TEXT_MUTED, "?"))

    # Regime from status
    regime = status.get("regime", "—").upper()
    confidence = status.get("confidence", 0.0)
    signal = status.get("signal", "—")

    regime_colors = {"BULL": GREEN, "BEAR": RED, "CHOP": YELLOW}
    r_color = TEXT_DIM
    for key, c in regime_colors.items():
        if key in regime:
            r_color = c
            break

    signal_colors = {"BUY": GREEN, "SELL": RED, "HOLD": TEXT_DIM, "SHORT": PURPLE}
    s_color = signal_colors.get(signal, TEXT_DIM)

    # Kill-switch bar — green if healthy
    kill_bar_color = GREEN if deg in ("healthy", "too_early", "unknown", "insufficient_data", "research") else RED
    kill_bar_width = "100%" if deg != "critical" else "30%"

    return html.Div([
        # Header row
        html.Div([
            html.Span(f"{icon} ", style={"color": color, "fontSize": "0.9rem"}),
            html.Span(project, className="project-name", style={"color": color}),
            html.Span(deg_label, className="health-badge",
                      style={"color": deg_color, "borderColor": deg_color,
                             "marginLeft": "auto", "float": "right"}),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(className="separator"),

        # Regime line
        html.Div([
            html.Span(regime, style={
                "color": r_color, "fontWeight": "600", "fontSize": "0.85rem",
                "letterSpacing": "0.05em",
            }),
            html.Span(f"  {confidence:.0%}", style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
            html.Span(f"  {signal}", style={
                "color": s_color, "fontWeight": "500", "fontSize": "0.7rem",
                "marginLeft": "12px",
            }),
        ], style={"marginBottom": "12px"}) if regime != "—" else html.Div(),

        # Index 100 line
        html.Div([
            html.Span("INDEX ", style={"color": TEXT_MUTED, "fontSize": "0.6rem",
                                        "letterSpacing": "0.1em"}),
            html.Span(f"{metrics.get('index_100', 100.0):.1f}", style={
                "color": GREEN if metrics.get("index_100", 100) >= 100 else RED,
                "fontSize": "1.3rem", "fontWeight": "700",
                "fontVariantNumeric": "tabular-nums",
            }),
            html.Span(f"  {metrics.get('index_100', 100.0) - 100:+.1f}%", style={
                "color": GREEN if metrics.get("index_100", 100) >= 100 else RED,
                "fontSize": "0.7rem", "fontWeight": "400",
            }),
        ], style={"marginBottom": "10px"}),

        # Metrics grid (2x2)
        html.Div([
            html.Div([
                html.Div("P&L", className="metric-label"),
                html.Div(f"${pnl:+,.2f}", style={
                    "color": pnl_color, "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("TRADES", className="metric-label"),
                html.Div(str(metrics["total_trades"]), style={
                    "color": TEXT, "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "8px"}),

        html.Div([
            html.Div([
                html.Div("WIN RATE", className="metric-label"),
                html.Div(f"{metrics['win_rate']:.0%}", style={
                    "color": GREEN if metrics["win_rate"] > 0.5 else
                             (YELLOW if metrics["win_rate"] > 0.4 else RED),
                    "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("SHARPE(20)", className="metric-label"),
                html.Div(f"{metrics['sharpe_20']:.3f}", style={
                    "color": GREEN if metrics["sharpe_20"] > 0.3 else
                             (YELLOW if metrics["sharpe_20"] > 0 else RED),
                    "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "8px"}),

        # Last trade
        html.Div([
            html.Span("LAST  ", style={"color": TEXT_MUTED, "fontSize": "0.6rem"}),
            html.Span(metrics["last_trade"], style={"color": TEXT_DIM, "fontSize": "0.65rem"}),
        ]),

        # Kill-switch bar
        html.Div(style={
            "width": kill_bar_width, "height": "2px",
            "background": kill_bar_color, "borderRadius": "1px",
            "marginTop": "10px", "opacity": "0.6",
            "transition": "width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
        }),

    ], className=f"bento-card anim-{anim}")


def _diamond_panel(summary: dict, index_100: float = 100.0,
                   anim: int = 5) -> html.Div:
    """DIAMOND project panel (different data structure)."""
    status = summary.get("status", "unknown")
    trades = summary.get("trades", 0)
    pnl = summary.get("pnl", 0)
    win_rate = summary.get("win_rate", 0)

    return html.Div([
        html.Div([
            html.Span(f"{PROJECT_ICONS['DIAMOND']} ", style={"color": PURPLE, "fontSize": "0.9rem"}),
            html.Span("DIAMOND", className="project-name", style={"color": PURPLE}),
            html.Span("PAPER", className="health-badge",
                      style={"color": TEXT_DIM, "borderColor": TEXT_DIM,
                             "marginLeft": "auto", "float": "right"}),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(className="separator"),

        html.Div([
            html.Span("●" if status == "active" else "○",
                      style={"color": GREEN if status == "active" else RED,
                             "marginRight": "6px"}),
            html.Span(status.upper(), style={"color": TEXT_DIM, "fontSize": "0.7rem",
                                              "letterSpacing": "0.1em"}),
        ], style={"marginBottom": "12px"}),

        # Index 100 line
        html.Div([
            html.Span("INDEX ", style={"color": TEXT_MUTED, "fontSize": "0.6rem",
                                        "letterSpacing": "0.1em"}),
            html.Span(f"{index_100:.1f}", style={
                "color": GREEN if index_100 >= 100 else RED,
                "fontSize": "1.3rem", "fontWeight": "700",
                "fontVariantNumeric": "tabular-nums",
            }),
            html.Span(f"  {index_100 - 100:+.1f}%", style={
                "color": GREEN if index_100 >= 100 else RED,
                "fontSize": "0.7rem", "fontWeight": "400",
            }),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Div([
                html.Div("P&L", className="metric-label"),
                html.Div(f"${pnl:+,.2f}", style={
                    "color": GREEN if pnl >= 0 else RED,
                    "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("TRADES", className="metric-label"),
                html.Div(str(trades), style={
                    "color": TEXT, "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("WIN RATE", className="metric-label"),
                html.Div(f"{win_rate:.0%}", style={
                    "color": GREEN if win_rate > 0.5 else (YELLOW if win_rate > 0.4 else RED),
                    "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}),

    ], className=f"bento-card anim-{anim}")


def _emerald_panel(metrics: dict, bankroll: pd.DataFrame,
                   anim: int = 6) -> html.Div:
    """EMERALD project panel — NBA sports predictions (simulated bankroll)."""
    trades = metrics["total_trades"]
    win_rate = metrics["win_rate"]

    # Current bankroll from bankroll table (authoritative for P&L)
    current_balance = 1000.0
    if bankroll is not None and not bankroll.empty and "balance" in bankroll.columns:
        current_balance = float(bankroll["balance"].iloc[-1])

    # P&L derived from bankroll (not raw prediction sums, which are uncapped/inflated)
    pnl = current_balance - 1000.0
    pnl_color = GREEN if pnl >= 0 else RED

    return html.Div([
        html.Div([
            html.Span(f"{PROJECT_ICONS['EMERALD']} ", style={"color": EMERALD_GREEN, "fontSize": "0.9rem"}),
            html.Span("EMERALD", className="project-name", style={"color": EMERALD_GREEN}),
            html.Span("RESEARCH", className="health-badge",
                      style={"color": TEXT_DIM, "borderColor": TEXT_DIM,
                             "marginLeft": "auto", "float": "right"}),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(className="separator"),

        # Status line
        html.Div([
            html.Span("●" if trades > 0 else "○",
                      style={"color": GREEN if trades > 0 else TEXT_DIM,
                             "marginRight": "6px"}),
            html.Span("NBA PREDICTIONS", style={"color": TEXT_DIM, "fontSize": "0.7rem",
                                                 "letterSpacing": "0.1em"}),
        ], style={"marginBottom": "12px"}),

        # Index 100 line
        html.Div([
            html.Span("INDEX ", style={"color": TEXT_MUTED, "fontSize": "0.6rem",
                                        "letterSpacing": "0.1em"}),
            html.Span(f"{metrics.get('index_100', 100.0):.1f}", style={
                "color": GREEN if metrics.get("index_100", 100) >= 100 else RED,
                "fontSize": "1.3rem", "fontWeight": "700",
                "fontVariantNumeric": "tabular-nums",
            }),
            html.Span(f"  {metrics.get('index_100', 100.0) - 100:+.1f}%", style={
                "color": GREEN if metrics.get("index_100", 100) >= 100 else RED,
                "fontSize": "0.7rem", "fontWeight": "400",
            }),
        ], style={"marginBottom": "10px"}),

        # Metrics grid
        html.Div([
            html.Div([
                html.Div("P&L", className="metric-label"),
                html.Div(f"${pnl:+,.2f}", style={
                    "color": pnl_color, "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("BANKROLL", className="metric-label"),
                html.Div(f"${current_balance:,.0f}", style={
                    "color": GREEN if current_balance >= 1000 else RED,
                    "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "8px"}),

        html.Div([
            html.Div([
                html.Div("BETS", className="metric-label"),
                html.Div(str(trades), style={
                    "color": TEXT, "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("WIN RATE", className="metric-label"),
                html.Div(f"{win_rate:.0%}", style={
                    "color": GREEN if win_rate > 0.5 else (YELLOW if win_rate > 0.4 else RED),
                    "fontSize": "1.1rem", "fontWeight": "600",
                    "fontVariantNumeric": "tabular-nums",
                }),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}),

    ], className=f"bento-card anim-{anim}")


# ── Dash App ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="HMM TRADER // PORTFOLIO",
)

# Inject custom CSS + font
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href=\"""" + GOOGLE_FONT_URL + """\" rel="stylesheet">
    {%css%}
    <style>""" + CUSTOM_CSS + """</style>
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


@app.server.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


ALL_PROJECTS = ["AGATE", "BERYL", "CITRINE", "DIAMOND", "EMERALD"]

app.layout = html.Div([
    dcc.Interval(id="refresh", interval=60_000),
    dcc.Store(id="project-filter", data=ALL_PROJECTS),

    # Header
    html.Div([
        html.Div([
            html.Span("HMM TRADER", style={
                "color": CYAN, "fontSize": "0.8rem", "fontWeight": "700",
                "letterSpacing": "0.3em",
            }),
            html.Span("  //  ", style={"color": TEXT_MUTED}),
            html.Span("PORTFOLIO OVERVIEW", style={
                "color": TEXT_DIM, "fontSize": "0.8rem", "fontWeight": "300",
                "letterSpacing": "0.2em",
            }),
        ], style={"textAlign": "center", "padding": "20px 0 4px"}),
        html.Div(style={
            "height": "1px", "margin": "0 auto",
            "width": "200px",
            "background": f"linear-gradient(90deg, transparent, {CYAN_DIM}, transparent)",
        }),
    ], className="header-scanline"),

    html.Div(id="dashboard-content"),
], style={
    "minHeight": "100vh",
    "padding": "0 24px 40px",
    "maxWidth": "1400px",
    "margin": "0 auto",
}, className="grid-bg")


@app.callback(
    Output("dashboard-content", "children"),
    Input("refresh", "n_intervals"),
    State("project-filter", "data"),
)
def update_dashboard(_, active_filter):
    import traceback as _tb
    try:
        return _update_dashboard_inner(active_filter)
    except Exception as e:
        return [html.Pre(
            f"CALLBACK ERROR:\n{_tb.format_exc()}",
            style={"color": RED, "fontSize": "0.8rem", "padding": "20px",
                   "whiteSpace": "pre-wrap", "fontFamily": "monospace"},
        )]


def _update_dashboard_inner(active_filter=None):
    if active_filter is None:
        active_filter = list(ALL_PROJECTS)

    # Load all data
    all_trades = {}
    all_metrics = {}
    for project, db_path in PROJECT_DBS.items():
        df = _load_trades(db_path)
        all_trades[project] = df
        all_metrics[project] = _compute_project_metrics(project, df)

    # DIAMOND: separate DB + schema, normalized into same pipeline
    diamond_df = _load_diamond_trades()
    all_trades["DIAMOND"] = diamond_df
    all_metrics["DIAMOND"] = _compute_project_metrics("DIAMOND", diamond_df)

    # EMERALD: predictions DB with pnl + resolved_at
    emerald_df = _load_emerald_trades()
    all_trades["EMERALD"] = emerald_df
    all_metrics["EMERALD"] = _compute_project_metrics("EMERALD", emerald_df)

    emerald_bankroll = _load_emerald_bankroll()
    snapshots = _load_citrine_snapshots()
    diamond_summary = _diamond_summary_from_trades(diamond_df)

    agate_status = _load_status("agate_status.json")
    beryl_status = _load_status("beryl_status.json")

    # Compute indexed equity (base 100) for each project
    for project, df in all_trades.items():
        capital = STARTING_CAPITAL.get(project, 10_000.0)
        if project == "CITRINE" and not snapshots.empty and "total_equity" in snapshots.columns:
            idx = (float(snapshots["total_equity"].iloc[-1]) / capital) * 100
        elif project == "EMERALD":
            bal = 1000.0
            if emerald_bankroll is not None and not emerald_bankroll.empty and "balance" in emerald_bankroll.columns:
                bal = float(emerald_bankroll["balance"].iloc[-1])
            idx = (bal / capital) * 100
        elif not df.empty and "pnl" in df.columns:
            idx = ((capital + float(df["pnl"].sum())) / capital) * 100
        else:
            idx = 100.0
        all_metrics[project]["index_100"] = idx

    # Aggregates — use bankroll-derived P&L for EMERALD (raw pnl sums are inflated)
    emerald_balance = 1000.0
    if emerald_bankroll is not None and not emerald_bankroll.empty and "balance" in emerald_bankroll.columns:
        emerald_balance = float(emerald_bankroll["balance"].iloc[-1])
    emerald_real_pnl = emerald_balance - 1000.0

    total_pnl = sum(
        m["total_pnl"] for name, m in all_metrics.items() if name != "EMERALD"
    ) + emerald_real_pnl
    total_trades = sum(m["total_trades"] for m in all_metrics.values())
    active_projects = sum(1 for m in all_metrics.values() if m["total_trades"] > 0)

    all_pnls = np.concatenate([
        df["pnl"].values.astype(float)
        for df in all_trades.values()
        if not df.empty and "pnl" in df.columns
    ]) if any(not df.empty and "pnl" in df.columns for df in all_trades.values()) else np.array([])
    combined_sharpe = _rolling_sharpe(all_pnls, window=20)

    citrine_equity = 0.0
    if not snapshots.empty and "total_equity" in snapshots.columns:
        citrine_equity = float(snapshots["total_equity"].iloc[-1])

    # Exclude EMERALD (research) from system health count
    live_metrics = {k: v for k, v in all_metrics.items() if k != "EMERALD"}
    degraded = sum(1 for m in live_metrics.values() if m["degradation"] == "critical")
    warnings = sum(1 for m in live_metrics.values() if m["degradation"] == "warning")
    health_color = RED if degraded > 0 else (YELLOW if warnings > 0 else GREEN)
    health_label = (f"{degraded} DEGRADED" if degraded > 0
                    else f"{warnings} DRIFT" if warnings > 0
                    else "NOMINAL")

    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")

    # ── Bento Layout ─────────────────────────────────────────────────────

    # Row 1: Key metrics — asymmetric widths
    metrics_row = html.Div([
        html.Div(
            _metric_cell("COMBINED P&L", f"${total_pnl:+,.2f}",
                         GREEN if total_pnl >= 0 else RED, anim=1),
            style={"gridColumn": "span 3"},
        ),
        html.Div(
            _metric_cell("CITRINE EQUITY", f"${citrine_equity:,.0f}",
                         GREEN if citrine_equity >= 25000 else RED,
                         sub="from $25,000", anim=2),
            style={"gridColumn": "span 3"},
        ),
        html.Div(
            _metric_cell("TOTAL TRADES", str(total_trades), TEXT,
                         sub=f"{active_projects}/{len(all_metrics)} projects", anim=3),
            style={"gridColumn": "span 2"},
        ),
        html.Div(
            _metric_cell("SHARPE(20)", f"{combined_sharpe:.3f}",
                         GREEN if combined_sharpe > 0.3 else
                         (YELLOW if combined_sharpe > 0 else RED), anim=4),
            style={"gridColumn": "span 2"},
        ),
        html.Div(
            _metric_cell("SYSTEM HEALTH", health_label, health_color, anim=5),
            style={"gridColumn": "span 2"},
        ),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(12, 1fr)",
        "gap": "10px",
        "marginTop": "20px",
        "marginBottom": "10px",
    })

    # Row 2: Project panels — 5 panels, auto-wrap (3+2 or 5-across)
    # Panels are direct grid items (no wrapper divs) so CSS grid stretch
    # makes all cards in the same row equal height.
    projects_row = html.Div([
        _project_panel("AGATE", all_metrics["AGATE"], agate_status, anim=2),
        _project_panel("BERYL", all_metrics["BERYL"], beryl_status, anim=3),
        _project_panel("CITRINE", all_metrics["CITRINE"], {}, anim=4),
        _diamond_panel(diamond_summary, index_100=all_metrics["DIAMOND"]["index_100"], anim=5),
        _emerald_panel(all_metrics["EMERALD"], emerald_bankroll, anim=6),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
        "gap": "10px",
        "marginBottom": "10px",
    })

    # Row 3: Charts — asymmetric split (8/4)
    charts_row = html.Div([
        html.Div([
            html.Div(
                dcc.Graph(id="equity-chart",
                          figure=_apply_filter(_build_equity_chart(all_trades, snapshots, emerald_bankroll), active_filter),
                          config={"displayModeBar": False},
                          style={"height": "280px"}),
                className="chart-container",
            ),
        ], className="bento-card anim-6", style={"gridColumn": "span 8"}),

        html.Div([
            html.Div(
                dcc.Graph(id="pnl-chart",
                          figure=_apply_filter(_build_pnl_distribution(all_trades), active_filter),
                          config={"displayModeBar": False},
                          style={"height": "280px"}),
                className="chart-container",
            ),
        ], className="bento-card anim-7", style={"gridColumn": "span 4"}),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(12, 1fr)",
        "gap": "10px",
        "marginBottom": "10px",
    })

    # Row 4: Metrics over time — 2×2 subplot (full width)
    metrics_time_row = html.Div([
        html.Div([
            html.Div(
                dcc.Graph(id="metrics-time-chart",
                          figure=_apply_filter(_build_metrics_over_time(all_trades), active_filter),
                          config={"displayModeBar": False},
                          style={"height": "380px"}),
                className="chart-container",
            ),
        ], className="bento-card anim-7", style={"gridColumn": "span 12"}),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(12, 1fr)",
        "gap": "10px",
        "marginBottom": "10px",
    })

    # Row 5: Model health + timestamp
    bottom_row = html.Div([
        html.Div([
            html.Div(
                dcc.Graph(id="health-chart",
                          figure=_apply_filter(_build_degradation_chart(all_trades), active_filter),
                          config={"displayModeBar": False},
                          style={"height": "220px"}),
                className="chart-container",
            ),
        ], className="bento-card anim-8", style={"gridColumn": "span 9"}),

        html.Div([
            html.Div("SYSTEM STATUS", className="metric-label",
                     style={"marginBottom": "12px"}),

            # Per-project filter toggles (clickable)
            *[html.Button([
                html.Span("●", className="status-dot", style={
                    "backgroundColor": GREEN if m["degradation"] not in ("critical",) else RED,
                }),
                html.Span(p, style={"color": PROJECT_COLORS[p], "fontSize": "0.7rem",
                                     "fontWeight": "600", "letterSpacing": "0.1em"}),
                html.Span(f"  {m['total_trades']}t", style={
                    "color": TEXT_MUTED, "fontSize": "0.6rem", "marginLeft": "4px"}),
            ], id={"type": "project-toggle", "project": p},
               className="project-toggle" if p in active_filter else "project-toggle dimmed",
               n_clicks=0,
               style={"marginBottom": "6px"})
              for p, m in all_metrics.items()],

            html.Div(style={"height": "1px", "background": BORDER, "margin": "12px 0"}),

            # Timestamp
            html.Div([
                html.Div("UPDATED", className="metric-label"),
                html.Div(now_str, style={
                    "color": TEXT_DIM, "fontSize": "0.7rem",
                    "fontVariantNumeric": "tabular-nums",
                }),
                html.Div("auto-refresh 60s", className="metric-sub"),
            ]),

        ], className="bento-card anim-8", style={"gridColumn": "span 3"}),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(12, 1fr)",
        "gap": "10px",
        "marginBottom": "10px",
    })

    return [metrics_row, projects_row, charts_row, metrics_time_row, bottom_row]


# ── Filter Callbacks ─────────────────────────────────────────────────────────

@app.callback(
    Output("project-filter", "data"),
    Input({"type": "project-toggle", "project": ALL}, "n_clicks"),
    State("project-filter", "data"),
    prevent_initial_call=True,
)
def toggle_project_filter(n_clicks_list, current_filter):
    """Toggle a project in/out of the active filter list."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    # Extract which project was clicked from the trigger id
    trigger = ctx.triggered[0]
    prop_id = trigger["prop_id"]  # e.g. '{"project":"AGATE","type":"project-toggle"}.n_clicks'
    try:
        btn_id = json.loads(prop_id.rsplit(".", 1)[0])
        clicked = btn_id["project"]
    except (json.JSONDecodeError, KeyError):
        return dash.no_update

    if current_filter is None:
        current_filter = list(ALL_PROJECTS)

    if clicked in current_filter:
        # Deselect — but don't allow empty filter (keep at least one)
        new_filter = [p for p in current_filter if p != clicked]
        if not new_filter:
            # If this was the last one, reset to all (toggle-all-off = show all)
            return list(ALL_PROJECTS)
        return new_filter
    else:
        # Select — add back
        return current_filter + [clicked]


# Clientside callback: trace filtering, trendlines, and zoom-aware recalculation
app.clientside_callback(
    """
    function(activeProjects, eqRelayout, healthRelayout, metricsRelayout,
             equityFig, pnlFig, healthFig, metricsTimeFig) {

        var NU = window.dash_clientside.no_update;
        var ctx = window.dash_clientside.callback_context;
        var trigger = '';
        if (ctx.triggered && ctx.triggered.length) {
            trigger = ctx.triggered[0].prop_id;
        }

        var single = activeProjects.length === 1;
        var uirev = activeProjects.slice().sort().join(',');

        // ── Helpers ─────────────────────────────────────────────

        function linreg(xs, ys) {
            var n = xs.length;
            if (n < 3) return null;
            var sx=0, sy=0, sxy=0, sxx=0;
            for (var i=0; i<n; i++) {
                sx += xs[i]; sy += ys[i];
                sxy += xs[i]*ys[i]; sxx += xs[i]*xs[i];
            }
            var denom = n*sxx - sx*sx;
            if (Math.abs(denom) < 1e-12) return null;
            var slope = (n*sxy - sx*sy) / denom;
            var intercept = (sy - slope*sx) / n;
            var ssRes=0, ssTot=0, yMean = sy/n;
            for (var i=0; i<n; i++) {
                var p = slope*xs[i]+intercept;
                ssRes += (ys[i]-p)*(ys[i]-p);
                ssTot += (ys[i]-yMean)*(ys[i]-yMean);
            }
            return {slope:slope, intercept:intercept, r2: ssTot>1e-12 ? 1-ssRes/ssTot : 0};
        }

        function hexToRgba(hex, alpha) {
            if (hex.charAt(0) === '#') {
                var r = parseInt(hex.slice(1,3),16);
                var g = parseInt(hex.slice(3,5),16);
                var b = parseInt(hex.slice(5,7),16);
                return 'rgba('+r+','+g+','+b+','+alpha+')';
            }
            if (hex.indexOf('rgba') === 0)
                return hex.replace(/[\\d.]+\\)$/, alpha+')');
            if (hex.indexOf('rgb') === 0)
                return hex.replace('rgb(','rgba(').replace(')',','+alpha+')');
            return 'rgba(0,232,255,'+alpha+')';
        }

        function trendColor(trace) {
            var c = (trace.line && trace.line.color) || '#00e8ff';
            return hexToRgba(c, 0.5);
        }

        function getRange(relay, prefix) {
            if (!relay) return null;
            if (relay[prefix+'.autorange']) return null;
            var r0 = relay[prefix+'.range[0]'];
            var r1 = relay[prefix+'.range[1]'];
            if (r0 !== undefined && r1 !== undefined) return [r0, r1];
            return null;
        }

        function filterData(trace, range) {
            if (!trace || !trace.y || trace.y.length < 3) return null;
            var rXs=[], rYs=[], rXO=[];
            for (var i=0; i<trace.y.length; i++) {
                if (trace.y[i]==null || !isFinite(trace.y[i])) continue;
                var inR = true;
                if (range) {
                    var xv = trace.x[i];
                    if (typeof range[0] === 'string') {
                        xv = new Date(xv).getTime();
                        inR = xv >= new Date(range[0]).getTime()
                           && xv <= new Date(range[1]).getTime();
                    } else {
                        xv = typeof xv === 'number' ? xv : parseFloat(xv);
                        inR = xv >= range[0] && xv <= range[1];
                    }
                }
                if (inR) { rXs.push(rXs.length); rYs.push(trace.y[i]); rXO.push(trace.x[i]); }
            }
            if (rXs.length < 3) return null;
            return {xs:rXs, ys:rYs, xOrig:rXO};
        }

        function makeTrend(fd, color, xaxis, yaxis) {
            if (!fd) return null;
            var reg = linreg(fd.xs, fd.ys);
            if (!reg) return null;
            var n = fd.xs.length;
            var t = {
                x: [fd.xOrig[0], fd.xOrig[n-1]],
                y: [reg.intercept, reg.slope*fd.xs[n-1]+reg.intercept],
                mode: 'lines',
                name: 'Trend (R\\u00b2=' + reg.r2.toFixed(2) + ')',
                line: {color:color, width:2, dash:'dash'},
                showlegend: false, hoverinfo: 'skip', visible: true
            };
            if (xaxis) t.xaxis = xaxis;
            if (yaxis) t.yaxis = yaxis;
            return t;
        }

        function isMatch(name, lg, active) {
            for (var k=0; k<active.length; k++) {
                if (name===active[k] || name.indexOf(active[k])===0 || lg===active[k])
                    return true;
            }
            return false;
        }

        // ── Process a single-panel chart ────────────────────────

        function processChart(fig, active, addTrend, xRange) {
            if (!fig || !fig.data) return fig;
            var nf = JSON.parse(JSON.stringify(fig));
            nf.layout.uirevision = uirev;
            nf.data = nf.data.filter(function(t){
                return !(t.name && t.name.indexOf('Trend')===0);
            });
            var vis = null;
            for (var i=0; i<nf.data.length; i++) {
                var nm = nf.data[i].name || '';
                if (nm === 'COMBINED') {
                    nf.data[i].visible = !single;
                } else {
                    var lg = nf.data[i].legendgroup || '';
                    var m = isMatch(nm, lg, active);
                    nf.data[i].visible = m;
                    if (m && single && addTrend) vis = nf.data[i];
                }
            }
            if (single && addTrend && vis) {
                var fd = filterData(vis, xRange);
                var trend = makeTrend(fd, trendColor(vis));
                if (trend) nf.data.push(trend);
            }
            return nf;
        }

        // ── Process the 2x2 metrics subplot ─────────────────────

        function processMetrics(fig, active, relayoutObj) {
            if (!fig || !fig.data) return fig;
            var nf = JSON.parse(JSON.stringify(fig));
            nf.layout.uirevision = uirev;
            nf.data = nf.data.filter(function(t){
                return !(t.name && t.name.indexOf('Trend')===0);
            });
            var targets = [];
            for (var i=0; i<nf.data.length; i++) {
                var t = nf.data[i]; var nm = t.name||''; var lg = t.legendgroup||'';
                if (nm === 'COMBINED') { t.visible = !single; continue; }
                var m = isMatch(nm, lg, active);
                t.visible = m;
                if (m && single) targets.push(t);
            }
            if (single) {
                for (var j=0; j<targets.length; j++) {
                    var tt = targets[j];
                    var xa = tt.xaxis || 'x';
                    var ya = tt.yaxis || 'y';
                    if (xa==='x2' && ya==='y2') continue;
                    var axKey = xa==='x' ? 'xaxis' : xa.replace('x','xaxis');
                    var range = relayoutObj ? getRange(relayoutObj, axKey) : null;
                    var fd = filterData(tt, range);
                    var trend = makeTrend(fd, trendColor(tt), xa, ya);
                    if (trend) nf.data.push(trend);
                }
            }
            return nf;
        }

        // ── Main dispatch ───────────────────────────────────────

        var allP = ['AGATE','BERYL','CITRINE','DIAMOND','EMERALD'];
        var styles = allP.map(function(p){
            return activeProjects.indexOf(p)!==-1 ? '' : 'project-toggle dimmed';
        });
        var NUS = [NU,NU,NU,NU,NU];

        // Zoom on equity chart
        if (trigger.indexOf('equity-chart.relayoutData') !== -1) {
            if (!single) return [NU,NU,NU,NU].concat(NUS);
            var r = getRange(eqRelayout, 'xaxis');
            return [processChart(equityFig, activeProjects, true, r),
                    NU, NU, NU].concat(NUS);
        }
        // Zoom on health chart
        if (trigger.indexOf('health-chart.relayoutData') !== -1) {
            if (!single) return [NU,NU,NU,NU].concat(NUS);
            var r = getRange(healthRelayout, 'xaxis');
            return [NU, NU,
                    processChart(healthFig, activeProjects, true, r),
                    NU].concat(NUS);
        }
        // Zoom on metrics chart
        if (trigger.indexOf('metrics-time-chart.relayoutData') !== -1) {
            if (!single) return [NU,NU,NU,NU].concat(NUS);
            return [NU, NU, NU,
                    processMetrics(metricsTimeFig, activeProjects, metricsRelayout)
                   ].concat(NUS);
        }

        // Filter change — recompute all charts (full range)
        var eq = processChart(equityFig, activeProjects, true, null);
        var pnl = processChart(pnlFig, activeProjects, false, null);
        var health = processChart(healthFig, activeProjects, true, null);
        var mt = processMetrics(metricsTimeFig, activeProjects, null);
        return [eq, pnl, health, mt].concat(styles);
    }
    """,
    [
        Output("equity-chart", "figure"),
        Output("pnl-chart", "figure"),
        Output("health-chart", "figure"),
        Output("metrics-time-chart", "figure"),
        Output({"type": "project-toggle", "project": "AGATE"}, "className"),
        Output({"type": "project-toggle", "project": "BERYL"}, "className"),
        Output({"type": "project-toggle", "project": "CITRINE"}, "className"),
        Output({"type": "project-toggle", "project": "DIAMOND"}, "className"),
        Output({"type": "project-toggle", "project": "EMERALD"}, "className"),
    ],
    [
        Input("project-filter", "data"),
        Input("equity-chart", "relayoutData"),
        Input("health-chart", "relayoutData"),
        Input("metrics-time-chart", "relayoutData"),
    ],
    [
        State("equity-chart", "figure"),
        State("pnl-chart", "figure"),
        State("health-chart", "figure"),
        State("metrics-time-chart", "figure"),
    ],
    prevent_initial_call=True,
)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HMM Trader — Portfolio Overview")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    print(f"Portfolio Overview: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
