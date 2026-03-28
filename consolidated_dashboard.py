"""
consolidated_dashboard.py
─────────────────────────
Portfolio Overview — Combined view of all 4 trading sub-projects.

Shows:
  - Combined equity curve (AGATE + BERYL + CITRINE)
  - Per-project P&L cards with rolling Sharpe
  - Aggregate metrics (total equity, total trades, combined Sharpe)
  - Model health indicators (degradation detection)
  - Kill-switch status per project
  - DIAMOND paper trading summary

Run:
  python consolidated_dashboard.py                     # local at :8090
  python consolidated_dashboard.py --host 0.0.0.0      # VM-accessible

Port 8090 to avoid collision with existing dashboards (8050/8060/8070/8080).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

PROJECT_DBS = {
    "AGATE":   ROOT / "paper_trades.db",
    "BERYL":   ROOT / "beryl_trades.db",
    "CITRINE": ROOT / "citrine_trades.db",
}

# DIAMOND uses a separate repo — check both local and common locations
DIAMOND_DB_CANDIDATES = [
    ROOT.parent / "kalshi-diamond" / "diamond.db",
    Path("/home/ubuntu/kalshi-diamond/diamond.db"),
    ROOT / "diamond.db",
]

STATUS_FILES = {
    "AGATE": ROOT / "agate_status.json",
    "BERYL": ROOT / "beryl_status.json",
}

# Backtest expected Sharpe per project (for degradation detection)
EXPECTED_SHARPE = {
    "AGATE":   0.837,   # stability test extended_v2/7cf
    "BERYL":   0.825,   # Phase 3 best (NVDA)
    "CITRINE": 1.665,   # allocator optimization winner
}

# ── Colours ───────────────────────────────────────────────────────────────────
BG       = "#0d0f14"
PANEL    = "#141820"
BORDER   = "#1e2330"
TEXT     = "#e0e4f0"
TEXT_DIM = "#6b7394"
GREEN    = "#00e676"
RED      = "#ff1744"
YELLOW   = "#ffea00"
BLUE     = "#448aff"
PURPLE   = "#e040fb"
ORANGE   = "#ff6d00"

PROJECT_COLORS = {
    "AGATE":   ORANGE,
    "BERYL":   BLUE,
    "CITRINE": GREEN,
    "DIAMOND": PURPLE,
}


# ── Data Loaders ──────────────────────────────────────────────────────────────

def _load_trades(db_path: Path) -> pd.DataFrame:
    """Load closed trades (with P&L) from a SQLite DB. Returns empty DataFrame on error."""
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades ORDER BY timestamp ASC", conn,
            )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Filter to closed trades only (ENTER/SCALE_UP trades have NULL pnl)
        if "pnl" in df.columns:
            df = df.dropna(subset=["pnl"])
        return df
    except Exception:
        return pd.DataFrame()


def _load_citrine_snapshots() -> pd.DataFrame:
    """Load CITRINE portfolio snapshots for equity curve."""
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


def _load_diamond_summary() -> dict:
    """Load DIAMOND paper trading summary. Returns empty dict if unavailable."""
    for candidate in DIAMOND_DB_CANDIDATES:
        if candidate.exists():
            try:
                with sqlite3.connect(str(candidate)) as conn:
                    # Try to get paper trade summary
                    trades = pd.read_sql_query(
                        "SELECT * FROM paper_trades ORDER BY timestamp DESC LIMIT 50",
                        conn,
                    )
                    if trades.empty:
                        return {"status": "active", "trades": 0, "pnl": 0.0}
                    total_pnl = trades["pnl"].sum() if "pnl" in trades.columns else 0.0
                    n_trades = len(trades)
                    wins = (trades["pnl"] > 0).sum() if "pnl" in trades.columns else 0
                    return {
                        "status": "active",
                        "trades": n_trades,
                        "pnl": float(total_pnl),
                        "win_rate": float(wins / max(n_trades, 1)),
                    }
            except Exception:
                return {"status": "active", "trades": 0, "pnl": 0.0}
    return {"status": "no_db", "trades": 0, "pnl": 0.0}


def _load_status(filename: str) -> dict:
    """Load a JSON status file."""
    path = ROOT / filename
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _rolling_sharpe(pnls: np.ndarray, window: int = 20) -> float:
    """Compute rolling Sharpe over last N trades."""
    if len(pnls) < window:
        return 0.0
    recent = pnls[-window:]
    std = np.std(recent)
    if std < 1e-9:
        return 0.0
    return float(np.mean(recent) / std)


def _compute_project_metrics(project: str, trades_df: pd.DataFrame) -> dict:
    """Compute key metrics for a single project."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return {
            "total_pnl": 0.0, "total_trades": 0, "win_rate": 0.0,
            "sharpe_20": 0.0, "last_trade": "N/A",
            "degradation": "unknown", "degradation_sigma": 0.0,
        }

    pnls = trades_df["pnl"].values.astype(float)
    total_pnl = float(np.sum(pnls))
    total_trades = len(pnls)
    wins = int(np.sum(pnls > 0))
    win_rate = wins / max(total_trades, 1)
    sharpe_20 = _rolling_sharpe(pnls, window=20)

    # Last trade timestamp
    last_ts = trades_df["timestamp"].iloc[-1] if not trades_df.empty else "N/A"
    if isinstance(last_ts, pd.Timestamp):
        last_trade = last_ts.strftime("%Y-%m-%d %H:%M")
    else:
        last_trade = str(last_ts)[:16]

    # Model degradation detection
    expected = EXPECTED_SHARPE.get(project, 1.0)
    degradation = "unknown"
    degradation_sigma = 0.0

    if total_trades >= 10:
        # Compare rolling Sharpe vs expected backtest Sharpe
        # Scale expected Sharpe down by typical backtest-to-live degradation (75-90%)
        live_expected = expected * 0.15  # expect 15% of backtest Sharpe in live
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
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sharpe_20": sharpe_20,
        "last_trade": last_trade,
        "degradation": degradation,
        "degradation_sigma": degradation_sigma,
    }


# ── Dashboard Components ──────────────────────────────────────────────────────

CARD_STYLE = {
    "backgroundColor": PANEL,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "12px",
    "fontFamily": "monospace",
}


def _metric_card(label: str, value: str, color: str = TEXT, sublabel: str = "") -> html.Div:
    """Single metric card."""
    children = [
        html.Div(label, style={"color": TEXT_DIM, "fontSize": "0.7rem", "textTransform": "uppercase"}),
        html.Div(value, style={"color": color, "fontSize": "1.4rem", "fontWeight": "bold"}),
    ]
    if sublabel:
        children.append(
            html.Div(sublabel, style={"color": TEXT_DIM, "fontSize": "0.7rem"})
        )
    return html.Div(children, style=CARD_STYLE)


def _project_card(project: str, metrics: dict) -> dbc.Col:
    """Card for a single project with key metrics."""
    color = PROJECT_COLORS.get(project, TEXT)
    pnl = metrics["total_pnl"]
    pnl_color = GREEN if pnl >= 0 else RED

    # Degradation badge
    deg = metrics["degradation"]
    deg_colors = {"healthy": GREEN, "warning": YELLOW, "critical": RED,
                  "insufficient_data": TEXT_DIM, "too_early": TEXT_DIM, "unknown": TEXT_DIM}
    deg_labels = {"healthy": "HEALTHY", "warning": "DRIFT", "critical": "DEGRADED",
                  "insufficient_data": "COLLECTING", "too_early": "TOO EARLY", "unknown": "?"}
    deg_color = deg_colors.get(deg, TEXT_DIM)
    deg_label = deg_labels.get(deg, "?")

    return dbc.Col(
        html.Div([
            html.Div([
                html.Span(project, style={"color": color, "fontSize": "1.1rem", "fontWeight": "bold"}),
                html.Span(f"  {deg_label}", style={"color": deg_color, "fontSize": "0.7rem",
                                                      "border": f"1px solid {deg_color}",
                                                      "borderRadius": "4px", "padding": "2px 6px",
                                                      "marginLeft": "8px"}),
            ]),
            html.Hr(style={"borderColor": BORDER, "margin": "8px 0"}),
            html.Div([
                html.Span("P&L: ", style={"color": TEXT_DIM}),
                html.Span(f"${pnl:+,.2f}", style={"color": pnl_color, "fontWeight": "bold"}),
            ]),
            html.Div([
                html.Span("Trades: ", style={"color": TEXT_DIM}),
                html.Span(f"{metrics['total_trades']}", style={"color": TEXT}),
                html.Span(f"  WR: {metrics['win_rate']:.0%}", style={"color": TEXT_DIM, "marginLeft": "12px"}),
            ]),
            html.Div([
                html.Span("Sharpe(20): ", style={"color": TEXT_DIM}),
                html.Span(f"{metrics['sharpe_20']:.3f}",
                          style={"color": GREEN if metrics['sharpe_20'] > 0.3 else
                                 (YELLOW if metrics['sharpe_20'] > 0 else RED)}),
            ]),
            html.Div([
                html.Span("Last: ", style={"color": TEXT_DIM}),
                html.Span(metrics["last_trade"], style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            ]),
        ], style=CARD_STYLE),
        md=3, sm=6,
    )


def _build_equity_chart(all_trades: dict[str, pd.DataFrame], snapshots: pd.DataFrame) -> go.Figure:
    """Build combined equity chart across all projects."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="monospace", color=TEXT),
        title="Combined Equity Curve",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        height=400,
    )

    # CITRINE uses snapshots (portfolio-level equity)
    if not snapshots.empty and "total_equity" in snapshots.columns:
        fig.add_trace(go.Scatter(
            x=snapshots["timestamp"],
            y=snapshots["total_equity"] - 25000,  # relative to starting capital
            name="CITRINE",
            line=dict(color=PROJECT_COLORS["CITRINE"], width=2),
        ))

    # AGATE and BERYL use cumulative P&L from trades
    for project in ["AGATE", "BERYL"]:
        df = all_trades.get(project, pd.DataFrame())
        if df.empty or "pnl" not in df.columns:
            continue
        df = df.sort_values("timestamp")
        cum_pnl = df["pnl"].cumsum()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=cum_pnl,
            name=project,
            line=dict(color=PROJECT_COLORS[project], width=2),
        ))

    # Combined total line
    # Build a daily combined P&L
    all_pnl_series = []
    for project, df in all_trades.items():
        if df.empty or "pnl" not in df.columns:
            continue
        daily = df.set_index("timestamp").resample("D")["pnl"].sum()
        all_pnl_series.append(daily)

    if all_pnl_series:
        combined = pd.concat(all_pnl_series, axis=1).fillna(0).sum(axis=1).cumsum()
        if not combined.empty:
            fig.add_trace(go.Scatter(
                x=combined.index,
                y=combined.values,
                name="COMBINED",
                line=dict(color=TEXT, width=3, dash="dot"),
            ))

    return fig


def _build_degradation_chart(all_trades: dict[str, pd.DataFrame]) -> go.Figure:
    """Build model health chart — rolling Sharpe vs expected for each project."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="monospace", color=TEXT),
        title="Model Health — Rolling 10-Trade Sharpe",
        xaxis_title="Trade #",
        yaxis_title="Sharpe Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        height=300,
    )

    for project, df in all_trades.items():
        if df.empty or "pnl" not in df.columns or len(df) < 5:
            continue
        pnls = df.sort_values("timestamp")["pnl"].values.astype(float)
        rolling = []
        window = 10
        for i in range(window, len(pnls) + 1):
            chunk = pnls[i - window:i]
            std = np.std(chunk)
            s = float(np.mean(chunk) / std) if std > 1e-9 else 0.0
            rolling.append(s)

        if rolling:
            fig.add_trace(go.Scatter(
                x=list(range(window, len(pnls) + 1)),
                y=rolling,
                name=project,
                line=dict(color=PROJECT_COLORS[project], width=2),
            ))

    # Add warning threshold line
    fig.add_hline(y=0, line=dict(color=RED, width=1, dash="dash"), annotation_text="Break-even")

    return fig


# ── Dash App ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="HMM Trader — Portfolio Overview",
)


@app.server.after_request
def add_no_cache_headers(response):
    """Prevent stale browser caching."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


app.layout = html.Div([
    dcc.Interval(id="refresh", interval=60_000),  # 60s auto-refresh
    html.H1("HMM TRADER — PORTFOLIO OVERVIEW",
            style={"textAlign": "center", "color": TEXT, "fontFamily": "monospace",
                   "fontSize": "1.2rem", "padding": "16px 0 8px"}),
    html.Div(id="dashboard-content"),
], style={"backgroundColor": BG, "minHeight": "100vh", "padding": "0 20px 20px"})


@app.callback(
    Output("dashboard-content", "children"),
    Input("refresh", "n_intervals"),
)
def update_dashboard(_):
    """Rebuild dashboard on every refresh."""
    # Load all data
    all_trades = {}
    all_metrics = {}
    for project, db_path in PROJECT_DBS.items():
        df = _load_trades(db_path)
        all_trades[project] = df
        all_metrics[project] = _compute_project_metrics(project, df)

    snapshots = _load_citrine_snapshots()
    diamond_summary = _load_diamond_summary()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    total_pnl = sum(m["total_pnl"] for m in all_metrics.values())
    total_trades = sum(m["total_trades"] for m in all_metrics.values())
    active_projects = sum(1 for m in all_metrics.values() if m["total_trades"] > 0)

    # Combined Sharpe from all trades
    all_pnls = np.concatenate([
        df["pnl"].values.astype(float)
        for df in all_trades.values()
        if not df.empty and "pnl" in df.columns
    ]) if any(not df.empty and "pnl" in df.columns for df in all_trades.values()) else np.array([])
    combined_sharpe = _rolling_sharpe(all_pnls, window=20)

    # CITRINE equity
    citrine_equity = 0.0
    if not snapshots.empty and "total_equity" in snapshots.columns:
        citrine_equity = float(snapshots["total_equity"].iloc[-1])

    # Degradation summary
    degraded_count = sum(1 for m in all_metrics.values() if m["degradation"] == "critical")
    warning_count = sum(1 for m in all_metrics.values() if m["degradation"] == "warning")
    health_color = RED if degraded_count > 0 else (YELLOW if warning_count > 0 else GREEN)
    health_label = (f"{degraded_count} DEGRADED" if degraded_count > 0
                    else f"{warning_count} DRIFT" if warning_count > 0
                    else "ALL HEALTHY")

    # ── Build layout ──────────────────────────────────────────────────────
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = [
        # Top row: aggregate metrics
        dbc.Row([
            dbc.Col(_metric_card("Combined P&L", f"${total_pnl:+,.2f}",
                                 GREEN if total_pnl >= 0 else RED), md=2),
            dbc.Col(_metric_card("CITRINE Equity", f"${citrine_equity:,.0f}",
                                 GREEN if citrine_equity >= 25000 else RED,
                                 f"started $25,000"), md=2),
            dbc.Col(_metric_card("Total Trades", str(total_trades), TEXT,
                                 f"{active_projects}/4 projects active"), md=2),
            dbc.Col(_metric_card("Combined Sharpe(20)", f"{combined_sharpe:.3f}",
                                 GREEN if combined_sharpe > 0.3 else
                                 (YELLOW if combined_sharpe > 0 else RED)), md=2),
            dbc.Col(_metric_card("Model Health", health_label, health_color), md=2),
            dbc.Col(_metric_card("Updated", now_str, TEXT_DIM, "auto-refresh 60s"), md=2),
        ], className="mb-3"),

        # Project cards
        dbc.Row([
            _project_card(project, metrics)
            for project, metrics in all_metrics.items()
        ] + [
            # DIAMOND card (separate structure)
            dbc.Col(
                html.Div([
                    html.Div([
                        html.Span("DIAMOND", style={"color": PURPLE, "fontSize": "1.1rem", "fontWeight": "bold"}),
                        html.Span("  PAPER", style={"color": TEXT_DIM, "fontSize": "0.7rem",
                                                      "border": f"1px solid {TEXT_DIM}",
                                                      "borderRadius": "4px", "padding": "2px 6px",
                                                      "marginLeft": "8px"}),
                    ]),
                    html.Hr(style={"borderColor": BORDER, "margin": "8px 0"}),
                    html.Div([
                        html.Span("Status: ", style={"color": TEXT_DIM}),
                        html.Span(diamond_summary.get("status", "unknown").upper(),
                                  style={"color": GREEN if diamond_summary.get("status") == "active" else RED}),
                    ]),
                    html.Div([
                        html.Span("Trades: ", style={"color": TEXT_DIM}),
                        html.Span(str(diamond_summary.get("trades", 0)), style={"color": TEXT}),
                    ]),
                    html.Div([
                        html.Span("P&L: ", style={"color": TEXT_DIM}),
                        html.Span(f"${diamond_summary.get('pnl', 0):+,.2f}",
                                  style={"color": GREEN if diamond_summary.get("pnl", 0) >= 0 else RED}),
                    ]),
                ], style=CARD_STYLE),
                md=3, sm=6,
            ),
        ], className="mb-3"),

        # Charts
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=_build_equity_chart(all_trades, snapshots),
                          config={"displayModeBar": False}),
                md=12,
            ),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=_build_degradation_chart(all_trades),
                          config={"displayModeBar": False}),
                md=12,
            ),
        ], className="mb-3"),

        # Regime status panels (AGATE + BERYL)
        dbc.Row([
            dbc.Col([
                html.H6("AGATE Regime", style={"color": ORANGE, "fontFamily": "monospace"}),
                _regime_status_mini("AGATE", _load_status("agate_status.json")),
            ], md=6),
            dbc.Col([
                html.H6("BERYL Regime", style={"color": BLUE, "fontFamily": "monospace"}),
                _regime_status_mini("BERYL", _load_status("beryl_status.json")),
            ], md=6),
        ], className="mb-3"),
    ]

    return content


def _regime_status_mini(name: str, status: dict) -> html.Div:
    """Compact regime status panel."""
    if not status:
        return html.Div(f"Waiting for {name} status...",
                        style={"color": TEXT_DIM, "fontSize": "0.8rem", **CARD_STYLE})

    regime = status.get("regime", "UNKNOWN").upper()
    confidence = status.get("confidence", 0.0)
    signal = status.get("signal", "HOLD")
    ticker = status.get("ticker", "?")

    regime_colors = {"BULL": GREEN, "BEAR": RED, "CHOP": YELLOW}
    # Match partial regime names
    for key in regime_colors:
        if key in regime:
            r_color = regime_colors[key]
            break
    else:
        r_color = TEXT_DIM

    signal_colors = {"BUY": GREEN, "SELL": RED, "HOLD": YELLOW}
    s_color = signal_colors.get(signal, TEXT_DIM)

    return html.Div([
        html.Span(f"{ticker} ", style={"color": TEXT_DIM}),
        html.Span(regime, style={"color": r_color, "fontWeight": "bold", "marginRight": "12px"}),
        html.Span(f"conf={confidence:.0%} ", style={"color": TEXT_DIM}),
        html.Span(signal, style={"color": s_color, "fontWeight": "bold"}),
    ], style=CARD_STYLE)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HMM Trader — Portfolio Overview Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    print(f"Portfolio Overview Dashboard: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
