"""
citrine_dashboard.py
────────────────────
CITRINE Portfolio Rotation — Live Dashboard.

Standalone Dash app that auto-refreshes every 60 seconds.
Reads from:
  - citrine_trades.db     (CITRINE trade history + portfolio snapshots)
  - citrine_wf_results.csv (walk-forward backtest results)

Run:
  python citrine_dashboard.py        # opens at http://127.0.0.1:8070

Port 8070 to avoid collision with AGATE (:8060) and backtest (:8050).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from daily_report import _agate_score, _beryl_score, _citrine_score, _score_bar

ROOT = Path(__file__).parent

# ── Colour palette (consistent with live_dashboard.py) ────────────────────────
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
CYAN     = "#00e5ff"

DB_PATH = ROOT / "citrine_trades.db"
WF_CSV  = ROOT / "citrine_wf_results.csv"


# ── Trading calendar helpers ─────────────────────────────────────────────────

def _us_market_holidays(year: int) -> set[date]:
    """Return set of US stock market holidays for a given year.

    Covers NYSE/NASDAQ closures: New Year's, MLK Day, Presidents' Day,
    Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day,
    Thanksgiving, Christmas. Observed-date rules applied.
    """
    from datetime import timedelta

    holidays: set[date] = set()

    def _nearest_weekday(d: date) -> date:
        """If holiday falls on Sat→Fri, Sun→Mon."""
        if d.weekday() == 5:  # Saturday
            return d - timedelta(days=1)
        if d.weekday() == 6:  # Sunday
            return d + timedelta(days=1)
        return d

    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Return the nth occurrence of weekday in month (1-indexed)."""
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))

    # New Year's Day (Jan 1)
    holidays.add(_nearest_weekday(date(year, 1, 1)))
    # MLK Day (3rd Monday in January)
    holidays.add(_nth_weekday(year, 1, 0, 3))  # Monday=0
    # Presidents' Day (3rd Monday in February)
    holidays.add(_nth_weekday(year, 2, 0, 3))
    # Good Friday (2 days before Easter Sunday)
    # Easter algorithm (Anonymous Gregorian)
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month_e = (h + l - 7 * m + 114) // 31
    day_e = ((h + l - 7 * m + 114) % 31) + 1
    easter = date(year, month_e, day_e)
    holidays.add(easter - timedelta(days=2))  # Good Friday
    # Memorial Day (last Monday in May)
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    holidays.add(d)
    # Juneteenth (June 19)
    holidays.add(_nearest_weekday(date(year, 6, 19)))
    # Independence Day (July 4)
    holidays.add(_nearest_weekday(date(year, 7, 4)))
    # Labor Day (1st Monday in September)
    holidays.add(_nth_weekday(year, 9, 0, 1))
    # Thanksgiving (4th Thursday in November)
    holidays.add(_nth_weekday(year, 11, 3, 4))  # Thursday=3
    # Christmas (December 25)
    holidays.add(_nearest_weekday(date(year, 12, 25)))

    return holidays


def _is_trading_day(d: date) -> bool:
    """Return True if d is a US stock market trading day (not weekend/holiday)."""
    if d.weekday() >= 5:  # Saturday or Sunday
        return False
    return d not in _us_market_holidays(d.year)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_trades() -> pd.DataFrame:
    """Load closed trades (EXITs) from citrine_trades.db."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE action = 'EXIT' "
                "ORDER BY timestamp DESC", conn,
            )
    except Exception:
        return pd.DataFrame()


def _load_all_trades() -> pd.DataFrame:
    """Load all trades from citrine_trades.db."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades ORDER BY timestamp DESC", conn,
            )
    except Exception:
        return pd.DataFrame()


def _load_snapshots() -> pd.DataFrame:
    """Load portfolio snapshots from citrine_trades.db."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            return pd.read_sql_query(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp ASC", conn,
            )
    except Exception:
        return pd.DataFrame()


def _load_wf_results() -> pd.DataFrame:
    """Load walk-forward backtest results."""
    if not WF_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(str(WF_CSV))
    except Exception:
        return pd.DataFrame()


# ── Chart builders ────────────────────────────────────────────────────────────

def _build_equity_curve(snap_df: pd.DataFrame) -> go.Figure:
    """Build portfolio equity curve from snapshots."""
    fig = go.Figure()

    if snap_df.empty:
        fig.add_annotation(
            text="No portfolio snapshots yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color=TEXT_DIM),
        )
    else:
        timestamps = pd.to_datetime(snap_df["timestamp"], errors="coerce")
        equity = snap_df["total_equity"]
        cash = snap_df["cash"]

        color = GREEN if equity.iloc[-1] >= 25000 else RED

        # Equity line
        fig.add_trace(go.Scatter(
            x=timestamps, y=equity,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.06)",
            name="Total Equity",
        ))

        # Cash line
        fig.add_trace(go.Scatter(
            x=timestamps, y=cash,
            mode="lines",
            line=dict(color=BLUE, width=1, dash="dot"),
            name="Cash",
        ))

        # Starting capital reference
        fig.add_hline(y=25000, line=dict(color=TEXT_DIM, width=1, dash="dot"),
                      annotation_text="$25k start")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        title=dict(text="Portfolio Equity", font=dict(size=13)),
        yaxis=dict(tickprefix="$", gridcolor=BORDER),
        xaxis=dict(gridcolor=BORDER),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def _build_positions_pie(snap_df: pd.DataFrame) -> go.Figure:
    """Build current allocation pie chart from latest snapshot."""
    fig = go.Figure()

    if snap_df.empty:
        fig.add_annotation(
            text="No positions yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=TEXT_DIM),
        )
    else:
        latest = snap_df.iloc[-1]
        try:
            positions = json.loads(latest["positions_json"]) if latest["positions_json"] else {}
        except Exception:
            positions = {}

        labels = list(positions.keys()) + ["Cash"]
        values = [p.get("value", 0) for p in positions.values()] + [latest["cash"]]
        colors = []
        for ticker in positions:
            d = positions[ticker].get("direction", "LONG")
            colors.append(GREEN if d == "LONG" else RED)
        colors.append(BLUE)

        fig.add_trace(go.Pie(
            labels=labels, values=values,
            hole=0.45,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=10, color=TEXT),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        title=dict(text="Portfolio Allocation", font=dict(size=13)),
        showlegend=False,
    )
    return fig


def _build_regime_heatmap(snap_df: pd.DataFrame) -> go.Figure:
    """Build sector-level regime distribution bar chart from latest snapshot."""
    fig = go.Figure()

    if snap_df.empty:
        fig.add_annotation(
            text="No data yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=TEXT_DIM),
        )
    else:
        latest = snap_df.iloc[-1]
        bull = latest.get("bull_count", 0)
        bear = latest.get("bear_count", 0)
        chop = latest.get("chop_count", 0)

        fig.add_trace(go.Bar(
            x=["BULL", "BEAR", "CHOP"],
            y=[bull, bear, chop],
            marker_color=[GREEN, RED, YELLOW],
            text=[str(bull), str(bear), str(chop)],
            textposition="auto",
            textfont=dict(size=14, color=BG, family="monospace"),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        height=200,
        title=dict(text="Market Regime Distribution", font=dict(size=13)),
        yaxis=dict(gridcolor=BORDER, title="Tickers"),
        xaxis=dict(gridcolor=BORDER),
        showlegend=False,
    )
    return fig


def _build_pnl_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """Build P&L histogram from closed trades."""
    fig = go.Figure()

    if trades_df.empty or "pnl" not in trades_df.columns:
        fig.add_annotation(
            text="No closed trades yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=TEXT_DIM),
        )
    else:
        pnls = trades_df["pnl"].dropna()
        if not pnls.empty:
            colors = [GREEN if p > 0 else RED for p in pnls]
            fig.add_trace(go.Histogram(
                x=pnls,
                marker=dict(color=BLUE, line=dict(color=TEXT_DIM, width=0.5)),
                nbinsx=20,
            ))
            fig.add_vline(x=0, line=dict(color=TEXT_DIM, width=1, dash="dot"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        height=200,
        title=dict(text="P&L Distribution", font=dict(size=13)),
        xaxis=dict(tickprefix="$", gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER, title="Count"),
        showlegend=False,
    )
    return fig


# ── Component builders ────────────────────────────────────────────────────────

def _metric_card(title: str, value: str, color: str = TEXT) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="card-title",
                   style={"fontSize": "0.7rem", "color": TEXT_DIM,
                          "marginBottom": "4px", "textTransform": "uppercase",
                          "letterSpacing": "0.08em"}),
            html.H4(value, style={"color": color, "fontFamily": "monospace",
                                   "fontWeight": "700", "marginBottom": "0"}),
        ]),
        style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
               "borderRadius": "8px", "padding": "4px"},
    )


def _kill_switch_panel(trades_df: pd.DataFrame) -> html.Div:
    """Build kill-switch status panel."""
    rules = []
    triggered = False

    if trades_df.empty:
        pnls = []
    else:
        pnls = trades_df["pnl"].dropna().tolist()

    # Rule 1: Total loss > 5% of $25k = $1,250
    total_pnl = sum(pnls)
    r1 = total_pnl < -1250
    rules.append(("Total loss > 5% ($1,250)", f"${total_pnl:+,.0f}", not r1))
    if r1: triggered = True

    # Rule 2: Rolling 20-trade Sharpe < 0.3
    sharpe20 = 0.0
    if len(pnls) >= 20:
        s = np.std(pnls[:20])
        sharpe20 = float(np.mean(pnls[:20]) / s) if s > 1e-9 else 0.0
    r2 = sharpe20 < 0.3 and len(pnls) >= 20
    rules.append(("Rolling 20-trade Sharpe < 0.3",
                  f"{sharpe20:.3f}" if len(pnls) >= 20 else "N/A", not r2))
    if r2: triggered = True

    # Rule 3: 0/10 wins
    if len(pnls) >= 10:
        r3 = all(p <= 0 for p in pnls[:10])
    else:
        r3 = False
    rules.append(("0 wins in last 10 trades",
                  f"{sum(1 for p in pnls[:10] if p > 0)}/10" if len(pnls) >= 10 else "N/A",
                  not r3))
    if r3: triggered = True

    overall_color = RED if triggered else GREEN
    overall_label = "TRIGGERED" if triggered else "OK"

    rule_rows = []
    for label, value, ok in rules:
        dot = GREEN if ok else RED
        rule_rows.append(html.Div([
            html.Span("● " if ok else "X ", style={"color": dot, "fontWeight": "700", "fontSize": "0.85rem"}),
            html.Span(label, style={"fontSize": "0.75rem", "color": TEXT}),
            html.Span(f"  {value}", style={"fontSize": "0.75rem", "color": dot,
                                           "fontFamily": "monospace", "marginLeft": "8px"}),
        ], style={"marginBottom": "6px"}))

    return html.Div([
        html.Div([
            html.Span("KILL-SWITCH ", style={"color": TEXT_DIM, "fontSize": "0.7rem", "letterSpacing": "0.1em"}),
            html.Span(overall_label, style={"color": overall_color, "fontWeight": "900",
                                            "fontSize": "1.1rem", "fontFamily": "monospace"}),
        ], style={"textAlign": "center", "marginBottom": "14px",
                  "border": f"2px solid {overall_color}", "borderRadius": "8px", "padding": "10px"}),
        html.Div(rule_rows),
    ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
              "borderRadius": "8px", "padding": "16px", "height": "100%"})


def _position_health_table(snap_df: pd.DataFrame) -> html.Div:
    """Sprint 3.6: Position Health table showing current holdings with P&L."""
    if snap_df.empty:
        return html.Div(
            "No snapshot data",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "30px",
                   "fontSize": "0.9rem", "fontFamily": "monospace"},
        )

    latest = snap_df.iloc[-1]
    positions_json = latest.get("positions_json", "[]")

    try:
        raw = json.loads(positions_json) if positions_json else {}
    except (json.JSONDecodeError, TypeError):
        raw = {}

    # positions_json is a dict: {"TICKER": {direction, shares, entry, current, value}}
    # Normalize to list of dicts with "ticker" key
    if isinstance(raw, dict):
        positions = [{"ticker": k, **v} for k, v in raw.items()]
    elif isinstance(raw, list):
        positions = raw
    else:
        positions = []

    if not positions:
        return html.Div(
            "No open positions",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "30px",
                   "fontSize": "0.9rem", "fontFamily": "monospace"},
        )

    rows = []
    for pos in positions:
        ticker = pos.get("ticker", "?")
        direction = pos.get("direction", "LONG")
        entry_price = pos.get("entry", pos.get("entry_price", 0))
        current_price = pos.get("current", pos.get("current_price", entry_price))
        shares = pos.get("shares", 0)

        # Compute unrealized P&L
        if direction == "LONG":
            unrealized = shares * (current_price - entry_price)
        else:
            unrealized = shares * (entry_price - current_price)

        notional = shares * current_price
        unrealized_pct = (unrealized / (shares * entry_price)) * 100 if entry_price > 0 and shares > 0 else 0

        rows.append({
            "TICKER": ticker,
            "DIR": direction[:1],
            "ENTRY": f"${entry_price:,.2f}",
            "CURRENT": f"${current_price:,.2f}",
            "SHARES": f"{shares:.1f}",
            "NOTIONAL": f"${notional:,.0f}",
            "P&L": f"${unrealized:+,.2f}",
            "P&L%": f"{unrealized_pct:+.1f}%",
            "SECTOR": pos.get("sector", ""),
        })

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto", "maxHeight": "300px", "backgroundColor": PANEL},
        style_header={"backgroundColor": BORDER, "color": TEXT_DIM,
                      "fontWeight": "600", "fontSize": "0.72rem",
                      "textTransform": "uppercase", "letterSpacing": "0.05em",
                      "border": f"1px solid {BORDER}"},
        style_cell={"backgroundColor": PANEL, "color": TEXT,
                    "border": f"1px solid {BORDER}",
                    "fontSize": "0.73rem", "fontFamily": "monospace",
                    "padding": "5px 8px", "textAlign": "right"},
        style_data_conditional=[
            {"if": {"filter_query": "{DIR} = L"}, "color": GREEN},
            {"if": {"filter_query": "{DIR} = S"}, "color": PURPLE},
            # Highlight losses
            {"if": {"column_id": "P&L", "filter_query": '{P&L} contains "-"'},
             "color": RED},
            {"if": {"column_id": "P&L%", "filter_query": '{P&L%} contains "-"'},
             "color": RED},
        ],
        page_size=15,
    )


def _signal_frequency_card(all_df: pd.DataFrame) -> html.Div:
    """Sprint 3.6: Signal frequency analysis for the last 7 days."""
    if all_df.empty:
        return html.Div(
            "No trades for frequency analysis",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "20px",
                   "fontSize": "0.85rem", "fontFamily": "monospace"},
        )

    # Parse timestamps
    all_df = all_df.copy()
    all_df["ts"] = pd.to_datetime(all_df["timestamp"], errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    last_7d = all_df[all_df["ts"] >= now - pd.Timedelta(days=7)]

    entries_7d = len(last_7d[last_7d["action"] == "ENTER"])
    exits_7d = len(last_7d[last_7d["action"] == "EXIT"])

    # Expected frequency (from backtest: ~302 trades / 8 windows / ~90 days per window)
    # ≈ 0.42 trades/day → ~2.9 per week. Round to ~3 per week for entries.
    expected_entries_per_week = 3.0

    freq_ratio = entries_7d / expected_entries_per_week if expected_entries_per_week > 0 else 0
    freq_deviation = abs(freq_ratio - 1.0) * 100

    if freq_deviation > 20:
        status_color = YELLOW
        status_text = f"DEVIANT ({freq_deviation:.0f}% off)"
    else:
        status_color = GREEN
        status_text = f"NORMAL ({freq_deviation:.0f}% off)"

    items = [
        html.Div(
            "SIGNAL FREQUENCY (7D)",
            style={"color": TEXT_DIM, "fontSize": "0.72rem", "letterSpacing": "0.1em",
                   "marginBottom": "10px"},
        ),
        html.Div([
            html.Span(f"Entries: {entries_7d}", style={"color": TEXT, "fontSize": "0.85rem"}),
            html.Span(f"  |  Exits: {exits_7d}", style={"color": TEXT_DIM, "fontSize": "0.85rem"}),
        ]),
        html.Div([
            html.Span(f"Expected: ~{expected_entries_per_week:.0f}/week", style={"color": TEXT_DIM, "fontSize": "0.8rem"}),
        ], style={"marginTop": "4px"}),
        html.Div([
            html.Span(f"Status: ", style={"color": TEXT_DIM, "fontSize": "0.85rem"}),
            html.Span(status_text, style={"color": status_color, "fontWeight": "700",
                                           "fontSize": "0.85rem"}),
        ], style={"marginTop": "8px"}),
    ]

    return html.Div(items, style={
        "backgroundColor": PANEL, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "16px",
    })


def _recent_trades_table(all_df: pd.DataFrame) -> html.Div:
    """Build recent trades table (last 15 trades)."""
    if all_df.empty:
        return html.Div(
            "No trades yet",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "30px",
                   "fontSize": "0.9rem", "fontFamily": "monospace"},
        )

    display = all_df.head(15).copy()
    cols = ["timestamp", "ticker", "action", "direction", "shares", "price", "notional", "pnl"]
    available = [c for c in cols if c in display.columns]
    display = display[available]

    # Format
    if "timestamp" in display.columns:
        display["timestamp"] = pd.to_datetime(display["timestamp"], errors="coerce").dt.strftime("%m/%d %H:%M")
    if "price" in display.columns:
        display["price"] = display["price"].apply(lambda v: f"${v:,.2f}")
    if "notional" in display.columns:
        display["notional"] = display["notional"].apply(lambda v: f"${v:,.0f}")
    if "shares" in display.columns:
        display["shares"] = display["shares"].apply(lambda v: f"{v:.2f}")
    if "pnl" in display.columns:
        display["pnl"] = display["pnl"].apply(
            lambda v: f"${v:+,.2f}" if pd.notna(v) else "—"
        )

    display.columns = [c.upper() for c in display.columns]

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        style_table={"overflowX": "auto", "maxHeight": "350px", "backgroundColor": PANEL},
        style_header={"backgroundColor": BORDER, "color": TEXT_DIM,
                      "fontWeight": "600", "fontSize": "0.72rem",
                      "textTransform": "uppercase", "letterSpacing": "0.05em",
                      "border": f"1px solid {BORDER}"},
        style_cell={"backgroundColor": PANEL, "color": TEXT,
                    "border": f"1px solid {BORDER}",
                    "fontSize": "0.73rem", "fontFamily": "monospace",
                    "padding": "5px 8px", "textAlign": "right"},
        style_data_conditional=[
            {"if": {"filter_query": "{ACTION} = ENTER"}, "color": GREEN},
            {"if": {"filter_query": "{ACTION} = EXIT"}, "color": ORANGE},
            {"if": {"filter_query": "{ACTION} = SCALE_UP"}, "color": CYAN},
            {"if": {"filter_query": "{DIRECTION} = SHORT"}, "color": PURPLE},
        ],
        page_size=15,
    )


# ── Dash App ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="CITRINE — Portfolio Dashboard",
)

# Prevent browser from caching stale dashboard pages
@app.server.after_request
def _no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.layout = html.Div([
    dcc.Interval(id="refresh-interval", interval=60_000, n_intervals=0),
    html.Div(id="header-container"),
    html.Div(id="body-container",
             style={"padding": "20px", "backgroundColor": BG, "minHeight": "100vh"}),
], style={"backgroundColor": BG, "fontFamily": "monospace"})


@app.callback(
    [Output("header-container", "children"),
     Output("body-container", "children")],
    [Input("refresh-interval", "n_intervals")],
)
def update_dashboard(_n):
    """Rebuild entire dashboard on each interval tick."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    trades_df = _load_trades()
    all_trades_df = _load_all_trades()
    snap_df = _load_snapshots()

    citrine_sc, citrine_stage = _citrine_score()

    # ── Header ─────────────────────────────────────────────────────────────
    header = html.Div([
        html.Div([
            html.Span("CITRINE ",
                       style={"color": ORANGE, "fontWeight": "900",
                              "fontSize": "1.2rem", "letterSpacing": "0.05em"}),
            html.Span("— Portfolio Rotation Dashboard",
                       style={"color": TEXT, "fontWeight": "300", "fontSize": "1.0rem"}),
        ]),
        html.Div([
            html.Span(f"Last refresh: {now_str}",
                       style={"color": TEXT_DIM, "fontSize": "0.68rem",
                              "letterSpacing": "0.08em"}),
            html.Span(f"   |   Maturity: {citrine_sc}% ({citrine_stage})",
                       style={"color": TEXT_DIM, "fontSize": "0.68rem"}),
            html.Span("   |   Auto-refresh: 60s",
                       style={"color": TEXT_DIM, "fontSize": "0.68rem"}),
        ]),
    ], style={
        "backgroundColor": PANEL,
        "borderBottom": f"1px solid {BORDER}",
        "padding": "14px 24px",
        "display": "flex", "flexDirection": "column",
    })

    # ── Top row: 5 metric cards ────────────────────────────────────────────
    if not snap_df.empty:
        latest = snap_df.iloc[-1]
        equity = latest["total_equity"]
        pnl_total = equity - 25000
        pnl_color = GREEN if pnl_total >= 0 else RED
        num_pos = latest["num_positions"]
        bull_count = latest.get("bull_count", 0)
        # Use actual cash / equity (not allocator target which can be misleading)
        cash_actual = latest.get("cash", equity)
        cash_pct_val = cash_actual / equity if equity > 0 else 1.0
    else:
        equity = 25000
        pnl_total = 0
        pnl_color = TEXT_DIM
        num_pos = 0
        bull_count = 0
        cash_pct_val = 1.0

    closed_count = len(trades_df)
    win_rate = 0
    if closed_count > 0:
        win_rate = (trades_df["pnl"] > 0).sum() / closed_count * 100

    top_row = dbc.Row([
        dbc.Col(_metric_card("Total Equity", f"${equity:,.0f}", pnl_color), md=2, sm=4),
        dbc.Col(_metric_card("Total P&L", f"${pnl_total:+,.0f}", pnl_color), md=2, sm=4),
        dbc.Col(_metric_card("Positions", str(num_pos), BLUE), md=2, sm=4),
        dbc.Col(_metric_card("Win Rate", f"{win_rate:.0f}%" if closed_count > 0 else "N/A",
                             GREEN if win_rate > 50 else (YELLOW if win_rate > 40 else RED)),
                md=2, sm=4),
        dbc.Col(_metric_card("BULL Tickers", str(bull_count), GREEN), md=2, sm=4),
        dbc.Col(_metric_card("Cash", f"{cash_pct_val:.0%}", BLUE), md=2, sm=4),
    ], className="g-2", style={"marginBottom": "8px"})

    # ── 2nd row: 5-day session P&L history ──────────────────────────────
    # Each tile shows one trading day's P&L (Day 0 = most recent started session)
    # Weekends and US market holidays are excluded — only real trading days shown.
    day_cards = []
    if not snap_df.empty:
        snap_df["_date"] = pd.to_datetime(snap_df["timestamp"]).dt.date
        unique_dates = sorted(snap_df["_date"].unique())

        # Filter to trading days only (exclude weekends + US market holidays)
        trading_dates = [d for d in unique_dates if _is_trading_day(d)]

        # Build per-day equity: last snapshot equity for each trading date
        day_equities = []
        for d in trading_dates:
            day_snap = snap_df[snap_df["_date"] == d].iloc[-1]
            day_equities.append((d, day_snap["total_equity"]))

        # Compute daily P&L for up to last 10 trading days
        # Day 0 uses previous day close (or $25k start) as baseline
        num_days = min(len(day_equities), 10)
        for i in range(num_days):
            idx = len(day_equities) - num_days + i  # index into day_equities
            d, eq = day_equities[idx]
            if idx == 0:
                prev_eq = 25000  # first day ever — baseline is starting capital
            else:
                prev_eq = day_equities[idx - 1][1]

            d_pnl = eq - prev_eq
            d_pct = (d_pnl / prev_eq) * 100 if prev_eq > 0 else 0
            d_color = GREEN if d_pnl >= 0 else RED
            day_offset = num_days - 1 - i  # 0 = most recent
            d_label = f"Day {-day_offset}" if day_offset > 0 else "Day 0 (today)"
            d_sublabel = d.strftime("%m/%d")
            d_str = f"${d_pnl:+,.0f} ({d_pct:+.2f}%)"

            day_cards.append(
                dbc.Col(
                    _metric_card(f"{d_label}  {d_sublabel}", d_str, d_color),
                    md=True, sm=4,
                )
            )

        snap_df.drop(columns=["_date"], inplace=True, errors="ignore")

    if not day_cards:
        # No data yet — show placeholder
        for i in range(10):
            offset = -(9 - i)
            label = f"Day {offset}" if offset < 0 else "Day 0 (today)"
            day_cards.append(
                dbc.Col(_metric_card(label, "—", TEXT_DIM), md=True, sm=4)
            )

    # Pad to exactly 10 columns if fewer days exist
    while len(day_cards) < 10:
        day_cards.insert(0, dbc.Col(_metric_card("—", "—", TEXT_DIM), md=True, sm=4))

    day_row = dbc.Row(day_cards, className="g-2", style={"marginBottom": "16px"})

    # ── Mid row: equity + allocation pie ───────────────────────────────────
    equity_fig = _build_equity_curve(snap_df)
    pie_fig = _build_positions_pie(snap_df)

    mid_row = dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(figure=equity_fig, config={"displayModeBar": False}),
        ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "8px"}), md=7),
        dbc.Col(html.Div([
            dcc.Graph(figure=pie_fig, config={"displayModeBar": False}),
        ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "8px"}), md=5),
    ], className="g-2", style={"marginBottom": "16px"})

    # ── Bottom row: regime bar + P&L dist + kill-switch ────────────────────
    regime_fig = _build_regime_heatmap(snap_df)
    pnl_fig = _build_pnl_distribution(trades_df)

    bottom_charts = dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(figure=regime_fig, config={"displayModeBar": False}),
        ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "8px"}), md=4),
        dbc.Col(html.Div([
            dcc.Graph(figure=pnl_fig, config={"displayModeBar": False}),
        ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "8px"}), md=4),
        dbc.Col(_kill_switch_panel(trades_df), md=4),
    ], className="g-2", style={"marginBottom": "16px"})

    # ── Position Health + Signal Frequency (Sprint 3.6) ─────────────────
    health_row = dbc.Row([
        dbc.Col(html.Div([
            html.H6("POSITION HEALTH", style={"color": TEXT_DIM, "letterSpacing": "0.1em",
                                                "fontSize": "0.72rem", "marginBottom": "8px"}),
            _position_health_table(snap_df),
        ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "16px"}), md=8),
        dbc.Col(_signal_frequency_card(all_trades_df), md=4),
    ], className="g-2", style={"marginBottom": "16px"})

    # ── Trades table ───────────────────────────────────────────────────────
    trades_section = html.Div([
        html.H6("RECENT TRADES", style={"color": TEXT_DIM, "letterSpacing": "0.1em",
                                         "fontSize": "0.72rem", "marginBottom": "8px"}),
        _recent_trades_table(all_trades_df),
    ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
              "borderRadius": "8px", "padding": "16px"})

    body = html.Div([top_row, day_row, mid_row, bottom_charts, health_row, trades_section])
    return header, body


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--host", default="0.0.0.0")
    _p.add_argument("--port", type=int, default=8070)
    _args = _p.parse_args()
    print(f"CITRINE Portfolio Dashboard -- http://{_args.host}:{_args.port}")
    app.run(
        host=_args.host,
        port=_args.port,
        debug=False,
        use_reloader=False,
    )
