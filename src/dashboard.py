"""
dashboard.py
────────────
Plotly-Dash trading terminal.

Layout (dark theme)
────────────────────
  ┌──────────────────────────────────────────────────────────────────────┐
  │  HEADER  BTC/USD HMM Regime Trader   [Regime Badge]  [Confidence]   │
  ├──────────────────┬───────────────────────────────────────────────────┤
  │  SIGNAL PANEL    │  PRICE CHART + regime-coloured background         │
  │  (confirmations) │                                                   │
  ├──────────────────┴───────────────────────────────────────────────────┤
  │  EQUITY CURVE vs BUY-AND-HOLD                                        │
  ├──────────────────┬───────────────────────────────────────────────────┤
  │  METRICS CARDS   │  TRADE LOG TABLE                                  │
  └──────────────────┴───────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)

# Colour palette
BG        = "#0d0f14"
PANEL     = "#141820"
BORDER    = "#1e2330"
TEXT      = "#e0e4f0"
TEXT_DIM  = "#6b7394"
GREEN     = "#00e676"
RED       = "#ff1744"
YELLOW    = "#ffea00"
BLUE      = "#448aff"
PURPLE    = "#e040fb"
ORANGE    = "#ff6d00"


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _regime_color(cat: str) -> str:
    return {"BULL": GREEN, "BEAR": RED, "CHOP": YELLOW}.get(cat, TEXT_DIM)


def build_price_chart(df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    """Candlestick chart with regime-shaded background and trade markers."""
    fig = go.Figure()

    # ── Regime background bands ───────────────────────────────────────────
    # Rendered as 3 filled Scatter traces on a paper-coordinate y-axis so
    # we never call add_vrect() (thousands of shape objects crash Plotly 6.x).
    if "regime_cat" in df.columns:
        _BAND_COLOR = {
            "BULL": "rgba(0,230,118,0.06)",
            "BEAR": "rgba(255,23,68,0.07)",
            "CHOP": "rgba(255,234,0,0.04)",
        }
        # Compute change-point indices once with vectorised diff
        regime_s = df["regime_cat"]
        change_mask = regime_s != regime_s.shift()
        starts = df.index[change_mask].tolist()
        cats   = regime_s[change_mask].tolist()
        ends   = starts[1:] + [df.index[-1]]

        # Accumulate polygon vertices per category (None = pen-up)
        px: dict[str, list] = {"BULL": [], "BEAR": [], "CHOP": []}
        py: dict[str, list] = {"BULL": [], "BEAR": [], "CHOP": []}
        for cat, t0, t1 in zip(cats, starts, ends):
            bucket = px.get(cat)
            if bucket is None:
                continue
            px[cat] += [t0, t1, t1, t0, t0, None]
            py[cat] += [0,  0,  1,  1,  0,  None]

        for cat, color in _BAND_COLOR.items():
            if not px[cat]:
                continue
            fig.add_trace(go.Scatter(
                x=px[cat], y=py[cat],
                fill="toself",
                fillcolor=color,
                line=dict(width=0),
                yaxis="y2",
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── Candlesticks ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        name="BTC/USD",
    ))

    # ── SMA 50 ────────────────────────────────────────────────────────────
    sma_col = f"sma_{config.TREND_MA_PERIOD}"
    if sma_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[sma_col],
            line=dict(color=BLUE, width=1, dash="dot"),
            name=f"SMA {config.TREND_MA_PERIOD}",
        ))

    # ── Trade entry / exit markers ────────────────────────────────────────
    if not trades_df.empty:
        has_direction = "direction" in trades_df.columns

        if has_direction:
            # Long entries: green ▲
            long_entries = trades_df[
                (trades_df["direction"] == "LONG") &
                trades_df["entry_time"].notna()
            ]
            if not long_entries.empty:
                fig.add_trace(go.Scatter(
                    x=long_entries["entry_time"], y=long_entries["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color=GREEN,
                                line=dict(color="white", width=1)),
                    name="Long Entry",
                ))

            # Short entries: red ▼
            short_entries = trades_df[
                (trades_df["direction"] == "SHORT") &
                trades_df["entry_time"].notna()
            ]
            if not short_entries.empty:
                fig.add_trace(go.Scatter(
                    x=short_entries["entry_time"], y=short_entries["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color=PURPLE,
                                line=dict(color="white", width=1)),
                    name="Short Entry",
                ))
        else:
            # Legacy: all entries are long
            entries = trades_df.dropna(subset=["entry_time"])
            fig.add_trace(go.Scatter(
                x=entries["entry_time"], y=entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color=GREEN,
                            line=dict(color="white", width=1)),
                name="Entry",
            ))

        # Exit markers (same for both modes)
        exits = trades_df.dropna(subset=["exit_time"])
        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=exits["exit_time"], y=exits["exit_price"],
                mode="markers",
                marker=dict(symbol="x", size=10, color=RED,
                            line=dict(color="white", width=1)),
                name="Exit",
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        height=420,
        title=dict(text="BTC/USD — HMM Regime Overlay", font=dict(size=14)),
        yaxis2=dict(
            overlaying="y",
            range=[0, 1],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            fixedrange=True,
        ),
    )
    return fig


def build_equity_chart(
    equity: pd.Series,
    bh_equity: pd.Series,
    trades_df: pd.DataFrame,
) -> go.Figure:
    """Equity curve vs buy-and-hold with drawdown shading."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # ── Equity curve ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        line=dict(color=GREEN, width=2),
        name="Strategy",
        fill="tozeroy",
        fillcolor="rgba(0,230,118,0.08)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bh_equity.index, y=bh_equity,
        line=dict(color=BLUE, width=1.5, dash="dash"),
        name="Buy & Hold",
    ), row=1, col=1)

    # ── Drawdown ──────────────────────────────────────────────────────────
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max * 100

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        line=dict(color=RED, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,23,68,0.15)",
        name="Drawdown %",
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        height=350,
        title=dict(text="Equity Curve vs Buy & Hold", font=dict(size=14)),
    )
    fig.update_yaxes(title_text="USD", row=1, col=1,
                     tickprefix="$", gridcolor=BORDER)
    fig.update_yaxes(title_text="DD %", row=2, col=1,
                     ticksuffix="%", gridcolor=BORDER)
    fig.update_xaxes(gridcolor=BORDER, showgrid=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metric cards
# ─────────────────────────────────────────────────────────────────────────────

def _card(title: str, value: str, color: str = TEXT) -> dbc.Card:
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


def build_metric_cards(metrics: dict) -> list:
    def fmt(v, suffix="", prefix=""):
        if v is None:
            return "—"
        return f"{prefix}{v:,.2f}{suffix}"

    ret_color  = GREEN if metrics.get("total_return_pct", 0) >= 0 else RED
    alpha_col  = GREEN if metrics.get("alpha_pct", 0) >= 0 else RED
    dd_color   = RED   if metrics.get("max_drawdown_pct", 0) < -10 else YELLOW

    cards = [
        _card("Final Equity",      fmt(metrics.get("final_equity"), prefix="$"), GREEN),
        _card("Total Return",      fmt(metrics.get("total_return_pct"), suffix="%"), ret_color),
        _card("vs Buy & Hold (α)", fmt(metrics.get("alpha_pct"), suffix="%"), alpha_col),
        _card("Max Drawdown",      fmt(metrics.get("max_drawdown_pct"), suffix="%"), dd_color),
        _card("Sharpe Ratio",      fmt(metrics.get("sharpe_ratio")), BLUE),
        _card("Win Rate",          fmt(metrics.get("win_rate_pct"), suffix="%"),
              GREEN if (metrics.get("win_rate_pct") or 0) >= 50 else RED),
        _card("Total Trades",      str(metrics.get("n_trades", 0)), TEXT),
        _card("Profit Factor",     fmt(metrics.get("profit_factor")),
              GREEN if (metrics.get("profit_factor") or 0) >= 1 else RED),
    ]

    # Direction breakdown (multi-direction mode)
    long_t  = metrics.get("long_trades", 0)
    short_t = metrics.get("short_trades", 0)
    if long_t + short_t > 0:
        long_wr  = metrics.get("long_win_rate_pct", 0)
        short_wr = metrics.get("short_win_rate_pct", 0)
        cards.append(_card(
            "Long / Short",
            f"{long_t}L ({long_wr:.0f}%) / {short_t}S ({short_wr:.0f}%)",
            BLUE,
        ))

    return cards


# ─────────────────────────────────────────────────────────────────────────────
# Signal panel
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_panel(
    latest_row:  pd.Series,
    regime:      str,
    regime_cat:  str,
    confidence:  float,
    use_regime_mapper: bool = False,
) -> html.Div:
    reg_color = _regime_color(regime_cat)

    # Determine which check columns to display
    has_direction_checks = "check_momentum_long" in latest_row.index

    if has_direction_checks and use_regime_mapper:
        # Multi-direction: show the relevant direction's checks
        allowed_dir = latest_row.get("allowed_direction", "FLAT")

        if allowed_dir in ("LONG", "LONG_OR_FLAT"):
            checks = {
                "RSI":         latest_row.get("check_rsi",              False),
                "Momentum ▲":  latest_row.get("check_momentum_long",    False),
                "Volatility":  latest_row.get("check_volatility",       False),
                "Volume":      latest_row.get("check_volume",           False),
                "ADX":         latest_row.get("check_adx",              False),
                "Trend ▲":     latest_row.get("check_price_trend_long", False),
                "MACD ▲":      latest_row.get("check_macd_long",        False),
                "Stoch ▲":     latest_row.get("check_stochastic_long",  False),
            }
        elif allowed_dir in ("SHORT", "SHORT_OR_FLAT"):
            checks = {
                "RSI":         latest_row.get("check_rsi",               False),
                "Momentum ▼":  latest_row.get("check_momentum_short",    False),
                "Volatility":  latest_row.get("check_volatility",        False),
                "Volume":      latest_row.get("check_volume",            False),
                "ADX":         latest_row.get("check_adx",               False),
                "Trend ▼":     latest_row.get("check_price_trend_short", False),
                "MACD ▼":      latest_row.get("check_macd_short",        False),
                "Stoch ▼":     latest_row.get("check_stochastic_short",  False),
            }
        else:
            checks = {}   # FLAT — no checks needed
    else:
        # Legacy: long-only checks
        checks = {
            "RSI":         latest_row.get("check_rsi",         False),
            "Momentum":    latest_row.get("check_momentum",    False),
            "Volatility":  latest_row.get("check_volatility",  False),
            "Volume":      latest_row.get("check_volume",      False),
            "ADX":         latest_row.get("check_adx",         False),
            "Price Trend": latest_row.get("check_price_trend", False),
            "MACD":        latest_row.get("check_macd",        False),
            "Stochastic":  latest_row.get("check_stochastic",  False),
        }

    count = sum(checks.values()) if checks else 0
    action_color = GREEN if count >= config.MIN_CONFIRMATIONS else (
                   YELLOW if count >= config.MIN_CONFIRMATIONS - 1 else RED)

    if use_regime_mapper and has_direction_checks:
        allowed_dir = latest_row.get("allowed_direction", "FLAT")
        if allowed_dir == "FLAT":
            signal_label = "FLAT"
            action_color = YELLOW
        elif allowed_dir in ("LONG", "LONG_OR_FLAT") and count >= config.MIN_CONFIRMATIONS:
            signal_label = "LONG SIGNAL"
            action_color = GREEN
        elif allowed_dir in ("SHORT", "SHORT_OR_FLAT") and count >= config.MIN_CONFIRMATIONS_SHORT:
            signal_label = "SHORT SIGNAL"
            action_color = PURPLE
        else:
            signal_label = f"{count}/{config.TOTAL_SIGNALS} CONFIRMS"
    else:
        signal_label = (
            "BUY  SIGNAL" if regime_cat == "BULL" and count >= config.MIN_CONFIRMATIONS
            else "SELL SIGNAL" if regime_cat == "BEAR"
            else f"{count}/{config.TOTAL_SIGNALS} CONFIRMS"
        )

    rows = []
    for name, ok in checks.items():
        rows.append(html.Div([
            html.Span("● " if ok else "○ ",
                      style={"color": GREEN if ok else TEXT_DIM, "fontSize": "0.8rem"}),
            html.Span(name, style={"fontSize": "0.78rem", "color": TEXT if ok else TEXT_DIM}),
        ], style={"marginBottom": "3px"}))

    # Current indicator values (compact)
    vals = html.Div([
        html.Span(f"RSI {latest_row.get('rsi', 0):.1f}  "
                  f"ADX {latest_row.get('adx', 0):.1f}  "
                  f"Vol×{latest_row.get('volume_ratio', 0):.2f}",
                  style={"fontSize": "0.7rem", "color": TEXT_DIM,
                         "fontFamily": "monospace"}),
    ], style={"marginTop": "8px"})

    return html.Div([
        # Regime badge
        html.Div([
            html.Span("REGIME ", style={"color": TEXT_DIM, "fontSize": "0.7rem",
                                         "letterSpacing": "0.1em"}),
            html.Span(regime.upper().replace("_", " "),
                      style={"color": reg_color, "fontWeight": "700",
                             "fontSize": "0.85rem", "fontFamily": "monospace"}),
        ], style={"marginBottom": "4px"}),

        # Confidence bar
        html.Div([
            html.Span("CONF ", style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
            html.Div(style={
                "display": "inline-block",
                "width": f"{confidence*100:.0f}%",
                "maxWidth": "100px",
                "height": "8px",
                "backgroundColor": reg_color,
                "borderRadius": "4px",
                "verticalAlign": "middle",
                "marginLeft": "6px",
                "marginRight": "6px",
            }),
            html.Span(f"{confidence*100:.1f}%",
                      style={"color": reg_color, "fontSize": "0.75rem",
                             "fontFamily": "monospace"}),
        ], style={"marginBottom": "12px"}),

        # Signal action
        html.Div(signal_label, style={
            "color": action_color,
            "fontSize": "1.1rem",
            "fontWeight": "900",
            "fontFamily": "monospace",
            "letterSpacing": "0.05em",
            "marginBottom": "10px",
            "textAlign": "center",
            "border": f"1px solid {action_color}",
            "borderRadius": "6px",
            "padding": "6px 10px",
        }),

        # Indicator checklist
        html.Div(rows, style={"marginBottom": "4px"}),
        vals,
    ], style={
        "backgroundColor": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px",
        "minWidth": "200px",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Trade log table
# ─────────────────────────────────────────────────────────────────────────────

def build_trade_table(trades_df: pd.DataFrame) -> dash.dash_table.DataTable:
    from dash import dash_table

    has_direction = "direction" in trades_df.columns if not trades_df.empty else False
    base_cols = ["entry_time", "exit_time", "entry_price", "exit_price",
                 "pnl", "pnl_pct", "duration_h", "exit_reason"]
    if has_direction:
        base_cols.insert(0, "direction")

    if trades_df.empty:
        display_df = pd.DataFrame(columns=base_cols)
    else:
        display_df = trades_df[base_cols].copy()
        display_df["entry_time"] = display_df["entry_time"].dt.strftime("%m/%d %H:%M")
        display_df["exit_time"]  = display_df["exit_time"].dt.strftime("%m/%d %H:%M")
        display_df["pnl"]        = display_df["pnl"].apply(lambda v: f"${v:+,.0f}")
        display_df["pnl_pct"]    = display_df["pnl_pct"].apply(lambda v: f"{v:+.2f}%")
        display_df["duration_h"] = display_df["duration_h"].apply(lambda v: f"{v:.1f}h")
        display_df["entry_price"] = display_df["entry_price"].apply(lambda v: f"${v:,.0f}")
        display_df["exit_price"]  = display_df["exit_price"].apply(lambda v: f"${v:,.0f}")
        if has_direction:
            display_df.columns = ["Dir", "Entry", "Exit", "Entry $", "Exit $",
                                   "P&L", "P&L %", "Duration", "Reason"]
        else:
            display_df.columns = ["Entry", "Exit", "Entry $", "Exit $",
                                   "P&L", "P&L %", "Duration", "Reason"]

    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display_df.columns],
        style_table={"overflowX": "auto", "overflowY": "auto",
                     "maxHeight": "320px", "backgroundColor": PANEL},
        style_header={"backgroundColor": BORDER, "color": TEXT_DIM,
                      "fontWeight": "600", "fontSize": "0.72rem",
                      "textTransform": "uppercase", "letterSpacing": "0.05em",
                      "border": f"1px solid {BORDER}"},
        style_cell={"backgroundColor": PANEL, "color": TEXT,
                    "border": f"1px solid {BORDER}",
                    "fontSize": "0.75rem", "fontFamily": "monospace",
                    "padding": "6px 10px", "textAlign": "right"},
        style_data_conditional=[
            {"if": {"filter_query": "{P&L} contains '+'"},
             "color": GREEN},
            {"if": {"filter_query": "{P&L} contains '-'"},
             "color": RED},
        ],
        sort_action="native",
        page_size=20,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance chart (populated from feature_importance.json if present)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_importance_chart() -> "go.Figure | None":
    """
    Load feature_importance.json (written by optimize.py) and return a bar
    chart.  Returns None if the file does not exist yet.
    """
    import json
    from pathlib import Path

    imp_path = Path(__file__).parent.parent / "feature_importance.json"
    if not imp_path.exists():
        return None

    try:
        with open(imp_path) as f:
            data = json.load(f)
    except Exception:
        return None

    if not data:
        return None

    labels = list(data.keys())
    values = [data[k] for k in labels]
    colors = [GREEN if v >= max(values) * 0.5 else BLUE for v in values]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=60),
        height=260,
        title=dict(
            text="HMM Feature Importance — Regime Separation (top-10 OOS runs)",
            font=dict(size=13),
        ),
        yaxis=dict(tickformat=".0%", gridcolor=BORDER),
        xaxis=dict(gridcolor=BORDER),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(result, use_regime_mapper: bool = False) -> dash.Dash:
    """
    Build and return the Dash app given a BacktestResult.

    Parameters
    ----------
    result : BacktestResult  (from backtester.py)
    use_regime_mapper : bool  (if True, show multi-direction UI)
    """
    from src.backtester import BacktestResult   # avoid circular at module level

    df       = result.df
    equity   = result.equity_curve
    trades   = result.trades
    metrics  = result.metrics
    bh_eq    = metrics.get("bh_equity", pd.Series(dtype=float))

    latest      = df.iloc[-1]
    regime      = latest.get("regime",     "unknown")
    regime_cat  = latest.get("regime_cat", "CHOP")
    confidence  = float(latest.get("confidence", 0.0))

    # ── Dash layout ────────────────────────────────────────────────────────
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="HMM Regime Trader",
    )

    price_fig   = build_price_chart(df, trades)
    equity_fig  = build_equity_chart(equity, bh_eq, trades)
    signal_div  = build_signal_panel(latest, regime, regime_cat, confidence,
                                     use_regime_mapper=use_regime_mapper)
    trade_tbl   = build_trade_table(trades)
    metric_cds  = build_metric_cards(metrics)
    imp_fig     = build_feature_importance_chart()   # None if JSON not yet written

    app.layout = html.Div([

        # ── Header ─────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span(f"{config.TICKER.replace('X:', '')} ",
                          style={"color": ORANGE, "fontWeight": "900",
                                 "fontSize": "1.2rem", "letterSpacing": "0.05em"}),
                html.Span("HMM REGIME TRADER",
                           style={"color": TEXT, "fontWeight": "300", "fontSize": "1.0rem"}),
                *(
                    [html.Span("  MULTI-DIR",
                               style={"color": PURPLE, "fontWeight": "700",
                                      "fontSize": "0.75rem", "marginLeft": "12px",
                                      "border": f"1px solid {PURPLE}",
                                      "borderRadius": "4px", "padding": "2px 6px"})]
                    if use_regime_mapper else []
                ),
            ]),
            html.Div([
                html.Span(f"{config.N_STATES}-STATE GAUSSIAN HMM  ·  {config.LEVERAGE}× LEVERAGE  ·  {config.MIN_CONFIRMATIONS}/{config.TOTAL_SIGNALS} CONFIRMATIONS"
                          + (f" (SHORT: {config.MIN_CONFIRMATIONS_SHORT}/{config.TOTAL_SIGNALS})"
                             if use_regime_mapper and config.MIN_CONFIRMATIONS_SHORT != config.MIN_CONFIRMATIONS else "")
                          + f"  ·  {config.REGIME_CONFIDENCE_MIN*100:.0f}% CONF GATE",
                           style={"color": TEXT_DIM, "fontSize": "0.68rem",
                                  "letterSpacing": "0.08em"}),
            ]),
        ], style={
            "backgroundColor": PANEL,
            "borderBottom": f"1px solid {BORDER}",
            "padding": "14px 24px",
            "display": "flex",
            "flexDirection": "column",
        }),

        # ── Body ────────────────────────────────────────────────────────────
        html.Div([

            # Row 1: signal panel + price chart
            html.Div([
                html.Div(signal_div,
                         style={"width": "220px", "flexShrink": "0",
                                "marginRight": "16px"}),
                html.Div(dcc.Graph(figure=price_fig, config={"displayModeBar": False}),
                         style={"flex": "1"}),
            ], style={"display": "flex", "alignItems": "flex-start",
                      "marginBottom": "16px"}),

            # Row 2: equity curve
            dcc.Graph(figure=equity_fig, config={"displayModeBar": False},
                      style={"marginBottom": "16px"}),

            # Row 2b: feature importance (hidden until optimize.py has run)
            *(
                [dcc.Graph(figure=imp_fig, config={"displayModeBar": False},
                           style={"marginBottom": "16px"})]
                if imp_fig is not None else []
            ),

            # Row 3: metric cards
            html.Div(
                dbc.Row([dbc.Col(_c, md=3, sm=6, xs=12) for _c in metric_cds],
                        className="g-2"),
                style={"marginBottom": "16px"},
            ),

            # Row 4: trade log
            html.Div([
                html.H6("TRADE LOG",
                        style={"color": TEXT_DIM, "letterSpacing": "0.1em",
                               "fontSize": "0.72rem", "marginBottom": "8px"}),
                trade_tbl,
            ], style={
                "backgroundColor": PANEL,
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
                "padding": "16px",
            }),

        ], style={"padding": "20px", "backgroundColor": BG, "minHeight": "100vh"}),

    ], style={"backgroundColor": BG, "fontFamily": "monospace"})

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Launch helper
# ─────────────────────────────────────────────────────────────────────────────

def launch(
    result,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    use_regime_mapper: bool = False,
) -> None:
    """
    Build the Dash app from *result* and run the server.
    Prints a full traceback and re-raises on any failure so the caller
    (main.py) always sees the real error instead of a silent exit.
    """
    import traceback as _tb

    try:
        app = create_app(result, use_regime_mapper=use_regime_mapper)
        log.info("Dashboard ready — serving on http://%s:%d", host, port)
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,               # no Werkzeug child-process spawn
            dev_tools_silence_routes_logging=True,  # suppress per-request noise
        )
    except Exception as exc:
        log.error("Dashboard crashed: %s", exc)
        _tb.print_exc()
        raise
