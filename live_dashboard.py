"""
live_dashboard.py
-----------------
AGATE / BERYL -- Live Monitor Dashboard.

Standalone Dash app that auto-refreshes every 60 seconds.
Reads from:
  - paper_trades.db   (AGATE live trade history)
  - beryl_optimization_results.csv  (BERYL optimization progress)

Run:
  python live_dashboard.py        # opens at http://127.0.0.1:8060

Port 8060 to avoid collision with backtester dashboard on 8050.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from daily_report import _agate_score, _beryl_score, _score_bar

ROOT = Path(__file__).parent

# ── Colour palette (same as src/dashboard.py) ────────────────────────────────
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

DB_PATH       = ROOT / "paper_trades.db"
CITRINE_DB    = ROOT / "citrine_trades.db"
CSV_PATH      = ROOT / "beryl_optimization_results.csv"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_trades() -> pd.DataFrame:
    """Load all trades from paper_trades.db. Returns empty DataFrame on error."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades ORDER BY timestamp DESC", conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


def _load_beryl() -> pd.DataFrame:
    """Load BERYL optimization results. Returns empty DataFrame on error."""
    if not CSV_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(str(CSV_PATH))
    except Exception:
        return pd.DataFrame()


def _rolling_sharpe(df: pd.DataFrame, window: int = 20) -> float:
    """Compute rolling Sharpe over last *window* trades. Returns 0.0 if not enough."""
    if df.empty or len(df) < window:
        return 0.0
    pnls = df.head(window)["pnl"].values.astype(float)
    std = np.std(pnls)
    if std < 1e-9:
        return 0.0
    return float(np.mean(pnls) / std)


def _trades_today(df: pd.DataFrame) -> int:
    """Count trades whose timestamp starts with today's UTC date."""
    if df.empty:
        return 0
    today = datetime.now(tz=timezone.utc).date().isoformat()
    return int(df["timestamp"].astype(str).str.startswith(today).sum())


def _kill_switch_status(df: pd.DataFrame) -> dict:
    """
    Evaluate the three kill-switch rules.
    Returns dict with keys: overall (bool), rules (list of dicts).
    """
    rules = []
    triggered = False

    # Rule 1: daily loss > 2% of $10k = $200
    today_pnl = 0.0
    if not df.empty:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        today_df = df[df["timestamp"].astype(str).str.startswith(today)]
        today_pnl = today_df["pnl"].sum() if not today_df.empty else 0.0
    r1_tripped = today_pnl < -200
    rules.append({
        "label": "Daily loss > 2% ($200)",
        "value": f"${today_pnl:+.2f}",
        "ok": not r1_tripped,
    })
    if r1_tripped:
        triggered = True

    # Rule 2: rolling 5-trade Sharpe < 0.3
    sharpe5 = 0.0
    if not df.empty and len(df) >= 5:
        pnls5 = df.head(5)["pnl"].values.astype(float)
        std5 = np.std(pnls5)
        sharpe5 = float(np.mean(pnls5) / std5) if std5 > 1e-9 else 0.0
    r2_tripped = (sharpe5 < 0.3) and (len(df) >= 5)
    rules.append({
        "label": "Rolling 5-trade Sharpe < 0.3",
        "value": f"{sharpe5:.3f}" if len(df) >= 5 else "N/A",
        "ok": not r2_tripped,
    })
    if r2_tripped:
        triggered = True

    # Rule 3: cumulative loss > 2% ($200)
    cum_pnl = df["pnl"].sum() if not df.empty else 0.0
    r3_tripped = cum_pnl < -200
    rules.append({
        "label": "Cumulative loss > 2% ($200)",
        "value": f"${cum_pnl:+.2f}",
        "ok": not r3_tripped,
    })
    if r3_tripped:
        triggered = True

    return {"triggered": triggered, "rules": rules}


def _load_status(filename: str) -> dict:
    """Load a JSON status file (agate_status.json or beryl_status.json)."""
    path = ROOT / filename
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _regime_panel(name: str, status: dict) -> html.Div:
    """Build a regime status panel for AGATE or BERYL."""
    if not status:
        return html.Div(
            f"No {name} status — waiting for first signal",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "20px",
                   "fontSize": "0.8rem", "fontFamily": "monospace",
                   "backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px"},
        )

    regime = status.get("regime", "UNKNOWN")
    confidence = status.get("confidence", 0.0)
    confirmations = status.get("confirmations", 0)
    signal = status.get("signal", "HOLD")
    price = status.get("current_price", 0.0)
    ticker = status.get("ticker", "?")
    ts = status.get("timestamp", "")

    # Regime colour
    regime_upper = regime.upper() if regime else "UNKNOWN"
    if regime_upper in ("BULL", "BULL_STRONG", "BULL_MILD"):
        regime_color = GREEN
        regime_label = "BULL"
    elif regime_upper in ("BEAR", "BEAR_CRASH", "BEAR_MILD"):
        regime_color = RED
        regime_label = "BEAR"
    elif regime_upper in ("CHOP", "CHOP_HIGH_VOL", "CHOP_LOW_VOL", "CHOP_NEUTRAL"):
        regime_color = YELLOW
        regime_label = "CHOP"
    else:
        regime_color = TEXT_DIM
        regime_label = regime_upper

    # Signal colour
    signal_colors = {"BUY": GREEN, "SELL": RED, "HOLD": YELLOW}
    signal_color = signal_colors.get(signal, TEXT_DIM)

    # Confidence bar
    conf_pct = int(confidence * 100)
    conf_color = GREEN if confidence >= 0.70 else (YELLOW if confidence >= 0.50 else RED)

    # Position info
    position = status.get("position")
    pos_text = "No position"
    pos_color = TEXT_DIM
    if position:
        pos_text = f"{position['side']} {position.get('size', 0):.4f} @ ${position.get('entry_price', 0):.2f}"
        pos_color = GREEN if position["side"] == "BUY" else PURPLE

    # Timestamp age
    age_str = ""
    if ts:
        try:
            ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = (datetime.now(tz=timezone.utc) - ts_dt).total_seconds()
            if age < 3600:
                age_str = f"{int(age // 60)}m ago"
            else:
                age_str = f"{age / 3600:.1f}h ago"
        except Exception:
            age_str = ""

    # Ensemble indicator
    ensemble_text = ""
    if status.get("use_ensemble"):
        ensemble_text = " (Ensemble)"

    return html.Div([
        # Header: project name + ticker
        html.Div([
            html.Span(f"{name} ",
                       style={"color": ORANGE, "fontWeight": "900",
                              "fontSize": "0.85rem"}),
            html.Span(f"— {ticker}{ensemble_text}",
                       style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            html.Span(f"  {age_str}",
                       style={"color": TEXT_DIM, "fontSize": "0.65rem",
                              "fontStyle": "italic"}),
        ], style={"marginBottom": "12px"}),

        # Regime badge
        html.Div([
            html.Span(regime_label,
                       style={"color": regime_color, "fontWeight": "900",
                              "fontSize": "1.6rem", "fontFamily": "monospace",
                              "letterSpacing": "0.1em"}),
        ], style={"textAlign": "center", "marginBottom": "8px"}),

        # Confidence bar
        html.Div([
            html.Span("Confidence: ",
                       style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
            html.Span(f"{conf_pct}%",
                       style={"color": conf_color, "fontWeight": "700",
                              "fontSize": "0.8rem", "fontFamily": "monospace"}),
        ], style={"textAlign": "center", "marginBottom": "4px"}),
        html.Div(
            html.Div(style={"width": f"{conf_pct}%", "height": "6px",
                             "backgroundColor": conf_color,
                             "borderRadius": "3px",
                             "transition": "width 0.3s"}),
            style={"backgroundColor": BORDER, "borderRadius": "3px",
                   "height": "6px", "marginBottom": "12px"},
        ),

        # Signal + Confirmations + Price
        html.Div([
            html.Div([
                html.Span("Signal: ", style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
                html.Span(signal, style={"color": signal_color, "fontWeight": "700",
                                          "fontSize": "0.85rem"}),
            ]),
            html.Div([
                html.Span("Confirms: ", style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
                html.Span(f"{confirmations}/8",
                           style={"color": GREEN if confirmations >= 6 else YELLOW,
                                  "fontWeight": "700", "fontSize": "0.85rem",
                                  "fontFamily": "monospace"}),
            ]),
            html.Div([
                html.Span("Price: ", style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
                html.Span(f"${price:,.2f}",
                           style={"color": TEXT, "fontWeight": "700",
                                  "fontSize": "0.85rem", "fontFamily": "monospace"}),
            ]),
        ], style={"display": "flex", "justifyContent": "space-around",
                  "marginBottom": "10px"}),

        # Position
        html.Div([
            html.Span("Position: ", style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
            html.Span(pos_text, style={"color": pos_color, "fontSize": "0.75rem",
                                        "fontFamily": "monospace"}),
        ], style={"textAlign": "center"}),

        # Multi-ticker scan summary (AGATE only, when scan_summary present)
        *(_scan_summary_section(status) if status.get("scan_summary") else []),

    ], style={
        "backgroundColor": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px",
    })


def _scan_summary_section(status: dict) -> list:
    """Build scan summary rows for AGATE multi-ticker panel."""
    summary = status.get("scan_summary", [])
    if not summary:
        return []

    n_scanned = status.get("tickers_scanned", len(summary))
    n_total = status.get("tickers_total", n_scanned)
    n_converged = status.get("tickers_converged", n_scanned)
    n_bull = status.get("tickers_bull", 0)
    n_bear = status.get("tickers_bear", 0)
    n_chop = status.get("tickers_chop", 0)

    conv_pct = (n_converged / n_total * 100) if n_total > 0 else 0
    conv_color = GREEN if conv_pct >= 70 else (YELLOW if conv_pct >= 50 else RED)

    header = html.Div([
        html.Span(f"Scan: ", style={"color": TEXT_DIM, "fontSize": "0.65rem"}),
        html.Span(f"{n_converged}/{n_total} ", style={"color": conv_color, "fontSize": "0.65rem",
                                                       "fontFamily": "monospace", "fontWeight": "700"}),
        html.Span(f"converged ", style={"color": TEXT_DIM, "fontSize": "0.65rem"}),
        html.Span(f"({n_bull}", style={"color": GREEN, "fontSize": "0.65rem", "fontFamily": "monospace"}),
        html.Span(f"B/", style={"color": TEXT_DIM, "fontSize": "0.65rem", "fontFamily": "monospace"}),
        html.Span(f"{n_bear}", style={"color": RED, "fontSize": "0.65rem", "fontFamily": "monospace"}),
        html.Span(f"R/", style={"color": TEXT_DIM, "fontSize": "0.65rem", "fontFamily": "monospace"}),
        html.Span(f"{n_chop}", style={"color": YELLOW, "fontSize": "0.65rem", "fontFamily": "monospace"}),
        html.Span(f"C)", style={"color": TEXT_DIM, "fontSize": "0.65rem", "fontFamily": "monospace"}),
    ], style={"textAlign": "center", "marginTop": "10px", "marginBottom": "4px"})

    rows = [header]

    bull_tickers = [s for s in summary if s.get("regime") == "BULL"]
    if bull_tickers:
        bull_tickers.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        bull_items = []
        for s in bull_tickers[:8]:
            t = s["ticker"].replace("X:", "")
            cf = s.get("confirmations", 0)
            sig = s.get("signal", "HOLD")
            color = GREEN if sig == "BUY" else YELLOW
            bull_items.append(
                html.Span(f"{t}({cf}) ",
                           style={"color": color, "fontSize": "0.6rem",
                                  "fontFamily": "monospace"})
            )
        if len(bull_tickers) > 8:
            bull_items.append(html.Span(f"+{len(bull_tickers)-8}",
                                         style={"color": TEXT_DIM, "fontSize": "0.6rem"}))
        rows.append(html.Div([
            html.Span("BULL: ", style={"color": GREEN, "fontSize": "0.65rem", "fontWeight": "700"}),
            *bull_items,
        ], style={"textAlign": "center"}))

    bear_tickers = [s for s in summary if s.get("regime") == "BEAR"]
    if bear_tickers:
        bear_tickers.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        bear_items = []
        for s in bear_tickers[:8]:
            t = s["ticker"].replace("X:", "")
            cf = s.get("confirmations", 0)
            bear_items.append(
                html.Span(f"{t}({cf}) ",
                           style={"color": RED, "fontSize": "0.6rem",
                                  "fontFamily": "monospace"})
            )
        if len(bear_tickers) > 8:
            bear_items.append(html.Span(f"+{len(bear_tickers)-8}",
                                         style={"color": TEXT_DIM, "fontSize": "0.6rem"}))
        rows.append(html.Div([
            html.Span("BEAR: ", style={"color": RED, "fontSize": "0.65rem", "fontWeight": "700"}),
            *bear_items,
        ], style={"textAlign": "center"}))

    return rows


def _beryl_regime_panel(status: dict) -> html.Div:
    """Build a dedicated BERYL regime panel for 98-ticker ensemble scanning."""
    if not status:
        return html.Div(
            "No BERYL status -- waiting for first scan",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "20px",
                   "fontSize": "0.8rem", "fontFamily": "monospace",
                   "backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px"},
        )

    summary = status.get("scan_summary", [])
    positions = status.get("positions", [])
    ts = status.get("timestamp", "")

    n_total = status.get("tickers_total", len(summary))
    n_converged = status.get("tickers_converged", n_total)
    n_bull = status.get("tickers_bull", 0)
    n_bear = status.get("tickers_bear", 0)
    n_chop = status.get("tickers_chop", 0)
    pos_count = status.get("positions_count", len(positions))
    max_pos = status.get("max_positions", 3)

    # Timestamp age
    age_str = ""
    if ts:
        try:
            ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = (datetime.now(tz=timezone.utc) - ts_dt).total_seconds()
            if age < 3600:
                age_str = f"{int(age // 60)}m ago"
            else:
                age_str = f"{age / 3600:.1f}h ago"
        except Exception:
            age_str = ""

    # Ensemble indicator
    ensemble_text = " (Ensemble)" if status.get("use_ensemble") else ""

    # ---- Convergence rate ----
    conv_pct = (n_converged / n_total * 100) if n_total > 0 else 0
    conv_color = GREEN if conv_pct >= 90 else (YELLOW if conv_pct >= 70 else RED)

    # ---- BUY signals count ----
    buy_signals = [s for s in summary if s.get("signal") == "BUY"]
    n_buy = len(buy_signals)

    # ---- Top 5 BULL tickers (by confidence) ----
    bull_tickers = [s for s in summary if s.get("regime") == "BULL"]
    bull_tickers.sort(key=lambda s: s.get("confidence", 0), reverse=True)

    # ---- Top 5 BEAR tickers (by confidence) ----
    bear_tickers = [s for s in summary if s.get("regime") == "BEAR"]
    bear_tickers.sort(key=lambda s: s.get("confidence", 0), reverse=True)

    # ---- Build sections ----
    sections = []

    # Header
    sections.append(html.Div([
        html.Span("BERYL ",
                   style={"color": ORANGE, "fontWeight": "900", "fontSize": "0.85rem"}),
        html.Span(f"-- 98-Ticker Rotation{ensemble_text}",
                   style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
        html.Span(f"  {age_str}",
                   style={"color": TEXT_DIM, "fontSize": "0.65rem", "fontStyle": "italic"}),
    ], style={"marginBottom": "12px"}))

    # Convergence banner
    sections.append(html.Div([
        html.Span(f"{n_converged}/{n_total}",
                   style={"color": conv_color, "fontWeight": "900",
                          "fontSize": "1.1rem", "fontFamily": "monospace"}),
        html.Span(f" converged ({conv_pct:.0f}%)",
                   style={"color": conv_color, "fontSize": "0.8rem"}),
    ], style={"textAlign": "center", "marginBottom": "8px"}))

    # Regime split bar
    total_regime = n_bull + n_bear + n_chop
    if total_regime > 0:
        bull_w = n_bull / total_regime * 100
        bear_w = n_bear / total_regime * 100
        chop_w = n_chop / total_regime * 100
    else:
        bull_w = bear_w = chop_w = 33.3

    sections.append(html.Div([
        html.Div([
            html.Span(f"{n_bull} BULL", style={"color": GREEN, "fontWeight": "700",
                                                "fontSize": "0.75rem", "fontFamily": "monospace"}),
            html.Span(" / ", style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            html.Span(f"{n_bear} BEAR", style={"color": RED, "fontWeight": "700",
                                                "fontSize": "0.75rem", "fontFamily": "monospace"}),
            html.Span(" / ", style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            html.Span(f"{n_chop} CHOP", style={"color": YELLOW, "fontWeight": "700",
                                                "fontSize": "0.75rem", "fontFamily": "monospace"}),
        ], style={"textAlign": "center", "marginBottom": "4px"}),

        # Stacked bar
        html.Div([
            html.Div(style={"width": f"{bull_w}%", "height": "8px",
                             "backgroundColor": GREEN, "display": "inline-block"}),
            html.Div(style={"width": f"{bear_w}%", "height": "8px",
                             "backgroundColor": RED, "display": "inline-block"}),
            html.Div(style={"width": f"{chop_w}%", "height": "8px",
                             "backgroundColor": YELLOW, "display": "inline-block"}),
        ], style={"borderRadius": "4px", "overflow": "hidden", "marginBottom": "10px",
                  "fontSize": "0", "lineHeight": "0"}),
    ]))

    # BUY signals count
    buy_color = GREEN if n_buy > 0 else TEXT_DIM
    sections.append(html.Div([
        html.Span(f"{n_buy} BUY signal{'s' if n_buy != 1 else ''} this scan",
                   style={"color": buy_color, "fontWeight": "700",
                          "fontSize": "0.8rem", "fontFamily": "monospace"}),
    ], style={"textAlign": "center", "marginBottom": "10px",
              "border": f"1px solid {BORDER}", "borderRadius": "6px",
              "padding": "6px"}))

    # ---- Positions panel ----
    sections.append(html.Div("POSITIONS",
                    style={"color": TEXT_DIM, "fontSize": "0.65rem",
                           "letterSpacing": "0.1em", "marginBottom": "6px",
                           "borderTop": f"1px solid {BORDER}", "paddingTop": "10px"}))

    if positions:
        for pos in positions[:3]:
            pticker = pos.get("ticker", "?")
            pside = pos.get("side", "BUY")
            pentry = pos.get("entry_price", 0)
            psize = pos.get("size", 0)
            pos_color = GREEN if pside == "BUY" else PURPLE

            pos_items = [
                html.Span(f"{pticker} ",
                           style={"color": ORANGE, "fontWeight": "700",
                                  "fontSize": "0.8rem"}),
                html.Span(f"{pside} ",
                           style={"color": pos_color, "fontWeight": "700",
                                  "fontSize": "0.75rem"}),
                html.Span(f"@ ${pentry:,.2f}",
                           style={"color": TEXT, "fontSize": "0.75rem",
                                  "fontFamily": "monospace"}),
            ]
            # Show unrealized P&L if current price available from scan
            current_price = None
            for s in summary:
                if s.get("ticker") == pticker:
                    current_price = s.get("current_price")
                    break
            if current_price and pentry > 0:
                if pside == "BUY":
                    pnl_pct = (current_price - pentry) / pentry * 100
                else:
                    pnl_pct = (pentry - current_price) / pentry * 100
                pnl_color = GREEN if pnl_pct >= 0 else RED
                pos_items.append(
                    html.Span(f"  {pnl_pct:+.2f}%",
                               style={"color": pnl_color, "fontWeight": "700",
                                      "fontSize": "0.75rem", "fontFamily": "monospace"}))

            sections.append(html.Div(pos_items, style={"marginBottom": "4px"}))

        sections.append(html.Div(
            f"{pos_count}/{max_pos} slots used",
            style={"color": TEXT_DIM, "fontSize": "0.65rem", "marginTop": "4px"}))
    else:
        sections.append(html.Div("No positions",
                        style={"color": TEXT_DIM, "fontSize": "0.75rem",
                               "fontFamily": "monospace"}))

    # ---- Top 5 BULL tickers ----
    if bull_tickers:
        sections.append(html.Div("TOP 5 BULL",
                        style={"color": TEXT_DIM, "fontSize": "0.65rem",
                               "letterSpacing": "0.1em", "marginTop": "10px",
                               "marginBottom": "4px",
                               "borderTop": f"1px solid {BORDER}", "paddingTop": "8px"}))
        for s in bull_tickers[:5]:
            t = s.get("ticker", "?")
            conf = s.get("confidence", 0)
            cf = s.get("confirmations", 0)
            mcf = s.get("min_confirmations", 5)
            sig = s.get("signal", "HOLD")
            sig_color = GREEN if sig == "BUY" else TEXT_DIM
            conf_color = GREEN if conf >= 0.90 else (YELLOW if conf >= 0.70 else TEXT_DIM)
            sections.append(html.Div([
                html.Span(f"{t:<6s}", style={"color": ORANGE, "fontWeight": "700",
                                              "fontSize": "0.75rem", "fontFamily": "monospace"}),
                html.Span(f" {conf:.0%}",
                           style={"color": conf_color, "fontWeight": "700",
                                  "fontSize": "0.75rem", "fontFamily": "monospace"}),
                html.Span(f" {cf}/{mcf}cf",
                           style={"color": GREEN if cf >= mcf else TEXT_DIM,
                                  "fontSize": "0.7rem", "fontFamily": "monospace"}),
                html.Span(f"  {sig}",
                           style={"color": sig_color, "fontWeight": "700",
                                  "fontSize": "0.7rem", "fontFamily": "monospace"}),
            ], style={"marginBottom": "2px"}))

    # ---- Top 5 BEAR tickers ----
    if bear_tickers:
        sections.append(html.Div("TOP 5 BEAR",
                        style={"color": TEXT_DIM, "fontSize": "0.65rem",
                               "letterSpacing": "0.1em", "marginTop": "8px",
                               "marginBottom": "4px",
                               "borderTop": f"1px solid {BORDER}", "paddingTop": "8px"}))
        for s in bear_tickers[:5]:
            t = s.get("ticker", "?")
            conf = s.get("confidence", 0)
            cf = s.get("confirmations", 0)
            conf_color = RED if conf >= 0.90 else (YELLOW if conf >= 0.70 else TEXT_DIM)
            sections.append(html.Div([
                html.Span(f"{t:<6s}", style={"color": RED, "fontWeight": "700",
                                              "fontSize": "0.75rem", "fontFamily": "monospace"}),
                html.Span(f" {conf:.0%}",
                           style={"color": conf_color, "fontWeight": "700",
                                  "fontSize": "0.75rem", "fontFamily": "monospace"}),
                html.Span(f" {cf}cf",
                           style={"color": TEXT_DIM, "fontSize": "0.7rem",
                                  "fontFamily": "monospace"}),
            ], style={"marginBottom": "2px"}))

    return html.Div(sections, style={
        "backgroundColor": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px",
    })


# ── Chart builders ────────────────────────────────────────────────────────────

def _build_equity_curve(df: pd.DataFrame) -> go.Figure:
    """Build cumulative P&L line chart from trade history."""
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No trades yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=18, color=TEXT_DIM),
        )
    else:
        # Sort chronologically (df arrives DESC)
        sorted_df = df.sort_values("timestamp").reset_index(drop=True)
        cum_pnl = sorted_df["pnl"].cumsum()
        timestamps = pd.to_datetime(sorted_df["timestamp"], errors="coerce")

        color = GREEN if (cum_pnl.iloc[-1] >= 0) else RED

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cum_pnl,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
            name="Cumulative P&L",
        ))

        # Zero line
        fig.add_hline(y=0, line=dict(color=TEXT_DIM, width=1, dash="dot"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        title=dict(text="Equity Curve -- Cumulative P&L", font=dict(size=13)),
        yaxis=dict(tickprefix="$", gridcolor=BORDER),
        xaxis=dict(gridcolor=BORDER),
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


def _kill_switch_panel(ks: dict) -> html.Div:
    """Build kill-switch status indicator with rule breakdown."""
    overall_color = RED if ks["triggered"] else GREEN
    overall_label = "TRIGGERED" if ks["triggered"] else "OK"

    rule_rows = []
    for rule in ks["rules"]:
        dot_color = GREEN if rule["ok"] else RED
        rule_rows.append(html.Div([
            html.Span("● " if rule["ok"] else "X ",
                       style={"color": dot_color, "fontSize": "0.85rem",
                              "fontWeight": "700"}),
            html.Span(rule["label"],
                       style={"fontSize": "0.75rem", "color": TEXT}),
            html.Span(f"  {rule['value']}",
                       style={"fontSize": "0.75rem", "color": dot_color,
                              "fontFamily": "monospace", "marginLeft": "8px"}),
        ], style={"marginBottom": "6px"}))

    return html.Div([
        html.Div([
            html.Span("KILL-SWITCH ",
                       style={"color": TEXT_DIM, "fontSize": "0.7rem",
                              "letterSpacing": "0.1em"}),
            html.Span(overall_label,
                       style={"color": overall_color, "fontWeight": "900",
                              "fontSize": "1.1rem", "fontFamily": "monospace"}),
        ], style={"textAlign": "center", "marginBottom": "16px",
                  "border": f"2px solid {overall_color}",
                  "borderRadius": "8px", "padding": "12px"}),

        html.Div(rule_rows, style={"marginTop": "8px"}),

    ], style={
        "backgroundColor": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px",
        "height": "100%",
    })


def _trades_table(df: pd.DataFrame) -> html.Div:
    """Build recent trades table (last 10 trades)."""
    if df.empty:
        return html.Div(
            "No trades yet",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "40px",
                   "fontSize": "0.9rem", "fontFamily": "monospace"},
        )

    display_df = df.head(10).copy()
    display_df = display_df[["timestamp", "side", "entry_price", "exit_price",
                              "size", "pnl", "pnl_pct", "signal_strength"]]
    display_df["timestamp"] = pd.to_datetime(
        display_df["timestamp"], errors="coerce",
    ).dt.strftime("%m/%d %H:%M")
    display_df["entry_price"] = display_df["entry_price"].apply(lambda v: f"${v:,.4f}")
    display_df["exit_price"]  = display_df["exit_price"].apply(lambda v: f"${v:,.4f}")
    display_df["size"]        = display_df["size"].apply(lambda v: f"{v:.4f}")
    display_df["pnl"]         = display_df["pnl"].apply(lambda v: f"${v:+,.2f}")
    display_df["pnl_pct"]     = display_df["pnl_pct"].apply(lambda v: f"{v:+.2f}%")
    display_df.columns = ["Time", "Side", "Entry", "Exit", "Size",
                           "P&L", "P&L %", "Signals"]

    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display_df.columns],
        style_table={"overflowX": "auto", "overflowY": "auto",
                     "maxHeight": "300px", "backgroundColor": PANEL},
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
        page_size=10,
    )


def _beryl_panel(beryl_df: pd.DataFrame) -> html.Div:
    """Build BERYL optimization progress panel."""
    if beryl_df.empty:
        return html.Div(
            "No BERYL results yet",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "40px",
                   "fontSize": "0.9rem", "fontFamily": "monospace"},
        )

    total = len(beryl_df)
    positive = int((beryl_df["wf_sharpe"] > 0).sum())
    high = int((beryl_df["wf_sharpe"] > 1.0).sum())
    best_sharpe = beryl_df["wf_sharpe"].max()

    # Top 3 configs
    top3 = beryl_df.nlargest(3, "wf_sharpe")
    top_rows = []
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        ticker = row.get("ticker", "?")
        sharpe = row.get("wf_sharpe", 0)
        ret = row.get("wf_return", 0)
        n_st = row.get("n_states", "?")
        feat = row.get("feature_set", "?")
        ens = row.get("use_ensemble", False)

        sharpe_color = GREEN if sharpe > 1.0 else (YELLOW if sharpe > 0 else RED)
        top_rows.append(html.Div([
            html.Span(f"#{i} ", style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            html.Span(f"{ticker} ",
                       style={"color": ORANGE, "fontWeight": "700",
                              "fontSize": "0.8rem"}),
            html.Span(f"Sharpe {sharpe:.3f} ",
                       style={"color": sharpe_color, "fontWeight": "700",
                              "fontSize": "0.8rem", "fontFamily": "monospace"}),
            html.Span(f"Ret {ret:+.1f}% ",
                       style={"color": GREEN if ret > 0 else RED,
                              "fontSize": "0.75rem", "fontFamily": "monospace"}),
            html.Span(f"({n_st}st/{feat}" + ("/ens" if ens else "") + ")",
                       style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
        ], style={"marginBottom": "6px"}))

    return html.Div([
        # Summary row
        html.Div([
            html.Span("Trials: ", style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            html.Span(f"{total}", style={"color": TEXT, "fontWeight": "700",
                                          "fontSize": "0.85rem"}),
            html.Span("   Positive: ", style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            html.Span(f"{positive} ({positive/total*100:.0f}%)" if total > 0 else "0",
                       style={"color": GREEN if positive > 0 else RED,
                              "fontWeight": "700", "fontSize": "0.85rem"}),
            html.Span("   Sharpe>1: ", style={"color": TEXT_DIM, "fontSize": "0.75rem"}),
            html.Span(f"{high}",
                       style={"color": GREEN if high > 0 else TEXT_DIM,
                              "fontWeight": "700", "fontSize": "0.85rem"}),
        ], style={"marginBottom": "12px"}),

        # Best Sharpe highlight
        html.Div([
            html.Span("BEST SHARPE ",
                       style={"color": TEXT_DIM, "fontSize": "0.7rem",
                              "letterSpacing": "0.1em"}),
            html.Span(f"{best_sharpe:.3f}",
                       style={"color": GREEN if best_sharpe > 1.0 else YELLOW,
                              "fontWeight": "900", "fontSize": "1.0rem",
                              "fontFamily": "monospace"}),
        ], style={"textAlign": "center", "marginBottom": "14px",
                  "border": f"1px solid {BORDER}",
                  "borderRadius": "6px", "padding": "8px"}),

        # Top 3 configs header
        html.Div("TOP 3 CONFIGS",
                  style={"color": TEXT_DIM, "fontSize": "0.7rem",
                         "letterSpacing": "0.1em", "marginBottom": "8px"}),
        html.Div(top_rows),

    ], style={
        "backgroundColor": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px",
        "height": "100%",
    })


# ── Cross-project agreement ───────────────────────────────────────────────────

def _load_citrine_held_tickers() -> set[str]:
    """Load currently-held CITRINE tickers from the latest portfolio snapshot."""
    if not CITRINE_DB.exists():
        return set()
    try:
        with sqlite3.connect(str(CITRINE_DB)) as conn:
            row = conn.execute(
                "SELECT positions FROM portfolio_snapshots "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        if not row or not row[0]:
            return set()
        positions = json.loads(row[0])
        # positions may be dict-of-dicts or list-of-dicts
        tickers: set[str] = set()
        if isinstance(positions, dict):
            tickers = set(positions.keys())
        elif isinstance(positions, list):
            for p in positions:
                t = p.get("ticker") if isinstance(p, dict) else None
                if t:
                    tickers.add(t)
        return tickers
    except Exception:
        return set()


def _cross_project_panel(beryl_status: dict) -> html.Div:
    """Build a cross-project signal agreement panel (BERYL vs CITRINE)."""
    beryl_summary = beryl_status.get("scan_summary", [])
    citrine_held = _load_citrine_held_tickers()

    if not beryl_summary and not citrine_held:
        return html.Div(
            "No cross-project data -- waiting for BERYL scan and CITRINE positions",
            style={"color": TEXT_DIM, "textAlign": "center", "padding": "20px",
                   "fontSize": "0.8rem", "fontFamily": "monospace",
                   "backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px"},
        )

    # Build lookup: ticker -> regime from BERYL scan
    beryl_regimes: dict[str, dict] = {}
    for s in beryl_summary:
        ticker = s.get("ticker", "")
        if ticker:
            beryl_regimes[ticker] = {
                "regime": s.get("regime", "UNKNOWN"),
                "confidence": s.get("confidence", 0.0),
            }

    # Agreement: CITRINE holds it AND BERYL says BULL
    agreements: list[str] = []
    # Conflict: CITRINE holds it AND BERYL says BEAR
    conflicts: list[tuple[str, str, float]] = []  # (ticker, beryl_regime, confidence)
    # Neutral: CITRINE holds it AND BERYL says CHOP or no data
    neutral: list[str] = []

    for ticker in sorted(citrine_held):
        beryl_info = beryl_regimes.get(ticker)
        if not beryl_info:
            continue  # BERYL didn't scan this ticker
        regime = beryl_info["regime"].upper() if beryl_info["regime"] else "UNKNOWN"
        if regime in ("BULL", "BULL_STRONG", "BULL_MILD"):
            agreements.append(ticker)
        elif regime in ("BEAR", "BEAR_CRASH", "BEAR_MILD"):
            conflicts.append((ticker, regime, beryl_info["confidence"]))
        else:
            neutral.append(ticker)

    sections = []

    # Header
    sections.append(html.Div([
        html.Span("CROSS-PROJECT AGREEMENT",
                   style={"color": ORANGE, "fontWeight": "900",
                          "fontSize": "0.8rem", "letterSpacing": "0.1em"}),
        html.Span("  BERYL x CITRINE",
                   style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
    ], style={"marginBottom": "12px"}))

    # Summary counts
    sections.append(html.Div([
        html.Span(f"CITRINE holds {len(citrine_held)} ticker{'s' if len(citrine_held) != 1 else ''}",
                   style={"color": TEXT, "fontSize": "0.75rem"}),
        html.Span(f"   |   BERYL scanned {len(beryl_summary)} tickers",
                   style={"color": TEXT_DIM, "fontSize": "0.7rem"}),
    ], style={"marginBottom": "10px"}))

    # BULL Agreement
    agree_color = GREEN if agreements else TEXT_DIM
    agree_items = [
        html.Span(f"BULL Agreement ({len(agreements)}): ",
                   style={"color": agree_color, "fontWeight": "700",
                          "fontSize": "0.75rem"}),
    ]
    if agreements:
        agree_items.append(
            html.Span(", ".join(agreements),
                       style={"color": GREEN, "fontSize": "0.75rem",
                              "fontFamily": "monospace"}))
    else:
        agree_items.append(
            html.Span("none", style={"color": TEXT_DIM, "fontSize": "0.75rem",
                                      "fontFamily": "monospace"}))
    sections.append(html.Div(agree_items, style={"marginBottom": "6px"}))

    # Conflicts (BERYL BEAR vs CITRINE LONG)
    conflict_color = RED if conflicts else TEXT_DIM
    conflict_items = [
        html.Span(f"Conflict ({len(conflicts)}): ",
                   style={"color": conflict_color, "fontWeight": "700",
                          "fontSize": "0.75rem"}),
    ]
    if conflicts:
        conflict_parts = []
        for ticker, regime, conf in conflicts:
            conflict_parts.append(f"{ticker} (BERYL: {regime} {conf:.0%})")
        conflict_items.append(
            html.Span(", ".join(conflict_parts),
                       style={"color": RED, "fontSize": "0.75rem",
                              "fontFamily": "monospace"}))
    else:
        conflict_items.append(
            html.Span("none", style={"color": TEXT_DIM, "fontSize": "0.75rem",
                                      "fontFamily": "monospace"}))
    sections.append(html.Div(conflict_items, style={"marginBottom": "6px"}))

    # Neutral (BERYL CHOP vs CITRINE LONG)
    if neutral:
        sections.append(html.Div([
            html.Span(f"Neutral ({len(neutral)}): ",
                       style={"color": YELLOW, "fontWeight": "700",
                              "fontSize": "0.75rem"}),
            html.Span(", ".join(neutral),
                       style={"color": YELLOW, "fontSize": "0.75rem",
                              "fontFamily": "monospace"}),
        ]))

    return html.Div(sections, style={
        "backgroundColor": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px",
    })


# ── Dash App ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="AGATE / BERYL -- Live Monitor",
)

# Prevent browser from caching stale dashboard pages
@app.server.after_request
def _no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.layout = html.Div([

    # Auto-refresh interval (60 seconds)
    dcc.Interval(id="refresh-interval", interval=60_000, n_intervals=0),

    # ── Header ────────────────────────────────────────────────────────────
    html.Div(id="header-container"),

    # ── Body (updated by callback) ────────────────────────────────────────
    html.Div(id="body-container",
             style={"padding": "20px", "backgroundColor": BG,
                    "minHeight": "100vh"}),

], style={"backgroundColor": BG, "fontFamily": "monospace"})


@app.callback(
    [Output("header-container", "children"),
     Output("body-container", "children")],
    [Input("refresh-interval", "n_intervals")],
)
def update_dashboard(_n):
    """Rebuild the entire dashboard on each interval tick."""
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    trades_df = _load_trades()
    beryl_df = _load_beryl()
    agate_status = _load_status("agate_status.json")
    beryl_status = _load_status("beryl_status.json")

    # Maturity scores
    agate_sc, agate_stage = _agate_score()
    beryl_sc, beryl_stage = _beryl_score()

    # Metrics
    today_count = _trades_today(trades_df)
    sharpe_20 = _rolling_sharpe(trades_df, window=20)
    ks = _kill_switch_status(trades_df)

    # ── Header ────────────────────────────────────────────────────────────
    header = html.Div([
        html.Div([
            html.Span("AGATE / BERYL ",
                       style={"color": ORANGE, "fontWeight": "900",
                              "fontSize": "1.2rem", "letterSpacing": "0.05em"}),
            html.Span("-- Live Monitor",
                       style={"color": TEXT, "fontWeight": "300",
                              "fontSize": "1.0rem"}),
        ]),
        html.Div([
            html.Span(f"Last refresh: {now_str}",
                       style={"color": TEXT_DIM, "fontSize": "0.68rem",
                              "letterSpacing": "0.08em"}),
            html.Span("   |   Auto-refresh: 60s",
                       style={"color": TEXT_DIM, "fontSize": "0.68rem"}),
        ]),
    ], style={
        "backgroundColor": PANEL,
        "borderBottom": f"1px solid {BORDER}",
        "padding": "14px 24px",
        "display": "flex",
        "flexDirection": "column",
    })

    # ── Top row: 4 metric cards ───────────────────────────────────────────
    sharpe_color = GREEN if sharpe_20 >= 0.5 else (YELLOW if sharpe_20 > 0 else RED)
    agate_color = GREEN if agate_sc >= 55 else (YELLOW if agate_sc >= 25 else TEXT_DIM)
    beryl_color = GREEN if beryl_sc >= 55 else (YELLOW if beryl_sc >= 25 else TEXT_DIM)

    top_row = dbc.Row([
        dbc.Col(_metric_card("AGATE Maturity", f"{agate_sc}%", agate_color), md=3, sm=6, xs=12),
        dbc.Col(_metric_card("BERYL Maturity", f"{beryl_sc}%", beryl_color), md=3, sm=6, xs=12),
        dbc.Col(_metric_card("AGATE Trades Today", str(today_count), BLUE), md=3, sm=6, xs=12),
        dbc.Col(_metric_card("AGATE Rolling Sharpe (20)",
                             f"{sharpe_20:.3f}" if len(trades_df) >= 20 else "N/A",
                             sharpe_color), md=3, sm=6, xs=12),
    ], className="g-2", style={"marginBottom": "16px"})

    # ── Regime status row: AGATE + BERYL regime panels ──────────────────
    regime_row = dbc.Row([
        dbc.Col(_regime_panel("AGATE", agate_status), md=5, sm=12),
        dbc.Col(_beryl_regime_panel(beryl_status), md=7, sm=12),
    ], className="g-2", style={"marginBottom": "16px"})

    # ── Cross-project agreement row ────────────────────────────────────────
    cross_row = dbc.Row([
        dbc.Col(_cross_project_panel(beryl_status), md=12),
    ], className="g-2", style={"marginBottom": "16px"})

    # ── Middle row: equity curve + kill-switch ────────────────────────────
    equity_fig = _build_equity_curve(trades_df)
    middle_row = dbc.Row([
        dbc.Col(
            html.Div([
                dcc.Graph(figure=equity_fig, config={"displayModeBar": False}),
            ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                      "borderRadius": "8px", "padding": "8px"}),
            md=8,
        ),
        dbc.Col(
            _kill_switch_panel(ks),
            md=4,
        ),
    ], className="g-2", style={"marginBottom": "16px"})

    # ── Bottom row: trades table + BERYL progress ────────────────────────
    bottom_row = dbc.Row([
        dbc.Col(
            html.Div([
                html.H6("RECENT TRADES",
                         style={"color": TEXT_DIM, "letterSpacing": "0.1em",
                                "fontSize": "0.72rem", "marginBottom": "8px"}),
                _trades_table(trades_df),
            ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                      "borderRadius": "8px", "padding": "16px"}),
            md=7,
        ),
        dbc.Col(
            html.Div([
                html.H6("BERYL OPTIMIZATION",
                         style={"color": TEXT_DIM, "letterSpacing": "0.1em",
                                "fontSize": "0.72rem", "marginBottom": "8px"}),
                _beryl_panel(beryl_df),
            ]),
            md=5,
        ),
    ], className="g-2")

    body = html.Div([top_row, regime_row, cross_row, middle_row, bottom_row])

    return header, body


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--host", default="0.0.0.0")
    _p.add_argument("--port", type=int, default=8060)
    _args = _p.parse_args()
    print(f"AGATE / BERYL Live Monitor -- http://{_args.host}:{_args.port}")
    app.run(
        host=_args.host,
        port=_args.port,
        debug=False,
        use_reloader=False,
    )
