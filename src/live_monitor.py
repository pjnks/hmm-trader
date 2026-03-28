"""
live_monitor.py
───────────────
Live trading monitor for risk management + daily reporting.

Tracks:
- Rolling Sharpe (last 20 trades)
- Daily/weekly P&L
- Win rate
- Kill-switch triggers (auto-stop if degrading)
- Daily email alerts
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record for database."""
    timestamp: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    side: str  # BUY or SHORT
    size: float
    pnl: float
    pnl_pct: float
    signal_strength: int  # # confirmations
    ticker: str = ""  # Polygon ticker (e.g. "X:SOLUSD") — for multi-ticker rotation


class LiveMonitor:
    """
    Monitor live trading performance and detect kill-switch conditions.

    Kill-Switch Rules (automatic):
    1. Daily loss > 2% of account
    2. Rolling 20-trade Sharpe < 0.3
    3. No positive trades in 10 consecutive trades
    """

    def __init__(self, db_path: str = "paper_trades.db"):
        self.db_path = Path(db_path)
        self._init_db()
        self.kill_switch_triggered = False
        self.kill_switch_reason = None

    def _init_db(self) -> None:
        """Initialize SQLite database for trade logging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    signal_strength INTEGER NOT NULL,
                    ticker TEXT NOT NULL DEFAULT ''
                )
            """)
            # Migrate existing DBs: add ticker column if missing
            try:
                conn.execute("SELECT ticker FROM trades LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE trades ADD COLUMN ticker TEXT NOT NULL DEFAULT ''")

            # Open positions table — tracks entries for paper P&L and state recovery
            conn.execute("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    signal_strength INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    def log_trade(self, trade: TradeRecord) -> None:
        """Log a closed trade to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades
                (timestamp, entry_time, exit_time, entry_price, exit_price, side, size, pnl, pnl_pct, signal_strength, ticker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.timestamp,
                    trade.entry_time,
                    trade.exit_time,
                    trade.entry_price,
                    trade.exit_price,
                    trade.side,
                    trade.size,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.signal_strength,
                    trade.ticker,
                ),
            )
            conn.commit()
        log.info(f"Trade logged: {trade.ticker} {trade.side} ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")

    def log_entry(self, ticker: str, entry_price: float, size: float,
                  side: str = "BUY", signal_strength: int = 0) -> None:
        """Log an open position entry for paper P&L tracking.

        Deduplication: skips insert if ticker already exists in open_positions.
        This prevents phantom entries when log_entry() is called on every scan
        cycle instead of just the first entry.
        """
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM open_positions WHERE ticker = ?", (ticker,)
            ).fetchone()[0]
            if existing > 0:
                return  # Already tracked — skip duplicate
            ts = datetime.now(tz=timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO open_positions
                   (timestamp, ticker, entry_price, size, side, signal_strength)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ts, ticker, entry_price, size, side, signal_strength),
            )
            conn.commit()
        log.info(f"Entry logged: {ticker} {side} {size:.4f} @ ${entry_price:.2f}")

    def clear_entry(self, ticker: str) -> None:
        """Remove an open position entry (called on exit)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM open_positions WHERE ticker = ?", (ticker,))
            conn.commit()

    def get_open_positions(self) -> list[dict]:
        """Get all open position entries (for state recovery on restart)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM open_positions ORDER BY timestamp ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trades(self, since_hours: int = 24) -> pd.DataFrame:
        """Get trades from last N hours."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=since_hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC",
                conn,
                params=(cutoff,),
            )

        return df

    def check_kill_switch(self, account_equity: float, initial_equity: float = 10000.0) -> bool:
        """
        Check if kill-switch conditions are met.

        Returns
        -------
        bool
            True if any kill-switch rule triggered
        """
        trades = self.get_trades(since_hours=24)

        if len(trades) == 0:
            return False

        # ── Rule 1: Daily loss > 2% ─────────────────────────────────
        daily_pnl = trades["pnl"].sum()
        daily_loss_pct = (daily_pnl / initial_equity) * 100

        if daily_loss_pct < -2.0:
            self.kill_switch_triggered = True
            self.kill_switch_reason = f"Daily loss > 2%: {daily_loss_pct:.2f}%"
            log.error(f"🚨 KILL-SWITCH TRIGGERED: {self.kill_switch_reason}")
            return True

        # ── Rule 2: Rolling Sharpe (last 20 trades) < 0.3 ──────────
        if len(trades) >= 20:
            last_20_pnl = trades.head(20)["pnl"].values
            sharpe_20 = np.mean(last_20_pnl) / (np.std(last_20_pnl) + 1e-6)

            if sharpe_20 < 0.3:
                self.kill_switch_triggered = True
                self.kill_switch_reason = f"Rolling 20-trade Sharpe < 0.3: {sharpe_20:.3f}"
                log.error(f"🚨 KILL-SWITCH TRIGGERED: {self.kill_switch_reason}")
                return True

        # ── Rule 3: No wins in last 10 trades ───────────────────────
        if len(trades) >= 10:
            last_10_wins = (trades.head(10)["pnl"] > 0).sum()

            if last_10_wins == 0:
                self.kill_switch_triggered = True
                self.kill_switch_reason = "0/10 winning trades (streak detected)"
                log.error(f"🚨 KILL-SWITCH TRIGGERED: {self.kill_switch_reason}")
                return True

        return False

    def get_daily_metrics(self) -> dict:
        """
        Calculate daily performance metrics for email report.

        Returns
        -------
        {
            "date": str,
            "total_pnl": float,
            "pnl_pct": float,
            "num_trades": int,
            "win_rate": float,
            "sharpe_20": float,  # rolling 20-trade
            "max_dd_today": float,
        }
        """
        trades = self.get_trades(since_hours=24)

        if len(trades) == 0:
            return {
                "date": datetime.now().date().isoformat(),
                "total_pnl": 0.0,
                "pnl_pct": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "sharpe_20": 0.0,
                "max_dd_today": 0.0,
            }

        total_pnl = trades["pnl"].sum()
        num_trades = len(trades)
        win_rate = (trades["pnl"] > 0).sum() / num_trades * 100
        pnl_pct = (total_pnl / 10000.0) * 100  # as % of initial capital

        # Rolling Sharpe
        sharpe_20 = 0.0
        if len(trades) >= 20:
            last_20_pnl = trades.head(20)["pnl"].values
            sharpe_20 = np.mean(last_20_pnl) / (np.std(last_20_pnl) + 1e-6)

        # Max drawdown today (cumsum then drawdown)
        cumsum = trades["pnl"][::-1].cumsum()
        running_max = cumsum.cummax()
        dd = (cumsum - running_max)
        max_dd_today = dd.min() if not dd.empty else 0.0

        return {
            "date": datetime.now().date().isoformat(),
            "total_pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "num_trades": num_trades,
            "win_rate": round(win_rate, 1),
            "sharpe_20": round(sharpe_20, 3),
            "max_dd_today": round(max_dd_today, 2),
        }

    def generate_email_body(self, current_price: float = 0.0, position_size: float = 0.0) -> str:
        """
        Generate daily email report content.

        Parameters
        ----------
        current_price : float
            Current market price (for position P&L)
        position_size : float
            Current position size (if any)

        Returns
        -------
        str
            Email body (plain text)
        """
        metrics = self.get_daily_metrics()

        body = f"""
HMM TRADER — Daily Report
{'='*60}

Date: {metrics['date']}

Performance:
  Total P&L: ${metrics['total_pnl']:.2f} ({metrics['pnl_pct']:.2f}%)
  Trades: {metrics['num_trades']}
  Win Rate: {metrics['win_rate']:.1f}%
  Rolling Sharpe (20): {metrics['sharpe_20']:.3f}
  Max Drawdown Today: ${metrics['max_dd_today']:.2f}

Status: {'🚨 KILL-SWITCH TRIGGERED' if self.kill_switch_triggered else '✅ LIVE'}
{f'Reason: {self.kill_switch_reason}' if self.kill_switch_triggered else ''}

Current Position:
  Size: {position_size:.4f} SOL
  Current Price: ${current_price:.2f}
  Position P&L: ${position_size * current_price:.2f} (unrealised)

Next Check: Daily report at 9am UTC

{'='*60}
Review dashboard: http://localhost:8050
        """.strip()

        return body
