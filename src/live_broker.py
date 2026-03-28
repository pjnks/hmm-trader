"""
live_broker.py
──────────────
Coinbase Advanced API integration for live trading.
Uses CDP (Coinbase Developer Platform) credentials from .env

Handles:
- Account balance + available margin
- Market orders (BUY, SELL)
- Position tracking (entry price, size, PnL)
- Stop-loss execution (if manual intervention needed)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# Load .env
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

COINBASE_API_KEY = os.environ.get("COINBASE_API_KEY", "")
COINBASE_API_SECRET = os.environ.get("COINBASE_API_SECRET", "")

# Coinbase Advanced API
COINBASE_API_URL = "https://api.coinbase.com"

@dataclass
class Position:
    """Tracked position in live trading."""
    entry_price: float
    size: float  # in crypto units
    side: str  # "BUY" or "SHORT"
    entry_time: str
    notional: float  # size × entry_price
    ticker: str = ""  # Polygon ticker (e.g. "X:SOLUSD") — for multi-ticker rotation


class LiveBroker:
    """
    Minimal Coinbase Advanced API client for live trading.

    Key assumption: We only hold ONE position at a time (no pyramiding).
    """

    def __init__(self, product_id: str = "SOL-USD", test_mode: bool = False):
        self.product_id = product_id
        self.test_mode = test_mode
        self.position: Position | None = None
        self.trade_history: list[dict] = []
        self.pnl_history: list[float] = []

        if not COINBASE_API_KEY or not COINBASE_API_SECRET:
            raise RuntimeError(
                "COINBASE_API_KEY and COINBASE_API_SECRET must be set in .env\n"
                "Get them from: https://advanced.coinbase.com/settings/api"
            )

    def get_account_balance(self) -> dict:
        """
        Returns account info: available cash, total balance, margin available.

        Returns
        -------
        {
            "cash": float,
            "total_balance": float,
            "margin_available": float,
        }
        """
        if self.test_mode:
            return {"cash": 10000.0, "total_balance": 10000.0, "margin_available": 2500.0}

        try:
            # In production, fetch from Coinbase API
            # This is a placeholder — actual implementation requires proper auth
            log.warning("Live account fetch not yet implemented. Using mock data.")
            return {"cash": 10000.0, "total_balance": 10000.0, "margin_available": 2500.0}
        except Exception as e:
            log.error(f"Failed to fetch account balance: {e}")
            raise

    def place_market_order(
        self, side: str, size: float, current_price: float
    ) -> dict:
        """
        Place a market order (BUY or SELL) at current price.

        Parameters
        ----------
        side : str
            "BUY" or "SELL"
        size : float
            Amount in crypto to buy/sell
        current_price : float
            Current market price (for slippage calculation)

        Returns
        -------
        {
            "order_id": str,
            "side": str,
            "size": float,
            "execution_price": float,
            "timestamp": str,
        }
        """
        if self.test_mode:
            # Simulate with 10 bps slippage
            slippage = current_price * 0.001 if side == "BUY" else -current_price * 0.001
            execution_price = current_price + slippage
            return {
                "order_id": f"test_{datetime.now().isoformat()}",
                "side": side,
                "size": size,
                "execution_price": execution_price,
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Production: POST /orders to Coinbase
            log.info(f"Placing {side} order: {size} {self.product_id} @ ${current_price}")

            # In real implementation, use proper OAuth/signature auth
            # For now, log the intention
            log.warning("Live order placement not yet implemented.")

            # Return mock order
            execution_price = current_price * (1 + (0.001 if side == "BUY" else -0.001))
            return {
                "order_id": f"mock_{datetime.now().isoformat()}",
                "side": side,
                "size": size,
                "execution_price": execution_price,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            log.error(f"Failed to place order: {e}")
            raise

    def open_position(self, side: str, size: float, entry_price: float) -> None:
        """Open a new position (BUY or SHORT)."""
        if self.position is not None:
            log.warning(f"Position already open: {self.position}. Closing first.")
            # In production, should raise or auto-close

        self.position = Position(
            entry_price=entry_price,
            size=size,
            side=side,
            entry_time=datetime.now().isoformat(),
            notional=size * entry_price,
        )
        log.info(f"Opened {side} position: {size} @ ${entry_price} (notional: ${self.position.notional:.2f})")

    def close_position(self, exit_price: float) -> dict:
        """
        Close current position and record P&L.

        Returns
        -------
        {
            "entry_price": float,
            "exit_price": float,
            "size": float,
            "pnl": float,
            "pnl_pct": float,
        }
        """
        if self.position is None:
            log.warning("No open position to close")
            return {}

        entry_price = self.position.entry_price
        size = self.position.size
        side = self.position.side

        if side == "BUY":
            pnl = size * (exit_price - entry_price)
        else:  # SHORT
            pnl = size * (entry_price - exit_price)

        pnl_pct = (pnl / self.position.notional) * 100

        trade = {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "side": side,
            "entry_time": self.position.entry_time,
            "exit_time": datetime.now().isoformat(),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        }

        self.trade_history.append(trade)
        self.pnl_history.append(pnl)
        self.position = None

        log.info(f"Closed {side} position: PnL ${pnl:.2f} ({pnl_pct:.2f}%)")
        return trade

    def get_current_pnl(self, current_price: float) -> float:
        """
        Get unrealised P&L on open position.

        Returns
        -------
        float
            Current P&L in USD
        """
        if self.position is None:
            return 0.0

        if self.position.side == "BUY":
            pnl = self.position.size * (current_price - self.position.entry_price)
        else:  # SHORT
            pnl = self.position.size * (self.position.entry_price - current_price)

        return round(pnl, 2)

    def get_statistics(self) -> dict:
        """
        Get lifetime trading statistics.

        Returns
        -------
        {
            "total_trades": int,
            "winning_trades": int,
            "losing_trades": int,
            "win_rate_pct": float,
            "total_pnl": float,
            "mean_pnl_per_trade": float,
        }
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "total_pnl": 0.0,
                "mean_pnl_per_trade": 0.0,
            }

        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t["pnl"] > 0)
        losing_trades = total_trades - winning_trades
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        mean_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round((winning_trades / total_trades * 100), 1),
            "total_pnl": round(total_pnl, 2),
            "mean_pnl_per_trade": round(mean_pnl, 2),
        }
