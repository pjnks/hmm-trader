"""
diamond_bridge.py
─────────────────
Cross-reference layer between DIAMOND (Kalshi anomaly detection) and both
AGATE (crypto) and CITRINE (NDX equity portfolio).

Two interfaces:
  1. AGATE (crypto):   get_diamond_boost("X:BTCUSD") → float multiplier
  2. CITRINE (equity):  CitrineDiamondBridge.fetch_boosts(["NVDA","AAPL",...]) → dict[str,float]

Mapping categories for CITRINE:
  - Earnings: Kalshi earnings markets → individual equity tickers
  - Crypto: BTC/ETH anomalies → crypto-correlated equities (MSTR, COIN)
  - Energy: WTI oil anomalies → energy sector tickers
  - Sector: Chip export restrictions → semiconductor tickers
  - Macro: Fed/CPI/GDP → small broad NDX impact (all tickers, reduced weight)

The DIAMOND DB lives at:
  Local:  /Users/perryjenkins/Documents/quant/diamond/diamond_trades.db
  VM:     /home/ubuntu/kalshi-diamond/diamond_trades.db
"""

from __future__ import annotations

import json
import logging
import platform
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Shared: DIAMOND DB path detection
# ═══════════════════════════════════════════════════════════════════════════

_IS_MAC = platform.system() == "Darwin"

_DB_PATHS = [
    Path("/home/ubuntu/kalshi-diamond/diamond_trades.db"),
    Path("/Users/perryjenkins/Documents/quant/diamond/diamond_trades.db"),
]


def _find_db() -> Path | None:
    """Return the first existing DIAMOND DB path, or None."""
    for p in _DB_PATHS:
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════
# AGATE interface (crypto tickers) — backward-compatible
# ═══════════════════════════════════════════════════════════════════════════

KALSHI_CRYPTO_MAP = {
    "KXBTC":    "X:BTCUSD",
    "KXBTCD":   "X:BTCUSD",
    "KXBTCW":   "X:BTCUSD",
    "KXBTCM":   "X:BTCUSD",
    "KXETH":    "X:ETHUSD",
    "KXETHD":   "X:ETHUSD",
    "KXSOL":    "X:SOLUSD",
    "KXSOLD":   "X:SOLUSD",
    "KXXRP":    "X:XRPUSD",
    "KXCRYPTO":  None,
}

_CRYPTO_TO_KALSHI: dict[str, list[str]] = {}
for _kp, _ct in KALSHI_CRYPTO_MAP.items():
    if _ct:
        _CRYPTO_TO_KALSHI.setdefault(_ct, []).append(_kp)


def get_recent_anomalies(
    crypto_ticker: str,
    hours: int = 24,
    min_score: float = 0.3,
) -> list[dict]:
    """Query DIAMOND anomalies for a crypto ticker (AGATE interface)."""
    kalshi_prefixes = _CRYPTO_TO_KALSHI.get(crypto_ticker, [])
    if not kalshi_prefixes:
        return []

    db_path = _find_db()
    if db_path is None:
        log.debug("DIAMOND DB not found")
        return []

    cutoff = time.time() - (hours * 3600)

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        like_clauses = " OR ".join(
            f"ticker LIKE '{prefix}%'" for prefix in kalshi_prefixes
        )
        rows = conn.execute(
            f"SELECT ticker, score, alert_level, ts "
            f"FROM anomalies "
            f"WHERE ({like_clauses}) AND ts >= ? AND score >= ? "
            f"ORDER BY ts DESC LIMIT 50",
            (cutoff, min_score),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        log.debug(f"DIAMOND query failed for {crypto_ticker}: {e}")
        return []


def get_diamond_boost(crypto_ticker: str, hours: int = 24) -> float:
    """
    AGATE interface: confidence boost based on DIAMOND anomalies.
    Returns multiplier: 1.0 (neutral) to 1.3 (CRITICAL anomalies).
    """
    anomalies = get_recent_anomalies(crypto_ticker, hours=hours, min_score=0.3)
    if not anomalies:
        return 1.0

    alert_levels = [a.get("alert_level", "LOG") for a in anomalies]
    max_score = max(a.get("score", 0) for a in anomalies)

    if "CRITICAL" in alert_levels or max_score >= 0.9:
        boost = 1.3
    elif "ALERT" in alert_levels or max_score >= 0.7:
        boost = 1.2
    elif "NOTABLE" in alert_levels or max_score >= 0.5:
        boost = 1.1
    else:
        boost = 1.05

    log.info(f"  DIAMOND boost for {crypto_ticker.replace('X:','')}: "
             f"{boost:.1f}x ({len(anomalies)} anomalies, max_score={max_score:.2f})")
    return boost


def get_diamond_summary() -> dict:
    """Summary of recent DIAMOND activity across all crypto tickers."""
    summary = {}
    for crypto_ticker in _CRYPTO_TO_KALSHI:
        anomalies = get_recent_anomalies(crypto_ticker, hours=24, min_score=0.3)
        if anomalies:
            summary[crypto_ticker] = {
                "count": len(anomalies),
                "max_score": max(a.get("score", 0) for a in anomalies),
                "latest_ts": anomalies[0].get("ts", 0),
                "alert_levels": list(set(
                    a.get("alert_level", "LOG") for a in anomalies
                )),
            }
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# CITRINE interface (NDX equity tickers)
# ═══════════════════════════════════════════════════════════════════════════

# Kalshi prefix → equity mapping for indirect correlations
_CRYPTO_EQUITIES = ["MSTR", "COIN"]
_ENERGY_EQUITIES = ["FANG", "BKR"]
_SEMICONDUCTOR_TICKERS = [
    "NVDA", "AMD", "AVGO", "QCOM", "LRCX", "KLAC",
    "AMAT", "MCHP", "NXPI", "TXN", "ADI",
]

# (category, affected_tickers, direction_sign)
# direction_sign: +1 = anomaly is bullish for equities, -1 = bearish
_PREFIX_MAP: dict[str, tuple[str, list[str], float]] = {
    "KXBTC":  ("crypto",  _CRYPTO_EQUITIES, +1.0),
    "KXETH":  ("crypto",  _CRYPTO_EQUITIES, +1.0),
    "KXWTI":  ("energy",  _ENERGY_EQUITIES, +1.0),
    "KXCHIP": ("sector",  _SEMICONDUCTOR_TICKERS, -1.0),
}

# Earnings: substring in Kalshi ticker → NDX equity
_EARNINGS_MAP = {
    "NVDA": "NVDA", "AAPL": "AAPL", "MSFT": "MSFT",
    "AMZN": "AMZN", "GOOG": "GOOGL", "META": "META", "TSLA": "TSLA",
}

# Macro prefixes → small boost/penalty across ALL tickers
_MACRO_PREFIXES = {"KXFED", "KXCPI", "KXGDP", "KXUNRA"}
_MACRO_WEIGHT = 0.3


class CitrineDiamondBridge:
    """
    Maps DIAMOND Kalshi anomaly scores to CITRINE alt-data boost multipliers.

    Usage in live_trading_citrine.py:
        bridge = CitrineDiamondBridge()
        diamond_boosts = bridge.fetch_boosts(["NVDA", "AAPL", "MSTR", ...])
        # Returns: {"MSTR": 1.15, "COIN": 1.15}
    """

    def __init__(self, db_path: str | None = None, window_hours: int = 24):
        self.db_path = Path(db_path) if db_path else _find_db()
        self.window_seconds = window_hours * 3600

    def fetch_boosts(self, equity_tickers: list[str]) -> dict[str, float]:
        """
        Query recent DIAMOND anomalies and map to CITRINE boost multipliers.

        Returns dict[ticker → float] where float in [0.7, 1.5].
        Tickers absent from dict = 1.0 (neutral).
        Gracefully returns {} if DB unavailable.
        """
        if self.db_path is None or not self.db_path.exists():
            log.debug("DIAMOND DB not available for CITRINE bridge")
            return {}

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            cutoff = time.time() - self.window_seconds

            rows = conn.execute(
                "SELECT ticker, score, alert_level, ts FROM anomalies "
                "WHERE ts > ? AND alert_level IN ('NOTABLE', 'ALERT', 'CRITICAL') "
                "ORDER BY ts DESC LIMIT 500",
                (cutoff,),
            ).fetchall()
            conn.close()
        except Exception as e:
            log.warning(f"[DiamondBridge] DB read failed: {e}")
            return {}

        if not rows:
            return {}

        # Group anomaly scores by equity ticker
        ticker_set = set(equity_tickers)
        equity_scores: dict[str, list[float]] = {}

        for row in rows:
            kalshi_ticker = row["ticker"]
            score = row["score"]
            mapped = self._map_to_equities(kalshi_ticker, score, ticker_set)
            for eq_ticker, weighted_score in mapped:
                equity_scores.setdefault(eq_ticker, []).append(weighted_score)

        # Convert aggregated scores to boost multipliers
        boosts = {}
        for ticker, scores in equity_scores.items():
            max_score = max(scores, key=abs)
            # Map: score in [-1, 1] → boost in [0.7, 1.5]
            boost = 1.0 + max_score * 0.5
            boost = max(0.7, min(1.5, boost))
            boosts[ticker] = round(boost, 3)

        if boosts:
            log.info(f"  [DiamondBridge] {len(boosts)} equity boosts from "
                     f"{len(rows)} Kalshi anomalies:")
            for t, b in sorted(boosts.items(),
                               key=lambda x: abs(x[1] - 1.0), reverse=True):
                indicator = "+" if b > 1.0 else "-" if b < 1.0 else "="
                log.info(f"    {indicator} {t}: {b:.3f}x")

        return boosts

    def _map_to_equities(
        self, kalshi_ticker: str, score: float, universe: set[str],
    ) -> list[tuple[str, float]]:
        """Map a Kalshi ticker to (equity_ticker, weighted_score) pairs."""
        results = []
        upper = kalshi_ticker.upper()

        # 1. Earnings markets (e.g. "NVDA-3DEC-EARNINGS")
        for prefix, equity in _EARNINGS_MAP.items():
            if prefix in upper and "EARN" in upper:
                if equity in universe:
                    results.append((equity, score))
                return results

        # 2. Prefix-based: crypto, energy, sector
        for prefix, (category, tickers, direction) in _PREFIX_MAP.items():
            if upper.startswith(prefix):
                for t in tickers:
                    if t in universe:
                        results.append((t, score * direction))
                return results

        # 3. Macro prefixes → all tickers with reduced weight
        for macro_prefix in _MACRO_PREFIXES:
            if upper.startswith(macro_prefix):
                for t in universe:
                    results.append((t, score * _MACRO_WEIGHT))
                return results

        return results
