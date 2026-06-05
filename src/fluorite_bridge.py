"""
fluorite_bridge.py
──────────────────
Cross-reference layer between FLUORITE (alt-data alpha screener) and
BERYL/CITRINE (HMM regime traders).

Maps FLUORITE Tier 2 composite scores to confidence boost multipliers.
Designed as a shadow-logging experiment (6 weeks, mid-July 2026 eval)
to measure whether FLUORITE's cross-sectional signals improve BERYL/CITRINE's
time-series entry quality.

Data transport:
  Mac:  reads screening.db directly (FLUORITE and trading-core on same machine)
  VM:   reads fluorite_bridge_scores.json (scp'd from Mac by scorer script)

Fails open: returns {} on any error (neutral 1.0x for all tickers).
"""

from __future__ import annotations

import json
import logging
import platform
import sqlite3
import time
from datetime import datetime, date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_IS_MAC = platform.system() == "Darwin"

# JSON scores file (written by FLUORITE scorer, scp'd to VM)
_JSON_PATHS = [
    Path("/home/ubuntu/HMM-Trader/data/fluorite_bridge_scores.json"),
    Path("/Users/perryjenkins/Documents/quant/trading-core/data/fluorite_bridge_scores.json"),
]

# Direct DB access (Mac only — FLUORITE doesn't run on VM)
_DB_PATHS = [
    Path("/Users/perryjenkins/Documents/quant/fluorite/data/screening.db"),
]

# Multiplier thresholds — conservative for shadow phase.
# Recalibrate after 6-week eval using shadow journal P&L attribution.
_SCORE_TIERS = [
    (65.0, 1.15),   # strong signal (top ~30 tickers)
    (55.0, 1.08),   # moderate (top ~100)
    (45.0, 1.03),   # weak positive
]
_NEUTRAL = 1.0
_MAX_STALENESS_HOURS = 36


def _find_json() -> Path | None:
    for p in _JSON_PATHS:
        if p.exists():
            return p
    return None


def _find_db() -> Path | None:
    for p in _DB_PATHS:
        if p.exists():
            return p
    return None


def _score_to_multiplier(score: float) -> float:
    for threshold, mult in _SCORE_TIERS:
        if score >= threshold:
            return mult
    return _NEUTRAL


class FluoriteBridge:
    """
    Maps FLUORITE cross-sectional scores to BERYL/CITRINE boost multipliers.

    Usage in live_trading_citrine.py:
        bridge = FluoriteBridge()
        boosts = bridge.fetch_boosts(["NVDA", "AAPL", "MSTR", ...])
        # Returns: {"NVDA": 1.08, "MELI": 1.15}

    Usage with shadow logging:
        details = bridge.fetch_boosts_with_detail(["NVDA", "AAPL"])
        # Returns: {"NVDA": {"composite_score": 58.2, "boost": 1.08, ...}}
    """

    def __init__(
        self,
        json_path: str | None = None,
        db_path: str | None = None,
        max_age_hours: int = _MAX_STALENESS_HOURS,
    ):
        self._json_path = Path(json_path) if json_path else None
        self._db_path = Path(db_path) if db_path else None
        self._max_age_hours = max_age_hours

    def fetch_boosts(self, equity_tickers: list[str]) -> dict[str, float]:
        """
        Return dict[ticker → multiplier] for tickers in FLUORITE's universe.
        Multiplier in [1.0, 1.15]. Absent tickers = 1.0 (neutral).
        Returns {} on any error (fail-open).
        """
        detail = self.fetch_boosts_with_detail(equity_tickers)
        return {t: d["boost"] for t, d in detail.items() if d["boost"] != _NEUTRAL}

    def fetch_boosts_with_detail(
        self, equity_tickers: list[str],
    ) -> dict[str, dict]:
        """
        Return full signal detail for shadow logging.
        dict[ticker → {composite_score, insider_score, tripwire_score,
                        cluster_detected, has_earnings_event, boost}]
        """
        scores = self._load_scores(equity_tickers)
        if not scores:
            return {}

        result = {}
        for ticker in equity_tickers:
            if ticker not in scores:
                continue
            s = scores[ticker]
            result[ticker] = {
                "composite_score": s.get("composite_score", 0),
                "insider_score": s.get("insider_score", 50.0),
                "tripwire_score": s.get("tripwire_score", 50.0),
                "cluster_detected": s.get("cluster_detected", 0),
                "has_earnings_event": s.get("has_earnings_event", 0),
                "boost": _score_to_multiplier(s.get("composite_score", 0)),
            }

        boosted = {t: d for t, d in result.items() if d["boost"] != _NEUTRAL}
        if boosted:
            log.info(f"  [FluoriteBridge] {len(boosted)} equity boosts from "
                     f"{len(scores)} scored tickers:")
            for t, d in sorted(boosted.items(),
                               key=lambda x: x[1]["composite_score"],
                               reverse=True)[:10]:
                flags = []
                if d["cluster_detected"]:
                    flags.append("cluster")
                if d["has_earnings_event"]:
                    flags.append("earnings")
                flag_str = f" [{'+'.join(flags)}]" if flags else ""
                log.info(f"    {t}: {d['boost']:.2f}x "
                         f"(score={d['composite_score']:.1f}{flag_str})")

        return result

    def _load_scores(
        self, tickers: list[str],
    ) -> dict[str, dict]:
        """Load scores from best available source (JSON first, DB fallback)."""
        # Try JSON first (works on both Mac and VM)
        scores = self._load_from_json(tickers)
        if scores is not None:
            return scores

        # Fallback to direct DB (Mac only)
        scores = self._load_from_db(tickers)
        if scores is not None:
            return scores

        log.debug("[FluoriteBridge] No data source available")
        return {}

    def _load_from_json(
        self, tickers: list[str],
    ) -> dict[str, dict] | None:
        json_path = self._json_path or _find_json()
        if json_path is None or not json_path.exists():
            return None

        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            log.warning(f"[FluoriteBridge] JSON read failed: {e}")
            return None

        # Staleness check
        exported_at = data.get("exported_at", "")
        if exported_at:
            try:
                export_dt = datetime.fromisoformat(exported_at)
                age_hours = (datetime.now() - export_dt).total_seconds() / 3600
                if age_hours > self._max_age_hours:
                    log.warning(f"[FluoriteBridge] JSON is {age_hours:.0f}h old "
                                f"(limit {self._max_age_hours}h), returning neutral")
                    return {}
            except ValueError:
                pass

        all_scores = data.get("scores", {})
        ticker_set = set(tickers)
        return {t: s for t, s in all_scores.items() if t in ticker_set}

    def _load_from_db(
        self, tickers: list[str],
    ) -> dict[str, dict] | None:
        db_path = self._db_path or _find_db()
        if db_path is None or not db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(db_path), timeout=2)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=2000")
            conn.row_factory = sqlite3.Row

            # Get the most recent score date
            row = conn.execute(
                "SELECT MAX(date) as latest FROM tier2_scores"
            ).fetchone()
            if not row or not row["latest"]:
                conn.close()
                return None

            latest_date = row["latest"]

            # Staleness check
            try:
                score_dt = datetime.strptime(latest_date, "%Y-%m-%d")
                age_hours = (datetime.now() - score_dt).total_seconds() / 3600
                if age_hours > self._max_age_hours:
                    log.warning(f"[FluoriteBridge] DB scores from {latest_date} "
                                f"({age_hours:.0f}h old), returning neutral")
                    conn.close()
                    return {}
            except ValueError:
                pass

            placeholders = ",".join("?" for _ in tickers)
            rows = conn.execute(
                f"SELECT ticker, composite_score, insider_score, tripwire_score, "
                f"       cluster_detected, has_earnings_event "
                f"FROM tier2_scores "
                f"WHERE date = ? AND ticker IN ({placeholders})",
                [latest_date] + list(tickers),
            ).fetchall()
            conn.close()

            return {
                r["ticker"]: {
                    "composite_score": r["composite_score"] or 0,
                    "insider_score": r["insider_score"] or 50.0,
                    "tripwire_score": r["tripwire_score"] or 50.0,
                    "cluster_detected": r["cluster_detected"] or 0,
                    "has_earnings_event": r["has_earnings_event"] or 0,
                }
                for r in rows
            }
        except Exception as e:
            log.warning(f"[FluoriteBridge] DB read failed: {e}")
            return None
