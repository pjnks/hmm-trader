"""
regime_mapper.py
────────────────
Maps (regime_cat, confidence) → StrategyDirection.

The RegimeMapper is the central decision point for multi-direction trading.
It reads the mapping table from config.REGIME_DIRECTION_MAP and converts
each bar's regime classification + HMM posterior confidence into an allowed
trading direction.

Usage
─────
    from src.regime_mapper import RegimeMapper

    mapper = RegimeMapper()
    direction = mapper.get_direction("BULL", 0.92)
    # → StrategyDirection.LONG
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

from src.types import StrategyDirection


class RegimeMapper:
    """Stateless mapper: (regime_cat, confidence) → StrategyDirection."""

    def get_direction(
        self,
        regime_cat: str,
        confidence: float,
    ) -> StrategyDirection:
        """
        Look up the allowed trading direction for a regime + confidence.

        Parameters
        ----------
        regime_cat : "BULL" | "BEAR" | "CHOP"
        confidence : HMM posterior probability (0.0 – 1.0)

        Returns
        -------
        StrategyDirection
        """
        conf_level = (
            "high" if confidence >= config.CONFIDENCE_HIGH_THRESHOLD else "med"
        )

        # Try exact (regime, conf_level) match first
        direction_str = config.REGIME_DIRECTION_MAP.get((regime_cat, conf_level))

        # Fall back to (regime, "any") for regimes like CHOP
        if direction_str is None:
            direction_str = config.REGIME_DIRECTION_MAP.get((regime_cat, "any"))

        # Ultimate fallback: FLAT
        if direction_str is None:
            return StrategyDirection.FLAT

        return StrategyDirection(direction_str)
