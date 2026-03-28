"""
types.py
────────
Shared type definitions for the HMM Regime Trader.
"""

from __future__ import annotations
from enum import Enum


class StrategyDirection(Enum):
    """Allowed trading direction for a given regime + confidence level."""
    LONG          = "LONG"           # only long entries allowed
    SHORT         = "SHORT"          # only short entries allowed
    FLAT          = "FLAT"           # no entries, close any position
    LONG_OR_FLAT  = "LONG_OR_FLAT"   # long if indicators confirm, else flat
    SHORT_OR_FLAT = "SHORT_OR_FLAT"  # short if indicators confirm, else flat

    @property
    def allows_long(self) -> bool:
        return self in (StrategyDirection.LONG, StrategyDirection.LONG_OR_FLAT)

    @property
    def allows_short(self) -> bool:
        return self in (StrategyDirection.SHORT, StrategyDirection.SHORT_OR_FLAT)

    @property
    def is_flat(self) -> bool:
        return self == StrategyDirection.FLAT
