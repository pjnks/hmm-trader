"""
strategy.py
───────────
Signal generation engine — supports LONG + SHORT directions.

Entry rules (ALL must hold simultaneously)
───────────────────────────────────────────
  1. RegimeMapper maps (regime_cat, confidence) → allowed direction
  2. Not currently in a cooldown after the last exit
  3. Indicator confirmations pass for the relevant direction

     ┌──────────────────────────────────────────────────────────────────────┐
     │ # │ Indicator    │ LONG confirm if …           │ SHORT confirm if … │
     ├───┼──────────────┼─────────────────────────────┼────────────────────┤
     │ 1 │ RSI          │ 30 < rsi < 70 (same)        │ 30 < rsi < 70      │
     │ 2 │ Momentum     │ momentum > 0                │ momentum < 0       │
     │ 3 │ Volatility   │ vol < 2× median (same)      │ vol < 2× median    │
     │ 4 │ Volume       │ volume_ratio > 1.1× (same)  │ volume_ratio > 1.1×│
     │ 5 │ ADX          │ adx > 25 (same)             │ adx > 25           │
     │ 6 │ Price Trend  │ Close > SMA-50              │ Close < SMA-50     │
     │ 7 │ MACD         │ macd > signal               │ macd < signal      │
     │ 8 │ Stochastic   │ %K < 80                     │ %K > 20            │
     └──────────────────────────────────────────────────────────────────────┘

Exit rules
──────────
  Legacy mode (use_regime_mapper=False):
    1. Immediate exit when HMM regime category flips to BEAR.

  Multi-direction mode (use_regime_mapper=True):
    1. LONG position: exit when direction no longer allows LONG
    2. SHORT position: exit when direction no longer allows SHORT

Cooldown
────────
  After every exit, no new entry is allowed for COOLDOWN_HOURS hours.

Public API
──────────
  engine = SignalEngine(use_regime_mapper=True)
  signal = engine.evaluate(row, regime_cat, confidence, position_side, last_exit_ts)
    → SignalResult(action, confirmations, details, direction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.types import StrategyDirection

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    action:        str                # "BUY" | "SELL" | "SHORT" | "COVER" | "HOLD" | "COOLDOWN"
    confirmations: int                # number of indicators that fired
    score:         float              # confirmations / TOTAL_SIGNALS
    regime_cat:    str                # BULL / BEAR / CHOP
    confidence:    float              # HMM posterior for current state
    details:       dict[str, bool]    = field(default_factory=dict)
    reason:        str                = ""
    direction:     Optional[str]      = None   # "LONG" | "SHORT" | None

    @property
    def is_entry(self) -> bool:
        return self.action in ("BUY", "SHORT")

    @property
    def is_exit(self) -> bool:
        return self.action in ("SELL", "COVER")

    @property
    def is_long_entry(self) -> bool:
        return self.action == "BUY"

    @property
    def is_short_entry(self) -> bool:
        return self.action == "SHORT"


# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """Signal evaluator — supports both legacy long-only and multi-direction."""

    def __init__(self, use_regime_mapper: bool = False) -> None:
        self.use_regime_mapper = use_regime_mapper
        if use_regime_mapper:
            from src.regime_mapper import RegimeMapper
            self._mapper = RegimeMapper()

    # ── LONG entry checks (return True = confirmed) ──────────────────────

    @staticmethod
    def _check_rsi(row: pd.Series) -> bool:
        v = row.get("rsi")
        if pd.isna(v):
            return False
        return config.RSI_LOWER < v < config.RSI_UPPER

    @staticmethod
    def _check_momentum_long(row: pd.Series) -> bool:
        v = row.get("momentum")
        if pd.isna(v):
            return False
        return v > 0

    @staticmethod
    def _check_volatility(row: pd.Series) -> bool:
        vol     = row.get("volatility")
        median  = row.get("vol_median")
        if pd.isna(vol) or pd.isna(median) or median == 0:
            return False
        return vol < config.VOLATILITY_MULT * median

    @staticmethod
    def _check_volume(row: pd.Series) -> bool:
        v = row.get("volume_ratio")
        if pd.isna(v):
            return False
        return v > config.VOLUME_MULT

    @staticmethod
    def _check_adx(row: pd.Series) -> bool:
        v = row.get("adx")
        if pd.isna(v):
            return False
        return v > config.ADX_MIN

    @staticmethod
    def _check_price_trend_long(row: pd.Series) -> bool:
        close = row.get("Close")
        sma   = row.get(f"sma_{config.TREND_MA_PERIOD}")
        if pd.isna(close) or pd.isna(sma):
            return False
        return close > sma

    @staticmethod
    def _check_macd_long(row: pd.Series) -> bool:
        m = row.get("macd")
        s = row.get("macd_signal")
        if pd.isna(m) or pd.isna(s):
            return False
        return m > s

    @staticmethod
    def _check_stochastic_long(row: pd.Series) -> bool:
        k = row.get("stoch_k")
        if pd.isna(k):
            return False
        return k < config.STOCH_UPPER

    # ── SHORT entry checks (4 inverted + 1 asymmetric) ──────────────────

    @staticmethod
    def _check_adx_short(row: pd.Series) -> bool:
        v = row.get("adx")
        if pd.isna(v):
            return False
        return v > config.ADX_MIN_SHORT

    @staticmethod
    def _check_momentum_short(row: pd.Series) -> bool:
        v = row.get("momentum")
        if pd.isna(v):
            return False
        return v < 0

    @staticmethod
    def _check_price_trend_short(row: pd.Series) -> bool:
        close = row.get("Close")
        sma   = row.get(f"sma_{config.TREND_MA_PERIOD}")
        if pd.isna(close) or pd.isna(sma):
            return False
        return close < sma

    @staticmethod
    def _check_macd_short(row: pd.Series) -> bool:
        m = row.get("macd")
        s = row.get("macd_signal")
        if pd.isna(m) or pd.isna(s):
            return False
        return m < s

    @staticmethod
    def _check_stochastic_short(row: pd.Series) -> bool:
        k = row.get("stoch_k")
        if pd.isna(k):
            return False
        return k > (100 - config.STOCH_UPPER)  # mirror: > 20 when STOCH_UPPER=80

    # ── Run all checks for a direction ───────────────────────────────────

    def _run_long_checks(self, row: pd.Series) -> tuple[int, dict[str, bool]]:
        checks: dict[str, bool] = {
            "rsi":         self._check_rsi(row),
            "momentum":    self._check_momentum_long(row),
            "volatility":  self._check_volatility(row),
            "volume":      self._check_volume(row),
            "adx":         self._check_adx(row),
            "price_trend": self._check_price_trend_long(row),
            "macd":        self._check_macd_long(row),
            "stochastic":  self._check_stochastic_long(row),
        }
        return sum(checks.values()), checks

    def _run_short_checks(self, row: pd.Series) -> tuple[int, dict[str, bool]]:
        checks: dict[str, bool] = {
            "rsi":         self._check_rsi(row),            # same
            "momentum":    self._check_momentum_short(row), # inverted
            "volatility":  self._check_volatility(row),     # same
            "volume":      self._check_volume(row),          # same
            "adx":         self._check_adx_short(row),       # asymmetric (ADX_MIN_SHORT)
            "price_trend": self._check_price_trend_short(row),  # inverted
            "macd":        self._check_macd_short(row),         # inverted
            "stochastic":  self._check_stochastic_short(row),   # inverted
        }
        return sum(checks.values()), checks

    # ── Cooldown helper ──────────────────────────────────────────────────

    @staticmethod
    def in_cooldown(current_ts: datetime,
                    last_exit:  Optional[datetime],
                    cooldown_hours: float = None) -> bool:
        if last_exit is None:
            return False
        if cooldown_hours is None:
            cooldown_hours = config.COOLDOWN_HOURS
        delta_h = (current_ts - last_exit).total_seconds() / 3600
        return delta_h < cooldown_hours

    # ── Main evaluate ────────────────────────────────────────────────────

    def evaluate(
        self,
        row:            pd.Series,
        regime_cat:     str,
        confidence:     float,
        in_position:    bool       = False,
        last_exit_ts:   Optional[datetime] = None,
        position_side:  str        = "FLAT",    # "FLAT" | "LONG" | "SHORT"
        last_exit_side: str        = "FLAT",    # direction of last closed position
    ) -> SignalResult:
        """
        Evaluate a single bar.

        Parameters
        ----------
        row             : A row from the enriched DataFrame (Close, indicators ...)
        regime_cat      : "BULL" | "BEAR" | "CHOP"
        confidence      : HMM posterior probability of current regime state
        in_position     : whether we currently hold a position (legacy compat)
        last_exit_ts    : timestamp of the last exit (or None)
        position_side   : "FLAT" | "LONG" | "SHORT" (multi-direction mode)
        last_exit_side  : "FLAT" | "LONG" | "SHORT" — direction of last closed position
                          (used for direction-specific cooldown in multi-dir mode)

        Returns
        -------
        SignalResult with action = "BUY" | "SELL" | "SHORT" | "COVER" | "HOLD" | "COOLDOWN"
        """
        if self.use_regime_mapper:
            return self._evaluate_multi_direction(
                row, regime_cat, confidence, position_side,
                last_exit_ts, last_exit_side,
            )
        else:
            return self._evaluate_legacy(
                row, regime_cat, confidence, in_position, last_exit_ts,
            )

    # ── Legacy evaluate (long-only, backward compatible) ─────────────────

    def _evaluate_legacy(
        self,
        row:          pd.Series,
        regime_cat:   str,
        confidence:   float,
        in_position:  bool,
        last_exit_ts: Optional[datetime],
    ) -> SignalResult:
        ts = row.name
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # ── Exit logic: immediate on BEAR ────────────────────────────────
        if in_position and regime_cat == "BEAR":
            regime_label = row.get("regime", "")
            if not config.EXIT_ON_BEAR_CRASH_ONLY or regime_label == "bear_crash":
                return SignalResult(
                    action="SELL",
                    confirmations=0,
                    score=0.0,
                    regime_cat=regime_cat,
                    confidence=confidence,
                    reason=f"regime_flip_{regime_label}",
                    direction="LONG",
                )

        # ── No entry conditions ──────────────────────────────────────────
        if regime_cat != "BULL":
            return SignalResult(
                action="HOLD",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason="not_bull_regime",
            )

        if self.in_cooldown(ts, last_exit_ts):
            remaining = config.COOLDOWN_HOURS - (
                (ts - last_exit_ts).total_seconds() / 3600
            )
            return SignalResult(
                action="COOLDOWN",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason=f"cooldown_{remaining:.1f}h_remaining",
            )

        if confidence < config.REGIME_CONFIDENCE_MIN:
            return SignalResult(
                action="HOLD",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason=f"low_confidence_{confidence:.2f}<{config.REGIME_CONFIDENCE_MIN}",
            )

        if in_position:
            return SignalResult(
                action="HOLD",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason="already_in_position",
            )

        # ── Check all 8 indicators ──────────────────────────────────────
        confirmed, checks = self._run_long_checks(row)
        score = confirmed / config.TOTAL_SIGNALS

        if confirmed >= config.MIN_CONFIRMATIONS:
            return SignalResult(
                action="BUY",
                confirmations=confirmed,
                score=score,
                regime_cat=regime_cat,
                confidence=confidence,
                details=checks,
                reason=f"{confirmed}/{config.TOTAL_SIGNALS}_confirmations",
                direction="LONG",
            )

        return SignalResult(
            action="HOLD",
            confirmations=confirmed,
            score=score,
            regime_cat=regime_cat,
            confidence=confidence,
            details=checks,
            reason=f"only_{confirmed}/{config.TOTAL_SIGNALS}_confirmations",
        )

    # ── Multi-direction evaluate (regime mapper) ─────────────────────────

    def _evaluate_multi_direction(
        self,
        row:            pd.Series,
        regime_cat:     str,
        confidence:     float,
        position_side:  str,          # "FLAT" | "LONG" | "SHORT"
        last_exit_ts:   Optional[datetime],
        last_exit_side: str = "FLAT", # direction of last closed position
    ) -> SignalResult:
        ts = row.name
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        direction = self._mapper.get_direction(regime_cat, confidence)

        # ── Exit logic (direction-aware) ─────────────────────────────────
        if position_side == "LONG" and not direction.allows_long:
            return SignalResult(
                action="SELL",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason=f"direction_flip_{direction.value}",
                direction="LONG",
            )

        if position_side == "SHORT" and not direction.allows_short:
            return SignalResult(
                action="COVER",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason=f"direction_flip_{direction.value}",
                direction="SHORT",
            )

        # ── Already in correct position ──────────────────────────────────
        if position_side != "FLAT":
            return SignalResult(
                action="HOLD",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason=f"already_{position_side.lower()}",
            )

        # ── FLAT: check for entry ────────────────────────────────────────

        # Direction-aware cooldown: use SHORT-specific hours if last exit was SHORT
        cooldown_h = (config.COOLDOWN_HOURS_SHORT
                      if last_exit_side == "SHORT"
                      else config.COOLDOWN_HOURS)
        if self.in_cooldown(ts, last_exit_ts, cooldown_hours=cooldown_h):
            remaining = cooldown_h - (
                (ts - last_exit_ts).total_seconds() / 3600
            )
            return SignalResult(
                action="COOLDOWN",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason=f"cooldown_{remaining:.1f}h_remaining",
            )

        # No entries in FLAT direction
        if direction.is_flat:
            return SignalResult(
                action="HOLD",
                confirmations=0,
                score=0.0,
                regime_cat=regime_cat,
                confidence=confidence,
                reason="direction_flat",
            )

        # Try LONG entry
        if direction.allows_long:
            confirmed, checks = self._run_long_checks(row)
            score = confirmed / config.TOTAL_SIGNALS

            if confirmed >= config.MIN_CONFIRMATIONS:
                return SignalResult(
                    action="BUY",
                    confirmations=confirmed,
                    score=score,
                    regime_cat=regime_cat,
                    confidence=confidence,
                    details=checks,
                    reason=f"{confirmed}/{config.TOTAL_SIGNALS}_confirmations",
                    direction="LONG",
                )
            return SignalResult(
                action="HOLD",
                confirmations=confirmed,
                score=score,
                regime_cat=regime_cat,
                confidence=confidence,
                details=checks,
                reason=f"only_{confirmed}/{config.TOTAL_SIGNALS}_long_confirmations",
            )

        # Try SHORT entry (uses SHORT-specific confirmation gate)
        if direction.allows_short:
            confirmed, checks = self._run_short_checks(row)
            score = confirmed / config.TOTAL_SIGNALS

            if confirmed >= config.MIN_CONFIRMATIONS_SHORT:
                return SignalResult(
                    action="SHORT",
                    confirmations=confirmed,
                    score=score,
                    regime_cat=regime_cat,
                    confidence=confidence,
                    details=checks,
                    reason=f"{confirmed}/{config.TOTAL_SIGNALS}_confirmations",
                    direction="SHORT",
                )
            return SignalResult(
                action="HOLD",
                confirmations=confirmed,
                score=score,
                regime_cat=regime_cat,
                confidence=confidence,
                details=checks,
                reason=f"only_{confirmed}/{config.TOTAL_SIGNALS}_short_confirmations",
            )

        # Fallback
        return SignalResult(
            action="HOLD",
            confirmations=0,
            score=0.0,
            regime_cat=regime_cat,
            confidence=confidence,
            reason="no_direction_match",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised helper: attach signal columns to the full DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_series(
    df: pd.DataFrame,
    use_regime_mapper: bool = False,
) -> pd.DataFrame:
    """
    Compute raw entry-signal mask columns on *df*.

    When use_regime_mapper=False (legacy):
        Adds: check_*, confirmation_count, raw_long_signal

    When use_regime_mapper=True (multi-direction):
        Adds: check_*_long, check_*_short, long_confirmation_count,
              short_confirmation_count, allowed_direction,
              raw_long_signal, raw_short_signal

    Returns df with new boolean/string columns.
    """
    df = df.copy()
    sma_col = f"sma_{config.TREND_MA_PERIOD}"
    stoch_lower = 100 - config.STOCH_UPPER  # mirror threshold (20 when STOCH_UPPER=80)

    if not use_regime_mapper:
        # ── Legacy mode: long-only (identical to original) ───────────────
        df["check_rsi"]         = df["rsi"].between(
            config.RSI_LOWER, config.RSI_UPPER, inclusive="neither"
        )
        df["check_momentum"]    = df["momentum"] > 0
        df["check_volatility"]  = df["volatility"] < config.VOLATILITY_MULT * df["vol_median"]
        df["check_volume"]      = df["volume_ratio"] > config.VOLUME_MULT
        df["check_adx"]         = df["adx"] > config.ADX_MIN
        df["check_price_trend"] = df["Close"] > df[sma_col]
        df["check_macd"]        = df["macd"] > df["macd_signal"]
        df["check_stochastic"]  = df["stoch_k"] < config.STOCH_UPPER

        check_cols = [c for c in df.columns if c.startswith("check_")]
        df["confirmation_count"] = df[check_cols].sum(axis=1)
        confidence_mask = (
            df["confidence"] >= config.REGIME_CONFIDENCE_MIN
            if "confidence" in df.columns
            else pd.Series(True, index=df.index)
        )
        df["raw_long_signal"] = (
            (df["regime_cat"] == "BULL") &
            confidence_mask &
            (df["confirmation_count"] >= config.MIN_CONFIRMATIONS)
        )

    else:
        # ── Multi-direction mode ─────────────────────────────────────────
        from src.regime_mapper import RegimeMapper
        mapper = RegimeMapper()

        # Shared checks (same for LONG and SHORT)
        df["check_rsi"]        = df["rsi"].between(
            config.RSI_LOWER, config.RSI_UPPER, inclusive="neither"
        )
        df["check_volatility"] = df["volatility"] < config.VOLATILITY_MULT * df["vol_median"]
        df["check_volume"]     = df["volume_ratio"] > config.VOLUME_MULT

        # ADX: asymmetric — LONG uses ADX_MIN, SHORT uses ADX_MIN_SHORT
        df["check_adx_long"]   = df["adx"] > config.ADX_MIN
        df["check_adx_short"]  = df["adx"] > config.ADX_MIN_SHORT

        # LONG-specific checks
        df["check_momentum_long"]    = df["momentum"] > 0
        df["check_price_trend_long"] = df["Close"] > df[sma_col]
        df["check_macd_long"]        = df["macd"] > df["macd_signal"]
        df["check_stochastic_long"]  = df["stoch_k"] < config.STOCH_UPPER

        # SHORT-specific checks (inverted)
        df["check_momentum_short"]    = df["momentum"] < 0
        df["check_price_trend_short"] = df["Close"] < df[sma_col]
        df["check_macd_short"]        = df["macd"] < df["macd_signal"]
        df["check_stochastic_short"]  = df["stoch_k"] > stoch_lower

        # Confirmation counts
        long_check_cols = [
            "check_rsi", "check_momentum_long", "check_volatility",
            "check_volume", "check_adx_long", "check_price_trend_long",
            "check_macd_long", "check_stochastic_long",
        ]
        short_check_cols = [
            "check_rsi", "check_momentum_short", "check_volatility",
            "check_volume", "check_adx_short", "check_price_trend_short",
            "check_macd_short", "check_stochastic_short",
        ]

        df["long_confirmation_count"]  = df[long_check_cols].sum(axis=1)
        df["short_confirmation_count"] = df[short_check_cols].sum(axis=1)

        # Also provide the generic confirmation_count (max of long/short)
        df["confirmation_count"] = df[["long_confirmation_count",
                                       "short_confirmation_count"]].max(axis=1)

        # Map each bar to an allowed direction
        df["allowed_direction"] = [
            mapper.get_direction(
                row["regime_cat"],
                row.get("confidence", 0.0),
            ).value
            for _, row in df.iterrows()
        ]

        # Raw signal masks (asymmetric confirmation gates)
        allows_long = df["allowed_direction"].isin(["LONG", "LONG_OR_FLAT"])
        allows_short = df["allowed_direction"].isin(["SHORT", "SHORT_OR_FLAT"])

        df["raw_long_signal"] = (
            allows_long &
            (df["long_confirmation_count"] >= config.MIN_CONFIRMATIONS)
        )
        df["raw_short_signal"] = (
            allows_short &
            (df["short_confirmation_count"] >= config.MIN_CONFIRMATIONS_SHORT)
        )

    return df
