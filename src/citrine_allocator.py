"""
citrine_allocator.py
────────────────────
Converts CITRINE scanner results into portfolio weights using the CITRINE
scoring formula, with hysteresis bands, configurable cooldowns, gradual
scaling, and adaptive cash management.

CITRINE Score Formula
─────────────────────
  citrine_score = confidence_weight × inverse_vol × indicator_quality × persistence_bonus

  Where:
    confidence_weight = clip((confidence - 0.65) / 0.25, 0, 1)
    inverse_vol       = clip(median_vol / realized_vol, 0.5, 2.0)
    indicator_quality = confirmations / 8
    persistence_bonus = 1.0 + min(max(persistence - 3, 0) × 0.10, 0.50)

Usage
─────
  from src.citrine_allocator import CitrineAllocator
  allocator = CitrineAllocator(capital=25000)
  weights, cash_pct = allocator.allocate(scans)
  allocator.update_holdings(weights)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── Path bootstrap ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.citrine_scanner import TickerScan

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PortfolioWeight:
    """Target allocation for one ticker in the CITRINE portfolio."""
    ticker:         str
    direction:      str         # "LONG" or "SHORT"
    raw_score:      float       # CITRINE score before normalization
    target_weight:  float       # normalized weight (sums to invested_pct)
    scaled_weight:  float       # after gradual scaling (day 1/3/5)
    notional_usd:   float       # dollar amount to allocate
    days_held:      int         # how many days in portfolio
    action:         str         # "ENTER", "HOLD", "SCALE_UP", "EXIT"
    sector:         str = ""
    confidence:     float = 0.0
    persistence:    int = 0


# ─────────────────────────────────────────────────────────────────────────────
class CitrineAllocator:
    """
    Converts TickerScan results into portfolio weights.

    Allocation Pipeline:
      1. Filter by regime (BULL → LONG, BEAR → SHORT, CHOP → skip)
      2. Apply hysteresis (entry at 80%, exit at 65%, dead zone in between)
      3. Apply cooldown (none / time-based / confidence-threshold)
      4. Compute CITRINE score for each qualifier
      5. Cap at MAX_POSITIONS by score
      6. Determine adaptive cash % from BULL/BEAR count
      7. Normalize weights to sum to invested %
      8. Apply gradual scaling (day 1=25%, day 3=50%, day 5=100%)
    """

    def __init__(
        self,
        capital: float = config.CITRINE_INITIAL_CAPITAL,
        long_only: bool = config.CITRINE_LONG_ONLY,
        cooldown_mode: str = config.CITRINE_COOLDOWN_MODE,
    ):
        self.capital = capital
        self.long_only = long_only
        self.cooldown_mode = cooldown_mode

        # Tracking state (persists across allocate() calls)
        self._holdings: dict[str, int] = {}          # ticker → days_held
        self._exit_day: dict[str, int] = {}           # ticker → day_counter when exited
        self._day_counter: int = 0                    # increments each allocate() call

    def allocate(
        self,
        scans: list[TickerScan],
        alt_data_boosts: dict[str, float] | None = None,
    ) -> tuple[list[PortfolioWeight], float]:
        """
        Main allocation pipeline.

        Returns
        -------
        weights : list[PortfolioWeight]
            Target allocations (may be empty if no qualifiers)
        cash_pct : float
            Fraction of portfolio to hold as cash (0-1)
        """
        self._day_counter += 1

        # Input validation
        if alt_data_boosts is None:
            alt_data_boosts = {}
        if not scans:
            log.warning("[CITRINE] allocate() called with empty or None scans — returning empty weights")
            return [], 1.0

        # Step 1: Filter by regime → determine direction
        candidates = self._filter_by_regime(scans)

        # Step 2: Apply hysteresis (different thresholds for new vs existing)
        candidates = self._apply_hysteresis(candidates)

        # Step 3: Apply cooldown
        candidates = self._apply_cooldown(candidates)

        # Step 4: Compute CITRINE score
        scored = self._compute_scores(candidates, scans, alt_data_boosts)

        # Step 5: Cap at MAX_POSITIONS (keep top by score)
        scored.sort(key=lambda x: x.raw_score, reverse=True)
        scored = scored[:config.CITRINE_MAX_POSITIONS]

        # Step 5b: Apply sector concentration limit (Sprint 3.3)
        scored = self._apply_sector_cap(scored)

        # Step 6: Determine adaptive cash %
        cash_pct = self._determine_cash_pct(scans)
        invested_pct = 1.0 - cash_pct

        # Step 7: Normalize weights
        total_score = sum(w.raw_score for w in scored)
        if total_score > 0:
            for w in scored:
                w.target_weight = (w.raw_score / total_score) * invested_pct
        else:
            for w in scored:
                w.target_weight = 0.0

        # Step 8: Apply gradual scaling
        scored = self._apply_gradual_scaling(scored)

        # Compute notional USD
        for w in scored:
            w.notional_usd = min(
                w.scaled_weight * self.capital,
                config.CITRINE_MAX_NOTIONAL,
            )

        # Determine exits: tickers in _holdings but not in scored
        held_tickers = set(self._holdings.keys())
        scored_tickers = {w.ticker for w in scored}
        exiting = held_tickers - scored_tickers

        # Create EXIT weights for positions being closed
        exits: list[PortfolioWeight] = []
        for ticker in exiting:
            exits.append(PortfolioWeight(
                ticker=ticker,
                direction="FLAT",
                raw_score=0.0,
                target_weight=0.0,
                scaled_weight=0.0,
                notional_usd=0.0,
                days_held=self._holdings.get(ticker, 0),
                action="EXIT",
                sector=config.CITRINE_SECTORS.get(ticker, ""),
            ))

        all_weights = scored + exits
        return all_weights, cash_pct

    def update_holdings(self, weights: list[PortfolioWeight]) -> None:
        """
        Update internal tracking after allocation is executed.
        Call this after executing trades based on allocate() results.
        """
        new_holdings: dict[str, int] = {}

        for w in weights:
            if w.action == "EXIT":
                # Record exit time for cooldown tracking
                self._exit_day[w.ticker] = self._day_counter
                # Remove from holdings
                continue

            if w.action == "ENTER":
                new_holdings[w.ticker] = 1
            elif w.action in ("HOLD", "SCALE_UP"):
                new_holdings[w.ticker] = self._holdings.get(w.ticker, 0) + 1

        self._holdings = new_holdings

    def reset(self) -> None:
        """Reset all tracking state (for backtester window boundaries)."""
        self._holdings.clear()
        self._exit_day.clear()
        self._day_counter = 0

    # ── Internal Pipeline Steps ──────────────────────────────────────────────

    def _filter_by_regime(
        self, scans: list[TickerScan]
    ) -> list[tuple[TickerScan, str]]:
        """
        Filter scans by regime → (scan, direction) pairs.
        BULL → LONG, BEAR → SHORT (if not long_only), CHOP → skip.
        """
        candidates: list[tuple[TickerScan, str]] = []
        for scan in scans:
            if scan.error is not None:
                continue
            if not scan.hmm_converged:
                continue
            if scan.regime_cat == "BULL":
                candidates.append((scan, "LONG"))
            elif scan.regime_cat == "BEAR" and not self.long_only:
                candidates.append((scan, "SHORT"))
            # CHOP → no position
        return candidates

    def _apply_hysteresis(
        self, candidates: list[tuple[TickerScan, str]]
    ) -> list[tuple[TickerScan, str]]:
        """
        Apply hysteresis bands:
          - New entries: confidence >= ENTRY_CONFIDENCE (0.80) + persistence >= PERSISTENCE_DAYS (3)
          - Existing holdings: stay as long as confidence >= EXIT_CONFIDENCE (0.65)
          - Dead zone (0.65-0.80): hold if already in, don't enter if not
        """
        result: list[tuple[TickerScan, str]] = []
        entry_conf = config.CITRINE_ENTRY_CONFIDENCE
        exit_conf = config.CITRINE_EXIT_CONFIDENCE
        min_persist = config.CITRINE_PERSISTENCE_DAYS

        for scan, direction in candidates:
            is_held = scan.ticker in self._holdings

            if is_held:
                # Existing position: use lower exit threshold
                if scan.confidence >= exit_conf:
                    result.append((scan, direction))
                # Below exit threshold → will be marked as EXIT
            else:
                # New position: need higher entry threshold + persistence
                if (scan.confidence >= entry_conf and
                        scan.persistence >= min_persist):
                    result.append((scan, direction))

        return result

    def _apply_cooldown(
        self, candidates: list[tuple[TickerScan, str]]
    ) -> list[tuple[TickerScan, str]]:
        """
        Apply cooldown rules for recently exited tickers.

        Modes:
          "none"      — no cooldown (hysteresis + persistence is enough)
          "time"      — block re-entry for COOLDOWN_DAYS after exit
          "threshold" — require REENTRY_CONFIDENCE instead of ENTRY_CONFIDENCE
        """
        if self.cooldown_mode == "none":
            return candidates

        result: list[tuple[TickerScan, str]] = []
        for scan, direction in candidates:
            # Only apply cooldown to new entries (not existing holdings)
            is_held = scan.ticker in self._holdings
            if is_held:
                result.append((scan, direction))
                continue

            # Check if this ticker was recently exited
            exit_day = self._exit_day.get(scan.ticker)
            if exit_day is None:
                # Never held before — no cooldown
                result.append((scan, direction))
                continue

            days_since_exit = self._day_counter - exit_day

            if self.cooldown_mode == "time":
                if days_since_exit >= config.CITRINE_COOLDOWN_DAYS:
                    result.append((scan, direction))
                # else: still in cooldown, skip

            elif self.cooldown_mode == "threshold":
                if scan.confidence >= config.CITRINE_REENTRY_CONFIDENCE:
                    result.append((scan, direction))
                # else: confidence not high enough for re-entry

        return result

    def _apply_sector_cap(
        self, scored: list[PortfolioWeight],
        max_per_sector: int | None = None,
    ) -> list[PortfolioWeight]:
        """
        Sprint 3.3: Enforce sector concentration limit.

        After position-cap, remove lowest-scoring positions from sectors
        that exceed max_per_sector. Prevents semiconductor-heavy or
        software-heavy portfolio bets.

        Scored list is assumed to be sorted by raw_score descending.
        """
        try:
            max_ps = max_per_sector or getattr(config, "CITRINE_MAX_PER_SECTOR", 4)

            sector_count: dict[str, int] = {}
            result: list[PortfolioWeight] = []

            for w in scored:
                sector = w.sector or config.CITRINE_SECTORS.get(w.ticker, "Unknown")
                count = sector_count.get(sector, 0)

                if count < max_ps:
                    result.append(w)
                    sector_count[sector] = count + 1
                else:
                    log.debug("Sector cap: dropping %s (sector=%s, already %d)",
                              w.ticker, sector, count)

            if len(result) < len(scored):
                dropped = len(scored) - len(result)
                log.info("Sector cap applied: dropped %d positions (max %d per sector)",
                         dropped, max_ps)

            return result
        except Exception as e:
            log.warning(f"[CITRINE] WARNING: Sector cap failed: {e} — returning input unchanged")
            return scored

    def _compute_scores(
        self,
        candidates: list[tuple[TickerScan, str]],
        all_scans: list[TickerScan],
        alt_data_boosts: dict[str, float] | None = None,
    ) -> list[PortfolioWeight]:
        """
        Compute CITRINE score for each candidate.

        Score = confidence_weight × inverse_vol × indicator_quality × persistence_bonus
        """
        # Compute median volatility across ALL scanned tickers (for risk parity)
        vols = [s.realized_vol for s in all_scans
                if s.error is None and s.realized_vol > 0]
        median_vol = float(np.median(vols)) if vols else 0.30  # fallback 30%

        weights: list[PortfolioWeight] = []
        for scan, direction in candidates:
            try:
                # 1. Confidence weight: maps [0.65, 0.90] → [0, 1]
                confidence_weight = np.clip((scan.confidence - 0.65) / 0.25, 0.0, 1.0)

                # 2. Inverse volatility factor (risk parity)
                if scan.realized_vol > 0:
                    inverse_vol = np.clip(median_vol / scan.realized_vol, 0.5, 2.0)
                else:
                    inverse_vol = 1.0

                # 3. Indicator quality
                if direction == "LONG":
                    indicator_quality = scan.confirmations / 8.0
                else:
                    indicator_quality = scan.confirmations_short / 8.0

                # 4. Sojourn decay (replaces persistence bonus — PhD review fix)
                # Regimes are metastable: expected remaining lifetime DECREASES as
                # regime ages. The transition matrix A[k,k] gives half-life.
                # Score decays from 1.0x to 0.5x as persistence exceeds half-life.
                half_life = max(getattr(scan, "regime_half_life", 30.0), 3.0)
                sojourn_ratio = scan.persistence / half_life
                sojourn_factor = max(0.5, 1.0 - 0.5 * max(0, sojourn_ratio - 0.5))

                # Alt-data boost (insider buying/selling signal)
                alt_boost = 1.0
                if alt_data_boosts is not None:
                    alt_boost = alt_data_boosts.get(scan.ticker, 1.0)

                # Final score (with sojourn decay + alt-data boost)
                score = (confidence_weight * inverse_vol *
                         indicator_quality * sojourn_factor * alt_boost)

                # Determine action
                is_held = scan.ticker in self._holdings
                days_held = self._holdings.get(scan.ticker, 0)

                if is_held:
                    # Check if scaling up (days 1→3→5)
                    if days_held < 5:
                        action = "SCALE_UP"
                    else:
                        action = "HOLD"
                else:
                    action = "ENTER"
                    days_held = 0

                weights.append(PortfolioWeight(
                    ticker=scan.ticker,
                    direction=direction,
                    raw_score=score,
                    target_weight=0.0,   # filled in step 7
                    scaled_weight=0.0,   # filled in step 8
                    notional_usd=0.0,    # filled after scaling
                    days_held=days_held + 1 if is_held else 1,
                    action=action,
                    sector=scan.sector,
                    confidence=scan.confidence,
                    persistence=scan.persistence,
                ))
            except Exception as e:
                log.error(f"[CITRINE] ERROR: Failed to compute score for {scan.ticker}: {e}")

        return weights

    def _determine_cash_pct(self, scans: list[TickerScan]) -> float:
        """
        Determine cash allocation using continuous scaling based on BULL ratio.

        Replaces rigid bucketed bands with a smooth function:
          cash_pct = max(0.10, 1.0 - bull_ratio^0.6)

        This avoids cliff-edge jumps at discrete thresholds (e.g., 14→15 BULL
        tickers causing cash to drop from 15%→10%). The exponent 0.6 gives a
        concave curve: conservative at low BULL counts, aggressive only when
        BULL ratio is high.

        Examples (82 tickers):
          5 BULL (6%) → cash 92% (very defensive)
          15 BULL (18%) → cash 70%
          30 BULL (37%) → cash 47%
          50 BULL (61%) → cash 27%
          70 BULL (85%) → cash 10% (floor)
        """
        valid = sum(1 for s in scans if s.error is None)
        bull_count = sum(
            1 for s in scans
            if s.error is None and s.regime_cat == "BULL"
        )

        if valid == 0:
            return 0.80  # No data = very defensive

        bull_ratio = bull_count / valid
        cash_pct = max(0.10, 1.0 - bull_ratio ** 0.6)
        return cash_pct

    def _apply_gradual_scaling(
        self, weights: list[PortfolioWeight]
    ) -> list[PortfolioWeight]:
        """
        Apply gradual position scaling based on days held.
        Day 1 = 25%, Day 3 = 50%, Day 5+ = 100% of target weight.
        """
        schedule = config.CITRINE_SCALE_SCHEDULE
        # Sort schedule by day (ascending)
        sorted_days = sorted(schedule.keys())

        for w in weights:
            # Find the applicable scale factor
            scale = 0.0
            for day in sorted_days:
                if w.days_held >= day:
                    scale = schedule[day]

            w.scaled_weight = w.target_weight * scale

        return weights


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with synthetic data
    print("\n" + "="*70)
    print("  CITRINE ALLOCATOR — Unit Test")
    print("="*70 + "\n")

    # Create mock scans
    mock_scans = [
        TickerScan("NVDA",  "BULL", 0.92, 8,  0.35, 7, 3, 950.0, "Semiconductors", True),
        TickerScan("AAPL",  "BULL", 0.85, 5,  0.20, 6, 2, 230.0, "Consumer Tech",  True),
        TickerScan("TSLA",  "BULL", 0.78, 2,  0.55, 5, 4, 380.0, "Consumer Tech",  True),
        TickerScan("MSFT",  "BULL", 0.82, 4,  0.18, 7, 2, 450.0, "Software",        True),
        TickerScan("AMGN",  "BEAR", 0.88, 6,  0.22, 3, 7, 310.0, "Biotech",         True),
        TickerScan("META",  "CHOP", 0.60, 1,  0.30, 4, 3, 600.0, "Internet",        True),
        TickerScan("INTC",  "BULL", 0.71, 1,  0.40, 5, 3, 25.0,  "Semiconductors",  True),
        TickerScan("GOOGL", "BULL", 0.88, 10, 0.22, 8, 2, 190.0, "Internet",        True),
    ]

    allocator = CitrineAllocator(capital=25000, long_only=False, cooldown_mode="none")
    weights, cash_pct = allocator.allocate(mock_scans)

    print(f"  Cash allocation: {cash_pct:.0%}")
    print(f"  Invested: {1 - cash_pct:.0%}")
    print(f"  Positions: {sum(1 for w in weights if w.action != 'EXIT')}")
    print()

    total_weight = 0.0
    for w in sorted(weights, key=lambda x: x.raw_score, reverse=True):
        if w.action == "EXIT":
            print(f"  ❌ {w.ticker:6s} EXIT")
        else:
            print(f"  {'🟢' if w.direction == 'LONG' else '🔴'} {w.ticker:6s} "
                  f"{w.direction:5s} score={w.raw_score:.3f}  "
                  f"target={w.target_weight:.1%}  scaled={w.scaled_weight:.1%}  "
                  f"${w.notional_usd:,.0f}  action={w.action}")
            total_weight += w.scaled_weight

    print(f"\n  Total scaled weight: {total_weight:.1%}  "
          f"(should be <= {1 - cash_pct:.0%})")

    # Verify constraints
    assert all(w.notional_usd <= config.CITRINE_MAX_NOTIONAL for w in weights), \
        "Max notional violated!"
    non_exit = [w for w in weights if w.action != "EXIT"]
    assert len(non_exit) <= config.CITRINE_MAX_POSITIONS, \
        f"Max positions violated: {len(non_exit)}"

    print("\n  ✅ All constraints passed!")
