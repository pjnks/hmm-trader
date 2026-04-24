"""
SPRINT 17 — CITRINE Shadow Tracker Friction Patch
==================================================

Status: PRE-STAGED — DO NOT APPLY UNTIL 2026-04-25 (post hands-off protocol)
Target: live_trading_citrine.py::ShadowTracker
Approved by: user directive 2026-04-23

PROBLEM
-------
ShadowTracker._log_shadow_trade (and its caller run_shadow_cycle) computes
entry_price = scan.current_price and exit_price = scan.current_price with
zero fee or slippage adjustment. The live engine applies:
  - _apply_slippage() at live_trading_citrine.py:1173
  - TAKER_FEE = 0.0004 per side at live_trading_citrine.py:1105

Every Grinold A/B to date has therefore carried a ~28 bps phantom tailwind:
  - fee component:      (entry_notional + exit_notional) * 0.0004
  - slippage component: (entry_notional + exit_notional) * 10/10000

At $833 notional (capital/30), total friction per round-trip = ~$2.34.
Current relaxed cohort avg gross = $2.21; strict cohort avg gross = $2.07.
Both BELOW the friction hurdle.

SOLUTION
--------
1. Add pnl_net column via safe ALTER TABLE (idempotent).
2. Compute pnl_net in run_shadow_cycle on EXIT events.
3. Pass pnl_net through _log_shadow_trade signature.
4. Persist both pnl (gross) and pnl_net for historical integrity.
   Pre-Sprint-17 rows keep pnl_net = NULL (meaningful: "friction untracked").

FILES TOUCHED
-------------
live_trading_citrine.py (one file, three surgical edits)
  - _init_shadow_table (add column to migration list)
  - run_shadow_cycle (compute pnl_net on EXIT)
  - _log_shadow_trade (add pnl_net param, INSERT into new column)

DEPLOYMENT
----------
scp this patch's logic into the Mac copy, commit, rsync to VM, restart service.
No DB migration downtime required — ALTER TABLE is idempotent via try/except.

=============================================================================
"""

# =============================================================================
# EDIT 1 — _init_shadow_table (currently at live_trading_citrine.py:~190-230)
# =============================================================================
# Current migration list at L209-215:
#   for col, typ in [
#       ("confirmations_short", "INTEGER"),
#       ("target_weight", "REAL"),
#       ("scaled_weight", "REAL"),
#       ("live_would_enter", "INTEGER"),
#       ("indicator_json", "TEXT"),
#   ]:
#
# ADD one more entry to this list:
#       ("pnl_net", "REAL"),      # <-- Sprint 17: net P&L after fees + slippage
#
# Result: new shadow_trades rows get a pnl_net column populated; old rows NULL.


# =============================================================================
# EDIT 2 — run_shadow_cycle (currently at live_trading_citrine.py:316-337)
# =============================================================================
# Current EXIT branch:
#
#   elif w.action == "EXIT" and w.ticker in self._positions:
#       # Shadow exit
#       pos = self._positions[w.ticker]
#       entry_price = pos["entry_price"]
#       exit_price = scan.current_price
#       if entry_price > 0 and exit_price > 0:
#           if pos["direction"] == "LONG":
#               pnl_pct = (exit_price - entry_price) / entry_price * 100
#           else:
#               pnl_pct = (entry_price - exit_price) / entry_price * 100
#           notional = pos.get("notional", 0)
#           pnl = notional * pnl_pct / 100
#       else:
#           pnl, pnl_pct = 0.0, 0.0
#
#       self._log_shadow_trade(
#           now, w.ticker, "EXIT", pos["direction"], scan,
#           weight=w, pnl=pnl, pnl_pct=pnl_pct,
#           alt_boost=(alt_data_boosts or {}).get(w.ticker, 1.0),
#       )
#
# REPLACE WITH:

"""
    elif w.action == "EXIT" and w.ticker in self._positions:
        # Shadow exit
        pos = self._positions[w.ticker]
        entry_price = pos["entry_price"]
        exit_price = scan.current_price
        notional = pos.get("notional", 0)
        pnl_net = None  # Sprint 17: friction-adjusted
        if entry_price > 0 and exit_price > 0:
            if pos["direction"] == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            pnl = notional * pnl_pct / 100

            # Sprint 17: apply same friction model as live engine
            # fee: both entry + exit notionals × TAKER_FEE
            # slippage: both entry + exit notionals × SLIPPAGE_BPS/10000
            entry_notional = notional
            exit_notional = abs(exit_price * (notional / entry_price))
            fee_cost = (entry_notional + exit_notional) * config.CITRINE_TAKER_FEE
            slippage_cost = (entry_notional + exit_notional) * (
                config.CITRINE_SLIPPAGE_BPS / 10000.0
            )
            pnl_net = pnl - fee_cost - slippage_cost
        else:
            pnl, pnl_pct = 0.0, 0.0

        self._log_shadow_trade(
            now, w.ticker, "EXIT", pos["direction"], scan,
            weight=w, pnl=pnl, pnl_pct=pnl_pct, pnl_net=pnl_net,
            alt_boost=(alt_data_boosts or {}).get(w.ticker, 1.0),
        )
"""


# =============================================================================
# EDIT 3 — _log_shadow_trade (currently at live_trading_citrine.py:355-~420)
# =============================================================================
# Current signature:
#
#   def _log_shadow_trade(
#       self, timestamp: str, ticker: str, action: str,
#       direction: str, scan: TickerScan,
#       weight: PortfolioWeight | None = None,
#       score: float = 0.0, pnl: float | None = None,
#       pnl_pct: float | None = None, alt_boost: float = 1.0,
#   ):
#
# ADD pnl_net parameter:
#
#   def _log_shadow_trade(
#       self, timestamp: str, ticker: str, action: str,
#       direction: str, scan: TickerScan,
#       weight: PortfolioWeight | None = None,
#       score: float = 0.0, pnl: float | None = None,
#       pnl_pct: float | None = None, pnl_net: float | None = None,  # <-- NEW
#       alt_boost: float = 1.0,
#   ):
#
# AND inside the INSERT statement at L385-399, add pnl_net to column list:
#
#   """INSERT INTO shadow_trades
#   (timestamp, ticker, action, direction, price, notional,
#    pnl, pnl_pct, pnl_net, regime, confidence, persistence,     # <-- add pnl_net
#    confirmations, confirmations_short, realized_vol,
#    citrine_score, sector, entry_atr, regime_half_life,
#    alt_boost, target_weight, scaled_weight,
#    live_would_enter, indicator_json)
#   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
#                                # ^^^ one extra ? (24 total, was 23)
#
# AND in the values tuple, add pnl_net in the right position (after pnl_pct):
#
#   round(pnl, 2) if pnl is not None else None,
#   round(pnl_pct, 2) if pnl_pct is not None else None,
#   round(pnl_net, 2) if pnl_net is not None else None,   # <-- NEW
#   scan.regime_cat,
#   ... (rest unchanged)


# =============================================================================
# VALIDATION (post-deploy, ~Apr 26)
# =============================================================================
# Expected: within first shadow cycle post-deploy, pnl_net appears on new EXIT
# rows. Pre-Sprint-17 rows stay NULL. Query:
#
#   SELECT
#     COUNT(*) FILTER (WHERE pnl_net IS NOT NULL) AS post_sprint17,
#     COUNT(*) FILTER (WHERE pnl_net IS NULL)     AS pre_sprint17
#   FROM shadow_trades
#   WHERE action='EXIT' AND pnl IS NOT NULL;
#
# Re-run Grinold A/B with pnl_net as the P&L field. Both cohorts should show
# average close to zero or negative, confirming the $0.10 net-of-friction
# result we predicted analytically.


# =============================================================================
# EXPECTED MATH CHECK
# =============================================================================
# Input: entry $100, exit $101, notional $833, direction LONG
#   raw pnl_pct = (101 - 100) / 100 * 100 = 1.0%
#   raw pnl     = 833 * 0.01 = $8.33
#   entry_notional = 833
#   exit_notional  = 101 * (833/100) = 841.33
#   fee_cost       = (833 + 841.33) * 0.0004 = $0.6697
#   slippage_cost  = (833 + 841.33) * 0.001  = $1.6743
#   pnl_net        = 8.33 - 0.67 - 1.67 = $5.99 (72% of gross)
#
# At the marginal case (pnl_pct = 0.28%, raw pnl = $2.33):
#   fee_cost      = (833 + 835.33) * 0.0004 = ~$0.667
#   slippage_cost = (833 + 835.33) * 0.001  = ~$1.668
#   pnl_net       = 2.33 - 0.67 - 1.67 = ~$0.00  <-- break-even threshold
#
# Consistent with the $2.34 friction hurdle identified in the forensic review.
