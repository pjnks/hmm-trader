# ─── HMM TRADER CONFIGURATION ────────────────────────────────────────────────

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER          = "X:SOLUSD"   # Ensemble winner (Sharpe 6.240)
INTERVAL        = "4h"
DAYS_HISTORY    = 730
TIMEFRAME       = "4h"         # Ensemble winner timeframe
TRAINING_DAYS   = 365          # bars used for IS/OOS split in optimizer

# ── HMM Feature Sets ──────────────────────────────────────────────────────────
FEATURE_SET = "extended_v2"     # A/B tested: +2.747 Sharpe vs extended (Sprint 3.1)

FEATURE_SETS: dict = {
    "base": [
        "log_return", "price_range", "volume_change",
    ],
    "volume_focused": [
        # vol_price_diverge excluded: binary (0/1) feature violates Gaussian HMM
        # assumption and causes degenerate covariance for n_states >= 5.
        "volume_change", "realized_vol_ratio",
    ],
    "extended": [
        # vol_price_diverge removed: binary {0,1} violates Gaussian HMM (see CLAUDE.md)
        "log_return", "price_range", "volume_change",
        "realized_vol_ratio", "return_autocorr",
    ],
    "full": [
        # vol_price_diverge removed: binary {0,1} violates Gaussian HMM (see CLAUDE.md)
        "log_return", "price_range", "volume_change",
        "realized_vol_ratio", "return_autocorr",
        "candle_body_ratio", "bb_width",
    ],
    # Sprint 3.1: v2 feature sets with new continuous replacements for vol_price_diverge
    "extended_v2": [
        "log_return", "price_range", "volume_change",
        "realized_vol_ratio", "return_autocorr",
        "realized_kurtosis",          # tail-risk signal (replaces binary vol_price_diverge)
    ],
    "full_v2": [
        "log_return", "price_range", "volume_change",
        "realized_vol_ratio", "return_autocorr",
        "candle_body_ratio", "bb_width",
        "realized_kurtosis", "volume_return_intensity", "return_momentum_ratio",
    ],
    # Sprint 8: ATR-normalized features — cross-asset universal config.
    # All inputs scaled by ATR so HMM sees vol-adjusted moves, not raw price moves.
    # Enables single config across BTC ($2k ATR) and HBAR ($0.001 ATR).
    "atr_normalized": [
        "atr_norm_return", "atr_norm_range", "atr_norm_volume",
        "realized_vol_ratio", "return_autocorr", "realized_kurtosis",
    ],
}

# ── HMM ───────────────────────────────────────────────────────────────────────
N_STATES        = 6
N_ITER          = 300
RANDOM_STATE    = 42
COV_TYPE        = "diag"       # Ensemble winner (diagonal covariance)

# ── Ensemble ─────────────────────────────────────────────────────────────────
ENSEMBLE_N_STATES      = [5, 6, 7]   # n_states variants for ensemble voting
ENSEMBLE_MIN_AGREEMENT = 2           # minimum models agreeing on regime (2 of 3)
ENSEMBLE_MIN_CONVERGED = 2           # minimum converged models to proceed

# Regime classification thresholds (based on per-state mean return z-score)
BULL_STATES     = 2            # top N states by mean return  → BULL
BEAR_STATES     = 2            # bottom N states by mean return → BEAR
# remaining middle states                                      → CHOP

REGIME_LABELS = {
    "BULL":  ["bull_strong", "bull_mild"],
    "BEAR":  ["bear_crash",  "bear_mild"],
    "CHOP":  ["chop_high_vol", "chop_low_vol", "chop_neutral"],
}

# Readable colour map for dashboard
REGIME_COLORS = {
    "bull_strong":   "#00ff88",
    "bull_mild":     "#66ffaa",
    "chop_high_vol": "#ffcc00",
    "chop_low_vol":  "#ffdd55",
    "chop_neutral":  "#ffee88",
    "bear_mild":     "#ff7744",
    "bear_crash":    "#ff2244",
}

# ── Strategy ──────────────────────────────────────────────────────────────────
MIN_CONFIRMATIONS        = 7        # Stability-tested: extended_v2/7cf → Sharpe +0.837 (was 6)
TOTAL_SIGNALS            = 8
REGIME_CONFIDENCE_MIN    = 0.70    # minimum HMM posterior before entering
EXIT_ON_BEAR_CRASH_ONLY  = False   # exit on any BEAR regime

# ── Regime-Direction Mapping (Phase 2: multi-direction) ──────────────────────
REGIME_DIRECTION_MAP = {
    ("BULL", "high"):  "LONG",           # BULL + conf >= CONFIDENCE_HIGH_THRESHOLD
    ("BULL", "med"):   "LONG_OR_FLAT",   # BULL + 0.70 <= conf < threshold
    ("BEAR", "high"):  "SHORT",          # BEAR + conf >= CONFIDENCE_HIGH_THRESHOLD
    ("BEAR", "med"):   "SHORT_OR_FLAT",  # BEAR + 0.70 <= conf < threshold
    ("CHOP", "any"):   "FLAT",           # CHOP at any confidence
}
CONFIDENCE_HIGH_THRESHOLD = 0.85  # above = "high", else "med"

# Individual indicator parameters
RSI_PERIOD          = 14
RSI_UPPER           = 70      # overbought ceiling
RSI_LOWER           = 30      # oversold floor (not used for entry filter, just reference)
MOMENTUM_PERIOD     = 10      # look-back periods for rate-of-change
VOLATILITY_PERIOD   = 20      # rolling std window
VOLATILITY_MULT     = 2.0     # confirm if vol < MULT × median vol
VOLUME_MA_PERIOD    = 20      # MA period for volume comparison
VOLUME_MULT         = 1.1     # confirm if volume > MULT × volume MA
ADX_PERIOD          = 14
ADX_MIN             = 20      # optimized (was 25; adx=20 → Sharpe 2.879 vs 2.455)
TREND_MA_PERIOD     = 50      # SMA period for price trend
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
STOCH_K             = 14
STOCH_D             = 3
STOCH_UPPER         = 80      # confirm if %K < STOCH_UPPER (room to run)

# ── Portfolio / Risk ──────────────────────────────────────────────────────────
INITIAL_CAPITAL     = 10_000.0
LEVERAGE            = 1.0     # Ensemble WF winner (was 1.5; live uses 0.25× for micro-capital)
COOLDOWN_HOURS      = 48      # Ensemble WF winner (was 72 — faster re-entry)
TAKER_FEE           = 0.0006  # 0.06% per side (Binance-like) — default for BTC/ETH

# ── Tiered Slippage by Crypto Liquidity (Sprint 9) ───────────────────────────
# Per-ticker transaction cost (fee + slippage) based on order book depth.
# Altcoins have thinner books, especially at regime-flip moments (high-vol).
# Realistic slippage prevents backtester from overstating altcoin edge.
AGATE_TICKER_FEES = {
    # Tier 1: Deep books — 6bps (base fee only)
    "X:BTCUSD": 0.0006,
    "X:ETHUSD": 0.0006,
    # Tier 2: Decent liquidity — 9bps (1.5× base)
    "X:SOLUSD": 0.0009,
    "X:BCHUSD": 0.0009,
    "X:LINKUSD": 0.0009,
    "X:XRPUSD": 0.0009,
    # Tier 3: Thin books — 12bps (2× base, per quant review recommendation)
    "X:LTCUSD":  0.0012,
    "X:SUIUSD":  0.0012,
    "X:DOGEUSD": 0.0012,
    "X:HBARUSD": 0.0012,
    "X:XLMUSD":  0.0012,
    "X:AVAXUSD": 0.0012,
    "X:ENAUSD":  0.0012,
}

def get_ticker_fee(ticker: str) -> float:
    """Return per-side transaction cost for a crypto ticker (fee + slippage)."""
    return AGATE_TICKER_FEES.get(ticker, TAKER_FEE)

# ── SHORT-specific overrides (multi-direction only) ─────────────────────
# When use_regime_mapper=True, these override the shared values for SHORT.
# Defaults match the shared values for backward compatibility.
MIN_CONFIRMATIONS_SHORT  = 7      # SHORT entry gate (default matches LONG)
COOLDOWN_HOURS_SHORT     = 48     # cooldown after SHORT exit (default matches LONG)
ADX_MIN_SHORT            = 25     # minimum ADX for SHORT entries (default matches LONG)

# ── AGATE Multi-Ticker Crypto Universe ────────────────────────────────────────
# Sprint 9 (2026-03-29): Narrowed from 14→6 based on Global Config Test.
# Only tickers that show positive Sharpe with UNIFIED config (base/4st/5cf/diag/4h)
# AND realistic tiered slippage survive. Per-ticker optimization was curve-fitting.
AGATE_TICKERS = [
    "X:LINKUSD",   # Sharpe +2.355, 4/5 windows, Tier 2 (9bps) — strongest
    "X:SOLUSD",    # Sharpe +1.931, 1/2 windows, Tier 2 (9bps)
    "X:HBARUSD",   # Sharpe +1.151, 2/5 windows, Tier 3 (12bps) — survives high fees
    "X:BCHUSD",    # Sharpe +1.028, 3/5 windows, Tier 2 (9bps) — most consistent
    "X:ETHUSD",    # Sharpe +0.377, 1/2 windows, Tier 1 (6bps) — marginal but liquid
    "X:SUIUSD",    # Sharpe +0.109, 2/5 windows, Tier 3 (12bps) — barely positive, on watch
    # ── Dropped (Sprint 9 Global Config Test — no edge with unified config) ──
    # X:BTCUSD  — Sharpe -3.370, too efficiently arbitraged
    # X:XRPUSD  — Sharpe -5.916, news-driven (regulatory), not regime-driven
    # X:AVAXUSD — Sharpe -5.750, no edge survives 12bps slippage
    # X:XLMUSD  — Sharpe -1.844, wild variance (return +35.9% but Sharpe negative)
    # X:DOGEUSD — Sharpe -1.367, no edge at 12bps
    # X:ENAUSD  — Sharpe -1.141, no edge at 12bps
    # X:LTCUSD  — Sharpe -0.128, flat
    # X:ADAUSD  — insufficient data (95d), dropped Sprint 8
    # X:SHIBUSD, X:DOTUSD — 0% positive in optimizer, dropped Sprint 6
]

# ── AGATE Unified Config (Sprint 9) ──────────────────────────────────────────
# Global Config Test proved per-ticker optimization was curve-fitting.
# Simple unified config (base/4st/5cf/diag/4h) shows real edge across all
# surviving tickers. Do NOT use per-ticker config overrides for AGATE.
AGATE_UNIFIED_CONFIG = {
    "n_states": 4,
    "feature_set": "base",
    "confirmations": 5,
    "cov_type": "diag",
    "timeframe": "4h",
}
AGATE_USE_UNIFIED_CONFIG = True  # Set False to revert to per-ticker configs

# Polygon ticker → Coinbase product ID mapping
CRYPTO_PRODUCT_MAP = {
    "X:BTCUSD":  "BTC-USD",
    "X:ETHUSD":  "ETH-USD",
    "X:SOLUSD":  "SOL-USD",
    "X:XRPUSD":  "XRP-USD",
    "X:LTCUSD":  "LTC-USD",
    "X:SUIUSD":  "SUI-USD",
    "X:DOGEUSD": "DOGE-USD",
    "X:ADAUSD":  "ADA-USD",
    "X:BCHUSD":  "BCH-USD",
    "X:LINKUSD": "LINK-USD",
    "X:HBARUSD": "HBAR-USD",
    "X:SHIBUSD": "1000SHIB-USD",   # Polygon SHIB ↔ Coinbase 1000SHIB
    "X:XLMUSD":  "XLM-USD",
    "X:AVAXUSD": "AVAX-USD",
    "X:DOTUSD":  "DOT-USD",
    "X:ENAUSD":  "ENA-USD",
}

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASH_HOST           = "127.0.0.1"
DASH_PORT           = 8050
REFRESH_INTERVAL_MS = 60_000  # auto-refresh every 60 s in live mode


# ══════════════════════════════════════════════════════════════════════════════
# CITRINE — Confidence-Weighted NDX100 Portfolio Rotation
# ══════════════════════════════════════════════════════════════════════════════

# ── Universe (full NDX100, Jan 2026 composition) ─────────────────────────────
CITRINE_UNIVERSE = [
    # ── Semiconductors ───────────────────────────────────────────────────────
    "NVDA", "AVGO", "AMD", "ASML", "MU", "AMAT", "LRCX", "INTC", "KLAC",
    "TXN", "QCOM", "ADI", "MRVL", "NXPI", "MCHP", "MPWR", "ARM",
    # ── Software / Cloud / Cybersecurity ─────────────────────────────────────
    "MSFT", "PLTR", "ADBE", "INTU", "SNPS", "CDNS", "ADSK", "PANW",
    "CRWD", "FTNT", "DDOG", "WDAY", "ZS", "TEAM", "CSGP", "CTSH",
    # ── Internet / Media / Entertainment ─────────────────────────────────────
    "GOOGL", "META", "NFLX", "CMCSA", "WBD", "CHTR", "EA", "TTWO",
    # ── E-Commerce / Consumer Tech / Travel ──────────────────────────────────
    "AMZN", "AAPL", "TSLA", "SHOP", "ABNB", "MELI", "BKNG", "MAR",
    "DASH", "ORLY", "ROST",
    # ── Telecom / Networking ─────────────────────────────────────────────────
    "TMUS", "CSCO",
    # ── Healthcare / Biotech / MedTech ───────────────────────────────────────
    "AMGN", "GILD", "ISRG", "VRTX", "REGN", "IDXX", "DXCM", "GEHC",
    "ALNY", "INSM",
    # ── Consumer Staples / Retail ────────────────────────────────────────────
    "PEP", "COST", "WMT", "MNST", "MDLZ", "KHC", "KDP", "SBUX",
    "CTAS", "CPRT",
    # ── Industrials / Services ───────────────────────────────────────────────
    "HON", "CSX", "PCAR", "FAST", "ODFL", "PAYX", "VRSK", "ROP", "ADP",
    # ── Energy / Utilities ───────────────────────────────────────────────────
    "CEG", "AEP", "XEL", "EXC", "BKR", "FANG", "LIN",
    # ── Fintech / Data / Other ───────────────────────────────────────────────
    "PYPL", "AXON", "MSTR", "APP", "STX", "WDC", "PDD", "TRI", "FER",
    "CCEP",
]

CITRINE_SECTORS = {
    # Semiconductors (17)
    "NVDA": "Semiconductors", "AVGO": "Semiconductors", "AMD": "Semiconductors",
    "ASML": "Semiconductors", "MU": "Semiconductors", "AMAT": "Semiconductors",
    "LRCX": "Semiconductors", "INTC": "Semiconductors", "KLAC": "Semiconductors",
    "TXN": "Semiconductors", "QCOM": "Semiconductors", "ADI": "Semiconductors",
    "MRVL": "Semiconductors", "NXPI": "Semiconductors", "MCHP": "Semiconductors",
    "MPWR": "Semiconductors", "ARM": "Semiconductors",
    # Software (16)
    "MSFT": "Software", "PLTR": "Software", "ADBE": "Software",
    "INTU": "Software", "SNPS": "Software", "CDNS": "Software",
    "ADSK": "Software", "PANW": "Cybersecurity", "CRWD": "Cybersecurity",
    "FTNT": "Cybersecurity", "DDOG": "Software", "WDAY": "Software",
    "ZS": "Cybersecurity", "TEAM": "Software", "CSGP": "Software",
    "CTSH": "Software",
    # Internet / Media (8)
    "GOOGL": "Internet", "META": "Internet", "NFLX": "Internet",
    "CMCSA": "Media", "WBD": "Media", "CHTR": "Media",
    "EA": "Gaming", "TTWO": "Gaming",
    # E-Commerce / Consumer Tech (11)
    "AMZN": "E-Commerce", "AAPL": "Consumer Tech", "TSLA": "Consumer Tech",
    "SHOP": "E-Commerce", "ABNB": "Travel", "MELI": "E-Commerce",
    "BKNG": "Travel", "MAR": "Travel", "DASH": "E-Commerce",
    "ORLY": "Retail", "ROST": "Retail",
    # Telecom (2)
    "TMUS": "Telecom", "CSCO": "Networking",
    # Healthcare (10)
    "AMGN": "Biotech", "GILD": "Biotech", "ISRG": "MedTech",
    "VRTX": "Biotech", "REGN": "Biotech", "IDXX": "MedTech",
    "DXCM": "MedTech", "GEHC": "MedTech", "ALNY": "Biotech",
    "INSM": "Biotech",
    # Consumer Staples (10)
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "WMT": "Consumer Staples", "MNST": "Consumer Staples",
    "MDLZ": "Consumer Staples", "KHC": "Consumer Staples",
    "KDP": "Consumer Staples", "SBUX": "Consumer Staples",
    "CTAS": "Industrials", "CPRT": "Industrials",
    # Industrials (9)
    "HON": "Industrials", "CSX": "Industrials", "PCAR": "Industrials",
    "FAST": "Industrials", "ODFL": "Industrials", "PAYX": "Industrials",
    "VRSK": "Industrials", "ROP": "Industrials", "ADP": "Industrials",
    # Energy / Utilities (7)
    "CEG": "Utilities", "AEP": "Utilities", "XEL": "Utilities",
    "EXC": "Utilities", "BKR": "Energy", "FANG": "Energy", "LIN": "Materials",
    # Other (10)
    "PYPL": "Fintech", "AXON": "Industrials", "MSTR": "Software",
    "APP": "Software", "STX": "Hardware", "WDC": "Hardware",
    "PDD": "E-Commerce", "TRI": "Data", "FER": "Materials", "CCEP": "Consumer Staples",
}

# ── Hysteresis Bands ─────────────────────────────────────────────────────────
CITRINE_ENTRY_CONFIDENCE    = 0.70    # Sprint 14: was 0.90. Calibration inversion proved
                                       # 0.90 selects WORST bucket (51.9% T+3 hit rate).
                                       # <70% bucket = 96.3% hit rate, 70-80% = 65.7%.
                                       # HMM alpha lives at regime transitions, not exhaustion.
CITRINE_EXIT_CONFIDENCE     = 0.50    # optimized (was 0.65; hold longer = Sharpe +1.665)
CITRINE_PERSISTENCE_DAYS    = 1       # Sprint 14: was 3. If HMM needs 3 days to confirm,
                                       # alpha has already decayed. Enter on day-1 transition.

# ── Cooldown (configurable, testable in backtest) ────────────────────────────
# "none"      — hysteresis + persistence is enough
# "time"      — block re-entry for COOLDOWN_DAYS after exit
# "threshold" — require REENTRY_CONFIDENCE instead of ENTRY_CONFIDENCE
CITRINE_COOLDOWN_MODE       = "none"
CITRINE_COOLDOWN_DAYS       = 5       # for "time" mode: trading days to wait
CITRINE_REENTRY_CONFIDENCE  = 0.85    # for "threshold" mode: higher bar to re-enter

# ── CITRINE Scoring ──────────────────────────────────────────────────────────
CITRINE_PERSISTENCE_BONUS   = 0.10    # +0.10 weight per BULL/BEAR day beyond 3
CITRINE_PERSISTENCE_CAP     = 0.50    # max bonus from persistence

# ── Gradual Scaling ──────────────────────────────────────────────────────────
CITRINE_SCALE_SCHEDULE = {
    1: 0.50,    # Day 1: 50% of target weight (optimized: fast scale)
    2: 1.00,    # Day 2+: 100% (optimized: aggressive ramp)
}

# ── Adaptive Cash Buffer ─────────────────────────────────────────────────────
# (min_bull, max_bull, cash_pct) — how much cash to hold based on BULL count
CITRINE_CASH_BANDS = [
    (15, 999, 0.10),   # 15+ BULL stocks → invest 90% (optimized: aggressive)
    (8,   14, 0.15),   # 8-14 BULL       → invest 85%
    (3,    7, 0.25),   # 3-7 BULL        → invest 75%
    (0,    2, 0.50),   # 0-2 BULL        → invest max 50%
]

# ── Position Limits ──────────────────────────────────────────────────────────
CITRINE_MAX_POSITIONS       = 15      # max simultaneous holdings
CITRINE_MAX_PER_SECTOR      = 4       # Sprint 3.3: max positions per sector (prevents concentration)
CITRINE_LONG_ONLY           = False   # True = only LONG positions; False = LONG + SHORT

# ── HMM Defaults (per-ticker; optimizer can override via JSON) ───────────────
CITRINE_HMM_DEFAULTS = {
    "n_states":      6,
    "feature_set":   "base",
    "cov_type":      "diag",
    "confirmations": 7,
}

# ── Risk / Capital ───────────────────────────────────────────────────────────
CITRINE_INITIAL_CAPITAL     = 25_000  # total portfolio capital
CITRINE_MAX_NOTIONAL        = 5_000   # per-ticker max position (micro-capital)
CITRINE_TAKER_FEE           = 0.0004  # 0.04% per side (equity commission)
CITRINE_SLIPPAGE_BPS        = 10      # 10bps simulated slippage

# ── Dashboard ────────────────────────────────────────────────────────────────
CITRINE_DASH_PORT           = 8070    # separate from AGATE (:8060) and backtest (:8050)
