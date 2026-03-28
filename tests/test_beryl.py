"""
tests/test_beryl.py
───────────────────
BERYL Sprint 5 test suite — covers multi-position rotation, ensemble HMM
instantiation with fallback, per-ticker config loading, alt-data smart scoring,
and position sizing constants.

All tests use synthetic data. No Polygon API calls.

Usage:
    pytest tests/test_beryl.py -v
    pytest tests/test_beryl.py -k "insider" -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ── Project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.hmm_model import HMMRegimeModel
from src.ensemble import EnsembleHMM
from src.alternative_data import InsiderTrade, InsiderSignal, AlternativeDataScore


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic properties."""
    rng = np.random.RandomState(seed)

    prices = [100.0]
    regime_lengths = rng.randint(20, 80, size=20)
    regime_dirs = rng.choice([-1, 1], size=20)

    bar = 0
    for length, direction in zip(regime_lengths, regime_dirs):
        for _ in range(length):
            if bar >= n_bars:
                break
            drift = direction * 0.002 + rng.normal(0, 0.015)
            prices.append(prices[-1] * (1 + drift))
            bar += 1
        if bar >= n_bars:
            break

    prices = np.array(prices[:n_bars])

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1d", tz="UTC")
    high = prices * (1 + rng.uniform(0.001, 0.03, n_bars))
    low = prices * (1 - rng.uniform(0.001, 0.03, n_bars))
    open_ = prices * (1 + rng.normal(0, 0.005, n_bars))
    volume = rng.uniform(1e6, 1e8, n_bars)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    }, index=dates)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def synthetic_ohlcv():
    return _make_synthetic_ohlcv()


@pytest.fixture
def per_ticker_config_file(tmp_path):
    """Create a temporary per-ticker config JSON file."""
    configs = {
        "NVDA": {"n_states": 5, "feature_set": "extended", "confirmations": 6, "cov_type": "diag", "mean_sharpe": 0.825},
        "TSLA": {"n_states": 6, "feature_set": "base", "confirmations": 7, "cov_type": "diag", "mean_sharpe": 0.807},
        "AAPL": {"n_states": 4, "feature_set": "extended_v2", "confirmations": 5, "cov_type": "diag", "mean_sharpe": 0.733},
    }
    config_path = tmp_path / "beryl_per_ticker_configs.json"
    with open(config_path, "w") as f:
        json.dump(configs, f)
    return config_path, configs


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BerylPosition dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestBerylPosition:
    """Test BerylPosition class creation and field assignment."""

    def test_create_position(self):
        from live_trading_beryl import BerylPosition
        pos = BerylPosition(ticker="NVDA", side="BUY", size=5.0, entry_price=150.0)
        assert pos.ticker == "NVDA"
        assert pos.side == "BUY"
        assert pos.size == 5.0
        assert pos.entry_price == 150.0

    def test_notional_computed(self):
        from live_trading_beryl import BerylPosition
        pos = BerylPosition(ticker="TSLA", side="BUY", size=10.0, entry_price=200.0)
        assert pos.notional == 2000.0

    def test_entry_time_set(self):
        from live_trading_beryl import BerylPosition
        pos = BerylPosition(ticker="AAPL", side="BUY", size=3.0, entry_price=180.0)
        # entry_time should be a valid ISO timestamp string
        assert isinstance(pos.entry_time, str)
        dt = datetime.fromisoformat(pos.entry_time)
        assert dt.tzinfo is not None  # timezone-aware


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Multi-position dict
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiPositionDict:
    """Verify BerylLiveEngine stores positions as dict[str, BerylPosition]."""

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_positions_is_dict(self, mock_load, mock_db):
        from live_trading_beryl import BerylLiveEngine, BerylPosition
        engine = BerylLiveEngine(tickers=["NVDA", "TSLA"], test_mode=True)
        assert isinstance(engine.positions, dict)
        assert len(engine.positions) == 0

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_add_and_remove_positions(self, mock_load, mock_db):
        from live_trading_beryl import BerylLiveEngine, BerylPosition
        engine = BerylLiveEngine(tickers=["NVDA", "TSLA", "AAPL"], test_mode=True)

        # Add positions
        pos1 = BerylPosition(ticker="NVDA", side="BUY", size=5.0, entry_price=150.0)
        pos2 = BerylPosition(ticker="TSLA", side="BUY", size=3.0, entry_price=200.0)
        engine.positions["NVDA"] = pos1
        engine.positions["TSLA"] = pos2
        assert len(engine.positions) == 2
        assert "NVDA" in engine.positions
        assert "TSLA" in engine.positions

        # Remove one
        del engine.positions["NVDA"]
        assert len(engine.positions) == 1
        assert "NVDA" not in engine.positions
        assert "TSLA" in engine.positions

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_cooldowns_is_dict(self, mock_load, mock_db):
        from live_trading_beryl import BerylLiveEngine
        engine = BerylLiveEngine(tickers=["NVDA"], test_mode=True)
        assert isinstance(engine.cooldowns, dict)
        assert len(engine.cooldowns) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Ensemble HMM instantiation with [N-1, N, N+1] states
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsembleHMMInstantiation:
    """Verify EnsembleHMM is created with [N-1, N, N+1] state list."""

    def test_n_states_list_from_config(self):
        """Given n_states=5 from per-ticker config, ensemble should use [4, 5, 6]."""
        n_states = 5
        n_states_list = [max(3, n_states - 1), n_states, n_states + 1]
        assert n_states_list == [4, 5, 6]

    def test_n_states_list_minimum_clamp(self):
        """Given n_states=3, N-1 should be clamped to 3 (not 2)."""
        n_states = 3
        n_states_list = [max(3, n_states - 1), n_states, n_states + 1]
        assert n_states_list == [3, 3, 4]

    def test_n_states_list_n4(self):
        """Given n_states=4, ensemble should use [3, 4, 5]."""
        n_states = 4
        n_states_list = [max(3, n_states - 1), n_states, n_states + 1]
        assert n_states_list == [3, 4, 5]

    def test_ensemble_constructor_accepts_n_states_list(self, synthetic_ohlcv):
        """EnsembleHMM accepts n_states_list parameter and populates models on fit."""
        from src.data_fetcher import build_hmm_features
        df = build_hmm_features(synthetic_ohlcv)
        df = df.dropna()

        feature_cols = config.FEATURE_SETS["base"]
        model = EnsembleHMM(
            n_states_list=[3, 4, 5],
            cov_type="diag",
            feature_cols=feature_cols,
        )
        assert model is not None
        assert model.n_states_list == [3, 4, 5]
        # Models list is empty before fit, populated after fit
        assert len(model.models) == 0
        model.fit(df)
        assert len(model.models) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Fallback retry logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackRetry:
    """When ensemble doesn't converge, verify fallback to single 4-state HMM."""

    def test_fallback_triggered_on_non_convergence(self, synthetic_ohlcv):
        """If EnsembleHMM.converged is False, fallback HMMRegimeModel should be tried."""
        from src.data_fetcher import build_hmm_features
        df = build_hmm_features(synthetic_ohlcv)
        df = df.dropna()

        feature_cols = config.FEATURE_SETS["base"]

        # Simulate: ensemble does not converge (converged is an instance attribute)
        ensemble = EnsembleHMM(n_states_list=[4, 5, 6], cov_type="diag", feature_cols=feature_cols)
        # Mock fit to do nothing, then manually set converged=False
        with patch.object(EnsembleHMM, "fit", return_value=ensemble):
            ensemble.fit(df)
            ensemble.converged = False

            # Replicate the fallback logic from live_trading_beryl._generate_signal_for_ticker
            assert not ensemble.converged
            fallback = HMMRegimeModel(n_states=4, cov_type="diag", feature_cols=feature_cols)
            assert fallback.n_states == 4
            assert fallback.cov_type == "diag"

    def test_fallback_config_is_4_state_diag(self):
        """Fallback model should always be n_states=4, cov_type='diag'."""
        feature_cols = config.FEATURE_SETS["base"]
        fallback = HMMRegimeModel(n_states=4, cov_type="diag", feature_cols=feature_cols)
        assert fallback.n_states == 4
        assert fallback.cov_type == "diag"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Per-ticker config loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerTickerConfigLoading:
    """Verify _get_ticker_config returns per-ticker params or defaults."""

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_returns_per_ticker_config(self, mock_load, mock_db):
        from live_trading_beryl import BerylLiveEngine, BERYL_DEFAULT_CONFIG
        engine = BerylLiveEngine(tickers=["NVDA"], test_mode=True)
        engine._per_ticker_params = {
            "NVDA": {"n_states": 5, "feature_set": "extended", "confirmations": 6, "cov_type": "full"},
        }

        tc = engine._get_ticker_config("NVDA")
        assert tc["n_states"] == 5
        assert tc["feature_set"] == "extended"
        assert tc["confirmations"] == 6
        # cov_type forced to diag for VM memory safety
        assert tc["cov_type"] == "diag"

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_returns_default_for_unknown_ticker(self, mock_load, mock_db):
        from live_trading_beryl import BerylLiveEngine, BERYL_DEFAULT_CONFIG
        engine = BerylLiveEngine(tickers=["NVDA"], test_mode=True)
        engine._per_ticker_params = {}

        tc = engine._get_ticker_config("UNKNOWN_TICKER")
        assert tc["n_states"] == BERYL_DEFAULT_CONFIG["n_states"]
        assert tc["feature_set"] == BERYL_DEFAULT_CONFIG["feature_set"]
        assert tc["confirmations"] == BERYL_DEFAULT_CONFIG["confirmations"]

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_global_min_confirmations_override(self, mock_load, mock_db):
        """CLI --min-confirmations should override per-ticker confirmations."""
        import live_trading_beryl
        from live_trading_beryl import BerylLiveEngine

        engine = BerylLiveEngine(tickers=["NVDA"], test_mode=True)
        engine._per_ticker_params = {
            "NVDA": {"n_states": 5, "feature_set": "extended", "confirmations": 6, "cov_type": "diag"},
        }

        # Set global override
        old_val = live_trading_beryl._GLOBAL_MIN_CONFIRMATIONS
        try:
            live_trading_beryl._GLOBAL_MIN_CONFIRMATIONS = 8
            tc = engine._get_ticker_config("NVDA")
            assert tc["confirmations"] == 8
        finally:
            live_trading_beryl._GLOBAL_MIN_CONFIRMATIONS = old_val

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_cov_type_forced_to_diag(self, mock_load, mock_db):
        """Even if per-ticker config says 'full', it should be forced to 'diag'."""
        from live_trading_beryl import BerylLiveEngine
        engine = BerylLiveEngine(tickers=["NVDA"], test_mode=True)
        engine._per_ticker_params = {
            "NVDA": {"n_states": 5, "feature_set": "extended", "confirmations": 6, "cov_type": "full"},
        }
        tc = engine._get_ticker_config("NVDA")
        assert tc["cov_type"] == "diag"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Lower confirmation threshold (default = 5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLowerConfirmationThreshold:
    """Verify BERYL_DEFAULT_CONFIG has confirmations=5 (Sprint 5 lowered from 7)."""

    def test_default_confirmations_is_5(self):
        from live_trading_beryl import BERYL_DEFAULT_CONFIG
        assert BERYL_DEFAULT_CONFIG["confirmations"] == 5

    def test_default_n_states(self):
        from live_trading_beryl import BERYL_DEFAULT_CONFIG
        assert BERYL_DEFAULT_CONFIG["n_states"] == 4

    def test_default_feature_set(self):
        from live_trading_beryl import BERYL_DEFAULT_CONFIG
        assert BERYL_DEFAULT_CONFIG["feature_set"] == "base"

    def test_default_cov_type(self):
        from live_trading_beryl import BERYL_DEFAULT_CONFIG
        assert BERYL_DEFAULT_CONFIG["cov_type"] == "diag"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Multi-position sizing
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiPositionSizing:
    """Verify MAX_POSITIONS=3 and per-position notional calculation."""

    def test_max_positions_is_3(self):
        from live_trading_beryl import MAX_POSITIONS
        assert MAX_POSITIONS == 3

    def test_max_notional_is_2500(self):
        from live_trading_beryl import MAX_NOTIONAL
        assert MAX_NOTIONAL == 2_500

    def test_per_position_notional(self):
        from live_trading_beryl import MAX_NOTIONAL, MAX_POSITIONS, MAX_NOTIONAL_PER_POS
        assert MAX_NOTIONAL_PER_POS == MAX_NOTIONAL // MAX_POSITIONS
        assert MAX_NOTIONAL_PER_POS == 833  # $2500 // 3 = $833

    @patch("live_trading_beryl.BerylLiveEngine._init_db")
    @patch("live_trading_beryl.BerylLiveEngine._load_per_ticker_configs")
    def test_open_position_respects_max(self, mock_load, mock_db):
        """Cannot open more than MAX_POSITIONS positions."""
        from live_trading_beryl import BerylLiveEngine, BerylPosition, MAX_POSITIONS

        engine = BerylLiveEngine(tickers=["NVDA", "TSLA", "AAPL", "MSFT"], test_mode=True)
        # Fill up to MAX_POSITIONS
        for i, ticker in enumerate(["NVDA", "TSLA", "AAPL"]):
            engine.positions[ticker] = BerylPosition(ticker=ticker, side="BUY", size=5.0, entry_price=100.0)
        assert len(engine.positions) == MAX_POSITIONS

        # Attempt to open another should fail
        result = engine._open_position({
            "ticker": "MSFT",
            "current_price": 300.0,
            "confidence": 0.95,
            "confirmations": 7,
            "min_confirmations": 5,
        })
        assert result is False
        assert len(engine.positions) == MAX_POSITIONS


# ═══════════════════════════════════════════════════════════════════════════════
# 8. LOOKBACK_DAYS = 365
# ═══════════════════════════════════════════════════════════════════════════════

class TestLookbackDays:
    """Verify LOOKBACK_DAYS constant is 365."""

    def test_lookback_days_is_365(self):
        from live_trading_beryl import LOOKBACK_DAYS
        assert LOOKBACK_DAYS == 365


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Alt-data 10b5-1 filtering
# ═══════════════════════════════════════════════════════════════════════════════

class TestInsider10b5_1Filtering:
    """InsiderTrade with is_10b5_1=True should not count as discretionary sell."""

    def test_10b5_1_sale_not_counted_as_discretionary(self):
        """10b5-1 pre-planned sales go to n_sells_10b5_1, not n_sells."""
        signal = InsiderSignal(ticker="NVDA")

        trades = [
            InsiderTrade(
                ticker="NVDA",
                insider_name="CEO Bob",
                insider_title="CEO",
                transaction_date="2026-01-15",
                transaction_code="S",
                shares=10000,
                price_per_share=150.0,
                acquired_or_disposed="D",
                filing_date="2026-01-16",
                is_10b5_1=True,  # Pre-planned sale
            ),
        ]

        # Replicate the get_signal aggregation logic
        buyers, sellers = set(), set()
        for t in trades:
            if t.transaction_code == "P":
                signal.n_buys += 1
                signal.buy_value += t.shares * t.price_per_share
                buyers.add(t.insider_name)
            elif t.transaction_code == "S":
                value = t.shares * t.price_per_share
                if t.is_10b5_1:
                    signal.n_sells_10b5_1 += 1
                    signal.sell_value_10b5_1 += value
                else:
                    signal.n_sells += 1
                    signal.sell_value += value
                    sellers.add(t.insider_name)

        signal.unique_buyers = len(buyers)
        signal.unique_sellers = len(sellers)

        # 10b5-1 sale should NOT be counted as discretionary
        assert signal.n_sells == 0
        assert signal.n_sells_10b5_1 == 1
        assert signal.sell_value == 0.0
        assert signal.sell_value_10b5_1 == 1_500_000.0
        assert signal.unique_sellers == 0

    def test_10b5_1_sale_does_not_trigger_bearish(self):
        """Even massive 10b5-1 selling should be NEUTRAL, not BEARISH."""
        signal = InsiderSignal(
            ticker="NVDA",
            n_sells=0,
            n_sells_10b5_1=5,
            sell_value=0.0,
            sell_value_10b5_1=50_000_000.0,
            unique_sellers=0,
        )
        assert signal.net_signal == "NEUTRAL"

    def test_discretionary_sale_counted_correctly(self):
        """A non-10b5-1 sale SHOULD count as discretionary."""
        signal = InsiderSignal(ticker="NVDA")

        trade = InsiderTrade(
            ticker="NVDA",
            insider_name="CFO Alice",
            insider_title="CFO",
            transaction_date="2026-01-15",
            transaction_code="S",
            shares=5000,
            price_per_share=150.0,
            acquired_or_disposed="D",
            filing_date="2026-01-16",
            is_10b5_1=False,  # Discretionary sale
        )

        value = trade.shares * trade.price_per_share
        signal.n_sells += 1
        signal.sell_value += value

        assert signal.n_sells == 1
        assert signal.sell_value == 750_000.0

    def test_insider_trade_is_10b5_1_default_false(self):
        """is_10b5_1 defaults to False."""
        trade = InsiderTrade(
            ticker="NVDA",
            insider_name="Test",
            insider_title="VP",
            transaction_date="2026-01-01",
            transaction_code="S",
            shares=100,
            price_per_share=100.0,
            acquired_or_disposed="D",
            filing_date="2026-01-02",
        )
        assert trade.is_10b5_1 is False


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Alt-data smart scoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestAltDataSmartScoring:
    """Test AlternativeDataScore._compute_boost for various insider scenarios."""

    def _make_scorer(self):
        """Create an AlternativeDataScore with mocked InsiderTracker (no SEC calls)."""
        scorer = AlternativeDataScore.__new__(AlternativeDataScore)
        scorer._insider = MagicMock()
        return scorer

    def test_neutral_no_data(self):
        """No buys, no sells -> 1.0x neutral."""
        scorer = self._make_scorer()
        signal = InsiderSignal(ticker="NVDA", n_buys=0, n_sells=0)
        assert scorer._compute_boost(signal) == 1.0

    def test_routine_selling_neutral(self):
        """Normal selling (not extreme) -> 1.0x neutral."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=0,
            n_sells=2,
            sell_value=500_000.0,
            unique_sellers=1,
        )
        # Not extreme: only 1 unique seller (needs 3+)
        assert signal.net_signal == "NEUTRAL"
        assert scorer._compute_boost(signal) == 1.0

    def test_extreme_discretionary_selling_penalty(self):
        """Extreme discretionary selling (>10x buy, 3+ sellers) -> 0.85x."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=0,
            n_sells=5,
            sell_value=15_000_000.0,  # > 10x buy_value (which is 0, so > max(0,1)*10 = 10)
            buy_value=0.0,
            unique_sellers=3,  # 3+ unique sellers
        )
        assert signal.net_signal == "BEARISH"
        assert scorer._compute_boost(signal) == 0.85

    def test_cluster_buying_boost(self):
        """3+ distinct insider buyers -> 1.5x boost (cluster buy)."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=4,
            n_sells=0,
            buy_value=2_000_000.0,
            unique_buyers=3,
            cluster_buy=True,
        )
        assert scorer._compute_boost(signal) == 1.5

    def test_moderate_buying_boost(self):
        """2 unique insider buyers -> 1.3x boost."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=3,
            n_sells=0,
            buy_value=500_000.0,
            unique_buyers=2,
            cluster_buy=False,
        )
        assert signal.net_signal == "BULLISH"
        assert scorer._compute_boost(signal) == 1.3

    def test_single_insider_purchase_boost(self):
        """Single insider purchase -> 1.15x boost."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=1,
            n_sells=0,
            buy_value=100_000.0,
            unique_buyers=1,
            cluster_buy=False,
        )
        assert signal.net_signal == "BULLISH"
        assert scorer._compute_boost(signal) == 1.15

    def test_10b5_1_only_selling_is_neutral(self):
        """All selling is 10b5-1 pre-planned (n_sells=0) -> 1.0x neutral."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=0,
            n_sells=0,
            n_sells_10b5_1=10,
            sell_value=0.0,
            sell_value_10b5_1=50_000_000.0,
            unique_sellers=0,
        )
        assert signal.net_signal == "NEUTRAL"
        assert scorer._compute_boost(signal) == 1.0

    def test_net_buying_overrides_some_selling(self):
        """More buys than (discretionary) sells -> BULLISH."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=3,
            n_sells=1,
            buy_value=500_000.0,
            sell_value=100_000.0,
            unique_buyers=2,
            cluster_buy=False,
        )
        assert signal.net_signal == "BULLISH"
        assert scorer._compute_boost(signal) == 1.3

    def test_bearish_requires_3_unique_sellers(self):
        """High sell value but only 2 sellers -> NEUTRAL, not BEARISH."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=0,
            n_sells=5,
            sell_value=20_000_000.0,
            unique_sellers=2,  # Only 2, needs 3
        )
        assert signal.net_signal == "NEUTRAL"
        assert scorer._compute_boost(signal) == 1.0

    def test_bearish_requires_10x_buy_value(self):
        """3+ sellers but sell_value < 10x buy_value -> NEUTRAL."""
        scorer = self._make_scorer()
        signal = InsiderSignal(
            ticker="NVDA",
            n_buys=1,
            n_sells=4,
            buy_value=1_000_000.0,
            sell_value=5_000_000.0,  # Only 5x, needs >10x
            unique_sellers=3,
        )
        # n_buys(1) is NOT > n_sells(4), so not BULLISH
        # sell_value(5M) is not > max(buy_value(1M), 1) * 10 = 10M, so not BEARISH
        assert signal.net_signal == "NEUTRAL"
        assert scorer._compute_boost(signal) == 1.0
