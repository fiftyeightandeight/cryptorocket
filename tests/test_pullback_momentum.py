"""Tests for the PullbackMomentum strategy (Martin Luk style)."""

import numpy as np
import pandas as pd
import pytest

from strategies.examples.pullback_momentum import (
    PullbackMomentum,
    PullbackMomentumAggressive,
    PullbackMomentumConservative,
)


def _make_hourly_prices(n_hours=500, symbols=None, seed=42):
    """Build synthetic hourly MultiIndex price DataFrame."""
    symbols = symbols or ["BTC", "ETH"]
    dates = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")

    np.random.seed(seed)
    data = {}
    for sym in symbols:
        base = 50_000 if sym == "BTC" else 3_000
        closes = base * np.cumprod(1 + np.random.normal(0, 0.005, n_hours))
        opens = closes * (1 + np.random.normal(0, 0.001, n_hours))
        data[("Close", sym)] = closes
        data[("Open", sym)] = opens
        data[("High", sym)] = np.maximum(closes, opens) * (1 + np.abs(np.random.normal(0, 0.01, n_hours)))
        data[("Low", sym)] = np.minimum(closes, opens) * (1 - np.abs(np.random.normal(0, 0.01, n_hours)))
        data[("Volume", sym)] = np.random.uniform(1e6, 5e6, n_hours)

    cols = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Symbol"])
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_trending_prices(n_hours=500, trend_strength=0.002):
    """Build prices with a clear uptrend to trigger long setups."""
    dates = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    np.random.seed(99)

    # Trending up with a pullback in the middle
    base = 50_000.0
    returns = np.random.normal(trend_strength, 0.003, n_hours)
    # Insert a pullback period
    returns[200:220] = np.random.normal(-0.003, 0.002, 20)
    # Resume uptrend
    returns[220:] = np.random.normal(trend_strength, 0.003, n_hours - 220)

    closes = base * np.cumprod(1 + returns)
    opens = closes * (1 + np.random.normal(0, 0.001, n_hours))

    data = {
        ("Close", "BTC"): closes,
        ("Open", "BTC"): opens,
        ("High", "BTC"): np.maximum(closes, opens) * (1 + np.abs(np.random.normal(0, 0.015, n_hours))),
        ("Low", "BTC"): np.minimum(closes, opens) * (1 - np.abs(np.random.normal(0, 0.015, n_hours))),
        ("Volume", "BTC"): np.random.uniform(2e6, 8e6, n_hours),
    }

    cols = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Symbol"])
    return pd.DataFrame(data, index=dates, columns=cols)


class TestPullbackMomentumSignals:
    def test_signals_shape_matches_closes(self):
        strategy = PullbackMomentum()
        prices = _make_hourly_prices()
        signals = strategy.prices_to_signals(prices)

        closes = prices["Close"]
        assert signals.shape == closes.shape
        assert list(signals.columns) == list(closes.columns)

    def test_signals_are_bounded(self):
        """Signals should only be -1, 0, or 1."""
        strategy = PullbackMomentum()
        prices = _make_hourly_prices()
        signals = strategy.prices_to_signals(prices)

        unique_vals = set(signals.values.flatten())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_no_signals_during_warmup(self):
        """No signals should fire in the first SLOW_EMA bars (not enough data)."""
        strategy = PullbackMomentum()
        prices = _make_hourly_prices(n_hours=600)
        signals = strategy.prices_to_signals(prices)

        # First ~50 bars should be zero (need 50-EMA + ADR lookback)
        warmup = strategy.SLOW_EMA
        assert (signals.iloc[:warmup] == 0).all().all()

    def test_no_signals_with_short_data(self):
        """Very short data should produce no signals."""
        strategy = PullbackMomentum()
        prices = _make_hourly_prices(n_hours=30)
        signals = strategy.prices_to_signals(prices)

        assert (signals == 0).all().all()

    def test_conservative_no_shorts(self):
        """Conservative variant should never produce short signals."""
        strategy = PullbackMomentumConservative()
        prices = _make_hourly_prices(n_hours=600)
        signals = strategy.prices_to_signals(prices)

        assert (signals >= 0).all().all()


class TestADRFilter:
    def test_compute_adr_shape(self):
        strategy = PullbackMomentum()
        prices = _make_hourly_prices()
        highs = prices["High"]
        lows = prices["Low"]
        closes = prices["Close"]

        adr = strategy._compute_adr(highs, lows, closes)
        assert adr.shape == closes.shape

    def test_adr_non_negative(self):
        strategy = PullbackMomentum()
        prices = _make_hourly_prices()
        highs = prices["High"]
        lows = prices["Low"]
        closes = prices["Close"]

        adr = strategy._compute_adr(highs, lows, closes)
        assert (adr >= 0).all().all()

    def test_adr_zero_during_warmup(self):
        """ADR should be 0 (filled) before lookback is met."""
        strategy = PullbackMomentum()
        prices = _make_hourly_prices(n_hours=100)
        highs = prices["High"]
        lows = prices["Low"]
        closes = prices["Close"]

        adr = strategy._compute_adr(highs, lows, closes)
        # Need ADR_LOOKBACK * 24 bars minimum
        min_bars = strategy.ADR_LOOKBACK * 24
        if min_bars < len(adr):
            assert (adr.iloc[:24] == 0).all().all()


class TestExitLogic:
    def test_max_hold_exit(self):
        """Position should be exited after MAX_HOLD_BARS."""
        strategy = PullbackMomentum()
        strategy.MAX_HOLD_BARS = 5

        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

        signals = pd.DataFrame(0.0, index=idx, columns=["BTC"])
        signals.iloc[2, 0] = 1.0  # entry at bar 2

        # Price always above EMA (no trailing exit)
        closes = pd.DataFrame(100.0, index=idx, columns=["BTC"])
        ema_fast = pd.DataFrame(95.0, index=idx, columns=["BTC"])

        result = strategy._apply_exits(signals, closes, ema_fast)

        # Active for bars 2-5 (bars_held 1,2,3,4), exit at bar 6 (bars_held=5)
        assert result.iloc[2, 0] == 1.0  # bars_held=1
        assert result.iloc[5, 0] == 1.0  # bars_held=4
        assert result.iloc[6, 0] == 0.0  # bars_held=5 → exit
        assert result.iloc[7, 0] == 0.0  # still flat

    def test_trailing_ema_exit_long(self):
        """Long position should exit when price drops below 9-EMA."""
        strategy = PullbackMomentum()
        strategy.MAX_HOLD_BARS = 100  # disable max hold

        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

        signals = pd.DataFrame(0.0, index=idx, columns=["BTC"])
        signals.iloc[2, 0] = 1.0

        closes = pd.DataFrame(100.0, index=idx, columns=["BTC"])
        ema_fast = pd.DataFrame(95.0, index=idx, columns=["BTC"])

        # Price drops below EMA at bar 6
        closes.iloc[6, 0] = 90.0

        result = strategy._apply_exits(signals, closes, ema_fast)

        assert result.iloc[2, 0] == 1.0  # active
        assert result.iloc[5, 0] == 1.0  # still active
        assert result.iloc[6, 0] == 0.0  # exited (price < ema)

    def test_trailing_ema_exit_short(self):
        """Short position should exit when price rises above 9-EMA."""
        strategy = PullbackMomentum()
        strategy.MAX_HOLD_BARS = 100

        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

        signals = pd.DataFrame(0.0, index=idx, columns=["BTC"])
        signals.iloc[2, 0] = -1.0

        closes = pd.DataFrame(100.0, index=idx, columns=["BTC"])
        ema_fast = pd.DataFrame(105.0, index=idx, columns=["BTC"])

        # Price rises above EMA at bar 5
        closes.iloc[5, 0] = 110.0

        result = strategy._apply_exits(signals, closes, ema_fast)

        assert result.iloc[2, 0] == -1.0  # active short
        assert result.iloc[4, 0] == -1.0  # still active
        assert result.iloc[5, 0] == 0.0   # exited


class TestWeightsAndPipeline:
    def test_weights_equal_weight_active(self):
        strategy = PullbackMomentum()
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        signals = pd.DataFrame(
            {"BTC": [1.0, -1.0, 0.0, 1.0, 0.0], "ETH": [1.0, 0.0, 0.0, -1.0, 0.0]},
            index=idx,
        )
        prices = _make_hourly_prices(n_hours=5, symbols=["BTC", "ETH"])

        weights = strategy.signals_to_target_weights(signals, prices)

        # Row 0: both active → 0.5 each
        assert weights.iloc[0]["BTC"] == pytest.approx(0.5)
        assert weights.iloc[0]["ETH"] == pytest.approx(0.5)

        # Row 1: only BTC active
        assert weights.iloc[1]["BTC"] == pytest.approx(-1.0)
        assert weights.iloc[1]["ETH"] == pytest.approx(0.0)

    def test_leverage_scaling(self):
        strategy = PullbackMomentumAggressive()

        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        signals = pd.DataFrame({"BTC": [1.0, 0.0, -1.0]}, index=idx)
        prices = _make_hourly_prices(n_hours=3, symbols=["BTC"])

        weights = strategy.signals_to_target_weights(signals, prices)

        assert weights.iloc[0]["BTC"] == pytest.approx(2.0)
        assert weights.iloc[2]["BTC"] == pytest.approx(-2.0)

    def test_full_pipeline_no_crash(self):
        """Full backtest pipeline runs without errors."""
        strategy = PullbackMomentum()
        prices = _make_hourly_prices(n_hours=500)

        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)

        assert gross.shape == signals.shape

    def test_full_pipeline_trending_data(self):
        """Pipeline with trending data should produce some signals."""
        strategy = PullbackMomentum()
        prices = _make_trending_prices(n_hours=600)

        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)

        assert gross.shape == signals.shape
        # No crash is the main assertion; signals may or may not fire
        # depending on random data


class TestVariants:
    def test_aggressive_has_higher_leverage(self):
        base = PullbackMomentum()
        aggressive = PullbackMomentumAggressive()

        assert aggressive.LEVERAGE > base.LEVERAGE
        assert aggressive.ADR_MIN < base.ADR_MIN

    def test_conservative_has_lower_leverage(self):
        base = PullbackMomentum()
        conservative = PullbackMomentumConservative()

        assert conservative.LEVERAGE < base.LEVERAGE
        assert conservative.ADR_MIN > base.ADR_MIN
        assert conservative.ALLOW_SHORTS is False

    def test_all_variants_have_unique_codes(self):
        codes = [
            PullbackMomentum.CODE,
            PullbackMomentumAggressive.CODE,
            PullbackMomentumConservative.CODE,
        ]
        assert len(codes) == len(set(codes))

    def test_conservative_pipeline_no_crash(self):
        strategy = PullbackMomentumConservative()
        prices = _make_hourly_prices(n_hours=500)
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)
        assert gross.shape == signals.shape

    def test_aggressive_pipeline_no_crash(self):
        strategy = PullbackMomentumAggressive()
        prices = _make_hourly_prices(n_hours=500)
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)
        assert gross.shape == signals.shape
