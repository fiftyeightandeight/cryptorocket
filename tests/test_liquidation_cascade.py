"""Tests for the LiquidationCascadeMomentum strategy."""

import numpy as np
import pandas as pd
import pytest

from strategies.examples.liquidation_cascade import LiquidationCascadeMomentum


def _make_hourly_prices(n_hours=200, symbols=None):
    """Build synthetic hourly MultiIndex price DataFrame."""
    symbols = symbols or ["BTC", "ETH"]
    dates = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")

    np.random.seed(42)
    data = {}
    for sym in symbols:
        base = 50_000 if sym == "BTC" else 3_000
        closes = base * np.cumprod(1 + np.random.normal(0, 0.005, n_hours))
        data[("Close", sym)] = closes
        data[("Open", sym)] = closes * (1 + np.random.normal(0, 0.001, n_hours))
        data[("High", sym)] = np.maximum(closes, data[("Open", sym)]) * 1.005
        data[("Low", sym)] = np.minimum(closes, data[("Open", sym)]) * 0.995
        data[("Volume", sym)] = np.random.uniform(1e6, 5e6, n_hours)

    cols = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Symbol"])
    return pd.DataFrame(data, index=dates, columns=cols)


class TestLiquidationCascadeSignals:
    def test_signals_shape_matches_closes(self):
        strategy = LiquidationCascadeMomentum()
        prices = _make_hourly_prices()
        signals = strategy.prices_to_signals(prices)

        closes = prices["Close"]
        assert signals.shape == closes.shape
        assert list(signals.columns) == list(closes.columns)

    def test_signals_are_bounded(self):
        """Signals should only be -1, 0, or 1."""
        strategy = LiquidationCascadeMomentum()
        prices = _make_hourly_prices()
        signals = strategy.prices_to_signals(prices)

        unique_vals = set(signals.values.flatten())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_no_signals_without_funding_data(self):
        """Without funding rates in DB, funding_extreme is all zeros,
        so no cascade signal should fire (requires all 3 conditions)."""
        strategy = LiquidationCascadeMomentum()
        prices = _make_hourly_prices()
        signals = strategy.prices_to_signals(prices)

        assert (signals == 0).all().all()

    def test_hold_signals_forward_fills(self):
        """_hold_signals should propagate a signal for HOLD_BARS."""
        strategy = LiquidationCascadeMomentum()
        strategy.HOLD_BARS = 3

        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        signals = pd.DataFrame(0.0, index=idx, columns=["BTC"])
        signals.iloc[2, 0] = 1.0  # signal at bar 2

        held = strategy._hold_signals(signals)

        assert held.iloc[2, 0] == 1.0  # original signal
        assert held.iloc[3, 0] == 1.0  # held
        assert held.iloc[4, 0] == 1.0  # held
        assert held.iloc[5, 0] == 0.0  # expired (3 bars total: 2,3,4)

    def test_hold_signals_newer_signal_overwrites(self):
        """A newer signal during the hold window should take precedence."""
        strategy = LiquidationCascadeMomentum()
        strategy.HOLD_BARS = 4

        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        signals = pd.DataFrame(0.0, index=idx, columns=["BTC"])
        signals.iloc[1, 0] = 1.0   # long at bar 1
        signals.iloc[3, 0] = -1.0  # short at bar 3 (during hold)

        held = strategy._hold_signals(signals)

        assert held.iloc[1, 0] == 1.0    # original long
        assert held.iloc[2, 0] == 1.0    # held long
        assert held.iloc[3, 0] == -1.0   # new short overwrites
        assert held.iloc[4, 0] == -1.0   # held short


class TestFundingExtreme:
    def test_returns_correct_shape(self):
        strategy = LiquidationCascadeMomentum()
        prices = _make_hourly_prices()
        closes = prices["Close"]
        result = strategy._funding_extreme(closes)

        assert result.shape == closes.shape

    def test_returns_zeros_without_db(self):
        """Without a database, should return all zeros gracefully."""
        strategy = LiquidationCascadeMomentum()
        prices = _make_hourly_prices()
        closes = prices["Close"]
        result = strategy._funding_extreme(closes)

        assert (result == 0).all().all()

    def test_values_are_bounded(self):
        strategy = LiquidationCascadeMomentum()
        prices = _make_hourly_prices()
        closes = prices["Close"]
        result = strategy._funding_extreme(closes)

        unique_vals = set(result.values.flatten())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_asymmetric_windows(self):
        """Verify short and long windows are different to avoid self-reference."""
        strategy = LiquidationCascadeMomentum()
        assert strategy.FUNDING_SHORT_WINDOW < strategy.FUNDING_LONG_WINDOW


class TestWeightsAndPipeline:
    def test_weights_equal_weight_active(self):
        strategy = LiquidationCascadeMomentum()
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

        # Row 1: only BTC active → weight -1.0
        assert weights.iloc[1]["BTC"] == pytest.approx(-1.0)
        assert weights.iloc[1]["ETH"] == pytest.approx(0.0)

        # Row 2: none active → all zero
        assert weights.iloc[2]["BTC"] == pytest.approx(0.0)

    def test_leverage_scaling(self):
        strategy = LiquidationCascadeMomentum()
        strategy.LEVERAGE = 2.0

        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        signals = pd.DataFrame({"BTC": [1.0, 0.0, -1.0]}, index=idx)
        prices = _make_hourly_prices(n_hours=3, symbols=["BTC"])

        weights = strategy.signals_to_target_weights(signals, prices)

        assert weights.iloc[0]["BTC"] == pytest.approx(2.0)
        assert weights.iloc[2]["BTC"] == pytest.approx(-2.0)

    def test_full_pipeline_no_crash(self):
        """Ensure the full backtest pipeline runs without errors
        even with no funding data."""
        strategy = LiquidationCascadeMomentum()
        prices = _make_hourly_prices(n_hours=100)

        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)

        assert gross.shape == signals.shape
        # With no funding data, all signals are zero → zero returns
        assert (gross == 0).all().all()

    def test_channel_breakout_requires_lookback(self):
        """Verify no signals fire during the warm-up period when
        min_periods hasn't been met."""
        strategy = LiquidationCascadeMomentum()
        # Use very short data — shorter than CHANNEL_LOOKBACK
        prices = _make_hourly_prices(n_hours=30)
        signals = strategy.prices_to_signals(prices)

        # With 48h channel lookback and 30h of data, breakout
        # conditions can never be met → no signals
        assert (signals == 0).all().all()
