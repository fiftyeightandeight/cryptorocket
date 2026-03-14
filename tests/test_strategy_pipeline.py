"""Tests for the CryptoMoonshot strategy pipeline."""

import numpy as np
import pandas as pd
import pytest

from backtest.commission import (
    HyperliquidPerpsCommission,
    HyperliquidSpotCommission,
    ZeroCommission,
)
from strategies.base import CryptoMoonshot


def _make_prices(n_days=60, symbols=None):
    """Build a synthetic MultiIndex price DataFrame."""
    symbols = symbols or ["BTC", "ETH"]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")

    np.random.seed(0)
    data = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for sym in symbols:
            if field == "Volume":
                data[(field, sym)] = np.random.uniform(1e6, 1e7, n_days)
            else:
                base = 50_000 if sym == "BTC" else 3_000
                data[(field, sym)] = base * np.cumprod(1 + np.random.normal(0, 0.02, n_days))

    cols = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Symbol"])
    prices = pd.DataFrame(data, index=dates, columns=cols)
    # Ensure High >= Close >= Low
    for sym in symbols:
        prices[("High", sym)] = prices[[("Open", sym), ("Close", sym)]].max(axis=1) * 1.01
        prices[("Low", sym)] = prices[[("Open", sym), ("Close", sym)]].min(axis=1) * 0.99
    return prices


class LongOnlyStub(CryptoMoonshot):
    """Always long with equal weight."""
    CODE = "test-long"
    SYMBOLS = ["BTC", "ETH"]
    INTERVAL = "1d"

    def prices_to_signals(self, prices):
        closes = prices["Close"]
        return pd.DataFrame(1.0, index=closes.index, columns=closes.columns)


class ShortOnlyStub(CryptoMoonshot):
    """Always short with equal weight."""
    CODE = "test-short"
    SYMBOLS = ["BTC", "ETH"]
    INTERVAL = "1d"

    def prices_to_signals(self, prices):
        closes = prices["Close"]
        return pd.DataFrame(-1.0, index=closes.index, columns=closes.columns)


class FlatStub(CryptoMoonshot):
    """Always flat — no signals."""
    CODE = "test-flat"
    SYMBOLS = ["BTC", "ETH"]
    INTERVAL = "1d"

    def prices_to_signals(self, prices):
        closes = prices["Close"]
        return pd.DataFrame(0.0, index=closes.index, columns=closes.columns)


class TestSignalsToTargetWeights:
    def test_equal_weight_long(self):
        strategy = LongOnlyStub()
        prices = _make_prices()
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)

        assert weights.shape == signals.shape
        assert weights.iloc[-1].tolist() == pytest.approx([0.5, 0.5])

    def test_equal_weight_short(self):
        strategy = ShortOnlyStub()
        prices = _make_prices()
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)

        assert weights.iloc[-1].tolist() == pytest.approx([-0.5, -0.5])

    def test_flat_weights_zero(self):
        strategy = FlatStub()
        prices = _make_prices()
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)

        assert (weights == 0).all().all()

    def test_leverage_scaling(self):
        strategy = LongOnlyStub()
        strategy.LEVERAGE = 2.0
        prices = _make_prices()
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)

        assert weights.iloc[-1].tolist() == pytest.approx([1.0, 1.0])


class TestPositionsToGrossReturns:
    def test_returns_shape(self):
        strategy = LongOnlyStub()
        prices = _make_prices()
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)

        assert gross.shape == positions.shape

    def test_flat_positions_zero_returns(self):
        strategy = FlatStub()
        prices = _make_prices()
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)

        assert (gross == 0).all().all()

    def test_first_row_is_zero(self):
        strategy = LongOnlyStub()
        prices = _make_prices()
        signals = strategy.prices_to_signals(prices)
        weights = strategy.signals_to_target_weights(signals, prices)
        positions = strategy.target_weights_to_positions(weights, prices)
        gross = strategy.positions_to_gross_returns(positions, prices)

        assert (gross.iloc[0] == 0).all()


class TestBacktestPipeline:
    def _run_with_db(self, strategy_cls, tmp_path):
        """Run a backtest using an in-memory-like DuckDB."""
        from data.schema import init_db, get_connection

        db_path = tmp_path / "test.duckdb"
        init_db(db_path)
        conn = get_connection(db_path)

        prices = _make_prices(n_days=90, symbols=["BTC", "ETH"])
        closes = prices["Close"]

        rows = []
        for sym in ["BTC", "ETH"]:
            for dt, val in closes[sym].items():
                rows.append([
                    sym, "1d", dt,
                    float(val * 0.99), float(val * 1.01),
                    float(val * 0.98), float(val), 1e6,
                ])
        conn.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

        strategy = strategy_cls()
        strategy.DB_PATH = str(db_path)
        return strategy.backtest()

    def test_backtest_returns_dict(self, tmp_path):
        results = self._run_with_db(LongOnlyStub, tmp_path)
        assert "returns" in results
        assert "positions" in results
        assert "signals" in results
        assert isinstance(results["returns"], pd.Series)

    def test_backtest_returns_not_all_nan(self, tmp_path):
        results = self._run_with_db(LongOnlyStub, tmp_path)
        assert not results["returns"].isna().all()

    def test_flat_strategy_zero_returns(self, tmp_path):
        results = self._run_with_db(FlatStub, tmp_path)
        assert (results["returns"] == 0).all()

    def test_commissions_reduce_returns(self, tmp_path):
        results = self._run_with_db(LongOnlyStub, tmp_path)
        gross_total = results["gross_returns"].sum(axis=1).sum()
        net_total = results["returns"].sum()
        assert net_total <= gross_total


class TestCommissionModels:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2024-01-01", periods=5)
        prices = pd.DataFrame(
            {"BTC": [50000, 51000, 49000, 52000, 50500]},
            index=dates,
        )
        changes = pd.DataFrame(
            {"BTC": [0.5, 0.0, -0.5, 0.5, 0.0]},
            index=dates,
        )
        return prices, changes

    def test_zero_commission(self, sample_data):
        prices, changes = sample_data
        model = ZeroCommission()
        result = model.get_commissions(prices, changes)
        assert (result == 0).all().all()

    def test_perps_commission_proportional(self, sample_data):
        prices, changes = sample_data
        model = HyperliquidPerpsCommission()
        result = model.get_commissions(prices, changes)
        assert (result >= 0).all().all()
        # Rows with no change should have zero commission
        assert result.loc[result.index[1], "BTC"] == 0.0
        assert result.loc[result.index[0], "BTC"] > 0.0

    def test_spot_higher_than_perps(self, sample_data):
        prices, changes = sample_data
        perp = HyperliquidPerpsCommission().get_commissions(prices, changes)
        spot = HyperliquidSpotCommission().get_commissions(prices, changes)
        assert (spot >= perp).all().all()

    def test_maker_cheaper_than_taker(self, sample_data):
        prices, changes = sample_data
        taker = HyperliquidPerpsCommission(assume_taker=True).get_commissions(prices, changes)
        maker = HyperliquidPerpsCommission(assume_taker=False).get_commissions(prices, changes)
        assert (taker >= maker).all().all()
