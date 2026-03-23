"""Tests for the cross-validation harness."""

import numpy as np
import pandas as pd
import pytest

from backtest.commission import HyperliquidPerpsCommission, ZeroCommission
from backtest.cross_validate import (
    CrossValidationReport,
    StageComparison,
    cross_validate,
    _compare_frames,
    _compare_series,
    _moonshot_positions,
    _moonshot_gross_returns,
    _moonshot_slippage,
)
from strategies.base import CryptoMoonshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n_days=60, symbols=None):
    """Build a synthetic MultiIndex price DataFrame."""
    symbols = symbols or ["BTC", "ETH"]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")

    np.random.seed(42)
    data = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for sym in symbols:
            if field == "Volume":
                data[(field, sym)] = np.random.uniform(1e6, 1e7, n_days)
            else:
                base = 50_000 if sym == "BTC" else 3_000
                data[(field, sym)] = base * np.cumprod(
                    1 + np.random.normal(0, 0.02, n_days)
                )

    cols = pd.MultiIndex.from_tuples(data.keys(), names=["Field", "Symbol"])
    prices = pd.DataFrame(data, index=dates, columns=cols)
    for sym in symbols:
        prices[("High", sym)] = prices[[("Open", sym), ("Close", sym)]].max(axis=1) * 1.01
        prices[("Low", sym)] = prices[[("Open", sym), ("Close", sym)]].min(axis=1) * 0.99
    return prices


# ---------------------------------------------------------------------------
# Stub strategies
# ---------------------------------------------------------------------------

class LongOnly(CryptoMoonshot):
    CODE = "xv-long"
    SYMBOLS = ["BTC", "ETH"]
    INTERVAL = "1d"
    SLIPPAGE_BPS = 5
    COMMISSION_CLASS = HyperliquidPerpsCommission

    def prices_to_signals(self, prices):
        closes = prices["Close"]
        return pd.DataFrame(1.0, index=closes.index, columns=closes.columns)


class Flat(CryptoMoonshot):
    CODE = "xv-flat"
    SYMBOLS = ["BTC", "ETH"]
    INTERVAL = "1d"
    SLIPPAGE_BPS = 0
    COMMISSION_CLASS = ZeroCommission

    def prices_to_signals(self, prices):
        closes = prices["Close"]
        return pd.DataFrame(0.0, index=closes.index, columns=closes.columns)


class MoonshotAligned(CryptoMoonshot):
    """Strategy whose framework methods match Moonshot defaults exactly."""

    CODE = "xv-aligned"
    SYMBOLS = ["BTC", "ETH"]
    INTERVAL = "1d"
    SLIPPAGE_BPS = 5
    COMMISSION_CLASS = HyperliquidPerpsCommission

    def prices_to_signals(self, prices):
        closes = prices["Close"]
        return pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    def target_weights_to_positions(self, target_weights, prices):
        return target_weights.shift()

    def positions_to_gross_returns(self, positions, prices):
        closes = prices["Close"]
        return (closes.pct_change() * positions.shift()).fillna(0)


# ---------------------------------------------------------------------------
# Moonshot reference function unit tests
# ---------------------------------------------------------------------------

class TestMoonshotReference:
    def test_positions_shifted_by_one(self):
        weights = pd.DataFrame(
            {"A": [0.5, 0.5, 0.0], "B": [0.5, 0.5, 0.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        pos = _moonshot_positions(weights)
        assert np.isnan(pos.iloc[0]["A"])
        assert pos.iloc[1]["A"] == 0.5
        assert pos.iloc[2]["A"] == 0.5

    def test_gross_returns_uses_lagged_positions(self):
        dates = pd.date_range("2024-01-01", periods=4, tz="UTC")
        close_data = {"A": [100.0, 110.0, 105.0, 115.0]}
        cols = pd.MultiIndex.from_tuples([("Close", "A")], names=["Field", "Symbol"])
        prices = pd.DataFrame(
            {("Close", "A"): [100.0, 110.0, 105.0, 115.0]},
            index=dates,
        )
        prices.columns = cols

        positions = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0]}, index=dates)
        gross = _moonshot_gross_returns(positions, prices)

        # Bar 2: pct_change = (105-110)/110, lagged pos = 1.0
        expected_bar2 = (105 - 110) / 110 * 1.0
        assert gross.iloc[2]["A"] == pytest.approx(expected_bar2, rel=1e-10)

    def test_slippage_symmetric(self):
        dates = pd.date_range("2024-01-01", periods=4)
        positions = pd.DataFrame({"A": [0.0, 0.5, 0.5, 0.0]}, index=dates)
        slip = _moonshot_slippage(positions, 10.0)

        # Entry bar 1: |0.5 - 0| * 10/10000
        assert slip.iloc[1]["A"] == pytest.approx(0.5 * 10 / 10_000)
        # Exit bar 3: |0 - 0.5| * 10/10000
        assert slip.iloc[3]["A"] == pytest.approx(0.5 * 10 / 10_000)
        # Hold bar 2: no turnover
        assert slip.iloc[2]["A"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Comparison helper unit tests
# ---------------------------------------------------------------------------

class TestCompareHelpers:
    def test_identical_frames_match(self):
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        result = _compare_frames("test", df, df, atol=1e-10)
        assert result.match is True
        assert result.max_abs_diff == 0.0

    def test_different_frames_dont_match(self):
        a = pd.DataFrame({"A": [1.0, 2.0]})
        b = pd.DataFrame({"A": [1.0, 3.0]})
        result = _compare_frames("test", a, b, atol=1e-10)
        assert result.match is False
        assert result.max_abs_diff == pytest.approx(1.0)

    def test_identical_series_match(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = _compare_series("test", s, s, atol=1e-10)
        assert result.match is True

    def test_correlation_near_one_for_scaled(self):
        a = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
        b = a * 2
        result = _compare_frames("test", a, b, atol=1e-10)
        assert result.match is False
        assert result.correlation == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Full cross-validation integration tests
# ---------------------------------------------------------------------------

class TestCrossValidate:
    def test_flat_strategy_all_match(self):
        """A flat strategy produces zero everywhere — both engines agree."""
        strategy = Flat()
        prices = _make_prices()
        report = cross_validate(strategy, prices=prices)

        for stage in report.stages:
            assert stage.match, f"{stage.name} should match for flat strategy"

    def test_long_strategy_signals_and_weights_match(self):
        """Shared stages must always be identical."""
        strategy = LongOnly()
        prices = _make_prices()
        report = cross_validate(strategy, prices=prices)

        sig = next(s for s in report.stages if s.name == "signals")
        wgt = next(s for s in report.stages if s.name == "target_weights")
        assert sig.match
        assert wgt.match

    def test_long_strategy_all_stages_match(self):
        """Our defaults now match Moonshot (shift + fillna-before-diff),
        so all stages should agree."""
        strategy = LongOnly()
        prices = _make_prices()
        report = cross_validate(strategy, prices=prices)

        for stage in report.stages:
            assert stage.match, f"{stage.name} should match now that defaults are aligned"

    def test_aligned_strategy_all_stages_match(self):
        """Explicit Moonshot-aligned overrides should also match exactly."""
        strategy = MoonshotAligned()
        prices = _make_prices()
        report = cross_validate(strategy, prices=prices)

        for stage in report.stages:
            assert stage.match, f"{stage.name} should match for aligned strategy"

    def test_report_contains_metrics(self):
        strategy = LongOnly()
        prices = _make_prices()
        report = cross_validate(strategy, prices=prices)

        assert "sharpe" in report.our_metrics
        assert "sharpe" in report.moonshot_metrics
        assert "cagr" in report.our_metrics

    def test_portfolio_returns_are_series(self):
        strategy = LongOnly()
        prices = _make_prices()
        report = cross_validate(strategy, prices=prices)

        assert isinstance(report.our_returns, pd.Series)
        assert isinstance(report.moonshot_returns, pd.Series)
        assert len(report.our_returns) == len(prices)

    def test_report_all_match_property(self):
        flat_report = cross_validate(Flat(), prices=_make_prices())
        assert flat_report.all_match

        long_report = cross_validate(LongOnly(), prices=_make_prices())
        assert long_report.all_match

    def test_returns_highly_correlated(self):
        """Even when positions differ by a shift, portfolio returns should be
        highly correlated (same signal, nearly same market path)."""
        strategy = LongOnly()
        prices = _make_prices(n_days=200)
        report = cross_validate(strategy, prices=prices)

        port = next(s for s in report.stages if s.name == "portfolio_returns")
        assert port.correlation > 0.9, "Returns should be highly correlated"
