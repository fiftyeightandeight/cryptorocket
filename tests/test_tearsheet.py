"""Tests for backtest.tearsheet — metric calculations."""

import numpy as np
import pandas as pd
import pytest

from backtest.tearsheet import compute_metrics


@pytest.fixture
def flat_returns():
    return pd.Series(0.0, index=pd.date_range("2024-01-01", periods=365))


@pytest.fixture
def positive_returns():
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0.001, 0.02, 365),
        index=pd.date_range("2024-01-01", periods=365),
    )


@pytest.fixture
def negative_returns():
    np.random.seed(42)
    return pd.Series(
        np.random.normal(-0.002, 0.02, 365),
        index=pd.date_range("2024-01-01", periods=365),
    )


class TestComputeMetrics:
    def test_empty_returns(self):
        result = compute_metrics(pd.Series([], dtype=float))
        assert result == {}

    def test_all_nan_returns(self):
        result = compute_metrics(pd.Series([np.nan, np.nan, np.nan]))
        assert result == {}

    def test_flat_returns_are_zero(self, flat_returns):
        m = compute_metrics(flat_returns)
        assert m["total_return"] == 0.0
        assert m["cagr"] == 0.0
        assert m["sharpe"] == 0
        assert m["max_drawdown"] == 0.0

    def test_metric_keys(self, positive_returns):
        m = compute_metrics(positive_returns)
        expected_keys = {
            "total_return", "cagr", "ann_volatility", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate", "profit_factor",
            "avg_win", "avg_loss", "n_periods", "n_years",
        }
        assert set(m.keys()) == expected_keys

    def test_positive_drift_metrics(self, positive_returns):
        m = compute_metrics(positive_returns)
        assert m["total_return"] > 0
        assert m["cagr"] > 0
        assert m["sharpe"] > 0
        assert m["ann_volatility"] > 0
        assert m["max_drawdown"] <= 0

    def test_negative_drift_metrics(self, negative_returns):
        m = compute_metrics(negative_returns)
        assert m["total_return"] < 0
        assert m["cagr"] < 0
        assert m["sharpe"] < 0

    def test_n_periods_matches(self, positive_returns):
        m = compute_metrics(positive_returns)
        assert m["n_periods"] == 365
        assert abs(m["n_years"] - 1.0) < 0.01

    def test_win_rate_range(self, positive_returns):
        m = compute_metrics(positive_returns)
        assert 0 <= m["win_rate"] <= 1

    def test_custom_periods_per_year(self, positive_returns):
        m_daily = compute_metrics(positive_returns, periods_per_year=365)
        m_hourly = compute_metrics(positive_returns, periods_per_year=8760)
        assert m_daily["n_years"] != m_hourly["n_years"]
        assert m_daily["ann_volatility"] != m_hourly["ann_volatility"]

    def test_perfect_wins(self):
        returns = pd.Series([0.01] * 100, index=pd.date_range("2024-01-01", periods=100))
        m = compute_metrics(returns)
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == float("inf")

    def test_perfect_losses(self):
        returns = pd.Series([-0.01] * 100, index=pd.date_range("2024-01-01", periods=100))
        m = compute_metrics(returns)
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0
