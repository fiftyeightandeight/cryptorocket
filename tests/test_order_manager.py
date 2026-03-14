"""Tests for execution.order_manager — order conversion and sizing."""

import pandas as pd
import pytest

from execution.order_manager import (
    round_price,
    round_size,
    stubs_to_hyperliquid_orders,
)


class TestRoundSize:
    def test_rounds_down(self):
        assert round_size(1.999, 2) == 1.99

    def test_zero_decimals(self):
        assert round_size(9.87, 0) == 9.0

    def test_negative_input_uses_abs(self):
        assert round_size(-5.678, 1) == 5.6

    def test_tiny_rounds_to_zero(self):
        assert round_size(0.001, 0) == 0.0


class TestRoundPrice:
    def test_five_sig_figs(self):
        assert round_price(12345.6789) == 12346.0

    def test_zero_returns_zero(self):
        assert round_price(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert round_price(-10.0) == 0.0

    def test_small_price(self):
        result = round_price(0.00012345)
        assert result == pytest.approx(0.00012345, rel=1e-2)

    def test_spot_more_decimals(self):
        px = 0.000001234
        perp = round_price(px, is_spot=False)
        spot = round_price(px, is_spot=True)
        assert len(str(spot).split(".")[-1]) <= 8
        assert len(str(perp).split(".")[-1]) <= 6


class TestStubsToHyperliquidOrders:
    @pytest.fixture
    def basic_stubs(self):
        return pd.DataFrame({
            "Symbol": ["BTC", "ETH"],
            "Weight": [0.5, 0.3],
            "Action": ["BUY", "SELL"],
            "Strategy": ["test", "test"],
            "order_type": ["market", "market"],
            "tif": ["Ioc", "Ioc"],
        })

    def test_converts_stubs_to_orders(self, basic_stubs):
        orders = stubs_to_hyperliquid_orders(
            basic_stubs,
            account_value=100_000,
            current_prices={"BTC": 50_000.0, "ETH": 3_000.0},
            sz_decimals={"BTC": 4, "ETH": 3},
        )
        assert len(orders) == 2

        btc_order = orders[0]
        assert btc_order["coin"] == "BTC"
        assert btc_order["is_buy"] is True
        assert btc_order["sz"] > 0
        assert btc_order["reduce_only"] is False

        eth_order = orders[1]
        assert eth_order["coin"] == "ETH"
        assert eth_order["is_buy"] is False

    def test_skips_missing_price(self, basic_stubs):
        orders = stubs_to_hyperliquid_orders(
            basic_stubs,
            account_value=100_000,
            current_prices={"BTC": 50_000.0},  # ETH missing
            sz_decimals={"BTC": 4, "ETH": 3},
        )
        assert len(orders) == 1
        assert orders[0]["coin"] == "BTC"

    def test_skips_zero_price(self, basic_stubs):
        orders = stubs_to_hyperliquid_orders(
            basic_stubs,
            account_value=100_000,
            current_prices={"BTC": 0.0, "ETH": 3_000.0},
            sz_decimals={"BTC": 4, "ETH": 3},
        )
        assert len(orders) == 1
        assert orders[0]["coin"] == "ETH"

    def test_size_matches_notional(self):
        stubs = pd.DataFrame({
            "Symbol": ["BTC"],
            "Weight": [0.1],
            "Action": ["BUY"],
            "Strategy": ["test"],
            "order_type": ["market"],
            "tif": ["Ioc"],
        })
        orders = stubs_to_hyperliquid_orders(
            stubs,
            account_value=100_000,
            current_prices={"BTC": 50_000.0},
            sz_decimals={"BTC": 4},
        )
        expected_size = round_size(10_000 / 50_000, 4)
        assert orders[0]["sz"] == expected_size

    def test_buy_limit_above_market(self):
        stubs = pd.DataFrame({
            "Symbol": ["ETH"],
            "Weight": [0.5],
            "Action": ["BUY"],
            "Strategy": ["test"],
            "order_type": ["market"],
            "tif": ["Ioc"],
        })
        orders = stubs_to_hyperliquid_orders(
            stubs,
            account_value=100_000,
            current_prices={"ETH": 3000.0},
            sz_decimals={"ETH": 3},
        )
        assert orders[0]["limit_px"] > 3000.0

    def test_sell_limit_below_market(self):
        stubs = pd.DataFrame({
            "Symbol": ["ETH"],
            "Weight": [0.5],
            "Action": ["SELL"],
            "Strategy": ["test"],
            "order_type": ["market"],
            "tif": ["Ioc"],
        })
        orders = stubs_to_hyperliquid_orders(
            stubs,
            account_value=100_000,
            current_prices={"ETH": 3000.0},
            sz_decimals={"ETH": 3},
        )
        assert orders[0]["limit_px"] < 3000.0


