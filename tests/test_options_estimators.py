"""Tests for Deribit client, options estimators, and serialization."""

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pytest

from backtest.options_estimators import (
    IVPremiumEstimator,
    SpreadEstimator,
    _bucket_key,
    _bucket_label,
)
from data.deribit_client import DeribitClient, parse_instrument_name
from data.schema import SCHEMA_SQL


# ---------------------------------------------------------------------------
# parse_instrument_name
# ---------------------------------------------------------------------------

class TestParseInstrumentName:
    def test_standard_put(self):
        result = parse_instrument_name("BTC-28MAR26-80000-P")
        assert result["underlying"] == "BTC"
        assert result["expiry_date"] == date(2026, 3, 28)
        assert result["strike"] == 80000.0
        assert result["option_type"] == "P"

    def test_standard_call(self):
        result = parse_instrument_name("BTC-1JAN27-100000-C")
        assert result["underlying"] == "BTC"
        assert result["expiry_date"] == date(2027, 1, 1)
        assert result["strike"] == 100000.0
        assert result["option_type"] == "C"

    def test_eth(self):
        result = parse_instrument_name("ETH-15JUN26-5000-P")
        assert result["underlying"] == "ETH"
        assert result["expiry_date"] == date(2026, 6, 15)
        assert result["strike"] == 5000.0

    def test_single_digit_day(self):
        result = parse_instrument_name("BTC-5FEB26-90000-C")
        assert result["expiry_date"] == date(2026, 2, 5)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_instrument_name("BTC-PERP")

    def test_invalid_format_too_few_parts(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_instrument_name("BTC-28MAR26-80000")


# ---------------------------------------------------------------------------
# DeribitClient response handling
# ---------------------------------------------------------------------------

class TestDeribitClient:
    @patch("data.deribit_client.requests.Session")
    def test_get_book_summary(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": [
                {
                    "instrument_name": "BTC-28MAR26-80000-P",
                    "bid_price": 0.015,
                    "ask_price": 0.020,
                    "mark_price": 0.017,
                    "mark_iv": 55.0,
                    "open_interest": 100.0,
                    "underlying_price": 85000.0,
                }
            ]
        }
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = DeribitClient(base_url="https://test.deribit.com/api/v2")
        result = client.get_book_summary("BTC", "option")

        assert len(result) == 1
        assert result[0]["instrument_name"] == "BTC-28MAR26-80000-P"

    @patch("data.deribit_client.requests.Session")
    def test_get_ticker(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {
                "instrument_name": "BTC-28MAR26-80000-P",
                "best_bid_price": 0.015,
                "best_ask_price": 0.020,
                "mark_price": 0.017,
                "mark_iv": 55.0,
                "bid_iv": 53.0,
                "ask_iv": 57.0,
                "greeks": {
                    "delta": -0.20,
                    "gamma": 0.0001,
                    "theta": -50.0,
                    "vega": 100.0,
                    "rho": -5.0,
                },
                "underlying_price": 85000.0,
                "open_interest": 100.0,
            }
        }
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = DeribitClient(base_url="https://test.deribit.com/api/v2")
        result = client.get_ticker("BTC-28MAR26-80000-P")

        assert result["bid_iv"] == 53.0
        assert result["greeks"]["delta"] == -0.20

    @patch("data.deribit_client.requests.Session")
    def test_api_error_raises(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "error": {"code": 10001, "message": "bad request"}
        }
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = DeribitClient(base_url="https://test.deribit.com/api/v2")
        with pytest.raises(RuntimeError, match="Deribit API error"):
            client.get_book_summary()


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

class TestBucketHelpers:
    def test_bucket_label_in_range(self):
        assert _bucket_label(0.05, [0, 0.10, 0.25]) == "0-0.1"

    def test_bucket_label_inf(self):
        assert "inf" in _bucket_label(100, [0, 40, 60, 80, float("inf")])

    def test_bucket_key_deterministic(self):
        k1 = _bucket_key(0.15, 20, 55)
        k2 = _bucket_key(0.15, 20, 55)
        assert k1 == k2
        assert "|" in k1


# ---------------------------------------------------------------------------
# Helper: in-memory DuckDB with synthetic snapshot data
# ---------------------------------------------------------------------------

def _create_test_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(SCHEMA_SQL)

    now = datetime.now(timezone.utc)
    rng = np.random.default_rng(42)

    # Synthetic DVOL
    conn.execute(
        "INSERT INTO dvol (timestamp, currency, open, high, low, close) VALUES (?, 'BTC', 55, 58, 52, 55)",
        [now],
    )

    # Synthetic realized vol
    conn.execute(
        "INSERT INTO realized_volatility (timestamp, currency, volatility) VALUES (?, 'BTC', 45)",
        [now],
    )

    # Synthetic snapshots across different buckets
    # Each row needs a unique (snapshot_ts, instrument_name) pair
    deltas = [0.05, 0.15, 0.30, 0.45]
    dtes = [3, 10, 20, 60, 120]
    row_idx = 0
    for d in deltas:
        for dte in dtes:
            mark = 0.01 + rng.uniform(0, 0.05)
            spread_frac = 0.05 + (0.5 - d) * 0.1 + rng.uniform(0, 0.02)
            half_spread = mark * spread_frac / 2
            bid = mark - half_spread
            ask = mark + half_spread
            mark_iv = 55 + rng.uniform(-5, 15)
            strike = 80000 + row_idx * 1000

            conn.execute(
                """INSERT INTO options_snapshots
                   (snapshot_ts, instrument_name, bid_price, ask_price,
                    mark_price, bid_iv, ask_iv, mark_iv, underlying_price,
                    delta, gamma, theta, vega, rho, open_interest,
                    strike, expiry_date, option_type, dte)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    now,
                    f"BTC-28MAR26-{int(strike)}-P",
                    float(bid),
                    float(ask),
                    float(mark),
                    float(mark_iv - 2),
                    float(mark_iv + 2),
                    float(mark_iv),
                    85000.0,
                    float(-d),
                    0.0001,
                    -50.0,
                    100.0,
                    -5.0,
                    50.0,
                    float(strike),
                    date(2026, 3, 28),
                    "P",
                    dte,
                ],
            )
            row_idx += 1

    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# SpreadEstimator
# ---------------------------------------------------------------------------

class TestSpreadEstimator:
    def test_fit_and_estimate(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = SpreadEstimator().fit(db_path=db_path)

        assert est._n_samples > 0
        assert est._global_median > 0

        spread = est.estimate(delta=0.20, dte=20, dvol=55)
        assert spread > 0
        assert spread < 2.0  # sanity bound

    def test_wider_spread_for_low_delta(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = SpreadEstimator().fit(db_path=db_path)

        low_delta = est.estimate(delta=0.05, dte=20, dvol=55)
        high_delta = est.estimate(delta=0.45, dte=20, dvol=55)
        # Low-delta (far OTM) options typically have wider spreads
        assert low_delta > high_delta

    def test_fallback_to_global_median(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = SpreadEstimator().fit(db_path=db_path)

        spread = est.estimate(delta=0.49, dte=300, dvol=99)
        assert spread == est._global_median

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(SCHEMA_SQL)
        conn.close()

        est = SpreadEstimator().fit(db_path=db_path)
        assert est._n_samples == 0
        assert est.estimate(delta=0.2, dte=30) == 0.0


# ---------------------------------------------------------------------------
# IVPremiumEstimator
# ---------------------------------------------------------------------------

class TestIVPremiumEstimator:
    def test_fit_and_estimate(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = IVPremiumEstimator().fit(db_path=db_path)

        assert est._n_samples > 0

        sigma = est.estimate(delta=0.20, dte=20, dvol=55, realized_vol=45)
        # IV should be higher than realized vol (positive premium)
        assert sigma > 45

    def test_iv_premium_positive_on_average(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = IVPremiumEstimator().fit(db_path=db_path)

        # Our synthetic data has mark_iv around 55-70 and realized_vol=45
        assert est._global_median > 0

    def test_returns_realized_plus_premium(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = IVPremiumEstimator().fit(db_path=db_path)

        rv1 = 30.0
        rv2 = 60.0
        s1 = est.estimate(delta=0.20, dte=20, dvol=55, realized_vol=rv1)
        s2 = est.estimate(delta=0.20, dte=20, dvol=55, realized_vol=rv2)
        assert s2 - s1 == pytest.approx(rv2 - rv1, abs=0.01)

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute(SCHEMA_SQL)
        conn.close()

        est = IVPremiumEstimator().fit(db_path=db_path)
        assert est._n_samples == 0
        assert est.estimate(delta=0.2, dte=30, realized_vol=50) == 50.0


# ---------------------------------------------------------------------------
# JSON round-trip serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_spread_save_load(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = SpreadEstimator().fit(db_path=db_path)

        save_path = tmp_path / "spread.json"
        est.save(save_path)

        loaded = SpreadEstimator.load(save_path)
        assert loaded._global_median == est._global_median
        assert loaded._buckets == est._buckets
        assert loaded._n_samples == est._n_samples

        data = json.loads(save_path.read_text())
        assert data["type"] == "SpreadEstimator"

    def test_iv_premium_save_load(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        est = IVPremiumEstimator().fit(db_path=db_path)

        save_path = tmp_path / "iv_premium.json"
        est.save(save_path)

        loaded = IVPremiumEstimator.load(save_path)
        assert loaded._global_median == est._global_median
        assert loaded._buckets == est._buckets

        data = json.loads(save_path.read_text())
        assert data["type"] == "IVPremiumEstimator"

    def test_loaded_estimator_produces_same_results(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        spread_est = SpreadEstimator().fit(db_path=db_path)

        save_path = tmp_path / "spread.json"
        spread_est.save(save_path)
        loaded = SpreadEstimator.load(save_path)

        for delta in [0.05, 0.15, 0.30, 0.45]:
            for dte in [5, 15, 60]:
                orig = spread_est.estimate(delta=delta, dte=dte, dvol=55)
                lo = loaded.estimate(delta=delta, dte=dte, dvol=55)
                assert orig == lo
