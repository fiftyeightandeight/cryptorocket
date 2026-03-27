"""Deribit JSON-RPC API client for public market data endpoints.

All endpoints used here are public (no authentication required).
Rate limiting follows Deribit's non-matching engine limit of 20 req/s.
"""

import logging
import os
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://www.deribit.com/api/v2"
_TESTNET_BASE_URL = "https://test.deribit.com/api/v2"

RATE_LIMIT_PER_SEC = 18  # stay under 20 req/s hard limit
MAX_RETRIES = 4
RETRY_BASE_DELAY = 1.0


class _RateLimiter:
    """Simple token-bucket limiting requests per second."""

    def __init__(self, max_per_sec: int = RATE_LIMIT_PER_SEC):
        self.min_interval = 1.0 / max_per_sec
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


def parse_instrument_name(name: str) -> dict:
    """Parse a Deribit instrument name into structured fields.

    Example: "BTC-28MAR26-80000-P" ->
        {"underlying": "BTC", "expiry_date": date(2026,3,28),
         "strike": 80000.0, "option_type": "P"}
    """
    parts = name.split("-")
    if len(parts) != 4:
        raise ValueError(f"Cannot parse instrument name: {name}")

    underlying = parts[0]
    expiry_str = parts[1]
    strike = float(parts[2])
    option_type = parts[3].upper()

    day = int(expiry_str[:len(expiry_str) - 5])
    month_str = expiry_str[len(expiry_str) - 5 : len(expiry_str) - 2].upper()
    year_str = expiry_str[-2:]

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = month_map[month_str]
    year = 2000 + int(year_str)

    from datetime import date
    expiry_date = date(year, month, day)

    return {
        "underlying": underlying,
        "expiry_date": expiry_date,
        "strike": strike,
        "option_type": option_type,
    }


class DeribitClient:
    """Thin wrapper around Deribit's public JSON-RPC API."""

    def __init__(self, base_url: Optional[str] = None):
        if base_url:
            self.base_url = base_url
        elif os.environ.get("DERIBIT_TESTNET", "").lower() == "true":
            self.base_url = _TESTNET_BASE_URL
        else:
            self.base_url = os.environ.get("DERIBIT_BASE_URL", _DEFAULT_BASE_URL)

        self._session = requests.Session()
        self._rate_limiter = _RateLimiter()

    def _call(self, method: str, params: Optional[dict] = None) -> dict:
        """Execute a JSON-RPC call with retry on rate limit (429)."""
        url = f"{self.base_url}/{method}"
        for attempt in range(MAX_RETRIES + 1):
            self._rate_limiter.acquire()
            try:
                resp = self._session.get(url, params=params or {}, timeout=15)
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Request error %s, retry %d/%d in %.1fs", e, attempt + 1, MAX_RETRIES, delay)
                time.sleep(delay)
                continue

            if resp.status_code == 429:
                if attempt == MAX_RETRIES:
                    resp.raise_for_status()
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("429 rate limited, retry %d/%d in %.1fs", attempt + 1, MAX_RETRIES, delay)
                time.sleep(delay)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise RuntimeError(f"Deribit API error: {data['error']}")

            return data["result"]

        raise RuntimeError("Exhausted retries")

    # ------------------------------------------------------------------
    # Market data endpoints
    # ------------------------------------------------------------------

    def get_book_summary(
        self, currency: str = "BTC", kind: str = "option"
    ) -> list[dict]:
        """Retrieve summary for all instruments of a given kind.

        Returns list of dicts with keys including: instrument_name,
        bid_price, ask_price, mark_price, mark_iv, open_interest,
        underlying_price, volume.
        """
        return self._call(
            "public/get_book_summary_by_currency",
            {"currency": currency, "kind": kind},
        )

    def get_ticker(self, instrument_name: str) -> dict:
        """Retrieve full ticker for one instrument.

        For options, includes: best_bid_price, best_ask_price,
        mark_price, mark_iv, bid_iv, ask_iv, greeks (delta, gamma,
        theta, vega, rho), underlying_price, open_interest.
        """
        return self._call("public/ticker", {"instrument_name": instrument_name})

    def get_historical_volatility(self, currency: str = "BTC") -> list:
        """Retrieve realized volatility time series.

        Returns list of [timestamp_ms, volatility] pairs.
        """
        return self._call(
            "public/get_historical_volatility", {"currency": currency}
        )

    def get_volatility_index_data(
        self,
        currency: str = "BTC",
        start_timestamp: int = 0,
        end_timestamp: Optional[int] = None,
        resolution: str = "3600",
    ) -> dict:
        """Retrieve DVOL index candles.

        resolution: "1" (1s), "60" (1m), "3600" (1h), "43200" (12h), "1D"
        Returns dict with "data" (list of [ts, o, h, l, c]) and
        "continuation" (next end_timestamp or None).
        """
        if end_timestamp is None:
            end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        return self._call(
            "public/get_volatility_index_data",
            {
                "currency": currency,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "resolution": resolution,
            },
        )

    def get_settlements(
        self,
        currency: str = "BTC",
        settlement_type: str = "delivery",
        count: int = 1000,
        continuation: Optional[str] = None,
        search_start_timestamp: Optional[int] = None,
    ) -> dict:
        """Retrieve settlement/delivery events, paginated.

        search_start_timestamp: if set, the API returns results going
        backwards from this timestamp (ms).  Use the MIN stored timestamp
        to skip already-fetched data and jump straight to the backfill
        frontier.

        Returns dict with "settlements" list and "continuation" token.
        """
        params: dict = {
            "currency": currency,
            "type": settlement_type,
            "count": count,
        }
        if continuation is not None:
            params["continuation"] = continuation
        if search_start_timestamp is not None:
            params["search_start_timestamp"] = search_start_timestamp
        return self._call(
            "public/get_last_settlements_by_currency", params
        )

    def get_delivery_prices(
        self,
        index_name: str = "btc_usd",
        count: int = 1000,
        offset: int = 0,
    ) -> dict:
        """Retrieve delivery (settlement) prices by index.

        Returns dict with "data" list of {date, delivery_price} and
        "records_total".
        """
        return self._call(
            "public/get_delivery_prices",
            {"index_name": index_name, "count": count, "offset": offset},
        )
