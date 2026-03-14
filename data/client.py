import time
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from hyperliquid.info import Info
from hyperliquid.utils.error import ClientError

from config import BASE_URL, CANDLE_LIMIT

logger = logging.getLogger(__name__)

# Conservative rate limit: use 800/min (vs 1200 nominal) to leave headroom
RATE_LIMIT_WEIGHT = 800
RATE_LIMIT_WINDOW = 60  # seconds
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2  # seconds


class RateLimiter:
    """Token-bucket rate limiter tracking request weight over a sliding window."""

    def __init__(self, max_weight: int = RATE_LIMIT_WEIGHT, window: float = RATE_LIMIT_WINDOW):
        self.max_weight = max_weight
        self.window = window
        self._requests: list[tuple[float, int]] = []  # (timestamp, weight)
        self._lock = threading.Lock()

    def _prune(self, now: float):
        cutoff = now - self.window
        self._requests = [(t, w) for t, w in self._requests if t > cutoff]

    def _current_weight(self, now: float) -> int:
        self._prune(now)
        return sum(w for _, w in self._requests)

    def acquire(self, weight: int = 1):
        """Block until we have capacity for the given weight."""
        while True:
            with self._lock:
                now = time.monotonic()
                current = self._current_weight(now)
                if current + weight <= self.max_weight:
                    self._requests.append((now, weight))
                    return

                # Calculate how long to wait for oldest request to expire
                oldest = self._requests[0][0]
                wait = (oldest + self.window) - now + 0.1

            logger.debug(f"Rate limit: {current + weight}/{self.max_weight} weight, sleeping {wait:.1f}s")
            time.sleep(max(wait, 0.1))

    def backoff(self, seconds: float):
        """Force a cooldown period after a 429 response."""
        with self._lock:
            # Add a large phantom weight to prevent new requests during cooldown
            now = time.monotonic()
            self._requests.append((now, self.max_weight))
        logger.info(f"Rate limit backoff: sleeping {seconds:.1f}s")
        time.sleep(seconds)


def _candle_weight(n_candles: int) -> int:
    """Estimate API weight for a candle snapshot request.

    From Hyperliquid docs: candleSnapshot costs 20 + 2 per 300 candles.
    """
    return 20 + 2 * ((n_candles + 299) // 300)


def _retry_on_rate_limit(func, rate_limiter: RateLimiter):
    """Execute func with retry + exponential backoff on 429 errors."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return func()
        except ClientError as e:
            if e.status_code == 429:
                if attempt == MAX_RETRIES:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"429 rate limited, retry {attempt + 1}/{MAX_RETRIES} in {delay}s")
                rate_limiter.backoff(delay)
            else:
                raise


class HyperliquidClient:
    """Thin wrapper around the Hyperliquid SDK Info class for data retrieval."""

    def __init__(self, base_url: Optional[str] = None):
        self.info = Info(base_url or BASE_URL, skip_ws=True)
        self.rate_limiter = RateLimiter()

    def _call(self, weight: int, func):
        """Acquire rate limit weight, call func, retry on 429."""
        self.rate_limiter.acquire(weight=weight)
        return _retry_on_rate_limit(func, self.rate_limiter)

    def get_perp_universe(self) -> list[dict]:
        """Return list of perp asset metadata dicts with keys: name, szDecimals, maxLeverage."""
        meta = self._call(1, lambda: self.info.meta_and_asset_ctxs())
        universe_info = meta[0]["universe"]
        asset_ctxs = meta[1]
        results = []
        for info, ctx in zip(universe_info, asset_ctxs):
            results.append({
                "name": info["name"],
                "szDecimals": info["szDecimals"],
                "maxLeverage": info.get("maxLeverage", 50),
                "markPx": ctx.get("markPx"),
                "dayNtlVlm": ctx.get("dayNtlVlm"),
                "openInterest": ctx.get("openInterest"),
                "funding": ctx.get("funding"),
            })
        return results

    def get_spot_universe(self) -> list[dict]:
        """Return list of spot pair metadata."""
        meta = self._call(1, lambda: self.info.spot_meta_and_asset_ctxs())
        spot_meta = meta[0]
        asset_ctxs = meta[1]
        tokens = {t["index"]: t for t in spot_meta["tokens"]}
        results = []
        for pair, ctx in zip(spot_meta["universe"], asset_ctxs):
            base_token = tokens[pair["tokens"][0]]
            quote_token = tokens[pair["tokens"][1]]
            results.append({
                "name": pair["name"],
                "baseName": base_token["name"],
                "quoteName": quote_token["name"],
                "szDecimals": base_token["szDecimals"],
                "markPx": ctx.get("markPx"),
                "dayNtlVlm": ctx.get("dayNtlVlm"),
            })
        return results

    def get_all_mids(self) -> dict[str, float]:
        """Return dict of symbol -> mid price."""
        raw = self._call(1, lambda: self.info.all_mids())
        return {k: float(v) for k, v in raw.items()}

    def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict]:
        """Fetch OHLCV candles with automatic pagination and rate limiting.

        Returns list of dicts with keys: timestamp, open, high, low, close, volume.
        """
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        all_candles = []
        current_start = start_ms

        while current_start < end_ms:
            estimated_weight = _candle_weight(CANDLE_LIMIT)
            raw = self._call(
                estimated_weight,
                lambda cs=current_start: self.info.candles_snapshot(
                    symbol, interval, cs, end_ms
                ),
            )
            if not raw:
                break

            for c in raw:
                all_candles.append({
                    "timestamp": datetime.fromtimestamp(c["t"] / 1000, tz=timezone.utc),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                })

            # Move start past the last candle we received
            last_ts = raw[-1]["t"]
            if last_ts <= current_start:
                break
            current_start = last_ts + 1

        return all_candles

    def get_funding_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict]:
        """Fetch funding rate history for a perp symbol."""
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        raw = self._call(1, lambda: self.info.funding_history(symbol, start_ms, end_ms))
        return [
            {
                "timestamp": datetime.fromtimestamp(r["time"] / 1000, tz=timezone.utc),
                "rate": float(r["fundingRate"]),
                "premium": float(r["premium"]),
            }
            for r in raw
        ]

    def get_l2_snapshot(self, symbol: str) -> dict:
        """Get L2 order book snapshot."""
        return self._call(1, lambda: self.info.l2_snapshot(symbol))
