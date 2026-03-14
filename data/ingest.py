import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import DEFAULT_INTERVALS, DEFAULT_LOOKBACK_DAYS
from data.client import HyperliquidClient
from data.schema import get_connection, init_db

logger = logging.getLogger(__name__)


def ingest_universe(client: Optional[HyperliquidClient] = None, db_path=None):
    """Fetch and store the current tradable universe (perps + spot)."""
    client = client or HyperliquidClient()
    init_db(db_path).close()
    conn = get_connection(db_path)

    # Perps
    perps = client.get_perp_universe()
    for p in perps:
        conn.execute(
            """INSERT OR REPLACE INTO universe (symbol, market_type, sz_decimals, max_leverage, mark_px, day_ntl_vlm, updated_at)
               VALUES (?, 'perp', ?, ?, ?, ?, current_timestamp)""",
            [p["name"], p["szDecimals"], p["maxLeverage"],
             float(p["markPx"]) if p["markPx"] else None,
             float(p["dayNtlVlm"]) if p["dayNtlVlm"] else None],
        )
    logger.info(f"Ingested {len(perps)} perp assets")

    # Spot
    spots = client.get_spot_universe()
    for s in spots:
        conn.execute(
            """INSERT OR REPLACE INTO universe (symbol, market_type, sz_decimals, max_leverage, mark_px, day_ntl_vlm, updated_at)
               VALUES (?, 'spot', ?, NULL, ?, ?, current_timestamp)""",
            [s["name"], s["szDecimals"],
             float(s["markPx"]) if s["markPx"] else None,
             float(s["dayNtlVlm"]) if s["dayNtlVlm"] else None],
        )
    logger.info(f"Ingested {len(spots)} spot pairs")
    conn.close()


def _parse_ts(val) -> datetime:
    """Convert a DuckDB timestamp value to a timezone-aware datetime."""
    if hasattr(val, "timestamp"):
        return val.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(val)).replace(tzinfo=timezone.utc)


def _get_stored_range(db_path, symbol: str, interval: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Query the first and last stored candle timestamps using a fresh connection."""
    conn = get_connection(db_path)
    try:
        result = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE symbol = ? AND interval = ?",
            [symbol, interval],
        ).fetchone()
        if result[0] is None:
            return None, None
        return _parse_ts(result[0]), _parse_ts(result[1])
    finally:
        conn.close()


def _insert_candles(db_path, symbol: str, interval: str, candles: list[dict]):
    """Batch-insert candles using a fresh connection."""
    conn = get_connection(db_path)
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO candles
               (symbol, interval, timestamp, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                [symbol, interval, c["timestamp"],
                 c["open"], c["high"], c["low"], c["close"], c["volume"]]
                for c in candles
            ],
        )
    finally:
        conn.close()


def ingest_candles(
    symbols: list[str],
    intervals: Optional[list[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    client: Optional[HyperliquidClient] = None,
    db_path=None,
):
    """Fetch and store candle data for given symbols and intervals."""
    client = client or HyperliquidClient()
    # Ensure tables exist
    init_db(db_path).close()
    intervals = intervals or DEFAULT_INTERVALS

    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    for symbol in symbols:
        for interval in intervals:
            first_stored, last_stored = _get_stored_range(db_path, symbol, interval)
            total_ingested = 0

            # Backfill: if requested start is before our earliest stored candle
            if first_stored is not None and start_date < first_stored:
                logger.info(f"{symbol}/{interval}: backfilling {start_date} → {first_stored}")
                candles = client.get_candles(symbol, interval, start_date, first_stored)
                if candles:
                    _insert_candles(db_path, symbol, interval, candles)
                    total_ingested += len(candles)

            # Forward fill: fetch from last stored (or start_date if no data)
            if last_stored is not None:
                effective_start = last_stored
                logger.info(f"{symbol}/{interval}: resuming from {effective_start}")
            else:
                effective_start = start_date

            candles = client.get_candles(symbol, interval, effective_start, end_date)
            if candles:
                _insert_candles(db_path, symbol, interval, candles)
                total_ingested += len(candles)

            if total_ingested > 0:
                logger.info(f"{symbol}/{interval}: ingested {total_ingested} candles total")
            else:
                logger.info(f"{symbol}/{interval}: no new candles")


def ingest_funding(
    symbols: list[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    client: Optional[HyperliquidClient] = None,
    db_path=None,
):
    """Fetch and store funding rate history for perp symbols."""
    client = client or HyperliquidClient()
    init_db(db_path).close()

    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    for symbol in symbols:
        rates = client.get_funding_history(symbol, start_date, end_date)
        if not rates:
            logger.info(f"{symbol}: no funding data")
            continue

        conn = get_connection(db_path)
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO funding_rates (symbol, timestamp, rate, premium)
                   VALUES (?, ?, ?, ?)""",
                [[symbol, r["timestamp"], r["rate"], r["premium"]] for r in rates],
            )
        finally:
            conn.close()

        logger.info(f"{symbol}: ingested {len(rates)} funding records")
