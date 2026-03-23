"""Deribit data ingestion — options snapshots, DVOL, realized vol, settlements.

Mirrors the pattern established in data/ingest.py for Hyperliquid data.
"""

import logging
from datetime import datetime, date, timezone
from typing import Optional

from data.deribit_client import DeribitClient, parse_instrument_name
from data.schema import get_connection, init_db

logger = logging.getLogger(__name__)


def ingest_options_snapshots(
    client: Optional[DeribitClient] = None,
    db_path=None,
    currency: str = "BTC",
) -> int:
    """Snapshot the full live options chain: book_summary + ticker for Greeks.

    For each instrument with nonzero bid *or* ask, fetches the full ticker
    to get bid_iv, ask_iv, and Greeks.  Returns the number of rows inserted.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    summaries = client.get_book_summary(currency=currency, kind="option")
    now_ts = datetime.now(timezone.utc)
    rows_inserted = 0

    for s in summaries:
        name = s["instrument_name"]
        bid = s.get("bid_price")
        ask = s.get("ask_price")

        if not bid and not ask:
            continue

        try:
            parsed = parse_instrument_name(name)
        except ValueError:
            logger.debug("Skipping unparseable instrument: %s", name)
            continue

        expiry = parsed["expiry_date"]
        dte = (expiry - now_ts.date()).days

        bid_iv = ask_iv = None
        greeks = {}
        try:
            ticker = client.get_ticker(name)
            bid_iv = ticker.get("bid_iv")
            ask_iv = ticker.get("ask_iv")
            greeks = ticker.get("greeks", {}) or {}
        except Exception as exc:
            logger.warning("Ticker failed for %s: %s", name, exc)

        conn.execute(
            """INSERT OR REPLACE INTO options_snapshots
               (snapshot_ts, instrument_name, bid_price, ask_price,
                mark_price, bid_iv, ask_iv, mark_iv, underlying_price,
                delta, gamma, theta, vega, rho, open_interest,
                strike, expiry_date, option_type, dte)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                now_ts,
                name,
                bid,
                ask,
                s.get("mark_price"),
                bid_iv,
                ask_iv,
                s.get("mark_iv"),
                s.get("underlying_price"),
                greeks.get("delta"),
                greeks.get("gamma"),
                greeks.get("theta"),
                greeks.get("vega"),
                greeks.get("rho"),
                s.get("open_interest"),
                parsed["strike"],
                expiry,
                parsed["option_type"],
                dte,
            ],
        )

        _upsert_instrument(conn, name, parsed)
        rows_inserted += 1

    logger.info("Ingested %d options snapshots for %s", rows_inserted, currency)
    return rows_inserted


def _upsert_instrument(conn, name: str, parsed: dict):
    conn.execute(
        """INSERT OR REPLACE INTO options_instruments
           (instrument_name, underlying, expiry_date, strike, option_type)
           VALUES (?, ?, ?, ?, ?)""",
        [name, parsed["underlying"], parsed["expiry_date"],
         parsed["strike"], parsed["option_type"]],
    )


def ingest_dvol(
    client: Optional[DeribitClient] = None,
    db_path=None,
    currency: str = "BTC",
    start_timestamp: int = 0,
    end_timestamp: Optional[int] = None,
    resolution: str = "3600",
    incremental: bool = False,
) -> int:
    """Fetch DVOL (volatility index) candles.

    When incremental=True, only fetches candles newer than the latest
    stored timestamp (suitable for hourly cron).  When False, fetches
    from start_timestamp (suitable for full backfill).

    Paginates using the continuation token until all history is fetched.
    Returns the total number of candle rows inserted.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    if incremental and start_timestamp == 0:
        row = conn.execute(
            "SELECT MAX(timestamp) FROM dvol WHERE currency = ?",
            [currency],
        ).fetchone()
        if row and row[0] is not None:
            last_ts = row[0]
            if hasattr(last_ts, "timestamp"):
                start_timestamp = int(last_ts.timestamp() * 1000)
            else:
                start_timestamp = int(last_ts)

    if end_timestamp is None:
        end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

    total = 0
    current_end = end_timestamp

    while True:
        result = client.get_volatility_index_data(
            currency=currency,
            start_timestamp=start_timestamp,
            end_timestamp=current_end,
            resolution=resolution,
        )
        candles = result.get("data", [])
        if not candles:
            break

        for c in candles:
            ts_ms, o, h, l, close = c[0], c[1], c[2], c[3], c[4]
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            conn.execute(
                """INSERT OR REPLACE INTO dvol
                   (timestamp, currency, open, high, low, close)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [ts, currency, o, h, l, close],
            )
        total += len(candles)

        continuation = result.get("continuation")
        if not continuation or continuation >= current_end:
            break
        current_end = continuation

    logger.info("Ingested %d DVOL candles for %s", total, currency)
    return total


def ingest_realized_volatility(
    client: Optional[DeribitClient] = None,
    db_path=None,
    currency: str = "BTC",
) -> int:
    """Fetch the full realized volatility history.

    Returns list of [timestamp_ms, vol] pairs from Deribit; each is
    inserted/updated.  Returns number of rows inserted.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    data = client.get_historical_volatility(currency=currency)
    if not data:
        logger.info("No realized volatility data returned for %s", currency)
        return 0

    for pair in data:
        ts_ms, vol = pair[0], pair[1]
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        conn.execute(
            """INSERT OR REPLACE INTO realized_volatility
               (timestamp, currency, volatility)
               VALUES (?, ?, ?)""",
            [ts, currency, vol],
        )

    logger.info("Ingested %d realized vol records for %s", len(data), currency)
    return len(data)


def ingest_settlements(
    client: Optional[DeribitClient] = None,
    db_path=None,
    currency: str = "BTC",
    settlement_type: str = "delivery",
) -> int:
    """Paginate through all settlement/delivery events.

    Returns total number of settlement rows inserted.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    total = 0
    continuation = None

    while True:
        result = client.get_settlements(
            currency=currency,
            settlement_type=settlement_type,
            continuation=continuation,
        )
        settlements = result.get("settlements", [])
        if not settlements:
            break

        for s in settlements:
            name = s.get("instrument_name", "")
            ts_ms = s.get("timestamp", 0)
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            conn.execute(
                """INSERT OR REPLACE INTO options_settlements
                   (instrument_name, timestamp, settlement_type,
                    index_price, mark_price, delivery_price,
                    session_profit_loss, profit_loss)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    name,
                    ts,
                    s.get("type"),
                    s.get("index_price"),
                    s.get("mark_price"),
                    s.get("delivery"),
                    s.get("session_profit_loss"),
                    s.get("profit_loss"),
                ],
            )

            try:
                parsed = parse_instrument_name(name)
                _upsert_instrument(conn, name, parsed)
            except ValueError:
                pass

        total += len(settlements)
        continuation = result.get("continuation")
        if not continuation:
            break

    logger.info("Ingested %d %s settlement records for %s",
                total, settlement_type, currency)
    return total


def ingest_delivery_prices(
    client: Optional[DeribitClient] = None,
    db_path=None,
    index_name: str = "btc_usd",
) -> int:
    """Fetch all delivery (settlement) prices and store them."""
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    total = 0
    offset = 0

    while True:
        result = client.get_delivery_prices(
            index_name=index_name, count=1000, offset=offset,
        )
        data = result.get("data", [])
        if not data:
            break

        for row in data:
            d = row.get("date")
            if isinstance(d, str):
                delivery_date = date.fromisoformat(d)
            elif isinstance(d, (int, float)):
                delivery_date = datetime.fromtimestamp(
                    d / 1000, tz=timezone.utc
                ).date()
            else:
                delivery_date = d

            conn.execute(
                """INSERT OR REPLACE INTO delivery_prices
                   (delivery_date, index_name, delivery_price)
                   VALUES (?, ?, ?)""",
                [delivery_date, index_name, row.get("delivery_price")],
            )
        total += len(data)
        records_total = result.get("records_total", 0)
        offset += len(data)
        if offset >= records_total:
            break

    logger.info("Ingested %d delivery prices for %s", total, index_name)
    return total
