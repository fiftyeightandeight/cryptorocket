"""Deribit data ingestion — options snapshots, DVOL, realized vol, settlements.

Mirrors the pattern established in data/ingest.py for Hyperliquid data.
All DB writes use batch inserts (executemany) for performance.
"""

import logging
import math
from datetime import datetime, date, timezone
from typing import Optional

from data.deribit_client import DeribitClient, parse_instrument_name
from data.schema import get_connection, init_db

logger = logging.getLogger(__name__)


def _bs_delta(S: float, K: float, T: float, sigma: float, option_type: str) -> float:
    """Black-Scholes delta (r=0, no dividends).

    sigma is annualized vol as a fraction (e.g. 0.55 for 55%).
    Returns delta in [-1, 1].
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    if option_type == "C":
        return nd1
    return nd1 - 1.0


def ingest_options_snapshots(
    client: Optional[DeribitClient] = None,
    db_path=None,
    currency: str = "BTC",
    detailed: bool = False,
) -> int:
    """Snapshot the full live options chain from book_summary.

    By default (detailed=False), uses only get_book_summary (1 API call)
    and computes delta from Black-Scholes.  With detailed=True, also
    calls get_ticker per instrument to get bid_iv, ask_iv, and full Greeks.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    summaries = client.get_book_summary(currency=currency, kind="option")
    now_ts = datetime.now(timezone.utc)

    snapshot_rows = []
    instrument_rows = []

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

        mark_iv = s.get("mark_iv")
        underlying_price = s.get("underlying_price")
        bid_iv = ask_iv = None
        delta = gamma = theta = vega = rho = None

        if mark_iv and underlying_price and dte > 0:
            sigma = mark_iv / 100.0
            T = dte / 365.0
            delta = _bs_delta(underlying_price, parsed["strike"], T, sigma, parsed["option_type"])

        if detailed:
            try:
                ticker = client.get_ticker(name)
                bid_iv = ticker.get("bid_iv")
                ask_iv = ticker.get("ask_iv")
                greeks = ticker.get("greeks", {}) or {}
                delta = greeks.get("delta", delta)
                gamma = greeks.get("gamma")
                theta = greeks.get("theta")
                vega = greeks.get("vega")
                rho = greeks.get("rho")
            except Exception as exc:
                logger.warning("Ticker failed for %s: %s", name, exc)

        snapshot_rows.append([
            now_ts, name, bid, ask,
            s.get("mark_price"), bid_iv, ask_iv, mark_iv, underlying_price,
            delta, gamma, theta, vega, rho,
            s.get("open_interest"),
            parsed["strike"], expiry, parsed["option_type"], dte,
        ])
        instrument_rows.append([
            name, parsed["underlying"], expiry,
            parsed["strike"], parsed["option_type"],
        ])

    if snapshot_rows:
        conn.executemany(
            """INSERT OR REPLACE INTO options_snapshots
               (snapshot_ts, instrument_name, bid_price, ask_price,
                mark_price, bid_iv, ask_iv, mark_iv, underlying_price,
                delta, gamma, theta, vega, rho, open_interest,
                strike, expiry_date, option_type, dte)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            snapshot_rows,
        )
        conn.executemany(
            """INSERT OR REPLACE INTO options_instruments
               (instrument_name, underlying, expiry_date, strike, option_type)
               VALUES (?, ?, ?, ?, ?)""",
            instrument_rows,
        )

    logger.info("Ingested %d options snapshots for %s", len(snapshot_rows), currency)
    return len(snapshot_rows)


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

        rows = []
        for c in candles:
            ts_ms, o, h, l, close = c[0], c[1], c[2], c[3], c[4]
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            rows.append([ts, currency, o, h, l, close])

        conn.executemany(
            """INSERT OR REPLACE INTO dvol
               (timestamp, currency, open, high, low, close)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
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
    """Fetch the full realized volatility history."""
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    data = client.get_historical_volatility(currency=currency)
    if not data:
        logger.info("No realized volatility data returned for %s", currency)
        return 0

    rows = []
    for pair in data:
        ts_ms, vol = pair[0], pair[1]
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rows.append([ts, currency, vol])

    conn.executemany(
        """INSERT OR REPLACE INTO realized_volatility
           (timestamp, currency, volatility)
           VALUES (?, ?, ?)""",
        rows,
    )

    logger.info("Ingested %d realized vol records for %s", len(data), currency)
    return len(data)


def ingest_settlements(
    client: Optional[DeribitClient] = None,
    db_path=None,
    currency: str = "BTC",
    settlement_type: str = "delivery",
) -> int:
    """Paginate through all settlement/delivery events."""
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

        settle_rows = []
        instrument_rows = []
        for s in settlements:
            name = s.get("instrument_name", "")
            ts_ms = s.get("timestamp", 0)
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            settle_rows.append([
                name, ts, s.get("type"),
                s.get("index_price"), s.get("mark_price"), s.get("delivery"),
                s.get("session_profit_loss"), s.get("profit_loss"),
            ])
            try:
                parsed = parse_instrument_name(name)
                instrument_rows.append([
                    name, parsed["underlying"], parsed["expiry_date"],
                    parsed["strike"], parsed["option_type"],
                ])
            except ValueError:
                pass

        conn.executemany(
            """INSERT OR REPLACE INTO options_settlements
               (instrument_name, timestamp, settlement_type,
                index_price, mark_price, delivery_price,
                session_profit_loss, profit_loss)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            settle_rows,
        )
        if instrument_rows:
            conn.executemany(
                """INSERT OR REPLACE INTO options_instruments
                   (instrument_name, underlying, expiry_date, strike, option_type)
                   VALUES (?, ?, ?, ?, ?)""",
                instrument_rows,
            )

        total += len(settlements)
        logger.info("Settlements progress: %d records so far", total)
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

        rows = []
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
            rows.append([delivery_date, index_name, row.get("delivery_price")])

        conn.executemany(
            """INSERT OR REPLACE INTO delivery_prices
               (delivery_date, index_name, delivery_price)
               VALUES (?, ?, ?)""",
            rows,
        )
        total += len(data)
        records_total = result.get("records_total", 0)
        offset += len(data)
        if offset >= records_total:
            break

    logger.info("Ingested %d delivery prices for %s", total, index_name)
    return total
