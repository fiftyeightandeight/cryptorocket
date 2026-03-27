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
    max_pages: Optional[int] = None,
) -> int:
    """Fetch DVOL (volatility index) candles.

    When incremental=True, only fetches candles newer than the latest
    stored timestamp (suitable for hourly cron).  When False, fetches
    from start_timestamp (suitable for full backfill).  max_pages caps
    the number of API pages per invocation.
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
    pages = 0
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
        pages += 1

        if max_pages and pages >= max_pages:
            logger.info("DVOL reached page cap (%d), will resume next run", max_pages)
            break

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
    incremental: bool = False,
) -> int:
    """Fetch realized volatility history.

    The Deribit API returns a fixed sliding window (~20 days) with no
    pagination or date-range params.  When incremental=True, only new
    timestamps not already in the DB are written, so the table grows
    as the window slides forward across runs.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    existing_ms: set = set()
    if incremental:
        stored = conn.execute(
            "SELECT timestamp FROM realized_volatility WHERE currency = ?",
            [currency],
        ).fetchall()
        for r in stored:
            ts = r[0]
            existing_ms.add(int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else int(ts))

    data = client.get_historical_volatility(currency=currency)
    if not data:
        logger.info("No realized volatility data returned for %s", currency)
        return 0

    rows = []
    for pair in data:
        ts_ms, vol = pair[0], pair[1]
        if incremental and int(ts_ms) in existing_ms:
            continue
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rows.append([ts, currency, vol])

    if rows:
        conn.executemany(
            """INSERT OR REPLACE INTO realized_volatility
               (timestamp, currency, volatility)
               VALUES (?, ?, ?)""",
            rows,
        )

    logger.info("Ingested %d new realized vol records for %s", len(rows), currency)
    return len(rows)


def _get_backfill_state(conn, task_name: str) -> Optional[str]:
    """Load a saved continuation token for a backfill task."""
    row = conn.execute(
        "SELECT continuation FROM backfill_state WHERE task_name = ?",
        [task_name],
    ).fetchone()
    return row[0] if row else None


def _save_backfill_state(conn, task_name: str, continuation: Optional[str]) -> None:
    """Persist (or clear) the continuation token for a backfill task."""
    if continuation is None:
        conn.execute("DELETE FROM backfill_state WHERE task_name = ?", [task_name])
    else:
        conn.execute(
            """INSERT OR REPLACE INTO backfill_state (task_name, continuation, updated_at)
               VALUES (?, ?, current_timestamp)""",
            [task_name, str(continuation)],
        )


def ingest_settlements(
    client: Optional[DeribitClient] = None,
    db_path=None,
    currency: str = "BTC",
    settlement_type: str = "delivery",
    max_pages: Optional[int] = None,
    incremental: bool = False,
) -> int:
    """Paginate through settlement/delivery events.

    The API returns newest-first.  When incremental=True and we already
    have data, we use two strategies:

    1. If a saved continuation token exists, resume deep backfill from
       where the last run stopped.
    2. Otherwise, use search_start_timestamp set to our oldest stored
       record to skip the entire overlap zone in one shot, jumping
       straight to the backfill frontier.

    This avoids wasting pages re-scanning records we already have.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)
    task_key = f"settlements_{currency}_{settlement_type}"

    oldest_stored_ms: Optional[int] = None
    newest_stored_ms: Optional[int] = None
    if incremental:
        row = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM options_settlements"
        ).fetchone()
        if row and row[0] is not None:
            for i, attr in enumerate(["oldest", "newest"]):
                ts = row[i]
                ms = int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else int(ts)
                if attr == "oldest":
                    oldest_stored_ms = ms
                else:
                    newest_stored_ms = ms

    saved_continuation = _get_backfill_state(conn, task_key) if incremental else None

    search_start_ts: Optional[int] = None
    backfilling_old = False
    if saved_continuation:
        logger.info("Resuming deep backfill from saved cursor")
        backfilling_old = True
    elif oldest_stored_ms:
        search_start_ts = oldest_stored_ms
        backfilling_old = True
        logger.info(
            "No saved cursor — using search_start_timestamp=%d to skip overlap",
            oldest_stored_ms,
        )

    total = 0
    skipped = 0
    pages = 0
    continuation = saved_continuation
    backfill_done = False

    while True:
        kwargs: dict = {
            "currency": currency,
            "settlement_type": settlement_type,
            "continuation": continuation,
        }
        if search_start_ts and pages == 0 and not continuation:
            kwargs["search_start_timestamp"] = search_start_ts

        result = client.get_settlements(**kwargs)
        settlements = result.get("settlements", [])
        if not settlements:
            backfill_done = True
            break

        settle_rows = []
        instrument_rows = []
        for s in settlements:
            name = s.get("instrument_name", "")
            ts_ms = s.get("timestamp", 0)

            if backfilling_old:
                if incremental and oldest_stored_ms and ts_ms >= oldest_stored_ms:
                    skipped += 1
                    continue
            else:
                if incremental and newest_stored_ms and ts_ms <= newest_stored_ms:
                    skipped += 1
                    continue

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

        if settle_rows:
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

        total += len(settle_rows)
        pages += 1
        logger.info(
            "Settlements progress: %d new, %d skipped (%d pages)",
            total, skipped, pages,
        )

        continuation = result.get("continuation")
        if not continuation:
            backfill_done = True
            break

        if max_pages and pages >= max_pages:
            logger.info("Reached page cap (%d), saving cursor for next run", max_pages)
            break

    if backfill_done:
        _save_backfill_state(conn, task_key, None)
        logger.info("Settlements backfill complete — all pages fetched")
    else:
        _save_backfill_state(conn, task_key, continuation)

    logger.info("Ingested %d %s settlement records for %s (skipped %d existing)",
                total, settlement_type, currency, skipped)
    return total


def ingest_delivery_prices(
    client: Optional[DeribitClient] = None,
    db_path=None,
    index_name: str = "btc_usd",
    max_pages: Optional[int] = None,
    incremental: bool = False,
) -> int:
    """Fetch delivery (settlement) prices and store them.

    When incremental=True, skips dates already in the DB.
    max_pages caps the number of API pages per invocation.
    """
    client = client or DeribitClient()
    init_db(db_path)
    conn = get_connection(db_path)

    existing_dates: set = set()
    if incremental:
        rows = conn.execute(
            "SELECT delivery_date FROM delivery_prices WHERE index_name = ?",
            [index_name],
        ).fetchall()
        existing_dates = {r[0] for r in rows}

    total = 0
    pages = 0
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

            if incremental and delivery_date in existing_dates:
                continue
            rows.append([delivery_date, index_name, row.get("delivery_price")])

        if rows:
            conn.executemany(
                """INSERT OR REPLACE INTO delivery_prices
                   (delivery_date, index_name, delivery_price)
                   VALUES (?, ?, ?)""",
                rows,
            )
        total += len(rows)
        pages += 1
        records_total = result.get("records_total", 0)
        offset += len(data)

        if max_pages and pages >= max_pages:
            logger.info("Reached page cap (%d), will resume next run", max_pages)
            break
        if offset >= records_total:
            break

    logger.info("Ingested %d delivery prices for %s", total, index_name)
    return total
