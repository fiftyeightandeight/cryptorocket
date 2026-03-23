import threading

import duckdb
from pathlib import Path

from config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS universe (
    symbol      VARCHAR NOT NULL,
    market_type VARCHAR NOT NULL,     -- 'perp' or 'spot'
    sz_decimals INTEGER,
    max_leverage INTEGER,
    mark_px     DOUBLE,
    day_ntl_vlm DOUBLE,
    updated_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, market_type)
);

CREATE TABLE IF NOT EXISTS candles (
    symbol    VARCHAR NOT NULL,
    interval  VARCHAR NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open      DOUBLE,
    high      DOUBLE,
    low       DOUBLE,
    close     DOUBLE,
    volume    DOUBLE,
    PRIMARY KEY (symbol, interval, timestamp)
);

CREATE TABLE IF NOT EXISTS funding_rates (
    symbol    VARCHAR NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    rate      DOUBLE,
    premium   DOUBLE,
    PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    VARCHAR PRIMARY KEY,
    strategy    VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    side        VARCHAR NOT NULL,
    size        DOUBLE NOT NULL,
    order_type  VARCHAR NOT NULL,
    limit_price DOUBLE,
    status      VARCHAR NOT NULL DEFAULT 'pending',
    fill_price  DOUBLE,
    fill_size   DOUBLE,
    fee         DOUBLE,
    created_at  TIMESTAMP DEFAULT current_timestamp,
    filled_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    strategy    VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    size        DOUBLE NOT NULL,
    entry_price DOUBLE,
    unrealized_pnl DOUBLE,
    updated_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (strategy, symbol)
);

-- Deribit options ----------------------------------------------------------

CREATE TABLE IF NOT EXISTS options_instruments (
    instrument_name VARCHAR PRIMARY KEY,
    underlying      VARCHAR,
    expiry_date     DATE,
    strike          DOUBLE,
    option_type     VARCHAR       -- 'P' or 'C'
);

CREATE TABLE IF NOT EXISTS options_snapshots (
    snapshot_ts     TIMESTAMP WITH TIME ZONE NOT NULL,
    instrument_name VARCHAR NOT NULL,
    bid_price       DOUBLE,
    ask_price       DOUBLE,
    mark_price      DOUBLE,
    bid_iv          DOUBLE,
    ask_iv          DOUBLE,
    mark_iv         DOUBLE,
    underlying_price DOUBLE,
    delta           DOUBLE,
    gamma           DOUBLE,
    theta           DOUBLE,
    vega            DOUBLE,
    rho             DOUBLE,
    open_interest   DOUBLE,
    strike          DOUBLE,
    expiry_date     DATE,
    option_type     VARCHAR,
    dte             INTEGER,
    PRIMARY KEY (snapshot_ts, instrument_name)
);

CREATE TABLE IF NOT EXISTS dvol (
    timestamp   TIMESTAMP WITH TIME ZONE NOT NULL,
    currency    VARCHAR NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    PRIMARY KEY (timestamp, currency)
);

CREATE TABLE IF NOT EXISTS realized_volatility (
    timestamp   TIMESTAMP WITH TIME ZONE NOT NULL,
    currency    VARCHAR NOT NULL,
    volatility  DOUBLE,
    PRIMARY KEY (timestamp, currency)
);

CREATE TABLE IF NOT EXISTS options_settlements (
    instrument_name VARCHAR NOT NULL,
    timestamp       TIMESTAMP WITH TIME ZONE NOT NULL,
    settlement_type VARCHAR,
    index_price     DOUBLE,
    mark_price      DOUBLE,
    delivery_price  DOUBLE,
    session_profit_loss DOUBLE,
    profit_loss     DOUBLE,
    PRIMARY KEY (instrument_name, timestamp)
);

CREATE TABLE IF NOT EXISTS delivery_prices (
    delivery_date DATE PRIMARY KEY,
    index_name    VARCHAR,
    delivery_price DOUBLE
);
"""

_pool: dict[str, duckdb.DuckDBPyConnection] = {}
_pool_lock = threading.Lock()


def get_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection, reusing one per db_path within the same thread.

    Callers that previously called conn.close() still work: a closed connection
    is detected and replaced. For short-lived scripts this behaves identically
    to the old per-call approach; for the live loop it avoids repeated open/close.
    """
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    key = f"{threading.current_thread().ident}:{path}"

    with _pool_lock:
        conn = _pool.get(key)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                _pool.pop(key, None)

        conn = duckdb.connect(str(path))
        _pool[key] = conn
        return conn


def close_connection(db_path: Path | str | None = None) -> None:
    """Explicitly close and remove a pooled connection."""
    path = Path(db_path) if db_path else DB_PATH
    key = f"{threading.current_thread().ident}:{path}"

    with _pool_lock:
        conn = _pool.pop(key, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def init_db(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Initialize the database with all required tables."""
    conn = get_connection(db_path)
    conn.execute(SCHEMA_SQL)
    return conn
