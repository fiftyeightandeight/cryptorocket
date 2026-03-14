from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from data.schema import get_connection


def get_available_symbols(
    interval: str = "1h",
    db_path: Optional[Path | str] = None,
) -> list[str]:
    """Return all symbols available in the database for the given interval."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM candles WHERE interval = ? ORDER BY symbol",
        [interval],
    ).fetchall()
    return [r[0] for r in rows]


def get_prices(
    symbols: list[str],
    interval: str = "1d",
    fields: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from DuckDB in Moonshot-compatible MultiIndex format.

    Returns a DataFrame with:
        Index: DatetimeIndex (Date)
        Columns: MultiIndex (Field, Symbol)

    Usage:
        prices = get_prices(['BTC', 'ETH'], '1d')
        closes = prices.loc['Close']          # DataFrame: Date × Symbol
        btc_close = prices.loc['Close']['BTC'] # Series indexed by Date
    """
    if fields is None:
        fields = ["Open", "High", "Low", "Close", "Volume"]

    conn = get_connection(db_path)

    # Build query
    placeholders = ", ".join(["?"] * len(symbols))
    query = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol IN ({placeholders})
          AND interval = ?
    """
    params: list = list(symbols) + [interval]

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)

    query += " ORDER BY timestamp, symbol"

    df = conn.execute(query, params).fetchdf()

    if df.empty:
        idx = pd.DatetimeIndex([], name="Date")
        cols = pd.MultiIndex.from_product([fields, symbols], names=["Field", "Symbol"])
        return pd.DataFrame(index=idx, columns=cols)

    # Rename columns to match Moonshot convention
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    # Pivot to MultiIndex columns: (Field, Symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index(["timestamp", "symbol"])

    # Stack fields into MultiIndex columns
    result = df.unstack("symbol")  # columns become (Field, Symbol)

    # Keep only requested fields
    field_map = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
    available_fields = [f for f in fields if f in field_map and field_map[f] in result.columns.get_level_values(0)]
    result = result[available_fields]

    result.index.name = "Date"
    result.columns.names = ["Field", "Symbol"]

    return result


def get_funding_rates(
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Fetch funding rate history from DuckDB.

    Returns DataFrame indexed by timestamp with columns per symbol,
    values are the 8-hour funding rate.
    """
    conn = get_connection(db_path)

    placeholders = ", ".join(["?"] * len(symbols))
    query = f"""
        SELECT symbol, timestamp, rate
        FROM funding_rates
        WHERE symbol IN ({placeholders})
    """
    params: list = list(symbols)

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)

    query += " ORDER BY timestamp, symbol"

    df = conn.execute(query, params).fetchdf()

    if df.empty:
        idx = pd.DatetimeIndex([], name="Date")
        return pd.DataFrame(index=idx, columns=symbols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    result = df.pivot(index="timestamp", columns="symbol", values="rate")
    result.index.name = "Date"
    return result
