#!/usr/bin/env python3
"""Incremental data collection — designed to run on cron.

Example crontab entries:
  # Collect 1h candles every hour
  0 * * * * cd /home/user/quantrocket && python -m crypto.scripts.collect_data --intervals 1h

  # Collect 1d candles once daily at 00:05 UTC
  5 0 * * * cd /home/user/quantrocket && python -m crypto.scripts.collect_data --intervals 1d
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone

from data.client import HyperliquidClient
from data.ingest import ingest_candles, ingest_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Incremental candle collection")
    parser.add_argument("--symbols", nargs="+", help="Symbols (default: all perps)")
    parser.add_argument("--intervals", nargs="+", default=["1h", "1d"])
    parser.add_argument("--db-path", help="Custom database path")
    args = parser.parse_args()

    client = HyperliquidClient()

    # Refresh universe
    ingest_universe(client, db_path=args.db_path)

    # Determine symbols
    symbols = args.symbols
    if not symbols:
        perps = client.get_perp_universe()
        symbols = [p["name"] for p in perps]

    # Incremental ingest — ingest_candles auto-detects last stored timestamp
    end_date = datetime.now(timezone.utc)
    # Start from 3 days ago as fallback if no data exists yet
    start_date = end_date - timedelta(days=3)

    ingest_candles(
        symbols=symbols,
        intervals=args.intervals,
        start_date=start_date,
        end_date=end_date,
        client=client,
        db_path=args.db_path,
    )

    logger.info("Collection complete.")


if __name__ == "__main__":
    main()
