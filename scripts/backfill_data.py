#!/usr/bin/env python3
"""One-time historical data backfill for all Hyperliquid perp assets."""

import argparse
import logging
from datetime import datetime, timedelta, timezone

from data.client import HyperliquidClient
from data.ingest import ingest_candles, ingest_funding, ingest_universe
from config import DEFAULT_LOOKBACK_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill Hyperliquid candle data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to backfill (default: all perps)")
    parser.add_argument("--intervals", nargs="+", default=["1h", "1d"], help="Candle intervals")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Lookback days")
    parser.add_argument("--funding", action="store_true", help="Also backfill funding rates")
    parser.add_argument("--db-path", help="Custom database path")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    args = parser.parse_args()

    if args.testnet:
        from hyperliquid.utils.constants import TESTNET_API_URL
        client = HyperliquidClient(base_url=TESTNET_API_URL)
    else:
        client = HyperliquidClient()

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    # Ingest universe first
    logger.info("Ingesting universe metadata...")
    ingest_universe(client, db_path=args.db_path)

    # Determine symbols
    symbols = args.symbols
    if not symbols:
        perps = client.get_perp_universe()
        symbols = [p["name"] for p in perps]
        logger.info(f"Backfilling {len(symbols)} perp symbols")

    # Ingest candles
    logger.info(f"Backfilling candles from {start_date.date()} to {end_date.date()}")
    ingest_candles(
        symbols=symbols,
        intervals=args.intervals,
        start_date=start_date,
        end_date=end_date,
        client=client,
        db_path=args.db_path,
    )

    # Optionally ingest funding
    if args.funding:
        logger.info("Backfilling funding rates...")
        ingest_funding(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            client=client,
            db_path=args.db_path,
        )

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
