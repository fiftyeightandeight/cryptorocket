#!/usr/bin/env python3
"""Data collection script optimized for GitHub Actions.

Differences from collect_data.py:
- No wallet/key required (read-only Info API)
- Handles first-run gracefully with configurable backfill
- Outputs structured logs for GH Actions
- Exits non-zero on failure for workflow visibility
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from data.client import HyperliquidClient
from data.ingest import ingest_candles, ingest_funding, ingest_universe
from data.schema import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GitHub Actions data collector")
    parser.add_argument("--symbols", help="Comma-separated symbols (default: top perps by volume)")
    parser.add_argument("--intervals", default="1h,1d", help="Comma-separated intervals")
    parser.add_argument("--backfill-days", type=int, default=7, help="Days to backfill on first run")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top symbols by volume")
    args = parser.parse_args()

    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    client = HyperliquidClient()

    # Initialize DB
    logger.info("Initializing database...")
    init_db()

    # Refresh universe
    logger.info("Fetching universe metadata...")
    ingest_universe(client)

    # Determine symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        # Top N perps by 24h volume
        perps = client.get_perp_universe()
        perps_with_vol = [
            p for p in perps
            if p.get("dayNtlVlm") and float(p["dayNtlVlm"]) > 0
        ]
        perps_with_vol.sort(key=lambda p: float(p["dayNtlVlm"]), reverse=True)
        symbols = [p["name"] for p in perps_with_vol[:args.top_n]]
        logger.info(f"Selected top {len(symbols)} perps by volume: {symbols[:10]}...")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.backfill_days)

    logger.info(f"Collecting {len(symbols)} symbols, intervals={intervals}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    errors = []
    for symbol in symbols:
        try:
            ingest_candles(
                symbols=[symbol],
                intervals=intervals,
                start_date=start_date,
                end_date=end_date,
                client=client,
            )
        except Exception as e:
            logger.error(f"Failed to collect {symbol}: {e}")
            errors.append(symbol)

    if errors:
        logger.warning(f"Failed symbols: {errors}")
        # Don't fail the workflow for partial errors
        if len(errors) > len(symbols) * 0.5:
            logger.error("More than 50% of symbols failed, marking as error")
            sys.exit(1)

    logger.info(f"Collection complete. {len(symbols) - len(errors)}/{len(symbols)} symbols succeeded.")

    # Collect funding rates for carry strategy support
    logger.info("Collecting funding rate data...")
    funding_errors = []
    for symbol in symbols:
        try:
            ingest_funding(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                client=client,
            )
        except Exception as e:
            logger.error(f"Failed to collect funding for {symbol}: {e}")
            funding_errors.append(symbol)

    logger.info(
        f"Funding collection complete. "
        f"{len(symbols) - len(funding_errors)}/{len(symbols)} symbols succeeded."
    )


if __name__ == "__main__":
    main()
