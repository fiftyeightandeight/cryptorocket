#!/usr/bin/env python3
"""Deribit options data collection — designed for hourly cron / GitHub Actions.

Usage:
  python -m scripts.collect_deribit                        # hourly snapshot
  python -m scripts.collect_deribit --backfill-dvol        # one-time DVOL backfill
  python -m scripts.collect_deribit --backfill-rv          # one-time realized vol
  python -m scripts.collect_deribit --backfill-settlements # one-time settlements
  python -m scripts.collect_deribit --backfill-deliveries  # one-time delivery prices
  python -m scripts.collect_deribit --all                  # snapshot + all backfills
"""

import argparse
import logging

from data.deribit_client import DeribitClient
from data.deribit_ingest import (
    ingest_delivery_prices,
    ingest_dvol,
    ingest_options_snapshots,
    ingest_realized_volatility,
    ingest_settlements,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Deribit options data collection"
    )
    parser.add_argument(
        "--db-path", help="Custom database path"
    )
    parser.add_argument(
        "--backfill-dvol", action="store_true",
        help="Backfill DVOL candles (hourly resolution, full history)",
    )
    parser.add_argument(
        "--backfill-rv", action="store_true",
        help="Backfill realized volatility history",
    )
    parser.add_argument(
        "--backfill-settlements", action="store_true",
        help="Backfill all option settlement/delivery records",
    )
    parser.add_argument(
        "--backfill-deliveries", action="store_true",
        help="Backfill delivery prices by expiry date",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run snapshot + all backfills",
    )
    parser.add_argument(
        "--skip-snapshot", action="store_true",
        help="Skip the live options chain snapshot",
    )
    args = parser.parse_args()

    client = DeribitClient()
    db = args.db_path

    run_all = args.all
    is_backfill = (
        args.backfill_dvol or args.backfill_rv
        or args.backfill_settlements or args.backfill_deliveries
        or run_all
    )

    if not args.skip_snapshot or run_all:
        n = ingest_options_snapshots(client, db_path=db)
        logger.info("Snapshot complete: %d instruments", n)

    if args.backfill_dvol or run_all:
        n = ingest_dvol(client, db_path=db)
        logger.info("DVOL backfill complete: %d candles", n)
    elif not is_backfill:
        n = ingest_dvol(client, db_path=db, incremental=True)
        logger.info("DVOL incremental update: %d candles", n)

    if args.backfill_rv or run_all:
        n = ingest_realized_volatility(client, db_path=db)
        logger.info("Realized vol backfill complete: %d records", n)
    elif not is_backfill:
        n = ingest_realized_volatility(client, db_path=db)
        logger.info("Realized vol update: %d records", n)

    if args.backfill_settlements or run_all:
        n = ingest_settlements(client, db_path=db)
        logger.info("Settlements backfill complete: %d records", n)

    if args.backfill_deliveries or run_all:
        n = ingest_delivery_prices(client, db_path=db)
        logger.info("Delivery prices backfill complete: %d records", n)

    logger.info("Deribit collection done.")


if __name__ == "__main__":
    main()
