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
from data.schema import close_connection

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
        "--dvol-pages", type=int, default=None,
        help="Max API pages per DVOL backfill run (default: unlimited)",
    )
    parser.add_argument(
        "--backfill-rv", action="store_true",
        help="Backfill realized volatility history",
    )
    parser.add_argument(
        "--backfill-settlements", action="store_true",
        help="Backfill option settlement/delivery records (incremental)",
    )
    parser.add_argument(
        "--settlement-pages", type=int, default=50,
        help="Max API pages per settlements backfill run (default: 50)",
    )
    parser.add_argument(
        "--backfill-deliveries", action="store_true",
        help="Backfill delivery prices by expiry date (incremental)",
    )
    parser.add_argument(
        "--delivery-pages", type=int, default=50,
        help="Max API pages per delivery prices backfill run (default: 50)",
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

    if not args.skip_snapshot:
        n = ingest_options_snapshots(client, db_path=db)
        logger.info("Snapshot complete: %d instruments", n)

    if args.backfill_dvol or run_all:
        n = ingest_dvol(client, db_path=db, max_pages=args.dvol_pages)
        logger.info("DVOL backfill complete: %d candles", n)
    elif not is_backfill:
        n = ingest_dvol(client, db_path=db, incremental=True)
        logger.info("DVOL incremental update: %d candles", n)

    if args.backfill_rv or run_all:
        n = ingest_realized_volatility(client, db_path=db, incremental=True)
        logger.info("Realized vol backfill complete: %d records", n)
    elif not is_backfill:
        n = ingest_realized_volatility(client, db_path=db, incremental=True)
        logger.info("Realized vol update: %d records", n)

    if args.backfill_settlements or run_all:
        n = ingest_settlements(
            client, db_path=db,
            max_pages=args.settlement_pages,
            incremental=True,
        )
        logger.info("Settlements backfill complete: %d records", n)

    if args.backfill_deliveries or run_all:
        n = ingest_delivery_prices(
            client, db_path=db,
            max_pages=args.delivery_pages,
            incremental=True,
        )
        logger.info("Delivery prices backfill complete: %d records", n)

    close_connection(db)
    logger.info("Deribit collection done.")


if __name__ == "__main__":
    main()
