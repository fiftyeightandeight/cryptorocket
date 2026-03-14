import logging
from typing import Optional

import pandas as pd

from strategies.base import CryptoMoonshot
from backtest.tearsheet import compute_metrics, print_tearsheet

logger = logging.getLogger(__name__)


def run_backtest(
    strategy: CryptoMoonshot,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    plot: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """Run a backtest for the given strategy and return results with metrics.

    Args:
        strategy: CryptoMoonshot subclass instance
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        plot: If True, generate equity curve plot
        output_path: Path to save plot (PNG or PDF)

    Returns:
        dict with backtest results and performance metrics
    """
    logger.info(f"Running backtest for {strategy.CODE}")
    logger.info(f"  Symbols: {strategy.SYMBOLS}")
    logger.info(f"  Interval: {strategy.INTERVAL}")
    logger.info(f"  Period: {start_date or 'earliest'} to {end_date or 'latest'}")

    # Run strategy pipeline
    results = strategy.backtest(start_date, end_date)

    # Compute performance metrics (use strategy's annualization factor)
    periods_per_year = results.get("periods_per_year", 365)
    metrics = compute_metrics(results["returns"], periods_per_year=periods_per_year)
    results["metrics"] = metrics

    # Print tearsheet
    print_tearsheet(strategy.CODE, metrics, results)

    # Optionally plot
    if plot or output_path:
        try:
            from backtest.tearsheet import plot_equity_curve
            plot_equity_curve(
                strategy.CODE,
                results["returns"],
                results.get("positions"),
                output_path=output_path,
            )
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")

    return results
