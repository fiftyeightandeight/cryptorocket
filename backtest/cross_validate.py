"""Cross-validation harness: CryptoMoonshot vs upstream Moonshot defaults.

Runs a strategy through both engines on identical price data and compares
intermediate outputs at each pipeline stage.  Shared stages (signals,
target weights) use the strategy's own methods.  The fork happens at
positions / returns where the two frameworks' defaults diverge.

Key behavioral differences tested:
  - Position entry timing:  ours = instant fill,  Moonshot = shift(1)
  - Slippage basis:         ours = diff().fillna(0),  Moonshot = fillna(0).diff()
  - Commission inputs:      both use the strategy's commission model, but
                            position-change vectors differ due to the shift
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import CryptoMoonshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StageComparison:
    """Comparison metrics for one pipeline stage."""

    name: str
    match: bool
    max_abs_diff: float
    mean_abs_diff: float
    correlation: float
    ours_mean: float = 0.0
    ours_std: float = 0.0
    moonshot_mean: float = 0.0
    moonshot_std: float = 0.0


@dataclass
class CrossValidationReport:
    """Full cross-validation results across all stages."""

    strategy_code: str
    stages: list[StageComparison]
    our_returns: pd.Series
    moonshot_returns: pd.Series
    our_metrics: dict = field(default_factory=dict)
    moonshot_metrics: dict = field(default_factory=dict)

    @property
    def all_match(self) -> bool:
        return all(s.match for s in self.stages)


# ---------------------------------------------------------------------------
# Moonshot reference implementations (upstream defaults)
# ---------------------------------------------------------------------------

def _moonshot_positions(weights: pd.DataFrame) -> pd.DataFrame:
    """Upstream Moonshot default: enter position the period after allocation."""
    return weights.shift()


def _moonshot_gross_returns(
    positions: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Upstream Moonshot default: close-to-close pct_change * lagged position."""
    closes = prices["Close"]
    return closes.pct_change() * positions.shift()


def _moonshot_turnover(positions: pd.DataFrame) -> pd.DataFrame:
    """Upstream Moonshot turnover: fillna *before* diff."""
    return positions.fillna(0).diff().abs()


def _moonshot_slippage(
    positions: pd.DataFrame, slippage_bps: float
) -> pd.DataFrame:
    """Upstream Moonshot fixed slippage: turnover * one-way bps."""
    return _moonshot_turnover(positions) * (slippage_bps / 10_000)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_frames(
    name: str,
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    atol: float,
) -> StageComparison:
    """Element-wise comparison of two identically-shaped DataFrames."""
    common_idx = ours.index.intersection(theirs.index)
    common_cols = ours.columns.intersection(theirs.columns)

    a = ours.loc[common_idx, common_cols].astype(float)
    b = theirs.loc[common_idx, common_cols].astype(float)

    diff = (a - b).fillna(0)
    abs_diff = diff.abs()

    max_abs = float(abs_diff.max().max()) if not abs_diff.empty else 0.0
    mean_abs = float(abs_diff.mean().mean()) if not abs_diff.empty else 0.0

    a_flat = a.values.flatten()
    b_flat = b.values.flatten()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() > 1 and np.std(a_flat[mask]) > 0 and np.std(b_flat[mask]) > 0:
        corr = float(np.corrcoef(a_flat[mask], b_flat[mask])[0, 1])
    else:
        corr = float("nan")

    return StageComparison(
        name=name,
        match=max_abs <= atol,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        correlation=corr,
        ours_mean=float(np.nanmean(a.values)),
        ours_std=float(np.nanstd(a.values)),
        moonshot_mean=float(np.nanmean(b.values)),
        moonshot_std=float(np.nanstd(b.values)),
    )


def _compare_series(
    name: str,
    ours: pd.Series,
    theirs: pd.Series,
    atol: float,
) -> StageComparison:
    """Element-wise comparison of two identically-indexed Series."""
    common_idx = ours.index.intersection(theirs.index)
    a = ours.loc[common_idx].astype(float)
    b = theirs.loc[common_idx].astype(float)

    diff = (a - b).fillna(0)
    abs_diff = diff.abs()

    max_abs = float(abs_diff.max()) if not abs_diff.empty else 0.0
    mean_abs = float(abs_diff.mean()) if not abs_diff.empty else 0.0

    mask = np.isfinite(a.values) & np.isfinite(b.values)
    if mask.sum() > 1 and np.std(a.values[mask]) > 0 and np.std(b.values[mask]) > 0:
        corr = float(np.corrcoef(a.values[mask], b.values[mask])[0, 1])
    else:
        corr = float("nan")

    return StageComparison(
        name=name,
        match=max_abs <= atol,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        correlation=corr,
        ours_mean=float(np.nanmean(a.values)),
        ours_std=float(np.nanstd(a.values)),
        moonshot_mean=float(np.nanmean(b.values)),
        moonshot_std=float(np.nanstd(b.values)),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def cross_validate(
    strategy: CryptoMoonshot,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    prices: Optional[pd.DataFrame] = None,
    atol: float = 1e-10,
) -> CrossValidationReport:
    """Run *strategy* through both engines on the same data and compare.

    Parameters
    ----------
    strategy : CryptoMoonshot
        Instantiated strategy.
    start_date, end_date : str, optional
        Date range (ignored when *prices* is supplied).
    prices : DataFrame, optional
        Pre-built MultiIndex (Field, Symbol) price DataFrame.  When given
        the harness skips the database fetch, which is useful for tests.
    atol : float
        Absolute tolerance for declaring two stages "matching".

    Returns
    -------
    CrossValidationReport
    """
    if prices is None:
        prices = strategy._get_prices(start_date, end_date)
    if prices.empty:
        raise ValueError(f"No price data for {strategy.CODE}")

    logger.info("Cross-validating %s  (%d bars)", strategy.CODE, len(prices))

    # ------------------------------------------------------------------
    # Shared stages (identical code path)
    # ------------------------------------------------------------------
    signals = strategy.prices_to_signals(prices)
    weights = strategy.signals_to_target_weights(signals, prices)

    # ------------------------------------------------------------------
    # Our pipeline
    # ------------------------------------------------------------------
    our_positions = strategy.target_weights_to_positions(weights, prices)
    our_gross = strategy.positions_to_gross_returns(our_positions, prices)

    our_pos_changes = our_positions.fillna(0).diff().fillna(0)
    our_slippage = our_pos_changes.abs() * (strategy.SLIPPAGE_BPS / 10_000)

    commission_model = strategy.COMMISSION_CLASS()
    closes = prices["Close"]
    our_commissions = commission_model.get_commissions(closes, our_pos_changes)

    our_net = our_gross - our_slippage - our_commissions
    our_portfolio = our_net.sum(axis=1)

    # ------------------------------------------------------------------
    # Moonshot reference pipeline
    # ------------------------------------------------------------------
    ms_positions = _moonshot_positions(weights)
    ms_gross = _moonshot_gross_returns(ms_positions, prices)

    ms_slippage = _moonshot_slippage(ms_positions, strategy.SLIPPAGE_BPS)

    ms_pos_changes = ms_positions.fillna(0).diff().fillna(0)
    ms_commissions = commission_model.get_commissions(closes, ms_pos_changes)

    ms_net = ms_gross.fillna(0) - ms_slippage - ms_commissions
    ms_portfolio = ms_net.sum(axis=1)

    # ------------------------------------------------------------------
    # Stage-by-stage comparison
    # ------------------------------------------------------------------
    stages = [
        _compare_frames("signals", signals, signals, atol),
        _compare_frames("target_weights", weights, weights, atol),
        _compare_frames("positions", our_positions, ms_positions.fillna(0), atol),
        _compare_frames("gross_returns", our_gross, ms_gross.fillna(0), atol),
        _compare_frames("slippage", our_slippage, ms_slippage, atol),
        _compare_frames("commissions", our_commissions, ms_commissions, atol),
        _compare_frames("net_returns", our_net, ms_net, atol),
        _compare_series("portfolio_returns", our_portfolio, ms_portfolio, atol),
    ]

    # ------------------------------------------------------------------
    # Compute tearsheet metrics for both
    # ------------------------------------------------------------------
    from backtest.tearsheet import compute_metrics

    periods_per_year = strategy._periods_per_year
    our_m = compute_metrics(our_portfolio, periods_per_year=periods_per_year)
    ms_m = compute_metrics(ms_portfolio, periods_per_year=periods_per_year)

    return CrossValidationReport(
        strategy_code=strategy.CODE,
        stages=stages,
        our_returns=our_portfolio,
        moonshot_returns=ms_portfolio,
        our_metrics=our_m,
        moonshot_metrics=ms_m,
    )


# ---------------------------------------------------------------------------
# Pretty-printed report
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    ("total_return", "%"),
    ("cagr", "%"),
    ("ann_volatility", "%"),
    ("sharpe", "f"),
    ("sortino", "f"),
    ("max_drawdown", "%"),
    ("calmar", "f"),
]


def print_cross_validation_report(report: CrossValidationReport) -> None:
    """Print a formatted comparison of both engines to stdout."""
    w = 72
    print(f"\n{'=' * w}")
    print(f"  Cross-Validation: {report.strategy_code}")
    print(f"{'=' * w}")

    # ---- stage table ----
    print()
    hdr = f"  {'Stage':<22} {'Match':>7} {'Max|Δ|':>12} {'Mean|Δ|':>12} {'Corr':>8}"
    print(hdr)
    print(f"  {'-' * 22} {'-' * 7} {'-' * 12} {'-' * 12} {'-' * 8}")

    for s in report.stages:
        tag = "OK" if s.match else "DIFF"
        corr = f"{s.correlation:.6f}" if np.isfinite(s.correlation) else "n/a"
        print(
            f"  {s.name:<22} {tag:>7} "
            f"{s.max_abs_diff:>12.2e} {s.mean_abs_diff:>12.2e} {corr:>8}"
        )

    # ---- metrics table ----
    om, mm = report.our_metrics, report.moonshot_metrics
    if om and mm:
        print()
        print(f"  {'Metric':<22} {'Ours':>12} {'Moonshot':>12} {'Delta':>12}")
        print(f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 12}")

        for key, fmt in _METRIC_KEYS:
            ov = om.get(key, 0)
            mv = mm.get(key, 0)
            d = ov - mv
            if fmt == "%":
                print(f"  {key:<22} {ov:>11.2%} {mv:>11.2%} {d:>+11.2%}")
            else:
                print(f"  {key:<22} {ov:>12.2f} {mv:>12.2f} {d:>+12.2f}")

    # ---- explanation of known differences ----
    pos_stage = next((s for s in report.stages if s.name == "positions"), None)
    if pos_stage and not pos_stage.match:
        print()
        print("  Known divergence: position entry timing")
        print("    Ours     : positions = target_weights        (instant fill)")
        print("    Moonshot : positions = target_weights.shift() (next-bar entry)")
        print("    This propagates into gross_returns, costs, and net_returns.")

    print(f"\n{'=' * w}\n")
