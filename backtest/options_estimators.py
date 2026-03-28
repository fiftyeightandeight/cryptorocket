"""Bid/ask spread and IV-premium estimators calibrated from live Deribit snapshots.

These estimators learn the *structure* of spreads and IV premiums from
accumulated snapshot data and produce point estimates suitable for augmenting
the Phase 1 (settlement-based) short-option backtest with realistic entry pricing.

Both estimators use a bucket-based approach:
  - |delta| buckets: [0, 0.10), [0.10, 0.25), [0.25, 0.40), [0.40, 0.50]
  - DTE buckets: [1, 7), [7, 14), [14, 30), [30, 90), [90, ∞)
  - DVOL buckets: [0, 40), [40, 60), [60, 80), [80, ∞)

For each bucket, the median of observed values is stored.  At estimation time,
the matching bucket is looked up; if empty, a global median fallback is used.

SpreadEstimator: spread expressed as fraction of mid IV, so it scales with
    the volatility regime.
IVPremiumEstimator: IV premium expressed as a ratio of realized vol
    (multiplicative), so the premium scales proportionally with RV.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.schema import get_connection

logger = logging.getLogger(__name__)

DELTA_EDGES = [0.0, 0.10, 0.25, 0.40, 0.50]
DTE_EDGES = [1, 7, 14, 30, 90, float("inf")]
DVOL_EDGES = [0, 40, 60, 80, float("inf")]


def _bucket_label(value: float, edges: list) -> str:
    """Return a human-readable label for the bucket containing *value*."""
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            hi = edges[i + 1]
            hi_str = "inf" if hi == float("inf") else str(hi)
            return f"{edges[i]}-{hi_str}"
    hi = edges[-1]
    hi_str = "inf" if hi == float("inf") else str(hi)
    return f"{edges[-2]}-{hi_str}"


def _bucket_key(delta_abs: float, dte: float, dvol: float) -> str:
    return (
        f"{_bucket_label(delta_abs, DELTA_EDGES)}"
        f"|{_bucket_label(dte, DTE_EDGES)}"
        f"|{_bucket_label(dvol, DVOL_EDGES)}"
    )


class SpreadEstimator:
    """Estimates bid/ask spread as a fraction of mid implied volatility.

    spread_iv_pct = (ask_iv - bid_iv) / mark_iv

    At estimation time, returns the IV-relative spread ratio.  To obtain
    the bid IV for a sold option: bid_iv = mark_iv * (1 - spread_iv_pct / 2).
    This causes the dollar spread to widen/tighten naturally with the IV regime.
    """

    def __init__(self):
        self._buckets: dict[str, float] = {}
        self._global_median: float = 0.0
        self._n_samples: int = 0

    def fit(self, db_path=None) -> "SpreadEstimator":
        """Calibrate from options_snapshots IV data + dvol."""
        conn = get_connection(db_path)

        df = conn.execute("""
            SELECT
                s.bid_iv,
                s.ask_iv,
                s.mark_iv,
                ABS(s.delta) AS delta_abs,
                s.dte,
                d.close AS dvol
            FROM options_snapshots s
            LEFT JOIN dvol d ON (
                d.currency = 'BTC'
                AND d.timestamp = (
                    SELECT MAX(d2.timestamp)
                    FROM dvol d2
                    WHERE d2.currency = 'BTC'
                      AND d2.timestamp <= s.snapshot_ts
                )
            )
            WHERE s.bid_iv > 0
              AND s.ask_iv > 0
              AND s.mark_iv > 0
        """).fetchdf()

        if df.empty:
            logger.warning("No IV data available for SpreadEstimator.fit()")
            return self

        df["spread_iv_pct"] = (df["ask_iv"] - df["bid_iv"]) / df["mark_iv"]
        df = df[df["spread_iv_pct"] > 0]

        if "dvol" not in df.columns or df["dvol"].isna().all():
            df["dvol"] = 50.0
        df["dvol"] = df["dvol"].fillna(50.0)

        df["key"] = df.apply(
            lambda r: _bucket_key(r["delta_abs"], r["dte"], r["dvol"]),
            axis=1,
        )

        medians = df.groupby("key")["spread_iv_pct"].median()
        self._buckets = medians.to_dict()
        self._global_median = float(df["spread_iv_pct"].median())
        self._n_samples = len(df)

        logger.info(
            "SpreadEstimator fitted: %d buckets from %d samples, "
            "global median=%.4f (IV-relative)",
            len(self._buckets), self._n_samples, self._global_median,
        )
        return self

    def estimate(
        self, delta: float, dte: float, dvol: float = 50.0
    ) -> float:
        """Return estimated spread as a fraction of mid IV."""
        key = _bucket_key(abs(delta), dte, dvol)
        return self._buckets.get(key, self._global_median)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "SpreadEstimator",
            "global_median": self._global_median,
            "n_samples": self._n_samples,
            "buckets": self._buckets,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SpreadEstimator":
        data = json.loads(Path(path).read_text())
        est = cls()
        est._buckets = data["buckets"]
        est._global_median = data["global_median"]
        est._n_samples = data.get("n_samples", 0)
        return est


class IVPremiumEstimator:
    """Estimates implied volatility as a multiple of realized volatility.

    Learns the typical ratio mark_iv / realized_vol, bucketed by delta,
    DTE, and DVOL regime.  At estimation time, returns
    sigma_implied = realized_vol * iv_ratio_bucket.

    This multiplicative form ensures the IV premium scales proportionally
    with RV rather than being a fixed additive constant.
    """

    def __init__(self):
        self._buckets: dict[str, float] = {}
        self._global_median: float = 1.0
        self._n_samples: int = 0

    def fit(self, db_path=None) -> "IVPremiumEstimator":
        """Calibrate from snapshots joined with realized vol and DVOL."""
        conn = get_connection(db_path)

        df = conn.execute("""
            SELECT
                s.mark_iv,
                ABS(s.delta) AS delta_abs,
                s.dte,
                d.close AS dvol,
                rv.volatility AS realized_vol
            FROM options_snapshots s
            LEFT JOIN dvol d ON (
                d.currency = 'BTC'
                AND d.timestamp = (
                    SELECT MAX(d2.timestamp)
                    FROM dvol d2
                    WHERE d2.currency = 'BTC'
                      AND d2.timestamp <= s.snapshot_ts
                )
            )
            LEFT JOIN realized_volatility rv ON (
                rv.currency = 'BTC'
                AND rv.timestamp = (
                    SELECT MAX(rv2.timestamp)
                    FROM realized_volatility rv2
                    WHERE rv2.currency = 'BTC'
                      AND rv2.timestamp <= s.snapshot_ts
                )
            )
            WHERE s.mark_iv IS NOT NULL
              AND s.mark_iv > 0
        """).fetchdf()

        if df.empty:
            logger.warning("No snapshot data available for IVPremiumEstimator.fit()")
            return self

        if "realized_vol" not in df.columns or df["realized_vol"].isna().all():
            logger.warning("No realized vol data; IV ratio defaults to 1.0")
            self._global_median = 1.0
            return self

        df["realized_vol"] = df["realized_vol"].fillna(0.0)
        df = df[df["realized_vol"] > 5.0]

        if df.empty:
            logger.warning("No rows with realized_vol > 5%%; IV ratio defaults to 1.0")
            return self

        if "dvol" not in df.columns or df["dvol"].isna().all():
            df["dvol"] = 50.0
        df["dvol"] = df["dvol"].fillna(50.0)

        df["iv_ratio"] = df["mark_iv"] / df["realized_vol"]

        df["key"] = df.apply(
            lambda r: _bucket_key(r["delta_abs"], r["dte"], r["dvol"]),
            axis=1,
        )

        medians = df.groupby("key")["iv_ratio"].median()
        self._buckets = medians.to_dict()
        self._global_median = float(df["iv_ratio"].median())
        self._n_samples = len(df)

        logger.info(
            "IVPremiumEstimator fitted: %d buckets from %d samples, "
            "global median ratio=%.2fx",
            len(self._buckets), self._n_samples, self._global_median,
        )
        return self

    def estimate(
        self,
        delta: float,
        dte: float,
        dvol: float = 50.0,
        realized_vol: float = 0.0,
    ) -> float:
        """Return estimated implied volatility (annualized %).

        sigma_implied = realized_vol * iv_ratio(bucket)
        Falls back to dvol if realized_vol is too small.
        """
        if realized_vol <= 5.0:
            return dvol
        key = _bucket_key(abs(delta), dte, dvol)
        ratio = self._buckets.get(key, self._global_median)
        return realized_vol * ratio

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "IVPremiumEstimator",
            "global_median": self._global_median,
            "n_samples": self._n_samples,
            "buckets": self._buckets,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "IVPremiumEstimator":
        data = json.loads(Path(path).read_text())
        est = cls()
        est._buckets = data["buckets"]
        est._global_median = data["global_median"]
        est._n_samples = data.get("n_samples", 0)
        return est
