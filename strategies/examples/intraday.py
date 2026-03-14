import numpy as np
import pandas as pd

from strategies.base import CryptoMoonshot
from backtest.commission import HyperliquidPerpsCommission


class IntradayMomentumUS(CryptoMoonshot):
    """Intraday momentum during US trading hours.

    Uses hourly bars filtered to the US session (14:00-22:00 UTC).
    Longs coins with positive session momentum, shorts those with
    negative. Positions are flattened at session end (no overnight).

    Similar to QuantRocket's intraday-ls-us pattern but adapted for
    24/7 crypto markets using a synthetic US session window.
    """

    CODE = "intraday-momentum-us"
    SYMBOLS = []  # empty = all available symbols in DB
    INTERVAL = "1h"
    SESSION_START_HOUR = 14  # 14:00 UTC ≈ NYSE open
    SESSION_END_HOUR = 22   # 22:00 UTC ≈ after NYSE close
    LOOKBACK = 3            # Hours of momentum lookback within session
    LEVERAGE = 1.0
    SLIPPAGE_BPS = 3
    COMMISSION_CLASS = HyperliquidPerpsCommission

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        closes = prices.loc["Close"]

        # Trailing return over LOOKBACK bars
        ret = closes.pct_change(self.LOOKBACK)

        # Long positive momentum, short negative
        signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        signals[ret > 0] = 1.0
        signals[ret < 0] = -1.0

        # Suppress signals in the first LOOKBACK bars of each session day
        # (not enough intraday history yet)
        dates = signals.index.date
        for date in np.unique(dates):
            day_mask = dates == date
            day_idx = signals.index[day_mask]
            if len(day_idx) > self.LOOKBACK:
                signals.loc[day_idx[: self.LOOKBACK]] = 0.0
            else:
                signals.loc[day_idx] = 0.0

        return signals


class IntradayMeanReversionUS(CryptoMoonshot):
    """Intraday mean reversion during US trading hours.

    Buys intraday dips and sells intraday rips within the US session.
    Uses the session VWAP as the fair value anchor.
    Positions flattened at session end.
    """

    CODE = "intraday-mr-us"
    SYMBOLS = []  # empty = all available symbols in DB
    INTERVAL = "1h"
    SESSION_START_HOUR = 14
    SESSION_END_HOUR = 22
    Z_ENTRY = 1.0     # Std devs from session mean to enter
    LEVERAGE = 0.5
    SLIPPAGE_BPS = 3
    COMMISSION_CLASS = HyperliquidPerpsCommission

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        closes = prices.loc["Close"]

        signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

        # Compute intraday z-score relative to expanding session mean/std
        dates = closes.index.date
        for date in np.unique(dates):
            day_mask = dates == date
            day_closes = closes.loc[day_mask]

            if len(day_closes) < 3:
                continue

            # Expanding mean and std within the session day
            session_mean = day_closes.expanding().mean()
            session_std = day_closes.expanding().std()
            session_std = session_std.replace(0, np.nan)

            z = (day_closes - session_mean) / session_std

            day_signals = pd.DataFrame(0.0, index=day_closes.index, columns=day_closes.columns)
            day_signals[z < -self.Z_ENTRY] = 1.0   # Buy the dip
            day_signals[z > self.Z_ENTRY] = -1.0    # Sell the rip

            signals.loc[day_mask] = day_signals.values

        return signals


class IntradayMomentumAsia(IntradayMomentumUS):
    """Same momentum strategy but for the Asia session (00:00-08:00 UTC)."""

    CODE = "intraday-momentum-asia"
    SESSION_START_HOUR = 0   # 00:00 UTC
    SESSION_END_HOUR = 8     # 08:00 UTC


class IntradayMomentumEurope(IntradayMomentumUS):
    """Same momentum strategy but for the Europe session (08:00-16:00 UTC)."""

    CODE = "intraday-momentum-eu"
    SESSION_START_HOUR = 8   # 08:00 UTC
    SESSION_END_HOUR = 16    # 16:00 UTC
