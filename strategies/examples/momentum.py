import numpy as np
import pandas as pd

from strategies.base import CryptoMoonshot
from backtest.commission import HyperliquidPerpsCommission


class CryptoMomentum(CryptoMoonshot):
    """Cross-sectional momentum strategy across major perps.

    Longs the top-N symbols by trailing return, equal-weighted.
    Rebalances daily.
    """

    CODE = "crypto-momentum"
    SYMBOLS = []  # empty = all available symbols in DB
    INTERVAL = "1d"
    LOOKBACK = 20
    TOP_N = 3
    LEVERAGE = 1.0
    SLIPPAGE_BPS = 5
    COMMISSION_CLASS = HyperliquidPerpsCommission

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        closes = prices.loc["Close"]
        # Trailing return over LOOKBACK periods
        momentum = closes.pct_change(self.LOOKBACK)
        # Rank: lowest rank = strongest momentum
        ranked = momentum.rank(axis=1, ascending=False)
        # Long top N
        signals = (ranked <= self.TOP_N).astype(float)
        return signals

    def signals_to_target_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        # Equal-weight top N
        count = signals.sum(axis=1).replace(0, 1)
        weights = signals.div(count, axis=0)
        return weights * self.LEVERAGE


class CryptoMeanReversion(CryptoMoonshot):
    """Simple mean reversion strategy.

    Longs symbols that have dropped below their rolling mean,
    shorts symbols that have risen above it.
    """

    CODE = "crypto-mean-reversion"
    SYMBOLS = []  # empty = all available symbols in DB
    INTERVAL = "1d"
    LOOKBACK = 20
    Z_ENTRY = 1.5  # Z-score threshold
    LEVERAGE = 0.5
    SLIPPAGE_BPS = 5
    COMMISSION_CLASS = HyperliquidPerpsCommission

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        closes = prices.loc["Close"]
        rolling_mean = closes.rolling(self.LOOKBACK).mean()
        rolling_std = closes.rolling(self.LOOKBACK).std()
        z_score = (closes - rolling_mean) / rolling_std

        signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        signals[z_score < -self.Z_ENTRY] = 1.0   # Long when oversold
        signals[z_score > self.Z_ENTRY] = -1.0    # Short when overbought
        return signals
