import numpy as np
import pandas as pd

from data.store import get_funding_rates
from strategies.base import CryptoMoonshot


class FundingRateCarry(CryptoMoonshot):
    """Delta-neutral funding rate carry strategy.

    Collects funding by taking the opposite side of crowded positioning:
    - When funding is positive (longs pay shorts): short perp + long spot
    - When funding is negative (shorts pay longs): long perp + short spot

    The spot leg keeps the position delta-neutral so P&L comes purely
    from funding payments, not directional exposure.

    Funding on Hyperliquid settles every 8 hours. This strategy evaluates
    the rolling average funding rate and enters/exits based on thresholds.
    """

    CODE = "funding-carry"
    SYMBOLS = []  # empty = all available perps in DB
    INTERVAL = "1h"
    LEVERAGE = 1.0

    # Funding rate thresholds (per 8h settlement)
    # Typical range: -0.01% to +0.03%, extremes can hit 0.1%+
    FUNDING_ENTRY_THRESHOLD = 0.0003   # 0.03% per 8h ≈ 32% annualized
    FUNDING_EXIT_THRESHOLD = 0.0001    # 0.01% per 8h — exit when carry fades
    FUNDING_LOOKBACK = 24              # Hours to average funding over (3 settlements)

    # Risk controls
    MAX_POSITIONS = 5                  # Max simultaneous carry positions
    MIN_DAILY_VOLUME = 1_000_000      # Skip illiquid pairs

    # Commission: spot legs cost more than perp-only
    SLIPPAGE_BPS = 7  # Higher for paired spot+perp execution

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate carry signals from funding rate data.

        Signal =  1: funding is very negative → long perp (receive funding)
        Signal = -1: funding is very positive → short perp (receive funding)
        Signal =  0: funding near zero → no carry edge
        """
        closes = prices.loc["Close"]
        symbols = list(closes.columns)

        # Fetch funding rate history aligned to price dates
        start = str(closes.index[0])
        end = str(closes.index[-1])
        funding = get_funding_rates(
            symbols=symbols,
            start_date=start,
            end_date=end,
            db_path=self.DB_PATH,
        )

        if funding.empty:
            return pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

        # Resample funding to hourly (funding settles every 8h, forward-fill between)
        funding = funding.reindex(closes.index, method="ffill")

        # Rolling average funding rate
        avg_funding = funding.rolling(self.FUNDING_LOOKBACK, min_periods=1).mean()

        signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

        # Positive funding → short perp (we receive funding from longs)
        signals[avg_funding > self.FUNDING_ENTRY_THRESHOLD] = -1.0

        # Negative funding → long perp (we receive funding from shorts)
        signals[avg_funding < -self.FUNDING_ENTRY_THRESHOLD] = 1.0

        # Exit zone: funding reverted to near-zero
        exit_mask = avg_funding.abs() < self.FUNDING_EXIT_THRESHOLD
        signals[exit_mask] = 0.0

        return signals

    def signals_to_target_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Equal-weight across active carry positions, capped at MAX_POSITIONS.

        Prioritize the highest absolute funding rates.
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for i in range(len(signals)):
            row = signals.iloc[i]
            active = row[row != 0]

            if active.empty:
                continue

            # If more candidates than MAX_POSITIONS, keep the strongest
            if len(active) > self.MAX_POSITIONS:
                active = active.reindex(
                    active.abs().nlargest(self.MAX_POSITIONS).index
                )

            n = len(active)
            weight_per_position = self.LEVERAGE / n
            weights.iloc[i, weights.columns.get_indexer(active.index)] = (
                active.values * weight_per_position
            )

        return weights

    def positions_to_gross_returns(
        self, positions: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Carry returns = funding income + price P&L.

        The position is delta-neutral (spot offsets perp), so the price
        P&L component should be near zero. The dominant return source
        is the funding rate collected on the perp leg.
        """
        closes = prices.loc["Close"]
        symbols = list(closes.columns)

        # 1. Price-based P&L (should be ~0 for delta-neutral, but model it)
        pct_returns = closes.pct_change()
        price_pnl = positions.shift(1) * pct_returns

        # 2. Funding income: position * funding_rate
        #    Short perp (position=-1) with positive funding → we receive funding
        #    Long perp (position=+1) with negative funding → we receive funding
        #    In both cases: income = -position * funding_rate
        start = str(closes.index[0])
        end = str(closes.index[-1])
        funding = get_funding_rates(
            symbols=symbols,
            start_date=start,
            end_date=end,
            db_path=self.DB_PATH,
        )

        if not funding.empty:
            funding = funding.reindex(closes.index, method="ffill").fillna(0)
            # Funding settles every 8h. With 1h bars, each bar gets 1/8 of the rate.
            hourly_funding = funding / 8
            # Income: we receive funding when positioned opposite the crowd
            funding_income = -positions.shift(1) * hourly_funding
        else:
            funding_income = pd.DataFrame(
                0.0, index=closes.index, columns=closes.columns
            )

        # Total gross return = funding income (price P&L cancels for delta-neutral)
        # We include price_pnl for accuracy in backtest (basis can drift)
        gross_returns = price_pnl.fillna(0) + funding_income.fillna(0)
        return gross_returns
