import numpy as np
import pandas as pd


class CommissionModel:
    """Base class for commission models."""

    def get_commissions(
        self, fill_prices: pd.DataFrame, position_changes: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate commissions for position changes.

        Args:
            fill_prices: DataFrame of fill prices (Date × Symbol)
            position_changes: DataFrame of position deltas (Date × Symbol), in weight terms

        Returns:
            DataFrame of commission costs as a fraction of portfolio value
        """
        raise NotImplementedError


class HyperliquidPerpsCommission(CommissionModel):
    """Hyperliquid perpetuals fee schedule.

    Default: VIP 0 tier (taker 3.5 bps, maker 1.0 bps).
    """

    TAKER_FEE = 0.000350  # 3.5 bps
    MAKER_FEE = 0.000100  # 1.0 bps

    def __init__(self, assume_taker: bool = True):
        self.fee_rate = self.TAKER_FEE if assume_taker else self.MAKER_FEE

    def get_commissions(
        self, fill_prices: pd.DataFrame, position_changes: pd.DataFrame
    ) -> pd.DataFrame:
        # Commission = |position_change_weight| * fee_rate
        # This is already in return-space since position_changes are in weight terms
        return position_changes.abs() * self.fee_rate


class HyperliquidSpotCommission(CommissionModel):
    """Hyperliquid spot trading fee schedule."""

    TAKER_FEE = 0.000700  # 7.0 bps
    MAKER_FEE = 0.000400  # 4.0 bps

    def __init__(self, assume_taker: bool = True):
        self.fee_rate = self.TAKER_FEE if assume_taker else self.MAKER_FEE

    def get_commissions(
        self, fill_prices: pd.DataFrame, position_changes: pd.DataFrame
    ) -> pd.DataFrame:
        return position_changes.abs() * self.fee_rate


class ZeroCommission(CommissionModel):
    """No commissions (for testing)."""

    def get_commissions(
        self, fill_prices: pd.DataFrame, position_changes: pd.DataFrame
    ) -> pd.DataFrame:
        return position_changes * 0.0
