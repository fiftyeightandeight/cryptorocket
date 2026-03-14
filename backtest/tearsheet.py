from typing import Optional

import numpy as np
import pandas as pd


def compute_metrics(returns: pd.Series, periods_per_year: float = 365) -> dict:
    """Compute standard performance metrics from a return series.

    Args:
        returns: Series of portfolio returns (daily by default)
        periods_per_year: Annualization factor (365 for daily crypto, 8760 for hourly)

    Returns:
        dict of performance metrics
    """
    returns = returns.dropna()
    if returns.empty:
        return {}

    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1

    # CAGR
    n_periods = len(returns)
    n_years = n_periods / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio (assuming 0 risk-free rate for crypto)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    # Sortino ratio
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 0
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    running_max = cum_returns.cummax()
    drawdowns = cum_returns / running_max - 1
    max_drawdown = drawdowns.min()

    # Calmar ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    n_positive = (returns > 0).sum()
    n_negative = (returns < 0).sum()
    n_trades = n_positive + n_negative
    win_rate = n_positive / n_trades if n_trades > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average win/loss
    avg_win = returns[returns > 0].mean() if n_positive > 0 else 0
    avg_loss = returns[returns < 0].mean() if n_negative > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "n_periods": n_periods,
        "n_years": n_years,
    }


def print_tearsheet(code: str, metrics: dict, results: dict):
    """Print a formatted tearsheet to stdout."""
    if not metrics:
        print(f"Strategy {code}: no data to report")
        return

    print(f"\n{'=' * 60}")
    print(f"  Strategy: {code}")
    print(f"{'=' * 60}")
    print(f"  Total Return:     {metrics['total_return']:>10.2%}")
    print(f"  CAGR:             {metrics['cagr']:>10.2%}")
    print(f"  Ann. Volatility:  {metrics['ann_volatility']:>10.2%}")
    print(f"  Sharpe Ratio:     {metrics['sharpe']:>10.2f}")
    print(f"  Sortino Ratio:    {metrics['sortino']:>10.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    print(f"  Calmar Ratio:     {metrics['calmar']:>10.2f}")
    print(f"  Win Rate:         {metrics['win_rate']:>10.2%}")
    print(f"  Profit Factor:    {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Win:          {metrics['avg_win']:>10.4%}")
    print(f"  Avg Loss:         {metrics['avg_loss']:>10.4%}")
    print(f"  Periods:          {metrics['n_periods']:>10d}")
    print(f"  Duration (yrs):   {metrics['n_years']:>10.2f}")
    print(f"{'=' * 60}")

    # Monthly returns table
    returns = results["returns"]
    if not returns.empty and hasattr(returns.index, "year"):
        monthly = _monthly_returns_table(returns)
        if monthly is not None:
            print("\n  Monthly Returns:")
            print(monthly.to_string())
            print()


def _monthly_returns_table(returns: pd.Series) -> Optional[pd.DataFrame]:
    """Build a year × month returns table."""
    try:
        returns.index = pd.to_datetime(returns.index)
        monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly.index.names = ["Year", "Month"]
        table = monthly.unstack("Month")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        table.columns = [month_names[m - 1] for m in table.columns]
        # Add yearly total
        table["Year"] = table.apply(lambda row: (1 + row.dropna()).prod() - 1, axis=1)
        return (table * 100).round(2)
    except Exception:
        return None


def plot_equity_curve(
    code: str,
    returns: pd.Series,
    positions: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
):
    """Plot equity curve and drawdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = cum_returns / running_max - 1

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Strategy: {code}", fontsize=14, fontweight="bold")

    # Equity curve
    ax1 = axes[0]
    ax1.plot(cum_returns.index, cum_returns.values, linewidth=1.5, color="#2196F3")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.4, color="#F44336")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {output_path}")
    else:
        plt.savefig("/tmp/backtest_equity_curve.png", dpi=150, bbox_inches="tight")
        print("  Plot saved to /tmp/backtest_equity_curve.png")

    plt.close()
