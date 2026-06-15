"""rsu-rebalancing: backtest a one-way threshold rebalancing strategy for RSU comp.

Public API:
    - Configs: :class:`GrantSchedule`, :class:`StrategyConfig`, :class:`SimConfig`,
      :class:`TaxConfig`
    - Data: :func:`get_prices`, :func:`get_price_frame`
    - Run: :func:`run_backtest`, :class:`SimResult`
    - Metrics: :func:`comparison_table`, :func:`summarize`, :func:`time_weighted_returns`,
      :func:`growth_of_one`, :func:`annualized_return`, :func:`annualized_volatility`,
      :func:`sharpe_ratio`, :func:`max_drawdown`
"""

from .config import GrantSchedule, SimConfig, StrategyConfig, TaxConfig
from .data import get_price_frame, get_prices
from .metrics import (
    annualized_return,
    annualized_volatility,
    comparison_table,
    growth_of_one,
    max_drawdown,
    sharpe_ratio,
    summarize,
    time_weighted_returns,
)
from .simulate import SimResult, run_backtest, run_rule
from .strategy import HoldEverything, SellAllAtVest, ThresholdRebalance

__all__ = [
    "GrantSchedule",
    "HoldEverything",
    "SellAllAtVest",
    "SimConfig",
    "SimResult",
    "StrategyConfig",
    "TaxConfig",
    "ThresholdRebalance",
    "annualized_return",
    "annualized_volatility",
    "comparison_table",
    "get_price_frame",
    "get_prices",
    "growth_of_one",
    "max_drawdown",
    "run_backtest",
    "run_rule",
    "sharpe_ratio",
    "summarize",
    "time_weighted_returns",
]

__version__ = "0.1.0"
