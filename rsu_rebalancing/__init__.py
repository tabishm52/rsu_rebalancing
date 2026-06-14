"""rsu-rebalancing: backtest a one-way threshold rebalancing strategy for RSU comp.

Public API:
    - Configs: :class:`GrantSchedule`, :class:`StrategyConfig`, :class:`SimConfig`
    - Data: :func:`get_prices`, :func:`get_price_frame`
    - Run: :func:`run_backtest`, :class:`SimResult`
    - Metrics: :func:`comparison_table`, :func:`summarize`, :func:`time_weighted_returns`,
      :func:`growth_of_one`
"""

from .config import GrantSchedule, SimConfig, StrategyConfig
from .data import get_price_frame, get_prices
from .metrics import (
    comparison_table,
    growth_of_one,
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
    "ThresholdRebalance",
    "comparison_table",
    "get_price_frame",
    "get_prices",
    "growth_of_one",
    "run_backtest",
    "run_rule",
    "summarize",
    "time_weighted_returns",
]

__version__ = "0.1.0"
