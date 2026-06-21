"""Presentation layer for the marimo backtest notebook.

The notebook's control panel, figures, and table formatting live here so the notebook
cells stay short and read as intent.
"""

from rsu_app.controls import build_backtest_controls, build_configs
from rsu_app.figures import (
    build_concentration_figure,
    build_performance_figure,
    outperformance_spans,
)
from rsu_app.tables import format_returns_table, format_trade_log

__all__ = [
    "build_backtest_controls",
    "build_concentration_figure",
    "build_configs",
    "build_performance_figure",
    "format_returns_table",
    "format_trade_log",
    "outperformance_spans",
]
