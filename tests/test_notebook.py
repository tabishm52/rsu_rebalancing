"""Smoke test: the backtest notebook runs end to end on its default settings.

This is the coverage the notebook otherwise lacks -- it executes every cell (config
assembly, the engine, both figures, and the tables) against the price fixture, so a
drift between the engine and the notebook's presentation layer fails CI instead of
silently breaking the app.
"""

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: figure cells must not open a window; set before pyplot loads

from rsu_app.fixtures import patch_yf  # noqa: E402 - import after the Agg backend is selected

_NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "rsu_backtest.py"


@pytest.fixture
def price_fixture() -> Iterator[None]:
    """Serve the checked-in price snapshot in place of live yfinance for a test."""
    with patch_yf():
        yield


def _load_notebook():
    spec = importlib.util.spec_from_file_location("rsu_backtest_nb", _NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_runs_on_defaults(price_fixture):
    notebook = _load_notebook()

    _, defs = notebook.app.run()

    # The backtest cell swallows failures into `error` + mo.stop, so "no exception from
    # app.run()" is not enough -- assert the happy path actually produced results.
    assert defs["error"] is None
    results = defs["results"]
    assert len(results) == 3
    assert defs["threshold_name"].startswith("Threshold")
    assert all(not r.market.values.empty for r in results.values())
