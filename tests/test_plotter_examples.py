import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from spotforecast2.manager.plotter import PredictionFigure

def test_make_plot_example_execution():
    """Validates the logic used in the make_plot docstring example."""
    # Setup synthetic data matching docstring example
    dates = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
    train_end = dates[70]
    y = pd.Series(np.random.rand(100) * 100, index=dates, name="load")
    p = y + np.random.normal(0, 5, 100)
    
    pkg = {
        "train_actual": y.loc[:train_end],
        "future_actual": y.loc[train_end:],
        "train_pred": p.loc[:train_end],
        "future_pred": p.loc[train_end:],
        "metrics_train": {"mae": 5.0, "mape": 0.1},
        "metrics_future": {"mae": 6.0, "mape": 0.12},
        "metrics_future_one_day": {"mae": 4.5, "mape": 0.08},
    }
    
    # Execute class and method
    fig_gen = PredictionFigure(pkg)
    fig = fig_gen.make_plot()
    
    # Assertions
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # Actual, Predicted, Last Week (at least)
    
    # Specific traces
    trace_names = [d.name for d in fig.data]
    assert any("Actual" in name for name in trace_names if name)
    assert any("prediction" in name.lower() for name in trace_names if name)
    assert any("last week" in name.lower() for name in trace_names if name)

def test_make_plot_with_benchmark():
    """Validates make_plot when a benchmark forecast is provided."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
    train_end = dates[70]
    y = pd.Series(range(100), index=dates, name="load")
    
    pkg = {
        "train_actual": y.loc[:train_end],
        "future_actual": y.loc[train_end:],
        "train_pred": y.loc[:train_end],
        "future_pred": y.loc[train_end:],
        "future_forecast": y.loc[train_end:] + 1,  # Benchmark
        "metrics_train": {"mae": 0, "mape": 0},
        "metrics_future": {"mae": 0, "mape": 0},
        "metrics_future_one_day": {"mae": 0, "mape": 0},
        "metrics_forecast": {"mae": 1, "mape": 0.01},
        "metrics_forecast_one_day": {"mae": 1, "mape": 0.01},
    }
    
    fig = PredictionFigure(pkg).make_plot()
    trace_names = [d.name for d in fig.data]
    assert any("Benchmark" in name for name in trace_names if name)
