# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Demo reporting metrics module.

This module provides the fixed-shape reporting helper that produces the
``{"MAE": float, "MSE": float}`` dictionary printed at the end of demo
and production runs.  It is intentionally separate from
``spotforecast2_safe.forecaster.metrics`` (the skforecast-aligned,
string-dispatch metric library used by the CV/backtesting engine).
"""

from typing import Dict

import pandas as pd


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate MAE and MSE for numeric evaluation.

    Computes Mean Absolute Error (MAE) and Mean Squared Error (MSE) between
    actual and predicted values. These metrics are essential for evaluating
    forecasting model performance in safety-critical applications.

    Args:
        actual: Series of actual observed values.
        predicted: Series of predicted values (must have same length as actual).

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'MAE': Mean Absolute Error
            - 'MSE': Mean Squared Error

    Raises:
        ValueError: If series have different lengths or contain NaN values.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.demo_metrics import calculate_metrics

        # Perfect predictions: both MAE and MSE should be zero
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = calculate_metrics(actual, predicted)
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")
        assert metrics["MAE"] == 0.0
        assert metrics["MSE"] == 0.0
        ```

        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.demo_metrics import calculate_metrics

        # Small symmetric errors: MAE == MSE == 1.0
        actual = pd.Series([10.0, 20.0, 30.0, 40.0])
        predicted = pd.Series([11.0, 19.0, 31.0, 39.0])
        metrics = calculate_metrics(actual, predicted)
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")
        assert metrics["MAE"] == 1.0
        assert metrics["MSE"] == 1.0
        ```

        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.demo_metrics import calculate_metrics

        # Larger asymmetric errors
        actual = pd.Series([100.0, 200.0, 300.0])
        predicted = pd.Series([95.0, 210.0, 290.0])
        metrics = calculate_metrics(actual, predicted)
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")
        assert abs(metrics["MAE"] - 25 / 3) < 1e-9
        assert abs(metrics["MSE"] - 75.0) < 1e-9
        ```

        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.demo_metrics import calculate_metrics

        # Safety-critical: validate metrics stay within acceptable bounds
        actual = pd.Series([50.0, 55.0, 60.0, 65.0, 70.0])
        predicted = pd.Series([51.0, 54.0, 61.0, 64.0, 71.0])
        metrics = calculate_metrics(actual, predicted)
        assert metrics["MAE"] < 2.0, "MAE exceeds safety threshold"
        assert metrics["MSE"] < 5.0, "MSE exceeds safety threshold"
        print("Safety validation passed")
        ```

        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.demo_metrics import calculate_metrics

        # Time series with a datetime index
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        actual = pd.Series([10.5, 11.2, 10.8, 11.5, 12.0], index=dates)
        predicted = pd.Series([10.3, 11.4, 10.9, 11.3, 12.1], index=dates)
        metrics = calculate_metrics(actual, predicted)
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")
        assert abs(metrics["MAE"] - 0.16) < 1e-9
        assert abs(metrics["MSE"] - 0.028) < 1e-9
        ```

        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.demo_metrics import calculate_metrics

        # Compare two models: lower MAE wins
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_model_a = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
        pred_model_b = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        metrics_a = calculate_metrics(actual, pred_model_a)
        metrics_b = calculate_metrics(actual, pred_model_b)
        winner = "A" if metrics_a["MAE"] < metrics_b["MAE"] else "B"
        print(f"Model A MAE: {metrics_a['MAE']:.4f}")
        print(f"Model B MAE: {metrics_b['MAE']:.4f}")
        print(f"Model {winner} has better MAE")
        assert winner == "A"
        ```
    """
    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual has {len(actual)} elements, "
            f"predicted has {len(predicted)}."
        )
    if actual.isna().any() or predicted.isna().any():
        raise ValueError(
            "Input series contain NaN values; metric computation requires "
            "complete data."
        )
    diff = actual - predicted
    mae = diff.abs().mean()
    mse = (diff**2).mean()
    return {"MAE": mae, "MSE": mse}
