# Time Series Visualization

Interactive visualization tools for time series data analysis and comparison.

## Overview

This module provides Plotly-based interactive visualizations for:
- Single and multiple time series
- Train/validation/test splits
- Model comparison
- Temporal patterns and trends

## Visualization Functions

### Visualize Time Series (Plotly)

::: spotforecast2_safe.preprocessing.time_series_visualization.visualize_ts_plotly
    options:
      docstring_style: google
      show_source: true

### Compare Time Series

::: spotforecast2_safe.preprocessing.time_series_visualization.visualize_ts_comparison
    options:
      docstring_style: google
      show_source: true

## Time Series Visualization Module

::: spotforecast2_safe.preprocessing.time_series_visualization
    options:
      docstring_style: google
      show_source: true

## Examples

### Basic Visualization

```python
import pandas as pd
import numpy as np
from spotforecast2_safe.preprocessing.time_series_visualization import visualize_ts_plotly

# Create sample time series
dates = pd.date_range('2024-01-01', periods=100, freq='h')
data_train = pd.DataFrame({
    'temperature': np.random.normal(20, 5, 100),
    'humidity': np.random.normal(60, 10, 100)
}, index=dates)

# Visualize
visualize_ts_plotly({'Train': data_train})
```

### Multi-Dataset Comparison

```python
from spotforecast2_safe.preprocessing.time_series_visualization import visualize_ts_comparison

dataframes = {
    'Train': data_train,
    'Validation': data_val,
    'Test': data_test
}

visualize_ts_comparison(
    dataframes,
    show_mean=True,
    title_suffix='[°C]'
)
```

## Features

- **Interactive Exploration**: Zoom, pan, and hover for detailed insights
- **Multiple Datasets**: Compare multiple time series side-by-side
- **Customization**: Control colors, templates, and figure sizes
- **Export Support**: Save visualizations as PNG or HTML
- **Responsive Design**: Works on desktop and mobile displays

## Requirements

Requires `plotly>=6.5.2` for visualization functionality.
