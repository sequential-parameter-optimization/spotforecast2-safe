# Outlier Detection and Handling

Guide for identifying and handling outliers in time series data.

## Overview

Outlier detection is crucial for time series forecasting as extreme values can distort model training and predictions. This module provides robust outlier detection and marking capabilities.

## Key Functions

### Mark Outliers

::: spotforecast2_safe.preprocessing.outlier.mark_outliers
    options:
      docstring_style: google
      show_source: true

### Visualize Outliers

::: spotforecast2_safe.preprocessing.outlier.visualize_outliers_plotly_scatter
    options:
      docstring_style: google
      show_source: true

## Outlier Module

::: spotforecast2_safe.preprocessing.outlier
    options:
      docstring_style: google
      show_source: true

## Examples

```python
import pandas as pd
from spotforecast2_safe.preprocessing.outlier import mark_outliers

# Create sample time series data
data = pd.DataFrame({
    'value': [1, 2, 100, 4, 5, 6, 7, 8, 9, 10],  # 100 is an outlier
})

# Mark outliers
result_data, outlier_mask = mark_outliers(
    data,
    contamination=0.1,  # Expect 10% contamination
    columns=['value']
)

print(f"Outliers marked: {outlier_mask.sum()} records")
```

## Detection Methods

This module uses isolation forest and other statistical methods to detect:
- Sudden spikes or drops
- Seasonal anomalies
- Drift in baseline values
- Sudden shifts in variance
