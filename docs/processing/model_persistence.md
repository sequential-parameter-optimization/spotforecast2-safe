# Model Persistence

Guide for saving and loading trained forecasting models.

## Overview

The model persistence functionality enables you to:
- Save trained forecasters to disk
- Load previously trained models
- Manage model caching
- Handle batch model operations

## Functions

### Saving Models

::: spotforecast2_safe.processing.n2n_predict_with_covariates._save_forecasters
    options:
      docstring_style: google
      show_source: true

### Loading Models

::: spotforecast2_safe.processing.n2n_predict_with_covariates._load_forecasters
    options:
      docstring_style: google
      show_source: true

### Model Directory Management

::: spotforecast2_safe.processing.n2n_predict_with_covariates._ensure_model_dir
    options:
      docstring_style: google
      show_source: true

::: spotforecast2_safe.processing.n2n_predict_with_covariates._model_directory_exists
    options:
      docstring_style: google
      show_source: true

## Examples

```python
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    _save_forecasters,
    _load_forecasters,
)

# Save trained models
trained_forecasters = {...}  # Your trained forecasters
_save_forecasters(trained_forecasters, model_dir="models/")

# Load previously trained models
loaded_forecasters = _load_forecasters(model_dir="models/")
```
