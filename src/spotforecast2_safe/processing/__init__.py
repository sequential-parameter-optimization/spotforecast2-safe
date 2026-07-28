# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Processing module for end-to-end forecasting pipelines."""

from .agg_predict import agg_predict
from .blend import blend_with_prior
from .forecast_scoring import (
    SUPPORTED_METRICS,
    aggregate_period_scores,
    mase_scaling_factors,
    score_forecasts,
    score_forecasts_by_period,
)
from .n2n_predict import n2n_predict
from .n2n_predict_with_covariates import n2n_predict_with_covariates
from .shape_check import (
    LevelCheckReport,
    ShapeCheckReport,
    apply_level_correction,
    check_forecast_level,
    check_forecast_shape,
)

__all__ = [
    "agg_predict",
    "aggregate_period_scores",
    "blend_with_prior",
    "mase_scaling_factors",
    "score_forecasts",
    "score_forecasts_by_period",
    "SUPPORTED_METRICS",
    "n2n_predict",
    "n2n_predict_with_covariates",
    "ShapeCheckReport",
    "check_forecast_shape",
    "LevelCheckReport",
    "check_forecast_level",
    "apply_level_correction",
]
