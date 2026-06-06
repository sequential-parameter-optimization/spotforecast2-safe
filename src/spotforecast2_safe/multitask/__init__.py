# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Multi-target forecasting pipeline orchestrator (safe subset).

This package provides a class hierarchy for multi-target time-series
forecasting pipelines in the ``spotforecast2-safe`` environment:

- ``BaseTask`` — shared data-preparation and helper logic.
- ``LazyTask`` — lazy fitting with default parameters; applies cached tuning
  results when available (from sibling-package tuning runs).
- ``DefaultsTask`` — fitting with default parameters; never reads the
  tuning-result cache.  Deterministic baseline.
- ``PredictTask`` — predict-only using saved models.
- ``CleanTask`` — removes all cached data from the pipeline cache directory.
- ``MultiTask`` — dispatcher that selects one of the tasks via a ``task``
  parameter.

Auto-tuning tasks (``OptunaTask``, ``SpotOptimTask``) and plotting helpers
are intentionally excluded from this package.  Use the ``spotforecast2``
sibling package for those capabilities.

Public API
----------
All classes are importable directly from ``spotforecast2_safe.multitask``.
"""

from spotforecast2_safe.multitask.base import BaseTask, agg_predictor
from spotforecast2_safe.multitask.clean import CleanTask
from spotforecast2_safe.multitask.defaults import DefaultsTask
from spotforecast2_safe.multitask.lazy import LazyTask
from spotforecast2_safe.multitask.multi import MultiTask
from spotforecast2_safe.multitask.predict import PredictTask
from spotforecast2_safe.multitask.runner import SAFE_PIPELINE_TASKS, run, run_with

__all__ = [
    "BaseTask",
    "CleanTask",
    "DefaultsTask",
    "LazyTask",
    "MultiTask",
    "PredictTask",
    "SAFE_PIPELINE_TASKS",
    "agg_predictor",
    "run",
    "run_with",
]
