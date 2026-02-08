# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Dataset configurations for spotforecast2_safe.

This module provides configuration dataclasses and data loading utilities
for managing parameters in forecasting demonstrations and production workflows.
"""

from spotforecast2_safe.manager.datasets.demo_data import DemoConfig
from spotforecast2_safe.manager.datasets.demo_loader import load_actual_combined

__all__ = ["DemoConfig", "load_actual_combined"]
