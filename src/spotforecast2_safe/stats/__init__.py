# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Statistical primitives for time-series diagnostics.

Pure compute, no plotting dependencies — safety-critical-friendly.
Visualization wrappers live in `spotforecast2.plots`.
"""

from spotforecast2_safe.stats.comparison import (
    PairedComparison,
    RepeatedMeasuresAnova,
    holm_adjust,
    paired_comparison,
    pairwise_paired_t,
    rank_concordance,
    rank_table,
    repeated_measures_anova,
)
from spotforecast2_safe.stats.errors import error_profile, error_summary
from spotforecast2_safe.stats.spectral import compute_periodogram
from spotforecast2_safe.stats.stationarity import augmented_dickey_fuller

__all__ = [
    "augmented_dickey_fuller",
    "compute_periodogram",
    "error_profile",
    "error_summary",
    "holm_adjust",
    "paired_comparison",
    "PairedComparison",
    "pairwise_paired_t",
    "rank_concordance",
    "rank_table",
    "repeated_measures_anova",
    "RepeatedMeasuresAnova",
]
