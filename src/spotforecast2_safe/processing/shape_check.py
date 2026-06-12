# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Forecast shape plausibility check — pure computation, no side effects.

Ports the correlation + daily-range-ratio logic from the operational script's
``warn_if_implausible_shape`` (``team4_4zones_submit.py``, lines ~934-965) as a
returning, raising-on-invalid-input function.  The decision to log warnings or
abort is entirely the operator's concern; this module provides only the
measurements.

The reference-selection fallback chain (ENTSO-E day-ahead -> actuals one week
earlier) stays in the operator layer so the library remains source-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ShapeCheckReport:
    """Immutable result of a forecast shape plausibility check.

    All numeric fields reflect the overlap intersection of ``y`` and
    ``reference``; ``n_overlap`` is the count of aligned index positions.

    Attributes:
        n_overlap: Number of aligned (overlapping) index positions used for
            the computation.  When this is below the evaluable minimum the
            ``skipped`` property returns ``True`` and ``corr`` /
            ``range_ratio`` are ``float('nan')``.
        corr: Pearson correlation between ``y`` and ``reference`` over the
            overlap.  ``float('nan')`` when ``n_overlap < min_overlap`` or
            when the computation fails (e.g. zero-variance series).
        range_ratio: ``(y.max() - y.min()) / (reference.max() - reference.min())``
            over the overlap.  ``float('nan')`` when ``n_overlap < min_overlap``
            or when the reference range is zero.
        min_corr: Minimum acceptable Pearson correlation (passed through from
            ``check_forecast_shape``).
        min_range_ratio: Minimum acceptable range ratio (passed through from
            ``check_forecast_shape``).

    Examples:
        ```{python}
        from spotforecast2_safe.processing.shape_check import ShapeCheckReport
        import math

        r = ShapeCheckReport(
            n_overlap=24, corr=0.85, range_ratio=0.9,
            min_corr=0.6, min_range_ratio=0.5,
        )
        assert r.plausible
        assert not r.skipped
        print("plausible:", r.plausible, "skipped:", r.skipped)

        # NaN correlation -> not plausible
        r_nan = ShapeCheckReport(
            n_overlap=24, corr=float("nan"), range_ratio=0.9,
            min_corr=0.6, min_range_ratio=0.5,
        )
        assert not r_nan.plausible
        print("NaN corr -> plausible:", r_nan.plausible)
        ```
    """

    n_overlap: int
    corr: float
    range_ratio: float
    min_corr: float
    min_range_ratio: float

    @property
    def plausible(self) -> bool:
        """Return ``True`` when both correlation and range ratio meet their thresholds.

        ``NaN`` in either metric is treated as a failure (returns ``False``).
        """
        if math.isnan(self.corr) or math.isnan(self.range_ratio):
            return False
        return self.corr >= self.min_corr and self.range_ratio >= self.min_range_ratio

    @property
    def skipped(self) -> bool:
        """Return ``True`` when the overlap is too small to evaluate.

        This happens when ``n_overlap < min_overlap`` (the threshold passed to
        `check_forecast_shape`).  In that case both ``corr`` and
        ``range_ratio`` are ``float('nan')``.
        """
        return (
            math.isnan(self.corr)
            and math.isnan(self.range_ratio)
            and self.n_overlap == 0
        )


def check_forecast_shape(
    y: pd.Series,
    reference: pd.Series,
    *,
    min_corr: float = 0.6,
    min_range_ratio: float = 0.5,
    min_overlap: int = 12,
) -> ShapeCheckReport:
    """Measure correlation and daily-range ratio between a forecast and its reference.

    Ports the plausibility metrics from the operational ``warn_if_implausible_shape``
    (script lines ~951-965): Pearson correlation on the index intersection and
    the ratio of forecast range to reference range.  A zero-reference range
    produces ``float('nan')`` for ``range_ratio`` (zero-range guard).

    This function is **pure**: it performs no logging, no warning emission, and
    no raising on an implausible result.  Callers receive a `ShapeCheckReport`
    and decide how to act.  Only invalid inputs (non-Series arguments, empty
    series) raise.

    Args:
        y: Forecast series (e.g. the 24-h submission).
        reference: Reference profile to compare against (e.g. ENTSO-E
            day-ahead forecast or actuals one week earlier).
        min_corr: Correlation threshold for ``ShapeCheckReport.plausible``.
            Default ``0.6`` matches the operational script.
        min_range_ratio: Range-ratio threshold for
            ``ShapeCheckReport.plausible``.  Default ``0.5`` matches the
            operational script.
        min_overlap: Minimum overlap length required to evaluate the metrics.
            Below this, ``corr`` and ``range_ratio`` are ``float('nan')``
            and ``ShapeCheckReport.skipped`` returns ``True``.

    Returns:
        `ShapeCheckReport` with the computed metrics and the passed thresholds.

    Raises:
        TypeError: When ``y`` or ``reference`` is not a ``pd.Series``.
        ValueError: When ``y`` or ``reference`` is empty.

    Examples:
        ```{python}
        import pandas as pd
        import numpy as np
        from spotforecast2_safe.processing.shape_check import check_forecast_shape

        idx = pd.date_range("2026-06-11 00:00", periods=24, freq="h", tz="UTC")
        profile = pd.Series([float(i % 12) for i in range(24)], index=idx)

        # Identical profile -> correlation 1.0, range ratio 1.0, plausible.
        report = check_forecast_shape(profile, profile)
        print(f"corr={report.corr:.2f}  range_ratio={report.range_ratio:.2f}  "
              f"plausible={report.plausible}")
        assert report.plausible

        # Flat forecast -> range_ratio < 0.5, not plausible.
        flat = pd.Series(5.0, index=idx)
        report_flat = check_forecast_shape(flat, profile)
        print(f"flat: range_ratio={report_flat.range_ratio:.2f}  "
              f"plausible={report_flat.plausible}")
        assert not report_flat.plausible

        # Short overlap -> skipped.
        short = profile.iloc[:5]
        report_short = check_forecast_shape(short, profile, min_overlap=12)
        print(f"short: skipped={report_short.skipped}")
        assert report_short.skipped
        ```
    """
    if not isinstance(y, pd.Series):
        raise TypeError(f"y must be a pd.Series, got {type(y).__name__!r}.")
    if not isinstance(reference, pd.Series):
        raise TypeError(
            f"reference must be a pd.Series, got {type(reference).__name__!r}."
        )
    if y.empty:
        raise ValueError("y is empty.")
    if reference.empty:
        raise ValueError("reference is empty.")

    common = y.index.intersection(reference.index)
    n_overlap = len(common)

    nan = float("nan")
    if n_overlap < min_overlap:
        return ShapeCheckReport(
            n_overlap=0,
            corr=nan,
            range_ratio=nan,
            min_corr=min_corr,
            min_range_ratio=min_range_ratio,
        )

    y_common = y.loc[common]
    ref_common = reference.loc[common]

    corr = float(y_common.corr(ref_common))
    if math.isnan(corr):
        corr = nan

    ref_range = float(ref_common.max() - ref_common.min())
    if ref_range <= 0:
        range_ratio = nan
    else:
        range_ratio = float((y_common.max() - y_common.min()) / ref_range)

    return ShapeCheckReport(
        n_overlap=n_overlap,
        corr=corr,
        range_ratio=range_ratio,
        min_corr=min_corr,
        min_range_ratio=min_range_ratio,
    )
