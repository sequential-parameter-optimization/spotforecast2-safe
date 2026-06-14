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

        By construction, ``check_forecast_shape`` always stores ``n_overlap=0``
        when skipping; manually-constructed reports with ``n_overlap > 0`` and
        NaN metrics report ``skipped=False``.
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


@dataclass(frozen=True)
class LevelCheckReport:
    """Immutable result of a forecast *level* (systematic-bias) check.

    Where `ShapeCheckReport` answers "does the forecast track the daily
    *profile*", this answers "does the forecast sit at the right *level*".  It
    captures a near-constant offset of the whole forecast against a reference —
    the failure mode behind the 2026-06-13 team_4 miss, where the forecast
    over-predicted every hour by a flat ~1.8 GW (``bias == MAE``).

    Attributes:
        n_overlap: Number of aligned (overlapping) index positions used. When
            below the evaluable minimum, ``skipped`` is ``True`` and the
            numeric fields are ``float('nan')``.
        statistic: Central-tendency statistic used for both levels — either
            ``"median"`` (robust, the default) or ``"mean"``.
        forecast_level: ``statistic`` of the forecast over the overlap.
        reference_level: ``statistic`` of the reference over the overlap.
        offset: ``forecast_level - reference_level`` (signed; positive means
            the forecast sits high, i.e. systematic over-prediction).
        rel_offset: ``offset / abs(reference_level)`` (signed). ``float('nan')``
            when the reference level is zero.
        tol: Relative-offset tolerance for ``biased`` (passed through from
            `check_forecast_level`).

    Examples:
        ```{python}
        from spotforecast2_safe.processing.shape_check import LevelCheckReport

        # Forecast sits 4 % high vs the reference -> biased at tol=0.02.
        r = LevelCheckReport(
            n_overlap=24, statistic="median",
            forecast_level=45_600.0, reference_level=43_858.0,
            offset=1_742.0, rel_offset=0.0397, tol=0.02,
        )
        print("biased:", r.biased, "rel_offset:", round(r.rel_offset, 4))
        assert r.biased and not r.skipped
        ```
    """

    n_overlap: int
    statistic: str
    forecast_level: float
    reference_level: float
    offset: float
    rel_offset: float
    tol: float

    @property
    def biased(self) -> bool:
        """Return ``True`` when ``abs(rel_offset)`` exceeds ``tol``.

        A ``NaN`` relative offset (zero reference level or skipped check) is
        treated as *not* biased (returns ``False``).
        """
        if math.isnan(self.rel_offset):
            return False
        return abs(self.rel_offset) > self.tol

    @property
    def skipped(self) -> bool:
        """Return ``True`` when the overlap was too small to evaluate.

        By construction `check_forecast_level` stores ``n_overlap=0`` and NaN
        levels when skipping.
        """
        return (
            self.n_overlap == 0
            and math.isnan(self.offset)
            and math.isnan(self.rel_offset)
        )


def _resolve_level_statistic(statistic: str):
    """Return the pandas reducer for ``statistic`` or raise ``ValueError``."""
    if statistic == "median":
        return lambda s: float(s.median())
    if statistic == "mean":
        return lambda s: float(s.mean())
    raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}.")


def check_forecast_level(
    y: pd.Series,
    reference: pd.Series,
    *,
    statistic: str = "median",
    tol: float = 0.02,
    min_overlap: int = 12,
) -> LevelCheckReport:
    """Measure the systematic level offset between a forecast and its reference.

    Complements `check_forecast_shape`: a forecast can track the daily profile
    perfectly (high correlation, good range ratio) yet sit at the wrong level —
    a flat over- or under-prediction.  This returns the signed offset of the
    forecast's central level against the reference's, in absolute and relative
    terms, and flags it as ``biased`` when the relative offset exceeds ``tol``.

    Like `check_forecast_shape`, this function is **pure**: no logging, no
    warning, no raising on a biased result.  Only invalid inputs raise.

    Args:
        y: Forecast series (e.g. the 24-h submission).
        reference: Reference profile (e.g. ENTSO-E day-ahead forecast or
            actuals one week earlier).
        statistic: Central-tendency statistic, ``"median"`` (default, robust)
            or ``"mean"``.
        tol: Relative-offset tolerance for ``LevelCheckReport.biased``. Default
            ``0.02`` (2 %).
        min_overlap: Minimum overlap length to evaluate. Below this the report
            is ``skipped`` with NaN levels.

    Returns:
        `LevelCheckReport` with the computed levels, offsets, and ``tol``.

    Raises:
        TypeError: When ``y`` or ``reference`` is not a ``pd.Series``.
        ValueError: When ``y`` or ``reference`` is empty, or ``statistic`` is
            not ``"median"`` / ``"mean"``.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.processing.shape_check import check_forecast_level

        idx = pd.date_range("2026-06-13 00:00", periods=24, freq="h", tz="UTC")
        actual = pd.Series([43_000.0 + 3_000.0 * (i % 12) / 12 for i in range(24)],
                           index=idx)

        # Forecast that sits a flat 1_800 MW too high -> biased.
        high = actual + 1_800.0
        rep = check_forecast_level(high, actual, tol=0.02)
        print(f"offset={rep.offset:.0f} MW  rel={rep.rel_offset:.3f}  biased={rep.biased}")
        assert rep.biased and rep.offset > 0

        # A well-centred forecast -> not biased.
        ok = actual + 50.0
        print("small offset biased:", check_forecast_level(ok, actual).biased)
        ```
    """
    reducer = _resolve_level_statistic(statistic)
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
        return LevelCheckReport(
            n_overlap=0,
            statistic=statistic,
            forecast_level=nan,
            reference_level=nan,
            offset=nan,
            rel_offset=nan,
            tol=tol,
        )

    forecast_level = reducer(y.loc[common])
    reference_level = reducer(reference.loc[common])
    offset = forecast_level - reference_level
    rel_offset = offset / abs(reference_level) if reference_level != 0 else nan

    return LevelCheckReport(
        n_overlap=n_overlap,
        statistic=statistic,
        forecast_level=forecast_level,
        reference_level=reference_level,
        offset=offset,
        rel_offset=rel_offset,
        tol=tol,
    )


def apply_level_correction(
    y: pd.Series,
    reference: pd.Series,
    *,
    statistic: str = "median",
    min_overlap: int = 12,
) -> pd.Series:
    """Shift a forecast so its central level matches a reference (debias).

    Estimates the constant offset ``statistic(y) - statistic(reference)`` over
    the index overlap and subtracts it from **every** value of ``y``, removing a
    systematic flat bias while preserving the daily shape.  This is the
    post-hoc correction for the failure `check_forecast_level` detects.

    The returned series keeps ``y``'s full index, name, and ordering; only the
    level is shifted.  The function is pure (no mutation of the inputs).

    Args:
        y: Forecast series to correct.
        reference: Reference whose level ``y`` should be aligned to.
        statistic: ``"median"`` (default) or ``"mean"`` — must match the
            estimator you would use in `check_forecast_level`.
        min_overlap: Minimum overlap required to estimate the offset.

    Returns:
        A new ``pd.Series`` equal to ``y - offset`` (same index/name as ``y``).

    Raises:
        TypeError: When ``y`` or ``reference`` is not a ``pd.Series``.
        ValueError: When ``y``/``reference`` is empty, ``statistic`` is invalid,
            or the overlap is smaller than ``min_overlap`` (no reliable offset).

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.processing.shape_check import (
            apply_level_correction, check_forecast_level,
        )

        idx = pd.date_range("2026-06-13 00:00", periods=24, freq="h", tz="UTC")
        actual = pd.Series([43_000.0 + 3_000.0 * (i % 12) / 12 for i in range(24)],
                           index=idx)
        biased = actual + 1_800.0  # flat over-prediction

        corrected = apply_level_correction(biased, actual)
        print("offset before:", round(check_forecast_level(biased, actual).offset))
        print("offset after :", round(check_forecast_level(corrected, actual).offset))
        assert abs(check_forecast_level(corrected, actual).offset) < 1.0
        ```
    """
    reducer = _resolve_level_statistic(statistic)
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
    if len(common) < min_overlap:
        raise ValueError(
            f"overlap {len(common)} < min_overlap {min_overlap}; cannot "
            f"estimate a level offset."
        )

    offset = reducer(y.loc[common]) - reducer(reference.loc[common])
    return y - offset
