# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Operational data-coverage guards for live forecasting pipelines.

These guards implement the battle-tested invariants from the team-4 operational
script (``team4_4zones_submit.py``).  All thresholds are parameters so callers
retain full control over the operational envelope; the script's ``Abort``
exception and exit-code conventions stay in the operator layer.

Guards 1-4 correspond to the four non-value-sanity checks in the script's
``assert_coverage`` function (lines ~457-572).  Guard 5 (value-sanity via
intra-hour range / adjacent-step / deviation rules) is already provided by
`spotforecast2_safe.preprocessing.target_corruption.apply_target_corruption_policy`
and is intentionally **not** duplicated here.

Binding invariant: every guard **raises** `~spotforecast2_safe.exceptions.CoverageError`
on violation.  `last_complete_hour` is a pure computation helper with no
side-effects; it raises `ValueError` on invalid input.
"""

from __future__ import annotations

import logging

import pandas as pd

from spotforecast2_safe.exceptions import CoverageError

logger = logging.getLogger(__name__)


def assert_frontier_fresh(
    index: pd.DatetimeIndex,
    required_last: pd.Timestamp,
) -> None:
    """Raise `~spotforecast2_safe.exceptions.CoverageError` if the data frontier is stale.

    Corresponds to the first guard in the operational ``assert_coverage``
    (script line ~457): ``index.max() < required_last`` means no new data has
    arrived since the required cutoff and forecasting would extrapolate into
    a data void.

    The comparison is strict (``<``): ``index.max() == required_last`` is a
    passing boundary (data is exactly as fresh as required).

    Args:
        index: DatetimeIndex of the interim data frame whose maximum timestamp
            is compared against the freshness requirement.
        required_last: Minimum acceptable value for ``index.max()``.  Callers
            typically pass ``today - pd.Timedelta(hours=1)`` (the last
            publishable complete UTC hour before ``now``).

    Raises:
        CoverageError: When ``index.max() < required_last``, naming both the
            observed maximum and the required value.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.coverage import assert_frontier_fresh
        from spotforecast2_safe.exceptions import CoverageError

        # Passing: index reaches the required timestamp.
        idx = pd.date_range("2026-06-10 00:00", periods=24, freq="h", tz="UTC")
        required = pd.Timestamp("2026-06-10 00:00", tz="UTC")
        assert_frontier_fresh(idx, required)   # no exception

        # Failing: data ends one hour before the requirement.
        try:
            assert_frontier_fresh(idx, pd.Timestamp("2026-06-10 23:00", tz="UTC"))
        except CoverageError as exc:
            print(exc)
        ```
    """
    last = index.max()
    if last < required_last:
        raise CoverageError(
            f"ENTSO-E coverage is stale: last data row {last} < required {required_last}."
        )


def assert_actual_lag_within(
    actual: pd.Series,
    now: pd.Timestamp,
    max_lag: pd.Timedelta,
) -> None:
    """Raise `~spotforecast2_safe.exceptions.CoverageError` if the last published actual is too old.

    Corresponds to the second guard in the operational ``assert_coverage``
    (script line ~462): the last non-NaN Actual Load observation is older than
    ``now - max_lag``, indicating ENTSO-E's publication pipeline has stalled.

    The comparison is strict (``<``): ``last_actual == now - max_lag`` is a
    passing boundary (lag is exactly at the tolerance).

    Args:
        actual: Series of Actual Load (or equivalent) values.  NaNs are
            stripped before computing the last published timestamp.
        now: Reference timestamp representing the current wall-clock time
            (tz-aware).
        max_lag: Maximum acceptable age of the last published observation.
            The operational default is ``pd.Timedelta(hours=36)``.

    Raises:
        CoverageError: When the last non-NaN observation is older than
            ``now - max_lag``, reporting the observed lag in whole hours.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.coverage import assert_actual_lag_within
        from spotforecast2_safe.exceptions import CoverageError

        now = pd.Timestamp("2026-06-11 12:00", tz="UTC")
        idx = pd.date_range("2026-06-10 00:00", periods=24, freq="h", tz="UTC")
        actual = pd.Series(range(24), index=idx, dtype=float)

        # Passing: last observation is 36 h old, tolerance is exactly 36 h.
        assert_actual_lag_within(actual, now, pd.Timedelta(hours=36))

        # Failing: tolerance is 35 h but last observation is 36 h old.
        try:
            assert_actual_lag_within(actual, now, pd.Timedelta(hours=35))
        except CoverageError as exc:
            print(exc)
        ```
    """
    clean = actual.dropna()
    if clean.empty:
        raise CoverageError("Actual Load series has no published (non-NaN) values.")
    last_actual = clean.index.max()
    deadline = now - max_lag
    if last_actual < deadline:
        lag_h = int((now - last_actual) / pd.Timedelta(hours=1))
        max_lag_h = int(max_lag / pd.Timedelta(hours=1))
        raise CoverageError(
            f"Actual Load is stale: last published {last_actual}, "
            f"{lag_h} h before now; tolerance {max_lag_h} h."
        )


def assert_no_interior_gaps(
    actual: pd.Series,
    now: pd.Timestamp,
    *,
    scan_window: pd.Timedelta,
    max_gap: pd.Timedelta,
) -> None:
    """Raise `~spotforecast2_safe.exceptions.CoverageError` if the recent actuals contain large holes.

    Corresponds to the third guard in the operational ``assert_coverage``
    (script lines ~471-483): a publication outage can hole the *middle* of
    the feed while both edge checks pass.  The 2026-06-02 incident — a full
    day of actuals published more than a day late — motivated this guard.

    The scan covers ``[now - scan_window, now]``; consecutive index differences
    exceeding ``max_gap`` are counted and the worst gap is reported.

    The comparison is strict (``>``): a difference exactly equal to ``max_gap``
    is acceptable.

    Args:
        actual: Series of Actual Load (or equivalent) values, NaNs excluded
            internally for gap detection.
        now: Reference timestamp for the scan window's right boundary.
        scan_window: How far back to look for interior gaps.  The operational
            default corresponds to ``pd.Timedelta(days=28)``.
        max_gap: Maximum acceptable consecutive index difference inside the
            scan window.  The operational default is ``pd.Timedelta(hours=12)``.

    Raises:
        CoverageError: When at least one gap exceeds ``max_gap``, reporting
            the count of oversized gaps and the worst gap's start/end pair.
            The message mirrors the operational script's format so operators
            recognise it.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.coverage import assert_no_interior_gaps
        from spotforecast2_safe.exceptions import CoverageError

        now = pd.Timestamp("2026-06-11 12:00", tz="UTC")
        # Clean hourly data for 30 days — no gaps.
        idx = pd.date_range("2026-05-12 00:00", periods=30 * 24, freq="h", tz="UTC")
        actual = pd.Series(1.0, index=idx)
        assert_no_interior_gaps(
            actual, now,
            scan_window=pd.Timedelta(days=28),
            max_gap=pd.Timedelta(hours=12),
        )

        # Inject a 24-hour interior gap (2026-06-02 incident pattern).
        gapped = actual.copy()
        drop = (gapped.index >= pd.Timestamp("2026-06-02 00:00", tz="UTC")) & \
               (gapped.index <  pd.Timestamp("2026-06-03 00:00", tz="UTC"))
        gapped = gapped[~drop]
        try:
            assert_no_interior_gaps(
                gapped, now,
                scan_window=pd.Timedelta(days=28),
                max_gap=pd.Timedelta(hours=12),
            )
        except CoverageError as exc:
            print(exc)
        ```
    """
    clean = actual.dropna()
    recent = clean.loc[now - scan_window :]
    if recent.empty:
        return
    gaps = recent.index.to_series().diff()
    oversized = gaps[gaps > max_gap]
    if not oversized.empty:
        worst_end = oversized.idxmax()
        worst_gap = oversized.max()
        gap_start = worst_end - worst_gap
        scan_days = int(scan_window / pd.Timedelta(days=1))
        max_gap_h = int(max_gap / pd.Timedelta(hours=1))
        raise CoverageError(
            f"Actual Load has {len(oversized)} interior gap(s) wider than "
            f"{max_gap_h} h in the last {scan_days} days "
            f"(worst: {gap_start} -> {worst_end}). "
            f"Re-download once the late actuals are published."
        )


def last_complete_hour(
    actual: pd.Series,
    *,
    samples_per_hour: int | None = None,
) -> pd.Timestamp:
    """Return the latest hour having a complete set of intra-hour samples.

    Implements the frontier-completeness guard from the operational
    ``assert_coverage`` (script lines ~485-497): only an hour with all of its
    quarter-hour samples published may safely anchor a live recursion.  A
    partial frontier hour averages to an anomalous level and drags the first
    forecast day (observed on the 2026-06-05 forecast).

    The expected sample count per hour is derived from the feed's own cadence
    (modal index difference) when ``samples_per_hour`` is ``None``.  For a
    15-min feed this evaluates to 4; for an hourly feed it evaluates to 1.

    The returned timestamp is the *floor* of the last complete hour (e.g.
    ``2026-06-11 10:00 UTC`` for a 15-min feed whose last complete hour ended
    at ``2026-06-11 10:45``).

    Args:
        actual: Series of Actual Load (or equivalent) values.  NaN values are
            excluded before computing the cadence and per-hour counts.
        samples_per_hour: Override for the expected sample count per hour.
            Pass ``None`` (default) to infer from the modal index difference.
            Must be a positive integer when provided.

    Returns:
        Timezone-aware `pd.Timestamp` floored to the hour of the last
        complete hour.

    Raises:
        ValueError: When ``actual`` is empty or all-NaN after dropping NaNs,
            or when ``samples_per_hour`` is provided but not a positive integer.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.preprocessing.coverage import last_complete_hour

        # 15-min feed: last hour has only 2 of 4 samples -> step back.
        idx_full = pd.date_range("2026-06-10 00:00", periods=24 * 4, freq="15min", tz="UTC")
        actual = pd.Series(1.0, index=idx_full)
        # Remove the last two slots of the last hour.
        partial = actual.iloc[:-2]
        result = last_complete_hour(partial)
        assert result == pd.Timestamp("2026-06-10 22:00", tz="UTC"), result
        print("last_complete_hour:", result)

        # Hourly feed: each hour has exactly 1 sample -> last hour is complete.
        idx_h = pd.date_range("2026-06-10 00:00", periods=24, freq="h", tz="UTC")
        actual_h = pd.Series(1.0, index=idx_h)
        result_h = last_complete_hour(actual_h)
        assert result_h == pd.Timestamp("2026-06-10 23:00", tz="UTC"), result_h
        print("last_complete_hour (hourly):", result_h)
        ```
    """
    clean = actual.dropna()
    if clean.empty:
        raise ValueError(
            "actual is empty or all-NaN; cannot compute last complete hour."
        )

    if samples_per_hour is not None:
        if not isinstance(samples_per_hour, int) or samples_per_hour < 1:
            raise ValueError(
                f"samples_per_hour must be a positive integer, got {samples_per_hour!r}."
            )
        sph = samples_per_hour
    else:
        cadence = clean.index.to_series().diff().mode().iloc[0]
        sph = int(pd.Timedelta(hours=1) / cadence)

    samples_by_hour = clean.resample("h").count()
    complete = samples_by_hour[samples_by_hour >= sph]
    if complete.empty:
        raise ValueError(
            "No complete hour found in actual; every hour has fewer than "
            f"{sph} sample(s)."
        )
    return complete.index.max()
