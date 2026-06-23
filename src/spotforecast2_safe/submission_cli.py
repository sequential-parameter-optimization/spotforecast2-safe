# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared CLI helpers for ENTSO-E day-ahead submission scripts.

This module promotes the five gate-helper functions that were duplicated
across ``scripts/team4_optuna_submit.py``, ``scripts/chronos.py``, and
``scripts/macl2l.py`` into a single, torch-free, minimal-dependency home
inside ``spotforecast2_safe``.

Public surface:

- :class:`AbortReason` — enum of named failure categories.
- :class:`SubmissionAbort` — exception carrying a ``reason`` + message; no
  integer exit code (exit-code mapping is caller policy).
- :func:`compute_dates_or_abort` — wraps
  :func:`spotforecast2_safe.downloader.entsoe.compute_submission_dates`.
- :func:`load_interim_or_abort` — wraps
  :func:`spotforecast2_safe.downloader.entsoe.load_interim`.
- :func:`assert_coverage_or_abort` — wraps
  :func:`spotforecast2_safe.preprocessing.coverage.assert_submission_coverage`;
  decides ``COVERAGE`` vs ``STALE_CACHE`` split internally.
- :func:`entsoe_predictions` — tolerant wrapper over
  :func:`spotforecast2_safe.downloader.entsoe.extract_entsoe_hourly_forecast`;
  does NOT abort.  ``pandas`` is imported lazily inside this function (and
  under ``TYPE_CHECKING`` for annotations).
- :func:`download_combined_or_abort` — wraps
  :func:`spotforecast2_safe.downloader.resilience.download_combined_with_fallback`;
  aborts on ``mode=None``.

Delegation note: this module is pure dispatch — it delegates every network
call and filesystem interaction to the modules that already implement and
threat-model them (``downloader.entsoe``, ``downloader.resilience``,
``preprocessing.coverage``).

Constraint: **all** ``spotforecast2_safe`` imports are lazy inside functions
so that ``configure_environment()`` can be called before ``get_data_home()``
is first resolved.  ``pandas`` is imported lazily inside
:func:`entsoe_predictions` (and under ``TYPE_CHECKING`` for annotations).
Tests monkeypatch the entsoe / resilience / coverage module attributes after
the lazy import inside each function.
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Public exception hierarchy
# ---------------------------------------------------------------------------


class AbortReason(enum.Enum):
    """Named failure categories for :class:`SubmissionAbort`.

    These correspond to the gate-check steps in the submission pipeline;
    the mapping to numeric exit codes is left to the caller.

    Members:

    - ``DATE`` — ``compute_submission_dates`` raised (bad date arithmetic).
    - ``INTERIM_MISSING`` — the combined interim CSV does not exist yet.
    - ``COVERAGE`` — a data-quality guard fired (gaps, staleness, corruption).
    - ``STALE_CACHE`` — a snapshot was restored and the data is older than
      yesterday 23:00 UTC (the ``"yesterday 23:00 UTC"`` message substring).
    - ``DOWNLOAD_UNRECOVERABLE`` — live download exhausted and no valid
      snapshot within the TTL.

    Examples:

        ```{python}
        from spotforecast2_safe.submission_cli import AbortReason

        print(list(AbortReason))
        print(AbortReason.DATE.name)
        ```
    """

    DATE = "DATE"
    INTERIM_MISSING = "INTERIM_MISSING"
    COVERAGE = "COVERAGE"
    STALE_CACHE = "STALE_CACHE"
    DOWNLOAD_UNRECOVERABLE = "DOWNLOAD_UNRECOVERABLE"


class SubmissionAbort(Exception):
    """Raised by the five gate helpers when a submission cannot proceed.

    Carries the :attr:`reason` enum member so callers can map to an
    integer exit code without parsing message strings.  No exit code is
    stored here — that is caller policy.

    Parameters
    ----------
    reason : AbortReason
        The category of the failure.
    message : str
        Human-readable description forwarded from the underlying exception.

    Attributes
    ----------
    reason : AbortReason
        The named failure category.

    Examples:

        ```{python}
        from spotforecast2_safe.submission_cli import AbortReason, SubmissionAbort

        try:
            raise SubmissionAbort(AbortReason.DATE, "bad date arithmetic")
        except SubmissionAbort as exc:
            print(exc.reason, str(exc))
        ```
    """

    def __init__(self, reason: AbortReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------


def compute_dates_or_abort(
    now: "pd.Timestamp",
    data_end: "str | None",
    train_years: int,
    start_margin_days: int,
    logger: logging.Logger,
) -> "spotforecast2_safe.downloader.entsoe.SubmissionDates":  # type: ignore[name-defined]  # noqa: F821
    """Compute the submission date window, aborting on invalid inputs.

    Delegates to
    :func:`spotforecast2_safe.downloader.entsoe.compute_submission_dates`
    (sf2-safe 22.10.0).  Wraps :exc:`ValueError` / :exc:`TypeError` in
    :exc:`SubmissionAbort` with ``reason=AbortReason.DATE`` so callers
    map that to their exit-code 1.

    Parameters
    ----------
    now : pd.Timestamp
        Current UTC timestamp (``pd.Timestamp.now(tz="UTC")``).
    data_end : str or None
        Override for ``END_DOWNLOAD`` in ``"YYYYMMDDHHMM"`` format, or
        ``None`` to derive automatically.
    train_years : int
        Training window length in years (drives the derived download start).
    start_margin_days : int
        Extra days subtracted from ``today - 365*train_years`` to cover
        leap-year shortfall, staleness, and truncation retraction.
    logger : logging.Logger
        Accepted for API symmetry with the other helpers; this function does
        not log.

    Returns
    -------
    SubmissionDates
        Named tuple with ``now``, ``today``, ``tomorrow``, ``last_target``,
        ``start_dl``, ``end_dl`` fields.

    Raises
    ------
    SubmissionAbort
        With ``reason=AbortReason.DATE`` when
        ``compute_submission_dates`` raises ``ValueError`` or ``TypeError``.

    Examples:

        ```{python}
        #| eval: false
        import logging
        import pandas as pd

        from spotforecast2_safe.submission_cli import compute_dates_or_abort

        logger = logging.getLogger("example")
        dates = compute_dates_or_abort(
            now=pd.Timestamp.now(tz="UTC"),
            data_end=None,
            train_years=3,
            start_margin_days=45,
            logger=logger,
        )
        print(dates.tomorrow.date())
        ```
    """
    import spotforecast2_safe.downloader.entsoe as _entsoe

    try:
        return _entsoe.compute_submission_dates(
            now,
            train_years=train_years,
            start_margin_days=start_margin_days,
            data_end=data_end,
        )
    except (ValueError, TypeError) as exc:
        raise SubmissionAbort(AbortReason.DATE, str(exc)) from exc


def load_interim_or_abort(
    logger: logging.Logger,
) -> "pd.DataFrame":
    """Read the combined DE-total interim CSV, aborting if it is missing.

    Delegates to
    :func:`spotforecast2_safe.downloader.entsoe.load_interim`
    (sf2-safe 22.10.0) with ``mode="combined"``.  Wraps
    :exc:`FileNotFoundError` in :exc:`SubmissionAbort` with
    ``reason=AbortReason.INTERIM_MISSING``.

    Parameters
    ----------
    logger : logging.Logger
        Script-level logger; emits an INFO line with row count and index
        bounds (byte-identical to the lecture-script log line).

    Returns
    -------
    pd.DataFrame
        The combined DE-total interim DataFrame indexed by UTC timestamps.

    Raises
    ------
    SubmissionAbort
        With ``reason=AbortReason.INTERIM_MISSING`` when the interim CSV
        does not exist.

    Examples:

        ```{python}
        #| eval: false
        import logging
        from spotforecast2_safe.submission_cli import load_interim_or_abort

        logger = logging.getLogger("example")
        interim = load_interim_or_abort(logger)
        print(interim.shape)
        ```
    """
    import spotforecast2_safe.downloader.entsoe as _entsoe

    try:
        interim = _entsoe.load_interim(mode="combined")
    except FileNotFoundError as exc:
        raise SubmissionAbort(AbortReason.INTERIM_MISSING, str(exc)) from exc
    logger.info(
        "interim: (%d rows, %s -> %s)",
        len(interim),
        interim.index.min(),
        interim.index.max(),
    )
    return interim


def assert_coverage_or_abort(
    interim: "pd.DataFrame",
    dates: "spotforecast2_safe.downloader.entsoe.SubmissionDates",  # type: ignore[name-defined]  # noqa: F821
    *,
    fallback: bool,
    corruption_mode: str,
    deviation_ref: str,
    max_actual_lag_hours: int,
    gap_scan_days: int,
    max_actual_gap_hours: int,
    corruption_policy: str,
    range_mw: float,
    step_mw: float,
    qc_window_days: int,
    deviation_mw: float,
    deviation_slots: int,
    logger: logging.Logger,
) -> "spotforecast2_safe.preprocessing.coverage.SubmissionCoverage":  # type: ignore[name-defined]  # noqa: F821
    """Assert data-quality coverage, aborting on guard violations.

    Delegates to
    :func:`spotforecast2_safe.preprocessing.coverage.assert_submission_coverage`
    (sf2-safe 22.10.0).  The ``corruption_mode`` parameter is **mandatory**:
    team4 passes ``"preview"`` (the authoritative QC happens later inside
    ``prepare_data``), while chronos / macl2l pass ``"authoritative"`` (they
    have no ``prepare_data``, so the coverage check is the only place the
    value-sanity policy applies).

    The ``COVERAGE`` vs ``STALE_CACHE`` split is decided internally by
    checking whether the :exc:`~spotforecast2_safe.exceptions.CoverageError`
    message contains the substring ``"yesterday 23:00 UTC"`` — the same heuristic
    the lecture scripts use.  Callers do NOT need to inspect the message.

    Parameters
    ----------
    interim : pd.DataFrame
        The combined DE-total interim DataFrame.
    dates : SubmissionDates
        Date window from :func:`compute_dates_or_abort`.
    fallback : bool
        ``True`` when a snapshot was restored (enables the staleness gate).
    corruption_mode : str
        ``"preview"`` or ``"authoritative"``; forwarded verbatim to
        ``assert_submission_coverage``.
    deviation_ref : str
        Column name for the deviation reference series (e.g.
        ``"Forecasted Load"``).
    max_actual_lag_hours : int
        Maximum allowed lag between the current time and the last actual.
    gap_scan_days : int
        Number of days back to scan for interior gaps.
    max_actual_gap_hours : int
        Maximum allowed interior gap length (hours).
    corruption_policy : str
        Target-corruption policy (``"truncate"``, ``"heal"``, or ``"abort"``).
    range_mw : float
        Maximum allowed intra-hour MW range (value-sanity guard).
    step_mw : float
        Maximum allowed adjacent-step MW jump (value-sanity guard).
    qc_window_days : int
        QC window length (days) for the deviation guard.
    deviation_mw : float
        Maximum allowed deviation from the reference series (MW).
    deviation_slots : int
        Number of consecutive slots that may exceed ``deviation_mw``.
    logger : logging.Logger
        Accepted for API symmetry with the other helpers; this function does
        not log.

    Returns
    -------
    SubmissionCoverage
        Coverage object whose ``n_steps`` and ``last_full_hour`` drive the
        downstream pipeline.

    Raises
    ------
    SubmissionAbort
        With ``reason=AbortReason.STALE_CACHE`` when the ``CoverageError``
        message contains ``"yesterday 23:00 UTC"`` (stale-snapshot gate);
        ``reason=AbortReason.COVERAGE`` for all other guard violations.

    Examples:

        ```{python}
        #| eval: false
        import logging
        from spotforecast2_safe.submission_cli import assert_coverage_or_abort

        logger = logging.getLogger("example")
        cov = assert_coverage_or_abort(
            interim=interim,
            dates=dates,
            fallback=False,
            corruption_mode="preview",
            deviation_ref="Forecasted Load",
            max_actual_lag_hours=36,
            gap_scan_days=28,
            max_actual_gap_hours=12,
            corruption_policy="truncate",
            range_mw=8_000,
            step_mw=6_000,
            qc_window_days=3,
            deviation_mw=11_000,
            deviation_slots=1,
            logger=logger,
        )
        print(cov.n_steps)
        ```
    """
    import spotforecast2_safe.preprocessing.coverage as _cov
    from spotforecast2_safe.exceptions import CoverageError

    try:
        return _cov.assert_submission_coverage(
            interim,
            dates,
            fallback=fallback,
            corruption_mode=corruption_mode,
            deviation_ref=deviation_ref,
            mode="combined",
            max_actual_lag_hours=max_actual_lag_hours,
            gap_scan_days=gap_scan_days,
            max_actual_gap_hours=max_actual_gap_hours,
            corruption_policy=corruption_policy,
            range_mw=range_mw,
            step_mw=step_mw,
            qc_window_days=qc_window_days,
            deviation_mw=deviation_mw,
            deviation_slots=deviation_slots,
        )
    except CoverageError as exc:
        msg = str(exc)
        if "yesterday 23:00 UTC" in msg:
            raise SubmissionAbort(AbortReason.STALE_CACHE, msg) from exc
        raise SubmissionAbort(AbortReason.COVERAGE, msg) from exc


def entsoe_predictions(
    interim: "pd.DataFrame",
    logger: logging.Logger,
) -> "pd.Series":
    """Return the ENTSO-E day-ahead hourly forecast series (tolerant).

    Delegates to
    :func:`spotforecast2_safe.downloader.entsoe.extract_entsoe_hourly_forecast`
    (sf2-safe 22.7.0).  An all-NaN column is **tolerated**: the library
    raises :exc:`ValueError` in that case; this function logs a WARNING and
    returns an empty :class:`~pandas.Series` instead.  A genuinely missing
    column (:exc:`KeyError`) propagates unchanged.

    This function does **not** abort.  ``preds_entsoe`` feeds only warn-only
    shape checks and diagnostics, never the model or the submission CSV.

    Parameters
    ----------
    interim : pd.DataFrame
        The combined DE-total interim DataFrame.
    logger : logging.Logger
        Script-level logger; emits a WARNING when the forecast is all-NaN
        (byte-identical to the lecture-script log line).

    Returns
    -------
    pd.Series
        Hourly ENTSO-E day-ahead forecast indexed by UTC timestamps, or an
        empty ``pd.Series(name="Forecasted Load", dtype="float64")`` when
        the column is all-NaN.

    Examples:

        ```{python}
        #| eval: false
        import logging
        from spotforecast2_safe.submission_cli import entsoe_predictions

        logger = logging.getLogger("example")
        preds = entsoe_predictions(interim, logger)
        print(preds.head())
        ```
    """
    import pandas as pd
    import spotforecast2_safe.downloader.entsoe as _entsoe

    try:
        return _entsoe.extract_entsoe_hourly_forecast(interim)
    except ValueError:
        logger.warning(
            "ENTSO-E day-ahead Forecasted Load unavailable (all-NaN); "
            "baseline skipped for shape check / diagnostics."
        )
        return pd.Series(name="Forecasted Load", dtype="float64")


def download_combined_or_abort(
    api_key: str,
    dates: "spotforecast2_safe.downloader.entsoe.SubmissionDates",  # type: ignore[name-defined]  # noqa: F821
    *,
    max_retries: int,
    backoff: float,
    timeout: "float | None",
    fallback_enabled: bool,
    country_code: str,
    logger: logging.Logger,
) -> bool:
    """Download the combined DE-total series with snapshot fallback, or abort.

    Delegates to
    :func:`spotforecast2_safe.downloader.resilience.download_combined_with_fallback`
    (sf2-safe 22.10.0).  Returns ``True`` when a snapshot was restored
    (triggers the yesterday-23:00 freshness gate downstream via
    ``result.fallback_gate``).  Raises :exc:`SubmissionAbort` with
    ``reason=AbortReason.DOWNLOAD_UNRECOVERABLE`` when ``result.mode is
    None`` (live download exhausted and no valid combined snapshot within
    the TTL).

    Parameters
    ----------
    api_key : str
        ENTSO-E Web API security token.
    dates : SubmissionDates
        Date window from :func:`compute_dates_or_abort`; supplies
        ``start_dl``, ``end_dl``, and ``now``.
    max_retries : int
        Number of outer download attempts.
    backoff : float
        Base seconds for exponential backoff between outer attempts
        (``backoff * 2**(attempt-1)``).
    timeout : float or None
        Per-socket read timeout (seconds); ``None`` disables.
    fallback_enabled : bool
        When ``False``, cached snapshots are never READ as a fallback
        (live data or abort), but they are still written on success.
    country_code : str
        ENTSO-E country code (e.g. ``"DE"``).
    logger : logging.Logger
        Accepted for API symmetry with the other helpers; this function does
        not log.

    Returns
    -------
    bool
        ``True`` when ``result.fallback_gate`` is set (a snapshot was
        restored); ``False`` when fresh live data was used.

    Raises
    ------
    SubmissionAbort
        With ``reason=AbortReason.DOWNLOAD_UNRECOVERABLE`` when the live
        download is exhausted and no valid snapshot is available.

    Examples:

        ```{python}
        #| eval: false
        import logging
        import pandas as pd

        from spotforecast2_safe.submission_cli import (
            compute_dates_or_abort,
            download_combined_or_abort,
        )

        logger = logging.getLogger("example")
        dates = compute_dates_or_abort(
            now=pd.Timestamp.now(tz="UTC"),
            data_end=None,
            train_years=3,
            start_margin_days=45,
            logger=logger,
        )
        fallback_gate = download_combined_or_abort(
            api_key="YOUR_API_KEY",
            dates=dates,
            max_retries=5,
            backoff=5.0,
            timeout=60.0,
            fallback_enabled=True,
            country_code="DE",
            logger=logger,
        )
        print("snapshot restored:", fallback_gate)
        ```
    """
    from spotforecast2_safe.downloader import resilience as _resil

    result = _resil.download_combined_with_fallback(
        api_key,
        start=dates.start_dl,
        end=dates.end_dl,
        now=dates.now,
        max_retries=max_retries,
        backoff=backoff,
        timeout=timeout,
        fallback_enabled=fallback_enabled,
        country_code=country_code,
    )
    if result.mode is None:
        raise SubmissionAbort(
            AbortReason.DOWNLOAD_UNRECOVERABLE,
            (
                "live download exhausted and no valid combined snapshot "
                f"within {_resil.SNAPSHOT_TTL}; operator intervention required."
                if fallback_enabled
                else "live download exhausted and --no-fallback is set."
            ),
        )
    return result.fallback_gate
