# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Target-side corruption detector and abort/heal/truncate policy.

ENTSO-E 15-min Actual Load occasionally contains physically-impossible
multi-GW intra-hour dropouts (e.g. 2026-06-03..05 incident) that poison
the recursive forecaster's anchor and lag features.  This module provides
a shared dynamics detector over the native-cadence (15-min) series and a
policy knob to handle detected corruptions.

Empirical gating: ENTSO-E does not retroactively correct gross intra-hour
dropouts of this class.  The recommended episode policy is ``"truncate"``;
``"heal"`` is safe only for short interior spans far from the anchor zone
(and is refused by design when either condition is violated); ``"abort"``
surfaces the error immediately for manual review.

By default all six knobs are off / at safe defaults, so the pipeline is
byte-identical to the pre-feature baseline.

Public API (re-exported from ``preprocessing.__init__``):
    - :class:`TargetCorruptionReport`
    - :func:`detect_target_corruption`
    - :func:`apply_target_corruption_policy`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from spotforecast2_safe.exceptions import TargetCorruptionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetCorruptionReport:
    """Immutable summary of a target-corruption detection and policy run.

    Args:
        fired: ``True`` when the detector identified at least one flagged
            hour-slot; ``False`` means no corruption was found (or the
            detector was not configured).
        n_flagged_cells: Number of native-cadence (e.g. 15-min) slots that
            were flagged across all target columns.
        n_flagged_hours: Number of distinct calendar hours that contain at
            least one flagged slot.
        spans: Contiguous flagged hour-runs as ``(start_iso, end_iso)``
            string pairs.  Empty list when the detector did not fire.
        action: The policy action taken: one of ``"abort"``, ``"heal"``,
            ``"truncate"``, or ``"noop"``.
        first_flagged_hour: The earliest flagged calendar hour, as a
            tz-aware ``pd.Timestamp`` floored to the hour, or ``None``
            when the detector did not fire.
        pre_policy_last_target: The last observed-target timestamp BEFORE
            any policy mutation.  ``None`` when the detector did not fire.
            Used by the ``"truncate"`` path in ``prepare_data`` to compute
            how many hours were retracted so that ``predict_size`` can be
            bumped idempotently.

    Examples:
        ```{python}
        from spotforecast2_safe.preprocessing.target_corruption import (
            TargetCorruptionReport,
        )

        report = TargetCorruptionReport(
            fired=False,
            n_flagged_cells=0,
            n_flagged_hours=0,
            spans=[],
            action="noop",
            first_flagged_hour=None,
            pre_policy_last_target=None,
        )
        print(report.fired, report.action)
        assert not report.fired
        assert report.action == "noop"
        ```
    """

    fired: bool
    n_flagged_cells: int
    n_flagged_hours: int
    spans: List[Tuple[str, str]]
    action: str
    first_flagged_hour: Optional[pd.Timestamp]
    pre_policy_last_target: Optional[pd.Timestamp]


def _compute_spans(
    flagged_hours: pd.DatetimeIndex,
) -> List[Tuple[str, str]]:
    """Collapse a sorted DatetimeIndex of flagged hours into contiguous runs.

    Args:
        flagged_hours: Sorted ``DatetimeIndex`` of hour-floored timestamps.

    Returns:
        List of ``(start_iso, end_iso)`` string pairs, one per contiguous
        run.  An empty list is returned when ``flagged_hours`` is empty.
    """
    if len(flagged_hours) == 0:
        return []

    spans: List[Tuple[str, str]] = []
    run_start = flagged_hours[0]
    prev = flagged_hours[0]
    for ts in flagged_hours[1:]:
        if ts - prev > pd.Timedelta(hours=1):
            spans.append((run_start.isoformat(), prev.isoformat()))
            run_start = ts
        prev = ts
    spans.append((run_start.isoformat(), prev.isoformat()))
    return spans


def detect_target_corruption(
    df: pd.DataFrame,
    *,
    targets: Sequence[str],
    range_mw: Optional[float],
    step_mw: Optional[float],
    window_days: Optional[int],
) -> pd.Series:
    """Detect physically-impossible target-column corruption in the native frame.

    Applies two independent rules on the native-cadence (e.g. 15-min) series
    within a rolling look-back window ending at the last observed target
    timestamp:

    - **Range rule** (sub-hourly cadence only): an hour is flagged when
      ``intra-hour max - intra-hour min > range_mw`` for any target column.
      Vacuously skipped for hourly-or-coarser cadence (intra-hour range is
      undefined on a single slot per hour).
    - **Step rule**: an hour is flagged when any ``|adjacent-slot diff|``
      that *touches* that hour exceeds ``step_mw`` for any target column.
      Applies to all cadences.

    Flags are OR-ed across target columns.  ALL native-cadence slots of a
    flagged calendar hour are marked ``True`` in the returned boolean
    ``Series``, so downstream NaN-ing operates on full hours rather than
    individual sub-hourly slots.

    The detector is **inert** (returns all-``False``) unless ``window_days``
    is set AND at least one of ``range_mw`` / ``step_mw`` is set.  If the
    data is shorter than ``window_days``, the window is clamped to
    ``df.index.min()`` without raising.

    Args:
        df: Native-cadence ``DataFrame`` indexed by a ``DatetimeIndex``.
            Must contain all columns listed in ``targets``.
        targets: Sequence of target column names to inspect.
        range_mw: Maximum allowed intra-hour range (MW).  ``None`` skips the
            range rule.
        step_mw: Maximum allowed absolute adjacent-slot difference (MW).
            ``None`` skips the step rule.
        window_days: Number of days before the last observed target to
            include in the scan.  ``None`` makes the detector inert.

    Returns:
        Boolean ``pd.Series`` aligned to ``df.index``.  ``True`` means the
        slot belongs to a flagged calendar hour.  All-``False`` when the
        detector is inert or no corruption is found.

    Examples:
        ```{python}
        import pandas as pd
        import numpy as np
        from spotforecast2_safe.preprocessing.target_corruption import (
            detect_target_corruption,
        )

        # 15-min cadence; one GW dropout at 12:15 inside the window
        idx = pd.date_range("2026-06-03", periods=48, freq="15min", tz="UTC")
        vals = [55_000.0] * 48
        vals[5] = 44_000.0          # 11 GW step drop  -> flags 12:00 hour
        df = pd.DataFrame({"load": vals}, index=idx)

        mask = detect_target_corruption(
            df, targets=["load"], range_mw=5_000, step_mw=8_000, window_days=3
        )
        # Slots in the 12:00 hour (index 4-7) are flagged
        assert mask.iloc[4:8].all(), "Slots in the flagged hour must be True"
        assert not mask.iloc[8:].any(), "Subsequent clean slots must be False"
        print("flagged:", mask.sum(), "slots")
        ```
    """
    # Detector is inert when the caller has not configured it.
    if window_days is None or (range_mw is None and step_mw is None):
        return pd.Series(False, index=df.index)

    # Derive cadence via mode of consecutive diffs (robust to irregular index).
    diffs = df.index.to_series().diff().dropna()
    if len(diffs) == 0:
        return pd.Series(False, index=df.index)
    cadence: pd.Timedelta = diffs.mode().iloc[0]
    is_sub_hourly = cadence < pd.Timedelta(hours=1)

    # Restrict the scan to the last `window_days` before the last observed target.
    last_target_ts = df[list(targets)].dropna(how="all").index.max()
    if pd.isna(last_target_ts):
        return pd.Series(False, index=df.index)

    window_start = max(df.index.min(), last_target_ts - pd.Timedelta(days=window_days))
    scan = df.loc[window_start:last_target_ts, list(targets)]

    # hour_flag will accumulate which hour-floored timestamps are corrupt.
    flagged_hours: set = set()

    for col in targets:
        if col not in scan.columns:
            continue
        series = scan[col].dropna()
        if len(series) < 2:
            continue

        # --- range rule (sub-hourly only) ---
        if is_sub_hourly and range_mw is not None:
            hourly_range = series.resample("h").agg(lambda x: x.max() - x.min())
            bad_hours = hourly_range.index[hourly_range > range_mw]
            flagged_hours.update(bad_hours.tolist())

        # --- step rule (all cadences) ---
        if step_mw is not None:
            abs_diff = series.diff().abs()
            # A step between slot i-1 and slot i flags the hour of slot i
            # *and* the hour of slot i-1.
            bad_mask = abs_diff > step_mw
            for ts in abs_diff.index[bad_mask]:
                flagged_hours.add(ts.floor("h"))
                # also flag the predecessor's hour
                loc = series.index.get_loc(ts)
                if loc > 0:
                    flagged_hours.add(series.index[loc - 1].floor("h"))

    if not flagged_hours:
        return pd.Series(False, index=df.index)

    # Expand flagged hours back to all native-cadence slots of that hour.
    floored = df.index.floor("h")
    flagged_set = set(flagged_hours)
    mask = pd.Series([fh in flagged_set for fh in floored], index=df.index, dtype=bool)
    return mask


def apply_target_corruption_policy(
    df: pd.DataFrame,
    *,
    targets: Sequence[str],
    policy: str,
    range_mw: Optional[float],
    step_mw: Optional[float],
    window_days: Optional[int],
    max_heal_hours: int,
    anchor_zone_hours: int,
    cutoff: Optional[pd.Timestamp],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, TargetCorruptionReport]:
    """Apply the configured corruption policy to the native-cadence frame.

    This is the single entry point for the target-corruption sub-pipeline.
    It runs :func:`detect_target_corruption` and then dispatches to the
    ``policy`` branch.  The function always acts only on target columns;
    exogenous columns are never modified.

    Policy semantics:

    - **noop** (detector inert or no flags): returns the **same** ``df``
      object (no copy) and a report with ``action="noop"``.
    - **abort**: raises :class:`~spotforecast2_safe.exceptions.TargetCorruptionError`
      with the span list and operator guidance.
    - **heal**: refuses and raises when (a) any flagged slot lies within
      ``anchor_zone_hours`` of ``cutoff``, or (b) ``n_flagged_hours >
      max_heal_hours``.  Otherwise sets flagged target slots to ``NaN`` so
      that ``apply_imputation("weighted_interp")`` can interpolate and
      zero-weight them.
    - **truncate**: sets ALL target columns to ``NaN`` from
      ``first_flagged_hour`` onward.  The existing 16.3.1 trailing-clamp in
      ``prepare_data`` then retracts ``data_end`` to the new last observed
      target.

    Args:
        df: Native-cadence ``DataFrame`` indexed by a ``DatetimeIndex``.
        targets: Target column names.
        policy: One of ``"abort"``, ``"heal"``, ``"truncate"``.
        range_mw: Range-rule threshold (MW); ``None`` skips that rule.
        step_mw: Step-rule threshold (MW); ``None`` skips that rule.
        window_days: Look-back window for the detector (days); ``None``
            makes the detector inert.
        max_heal_hours: Maximum number of flagged hours the ``"heal"``
            policy will accept.  ``0`` effectively disables healing.
        anchor_zone_hours: Hours before ``cutoff`` that are protected from
            healing.  Flagged slots inside this zone force a refusal.
            Default is ``168`` (one week).
        cutoff: The effective training cutoff timestamp used for the
            anchor-zone check.  ``None`` disables the zone check.
        logger: Standard-library logger for WARNING/INFO messages.

    Returns:
        Tuple of ``(df_out, report)`` where ``df_out`` is either the
        original ``df`` object (noop) or a mutated copy (heal/truncate),
        and ``report`` is a :class:`TargetCorruptionReport`.

    Raises:
        TargetCorruptionError: On ``policy="abort"`` when flags are found,
            or on ``policy="heal"`` when the heal guard refuses.

    Examples:
        ```{python}
        import logging
        import pandas as pd
        import numpy as np
        from spotforecast2_safe.preprocessing.target_corruption import (
            apply_target_corruption_policy,
        )

        log = logging.getLogger("demo")
        idx = pd.date_range("2026-06-03", periods=96, freq="15min", tz="UTC")
        vals = [55_000.0] * 96
        vals[5] = 44_000.0  # 11 GW step -> flags the 00:00 hour
        df = pd.DataFrame({"load": vals}, index=idx)

        df_out, report = apply_target_corruption_policy(
            df,
            targets=["load"],
            policy="truncate",
            range_mw=5_000,
            step_mw=8_000,
            window_days=3,
            max_heal_hours=0,
            anchor_zone_hours=168,
            cutoff=None,
            logger=log,
        )
        assert report.fired
        assert report.action == "truncate"
        # All target slots from first_flagged_hour onward are NaN
        nanned = df_out.loc[report.first_flagged_hour:, "load"]
        assert nanned.isna().all()
        print("action:", report.action, "flagged hours:", report.n_flagged_hours)
        ```
    """
    # Run detector.
    flag_mask = detect_target_corruption(
        df,
        targets=targets,
        range_mw=range_mw,
        step_mw=step_mw,
        window_days=window_days,
    )

    if not flag_mask.any():
        return df, TargetCorruptionReport(
            fired=False,
            n_flagged_cells=0,
            n_flagged_hours=0,
            spans=[],
            action="noop",
            first_flagged_hour=None,
            pre_policy_last_target=None,
        )

    # Compute summary stats from the mask.
    n_flagged_cells = int(flag_mask.sum())
    flagged_index = df.index[flag_mask]
    flagged_hours_index = pd.DatetimeIndex(sorted(set(flagged_index.floor("h"))))
    n_flagged_hours = len(flagged_hours_index)
    spans = _compute_spans(flagged_hours_index)
    first_flagged_hour = flagged_hours_index[0]

    # Record the last observed target BEFORE any mutation.
    target_df = df[list(targets)]
    pre_policy_last_target = target_df.dropna(how="all").index.max()
    if pd.isna(pre_policy_last_target):
        pre_policy_last_target = None

    # --- abort ---
    if policy == "abort":
        spans_str = str(spans[:3])
        raise TargetCorruptionError(
            f"target_corruption[abort]: {n_flagged_hours} corrupt hour(s) detected "
            f"in {len(spans)} span(s) (first 3: {spans_str}). "
            "ENTSO-E does not retroactively correct gross intra-hour dropouts of "
            "this class; the corruption will not heal itself: wait for it to age "
            "out of the QC window, or switch target_corruption_policy."
        )

    # --- heal ---
    if policy == "heal":
        # Check anchor-zone refusal.
        if cutoff is not None and anchor_zone_hours > 0:
            zone_start = cutoff - pd.Timedelta(hours=anchor_zone_hours)
            if bool((flag_mask & (df.index >= zone_start)).any()):
                raise TargetCorruptionError(
                    f"target_corruption[heal]: {n_flagged_hours} flagged hour(s) "
                    f"lie within the anchor zone ({anchor_zone_hours} h before "
                    f"cutoff={cutoff}). The anchor zone cannot be protected by "
                    "weights alone — use policy='truncate' instead."
                )
        # Check budget refusal.
        if n_flagged_hours > max_heal_hours:
            raise TargetCorruptionError(
                f"target_corruption[heal]: {n_flagged_hours} flagged hour(s) exceed "
                f"the heal budget of {max_heal_hours} h. Increase "
                "'target_max_heal_hours' or use policy='truncate'."
            )

        spans_str = str(spans[:3])
        logger.warning(
            "target_corruption[heal]: flagged %d cell(s) in %d span(s) -> NaN "
            "for weighted_interp imputation (budget %d h); spans: %s",
            n_flagged_cells,
            len(spans),
            max_heal_hours,
            spans_str,
        )
        df_out = df.copy()
        for col in targets:
            if col in df_out.columns:
                df_out.loc[flag_mask, col] = float("nan")
        return df_out, TargetCorruptionReport(
            fired=True,
            n_flagged_cells=n_flagged_cells,
            n_flagged_hours=n_flagged_hours,
            spans=spans,
            action="heal",
            first_flagged_hour=first_flagged_hour,
            pre_policy_last_target=pre_policy_last_target,
        )

    # --- truncate ---
    if policy == "truncate":
        # NaN all target columns from first_flagged_hour onward.
        truncate_mask = df.index >= first_flagged_hour
        logger.warning(
            "target_corruption[truncate]: corrupt target tail from %s; "
            "NaN-ed %d hour(s), trailing clamp will retract data_end",
            first_flagged_hour,
            n_flagged_hours,
        )
        df_out = df.copy()
        for col in targets:
            if col in df_out.columns:
                df_out.loc[truncate_mask, col] = float("nan")
        return df_out, TargetCorruptionReport(
            fired=True,
            n_flagged_cells=n_flagged_cells,
            n_flagged_hours=n_flagged_hours,
            spans=spans,
            action="truncate",
            first_flagged_hour=first_flagged_hour,
            pre_policy_last_target=pre_policy_last_target,
        )

    # Should not be reached; validate_config guards the policy value.
    raise ValueError(  # pragma: no cover
        f"Unknown target_corruption_policy: {policy!r}. "
        "Expected one of: 'abort', 'heal', 'truncate'."
    )
