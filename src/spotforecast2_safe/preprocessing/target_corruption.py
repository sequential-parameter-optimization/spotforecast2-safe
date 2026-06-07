# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Target-side corruption detector and abort/heal/truncate policy.

ENTSO-E 15-min Actual Load occasionally contains physically-impossible
multi-GW intra-hour dropouts (e.g. 2026-06-03..05 incident) that poison
the recursive forecaster's anchor and lag features.  This module provides
a shared detector over the native-cadence (15-min) series — two dynamics
rules (intra-hour range, adjacent-slot step) plus a deviation rule against
a published reference column such as the day-ahead load forecast — and a
policy knob to handle detected corruptions.

The deviation rule exists because the dropout class can stay *below* the
dynamics thresholds while sitting many GW under the day-ahead forecast
(observed 2026-06-07: 5.6 GW steps under a 6 GW limit at Actual − Forecast
= −11.6 GW).  Level-vs-reference is the discriminator dynamics cannot see.

Empirical gating: ENTSO-E does not retroactively correct gross intra-hour
dropouts of this class.  The recommended episode policy is ``"truncate"``;
``"heal"`` is safe only for short interior spans far from the anchor zone
(and is refused by design when either condition is violated); ``"abort"``
surfaces the error immediately for manual review.

By default all knobs are off / at safe defaults, so the pipeline is
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
    deviation_mw: Optional[float] = None,
    deviation_ref: Optional[str] = None,
    deviation_slots: int = 2,
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
    - **Deviation rule** (dropout-only, all cadences): an hour is flagged
      when ``target − reference < -deviation_mw`` holds for at least
      ``deviation_slots`` *consecutive* native-cadence slots within the
      scan window, where the reference is a published companion column
      such as the ENTSO-E day-ahead ``"Forecasted Load"``.  The rule is
      asymmetric by design: the known corruption class is exclusively a
      dropout *below* the day-ahead forecast, while actuals above the
      forecast are ordinary under-forecasting.  ``NaN`` in either column
      yields a ``NaN`` difference, which compares ``False`` — so the
      publication-lag frontier (forecast published, actual not yet) never
      flags, and a data gap breaks a consecutive run.  On hourly-or-coarser
      cadence the sustained requirement collapses to a single slot.  The
      rule is silently skipped when ``deviation_ref`` is missing from the
      frame (mirroring how absent target columns are skipped).

    Flags are OR-ed across target columns.  ALL native-cadence slots of a
    flagged calendar hour are marked ``True`` in the returned boolean
    ``Series``, so downstream NaN-ing operates on full hours rather than
    individual sub-hourly slots.

    The detector is **inert** (returns all-``False``) unless ``window_days``
    is set AND at least one of ``range_mw`` / ``step_mw`` / ``deviation_mw``
    is set.  If the data is shorter than ``window_days``, the window is
    clamped to ``df.index.min()`` without raising.

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
        deviation_mw: Maximum allowed dropout below the reference column
            (MW, positive magnitude): slots with
            ``target − reference < -deviation_mw`` are candidates.
            ``None`` skips the deviation rule.
        deviation_ref: Name of the reference column (e.g.
            ``"Forecasted Load"``).  The rule is skipped when ``None`` or
            when the column is absent from ``df``.  The reference column
            itself is never checked as a target by this rule.
        deviation_slots: Minimum number of *consecutive* sub-hourly slots
            the dropout must sustain before any hour is flagged (default
            ``2`` — a single-slot blip is more likely a metering glitch
            than the oscillating dropout class).  Clamped to ``1`` on
            hourly-or-coarser cadence.

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

        ```{python}
        # Deviation rule: a sub-threshold dropout the dynamics rules miss.
        import pandas as pd
        import numpy as np
        from spotforecast2_safe.preprocessing.target_corruption import (
            detect_target_corruption,
        )

        idx = pd.date_range("2026-06-07", periods=16, freq="15min", tz="UTC")
        forecast = pd.Series(48_000.0, index=idx)
        actual = forecast.copy()
        # Two consecutive slots 11.6 GW below the forecast, stepping by
        # only 5.8 GW per slot — below a 6 GW step rule, no range breach.
        actual.iloc[4] = forecast.iloc[4] - 5_800.0
        actual.iloc[5] = forecast.iloc[5] - 11_600.0
        actual.iloc[6] = forecast.iloc[6] - 11_600.0
        actual.iloc[7] = forecast.iloc[7] - 5_800.0
        # Publication-lag frontier: forecast published, actual not yet.
        actual.iloc[12:] = np.nan
        df = pd.DataFrame({"Actual Load": actual, "Forecasted Load": forecast})

        dyn_only = detect_target_corruption(
            df, targets=["Actual Load"],
            range_mw=15_000, step_mw=6_000, window_days=3,
        )
        with_dev = detect_target_corruption(
            df, targets=["Actual Load"],
            range_mw=15_000, step_mw=6_000, window_days=3,
            deviation_mw=8_000, deviation_ref="Forecasted Load",
        )
        assert not dyn_only.any(), "dynamics rules miss the dropout"
        assert with_dev.iloc[4:8].any(), "deviation rule catches it"
        assert not with_dev.iloc[12:].any(), "NaN frontier never flags"
        print("dynamics-only:", int(dyn_only.sum()), "| with deviation:",
              int(with_dev.sum()))
        ```
    """
    # Detector is inert when the caller has not configured it.
    if window_days is None or (
        range_mw is None and step_mw is None and deviation_mw is None
    ):
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
    # The deviation rule needs its reference column inside the scan slice
    # (same window, same UTC view) even when it is not a target.
    scan_cols = list(targets)
    if (
        deviation_mw is not None
        and deviation_ref is not None
        and deviation_ref in df.columns
        and deviation_ref not in scan_cols
    ):
        scan_cols.append(deviation_ref)
    scan = df.loc[window_start:last_target_ts, scan_cols]
    # DST safety: floor("h")/resample("h") on a tz-aware non-UTC index can
    # raise on ambiguous wall times (fall-back hour).  All hour bucketing is
    # therefore done on a UTC view; positions are unchanged, and for the
    # whole-hour-offset zones ENTSO-E uses, UTC hour bins coincide with wall
    # hour bins.  Naive indexes pass through untouched.
    if scan.index.tz is not None and str(scan.index.tz) != "UTC":
        scan = scan.tz_convert("UTC")

    # hour_flag will accumulate which hour-floored timestamps are corrupt.
    flagged_hours: set = set()

    for col in targets:
        if col not in scan.columns:
            continue
        series = scan[col]
        if series.notna().sum() < 2:
            continue

        # --- range rule (sub-hourly only) ---
        if is_sub_hourly and range_mw is not None:
            hourly_range = series.resample("h").agg(lambda x: x.max() - x.min())
            bad_hours = hourly_range.index[hourly_range > range_mw]
            flagged_hours.update(bad_hours.tolist())

        # --- step rule (all cadences) ---
        if step_mw is not None:
            abs_diff = series.diff().abs()
            # Only slot pairs exactly one cadence apart are "adjacent": a
            # value change across a data gap (missing rows) or a duplicate
            # timestamp is not a step — a legitimate ramp can span a gap and
            # must not flag.  NaN values yield NaN diffs, which compare
            # False, so gap *boundaries* (NaN -> value) never flag either —
            # matching the chapter tripwire's semantics.
            adjacent = scan.index.to_series().diff() == cadence
            bad_mask = (abs_diff > step_mw) & adjacent
            # A step between slot i-1 and slot i flags the hour of slot i
            # *and* the hour of slot i-1 (the left endpoint of the step lives
            # one cadence earlier).
            for ts in series.index[bad_mask]:
                flagged_hours.add(ts.floor("h"))
                flagged_hours.add((ts - cadence).floor("h"))

    # --- deviation rule (dropout vs reference, all cadences) ---
    # Silently skipped when the ref column is absent, mirroring the
    # `col not in scan.columns: continue` treatment of target columns.
    if (
        deviation_mw is not None
        and deviation_ref is not None
        and deviation_ref in scan.columns
    ):
        ref = scan[deviation_ref]
        # A single hourly slot already aggregates the dropout; the
        # consecutive-slots requirement is meaningful sub-hourly only.
        eff_slots = max(1, int(deviation_slots)) if is_sub_hourly else 1
        for col in targets:
            if col not in scan.columns or col == deviation_ref:
                continue
            # NaN in either column -> NaN difference -> compares False, so
            # the publication-lag frontier (forecast published, actual not
            # yet) never flags and a gap breaks a consecutive run.
            below = (scan[col] - ref < -float(deviation_mw)).to_numpy()
            if eff_slots > 1:
                # sustained[i]: slots i-eff_slots+1 .. i are all below.
                sustained = below.copy()
                for k in range(1, eff_slots):
                    sustained[k:] &= below[:-k]
                sustained[: eff_slots - 1] = False
                # Flag every slot of each qualifying run, not just its end.
                run_mask = sustained.copy()
                for k in range(1, eff_slots):
                    run_mask[:-k] |= sustained[k:]
            else:
                run_mask = below
            # A deviation is a level property of the slot itself — no
            # predecessor-hour semantics (unlike the step rule).
            for ts in scan.index[run_mask]:
                flagged_hours.add(ts.floor("h"))

    if not flagged_hours:
        return pd.Series(False, index=df.index)

    # Expand flagged hours back to all native-cadence slots of that hour
    # (positions are identical between df.index and the UTC view).
    if df.index.tz is not None and str(df.index.tz) != "UTC":
        floored = df.index.tz_convert("UTC").floor("h")
    else:
        floored = df.index.floor("h")
    mask = pd.Series(
        floored.isin(pd.DatetimeIndex(sorted(flagged_hours))),
        index=df.index,
        dtype=bool,
    )
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
    deviation_mw: Optional[float] = None,
    deviation_ref: Optional[str] = None,
    deviation_slots: int = 2,
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
        deviation_mw: Deviation-rule threshold (MW, positive magnitude):
            flags sustained dropouts ``target − reference < -deviation_mw``.
            ``None`` skips that rule.  See `detect_target_corruption`.
        deviation_ref: Reference column name for the deviation rule (e.g.
            ``"Forecasted Load"``).  When enabling this rule, scope
            ``targets`` to the actuals only (e.g. ``["Actual Load"]``) so
            that ``heal``/``truncate`` NaN only the actual and the
            reference survives as an exogenous prior.
        deviation_slots: Minimum consecutive sub-hourly slots for the
            deviation rule (default ``2``).

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
        deviation_mw=deviation_mw,
        deviation_ref=deviation_ref,
        deviation_slots=deviation_slots,
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

    # Compute summary stats from the mask (hour bucketing on a UTC view for
    # DST safety, mirroring detect_target_corruption).
    n_flagged_cells = int(flag_mask.sum())
    flagged_index = df.index[flag_mask]
    if flagged_index.tz is not None and str(flagged_index.tz) != "UTC":
        flagged_index = flagged_index.tz_convert("UTC")
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
