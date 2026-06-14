# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download-resilience decision tree for the spotforecast2-safe downloader.

This module is a first-class public submodule of
``spotforecast2_safe.downloader``.  Consumers import it as::

    from spotforecast2_safe.downloader import resilience as resil

    result = resil.download_with_fallback(
        api_key, start=..., end=..., now=..., max_retries=3,
        backoff=5.0, timeout=60.0, fallback_enabled=True,
    )

Design rationale.
    ENTSO-E's per-zone endpoint (``download_zone_loads``) can fail for
    individual control areas; ``on_zone_failure="collect"`` mode returns a
    structured ``ZoneResult`` per zone rather than raising.  This module
    consumes that API and adds the snapshot-fallback policy:

    1. Bootstrap the snapshot store by seeding from existing interim files
       (``SnapshotStore.seed_from_file`` per kind).
    2. Per outer retry round (``max_retries`` rounds total), call
       ``download_zone_loads(..., on_zone_failure="collect")`` for the zones
       that have not yet succeeded; one ``backoff * 2**(round-1)`` sleep after
       each failed round before the next; merge ``ZoneResult`` objects across
       rounds.
    3. Zone failure after all attempts: if a valid snapshot exists (within
       TTL), restore it and record status ``"cache"`` with age; otherwise
       ``"missing"``.
    4. If all four zones are in {live, cache}: ``mode="four_zone"``, done.
    5. Else (>= 1 ``"missing"``): try the live aggregated DE-total
       (``download_new_data``).  Success: snapshot + status ``"live"``.
       Fail + valid snapshot: restore + status ``"cache"``.
       Fail + no snapshot: status ``"missing"``.
    6. Combined in {live, cache}: ``mode="combined"``; else ``mode=None``
       (caller raises ``Abort(5, result.report())``).
    7. Prune the snapshot store at the end.

    The library (``spotforecast2_safe``) owns the mechanics: atomic writes,
    TTL checks, seed-from-file, restore.  This module owns the policy: when to
    fall back, what to log, what the decision tree means for the submission.

Snapshot store location.
    ``<data_home>/snapshots/``, NOT the cache home (the multitask clean task
    rmtrees the cache home; data home survives across cleaning runs).
    Sub-directories: ``snapshots/zone_<col>/`` for the four zones,
    ``snapshots/combined/`` for the combined series.  The ``SnapshotStore``
    manages timestamps, TTL, and atomicity.

Threat model (STRIDE).
    This module indirectly crosses a network boundary (by delegating to
    ``entsoe_module.download_zone_loads`` and ``download_new_data``) and
    directly crosses a local-filesystem trust boundary (by delegating to
    ``SnapshotStore.restore`` which writes a cached file into the trusted
    interim directory).  Contributors who change either surface MUST update
    this table in the same pull request; the rule is anchored in
    CONTRIBUTING.md ("Threat-model update rule").

    Data flow 1: outbound ENTSO-E HTTPS requests — delegated to
    ``entsoe_module.download_zone_loads`` / ``download_new_data`` (see
    ``downloader/entsoe.py``).  All TLS, retry, schema-validation, and
    token-redaction countermeasures live in ``entsoe.py``; the STRIDE analysis
    for that flow is recorded there.  This module's role is purely
    orchestration: it invokes those functions, catches every exception, and
    decides which fallback path to take.

    - Spoofing: delegated to entsoe.py (TLS + fixed endpoint).
    - Tampering: delegated to entsoe.py (TLS integrity + schema parser).
    - Repudiation: this module emits ``logging`` records for every attempt,
      retry, fallback decision, and snapshot restore.  No request-audit log;
      operators needing non-repudiation must capture these records at the
      host/SIEM level.
    - Information Disclosure: the ``api_key`` is passed directly to the
      entsoe module and is never logged by this module.
    - Denial of Service: bounded by the caller-supplied ``max_retries`` and
      ``backoff`` parameters; after the budget is exhausted the function
      returns a ``DownloadResult`` rather than looping silently.
    - Elevation of Privilege: no setuid boundary.  Not applicable.

    Data flow 2: local filesystem write — ``SnapshotStore.restore`` copies a
    cached snapshot back into ``interim/``, which is consumed by the
    downstream pipeline as trusted data.

    - Tampering: an attacker with write access to ``<data_home>/snapshots/``
      could plant a forged or stale snapshot file.  Countermeasures: (a) the
      TTL bound (``SNAPSHOT_TTL = 72 h``) rejects files whose embedded
      timestamp is too old; (b) snapshot filenames embed the UTC write
      timestamp so age forgery requires modifying the filename; (c) atomic
      writes in ``SnapshotStore`` prevent torn reads; (d) the data-home
      directory is under user control — operators must restrict write
      permissions to the pipeline user.  This module does not set filesystem
      permissions on behalf of the caller.
    - Information Disclosure: snapshot files may contain timestamped load
      series.  Not sensitive in isolation; operators running against non-public
      data must configure data-directory permissions.
    - Other STRIDE categories: not applicable for a local filesystem flow
      whose threat model is owned by the host operating system.

Note:
    This module MUST NOT import anything from ``spotforecast2`` (the full
    package) and MUST NOT import ``spotforecast2_safe`` at module top level
    (sf2-safe reads environment variables at call time;
    ``configure_environment()`` must run first). All sf2-safe imports are
    inside functions, referenced as attribute lookups on the lazily-imported
    module so monkeypatching works in tests.
"""

from __future__ import annotations

import logging
import time as _time_module
from dataclasses import dataclass
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# Single source of truth for the four zone column names.
# The scripts import this instead of defining their own ZONE_COLUMNS constant.
ZONE_COLUMNS: list[str] = [
    "load_amprion",
    "load_tennet",
    "load_transnetbw",
    "load_50hertz",
]

# Snapshot time-to-live: files older than this relative to ``now`` are ignored.
SNAPSHOT_TTL = pd.Timedelta(hours=72)

# Mapping from zone column name to ENTSO-E area identifier.
# Defined here so we never need to import GERMAN_TSO_ZONES at module top.
_ZONE_AREAS: dict[str, str] = {
    "load_amprion": "DE_AMPRION",
    "load_tennet": "DE_TENNET",
    "load_transnetbw": "DE_TRANSNET",
    "load_50hertz": "DE_50HZ",
}


# ---------------------------------------------------------------------------
# Snapshot store factory (lazy: data_home is resolved at call time)
# ---------------------------------------------------------------------------


def make_store() -> "SnapshotStore":  # type: ignore[name-defined]  # noqa: F821
    """Build a ``SnapshotStore`` rooted at ``<data_home>/snapshots/``.

    Lazy so that ``configure_environment()`` runs before ``get_data_home()``
    is called. Public: ``chronos.py`` builds its combined-DE-total fallback
    on the same store, so the snapshot root and TTL conventions stay defined
    once, here.

    Returns:
        SnapshotStore: A TTL-aware atomic snapshot store rooted at
        ``<data_home>/snapshots`` with a time-to-live of ``SNAPSHOT_TTL``.
    """
    from spotforecast2_safe.data.fetch_data import get_data_home
    from spotforecast2_safe.utils.snapshot_store import SnapshotStore

    return SnapshotStore(root=get_data_home() / "snapshots", ttl=SNAPSHOT_TTL)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ZoneOutcome:
    """Result for one zone's download attempt.

    Attributes:
        column: Zone column name (e.g. ``"load_amprion"``).
        status: One of ``"live"`` (fresh download), ``"cache"`` (snapshot
            restored), or ``"missing"`` (neither source available).
        age: Age of the snapshot at the time of restore, or ``None`` when
            status is ``"live"`` or ``"missing"``.
    """

    column: str
    status: str  # "live" | "cache" | "missing"
    age: pd.Timedelta | None = None


@dataclass
class DownloadResult:
    """Aggregate result of ``download_with_fallback``.

    Attributes:
        zones: Per-zone outcome, keyed by column name.
        combined_status: Status of the combined DE-total interim
            (``"n/a"`` when not attempted, ``"live"``, ``"cache"``, or
            ``"missing"``).
        combined_age: Age of the combined snapshot if restored, else ``None``.
        mode: ``"four_zone"`` when all zones are in {live, cache},
            ``"combined"`` when at least one zone is missing but the
            combined series is available, or ``None`` (unrecoverable).
    """

    zones: dict[str, ZoneOutcome]
    combined_status: str = "n/a"
    combined_age: pd.Timedelta | None = None
    mode: str | None = None  # "four_zone" | "combined" | None

    @property
    def fallback_gate(self) -> bool:
        """True when any data came from a snapshot (triggers the D-1 23:00 freshness gate).

        In ``four_zone`` mode: True if any zone has status ``"cache"``.
        In ``combined`` mode: True if ``combined_status == "cache"``.
        In ``None`` mode: always True (caller aborts before reaching the gate).
        """
        if self.mode == "four_zone":
            return any(z.status == "cache" for z in self.zones.values())
        if self.mode == "combined":
            return self.combined_status == "cache"
        return True

    def report(self) -> str:
        """Multi-line status report for log messages and Abort messages.

        Returns:
            Human-readable summary of every zone's status (with snapshot age
            for cache hits), the combined-series status, and what was attempted.
        """
        lines = ["Download resilience report:"]
        for col, z in self.zones.items():
            if z.status == "live":
                lines.append(f"  zone {col}: live (fresh download)")
            elif z.status == "cache":
                age_str = _fmt_age(z.age)
                lines.append(f"  zone {col}: CACHE ({age_str} old snapshot)")
            else:
                lines.append(f"  zone {col}: MISSING (no live data, no valid snapshot)")
        lines.append(
            f"  combined DE-total: {self.combined_status}"
            + (
                f" ({_fmt_age(self.combined_age)} old snapshot)"
                if self.combined_status == "cache"
                else ""
            )
        )
        lines.append(f"  mode: {self.mode!r}")
        return "\n".join(lines)


def _fmt_age(age: pd.Timedelta | None) -> str:
    if age is None:
        return "?"
    total_s = int(age.total_seconds())
    h, rem = divmod(total_s, 3600)
    m = rem // 60
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def download_with_fallback(
    api_key: str,
    *,
    start: str,
    end: str,
    now: pd.Timestamp,
    max_retries: int,
    backoff: float,
    timeout: float | None,
    fallback_enabled: bool,
    sleep: Callable[[float], None] = _time_module.sleep,
) -> DownloadResult:
    """Download zone loads with per-zone snapshot fallback.

    Per-zone decision tree:

    1. Bootstrap: ``SnapshotStore.seed_from_file`` for each zone kind and the
       combined kind so the first run after adding this layer still has
       snapshots for existing interim files.
    2. Per outer attempt (``max_retries``), call
       ``download_zone_loads(..., on_zone_failure="collect")`` for the zones
       still failing; wait ``backoff * 2**(attempt-1)`` between attempts.
    3. Success for a zone: write snapshot via ``SnapshotStore.write`` and
       record status ``"live"``.
    4. All attempts exhausted for a zone: if ``fallback_enabled`` and a valid
       snapshot exists: ``store.restore`` to ``interim/zone_<col>.csv``,
       record status ``"cache"`` with age; else status ``"missing"``.
    5. All four zones in {live, cache}: ``mode="four_zone"``; done.
    6. Else (>= 1 missing): attempt the live combined download
       (``download_new_data``, country_code="DE").  Success: snapshot + status
       ``"live"``.  Fail + valid snapshot: restore + status ``"cache"`` + age.
       Fail + no snapshot: status ``"missing"``.
    7. Combined in {live, cache}: ``mode="combined"``; else ``mode=None``
       (caller should raise ``Abort(5, result.report())``).
    8. ``SnapshotStore.prune(now)`` at the end.

    This function never raises for download failures; it only returns the
    ``DownloadResult`` describing what happened.

    Args:
        api_key: ENTSO-E Web API security token.
        start: Download start in ``"YYYYMMDDHHMM"`` format.
        end: Download end in ``"YYYYMMDDHHMM"`` format.
        now: Current UTC timestamp (used for snapshot naming and TTL checks).
        max_retries: Number of outer retry rounds (each round retries all
            still-failing zones via one collect-mode call; one
            ``backoff * 2**(round-1)`` sleep after each failed round).
        backoff: Base wait in seconds (exponential: ``backoff * 2**(attempt-1)``).
        timeout: Per-socket read timeout (seconds), or ``None`` to disable.
        fallback_enabled: When ``False``, snapshots are never READ as a
            fallback (live data or ``"missing"``), but they are still WRITTEN
            on success.
        sleep: Callable used for inter-attempt waits; injected so tests can
            pass ``sleep=lambda s: None`` for instant execution.

    Returns:
        DownloadResult: A record describing every zone's outcome and the
        overall ``mode`` (``"four_zone"``, ``"combined"``, or ``None``).

    Examples:
        ```{python}
        #| eval: false
        import pandas as pd

        from spotforecast2_safe.downloader import resilience as resil

        result = resil.download_with_fallback(
            api_key="YOUR_API_KEY",
            start="202301010000",
            end="202301050000",
            now=pd.Timestamp.now(tz="UTC"),
            max_retries=3,
            backoff=5.0,
            timeout=60.0,
            fallback_enabled=True,
        )
        print(result.report())
        if result.mode is None:
            raise SystemExit("unrecoverable: no live data and no valid snapshot")
        ```
    """
    # Lazy imports inside the function (sf2-safe reads env at call time;
    # configure_environment must run before this function is called).
    import spotforecast2_safe.downloader.entsoe as entsoe_module
    from spotforecast2_safe.data.fetch_data import get_data_home

    store = make_store()
    interim_dir = get_data_home() / "interim"

    # Bootstrap: seed snapshots from any existing interim files.
    for col in ZONE_COLUMNS:
        store.seed_from_file(f"zone_{col}", interim_dir / f"zone_{col}.csv", now)
    store.seed_from_file("combined", interim_dir / "energy_load.csv", now)

    zone_outcomes: dict[str, ZoneOutcome] = {}
    # Track which zones still need a successful download.
    failed_cols: set[str] = set(ZONE_COLUMNS)

    # --- per-zone attempts (collect mode, retrying only failed zones) ---
    for attempt in range(1, max_retries + 1):
        if not failed_cols:
            break
        zones_to_try = {col: _ZONE_AREAS[col] for col in failed_cols}
        try:
            results = entsoe_module.download_zone_loads(
                api_key=api_key,
                zones=zones_to_try,
                start=start,
                end=end,
                force=True,
                keep_forecast_future=True,
                timeout=timeout,
                on_zone_failure="collect",
            )
        except Exception as exc:  # noqa: BLE001 — argument-validation failures
            logger.warning(
                "download_zone_loads call failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            results = {col: None for col in failed_cols}  # type: ignore[assignment]

        if results is None:
            results = {}

        for col, zone_result in results.items():
            if zone_result is not None and zone_result.ok:
                zone_interim = interim_dir / f"zone_{col}.csv"
                snapped = store.write(f"zone_{col}", zone_interim, now)
                if snapped is None:
                    logger.warning(
                        "zone %s downloaded but interim file missing at %s; "
                        "no snapshot written — nothing to fall back to next run",
                        col,
                        zone_interim,
                    )
                zone_outcomes[col] = ZoneOutcome(column=col, status="live")
                failed_cols.discard(col)
                logger.info(
                    "zone %s: live download succeeded (attempt %d/%d)",
                    col,
                    attempt,
                    max_retries,
                )
            else:
                err = (
                    getattr(zone_result, "error", None)
                    if zone_result is not None
                    else None
                )
                logger.warning(
                    "zone %s: download attempt %d/%d failed: %s",
                    col,
                    attempt,
                    max_retries,
                    err,
                )

        if failed_cols and attempt < max_retries:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "retrying failed zones %s in %.0f s ...", list(failed_cols), wait
            )
            sleep(wait)

    # Apply fallback policy for zones still failed after all attempts.
    for col in failed_cols:
        snap = store.newest_valid(f"zone_{col}", now) if fallback_enabled else None
        if snap is not None:
            zone_interim = interim_dir / f"zone_{col}.csv"
            store.restore(snap, zone_interim)
            age = store.age_of(snap, now)
            logger.warning(
                "zone %s: live download failed; restored %s-old snapshot (%s)",
                col,
                _fmt_age(age),
                snap.name,
            )
            zone_outcomes[col] = ZoneOutcome(column=col, status="cache", age=age)
        else:
            if fallback_enabled:
                logger.warning(
                    "zone %s: live download failed and no valid snapshot; "
                    "status MISSING",
                    col,
                )
            else:
                logger.warning(
                    "zone %s: live download failed and --no-fallback is set; "
                    "status MISSING",
                    col,
                )
            zone_outcomes[col] = ZoneOutcome(column=col, status="missing")

    # --- mode decision ---
    all_ok = all(z.status in ("live", "cache") for z in zone_outcomes.values())
    if all_ok:
        result = DownloadResult(zones=zone_outcomes, mode="four_zone")
        store.prune(now)
        return result

    # --- combined DE-total fallback ---
    missing_zones = [col for col, z in zone_outcomes.items() if z.status == "missing"]
    logger.warning(
        "zone(s) missing: %s -- attempting combined DE-total fallback",
        missing_zones,
    )

    combined_interim = interim_dir / "energy_load.csv"
    combined_success = False
    for attempt in range(1, max_retries + 1):
        try:
            entsoe_module.download_new_data(
                api_key=api_key,
                country_code="DE",
                start=start,
                end=end,
                force=True,
                keep_forecast_future=True,
                timeout=timeout,
            )
            combined_success = True
            logger.info(
                "combined DE-total download succeeded (attempt %d/%d)",
                attempt,
                max_retries,
            )
            break
        except Exception as exc:  # noqa: BLE001
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "combined download attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                logger.warning("combined: retrying in %.0f s ...", wait)
                sleep(wait)

    if combined_success:
        snapped = store.write("combined", combined_interim, now)
        if snapped is None:
            logger.warning(
                "combined DE-total downloaded but interim file missing at %s; "
                "no snapshot written — nothing to fall back to next run",
                combined_interim,
            )
        result = DownloadResult(
            zones=zone_outcomes,
            combined_status="live",
            mode="combined",
        )
    else:
        snap = store.newest_valid("combined", now) if fallback_enabled else None
        if snap is not None:
            store.restore(snap, combined_interim)
            age = store.age_of(snap, now)
            logger.warning(
                "combined DE-total: live download failed; restored %s-old snapshot (%s)",
                _fmt_age(age),
                snap.name,
            )
            result = DownloadResult(
                zones=zone_outcomes,
                combined_status="cache",
                combined_age=age,
                mode="combined",
            )
        else:
            if fallback_enabled:
                logger.warning(
                    "combined DE-total: live download failed and no valid snapshot; "
                    "mode=None (unrecoverable)"
                )
            else:
                logger.warning(
                    "combined DE-total: live download failed and --no-fallback is set; "
                    "mode=None (unrecoverable)"
                )
            result = DownloadResult(
                zones=zone_outcomes,
                combined_status="missing",
                mode=None,
            )

    store.prune(now)
    return result
