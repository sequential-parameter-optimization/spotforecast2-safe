# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generic TTL-aware atomic snapshot store.

Provides a small, caller-configured store for maintaining timestamped CSV
snapshots of arbitrary named *kinds*.  The store knows nothing about what
the files represent, how they are produced, or how restoring them should be
interpreted — substitution policy stays operator-side.

The implementation is generalized from ``bart26k-lecture/scripts/_team4_resilience.py``
(mechanism only; all domain knowledge stripped).

```{python}
# Example (offline — uses tempfile)
import tempfile
from pathlib import Path
import pandas as pd
from spotforecast2_safe.utils.snapshot_store import SnapshotStore

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    store = SnapshotStore(root=root, ttl=pd.Timedelta(hours=72))
    now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")

    # Write a small source file
    source = root / "data.csv"
    source.write_text("a,b\\n1,2\\n3,4\\n")

    snap = store.write("demo", source, now)
    print("written:", snap)

    valid = store.newest_valid("demo", now)
    print("newest valid:", valid)

    pruned = store.prune(now)
    print("pruned (should be 0):", pruned)
```
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Timestamp format embedded in snapshot filenames; must be filename-safe.
_TS_FMT = "%Y%m%dT%H%M%SZ"


def parse_snapshot_timestamp(path: Path) -> pd.Timestamp | None:
    """Parse the UTC timestamp encoded in a snapshot filename stem.

    The stem must be formatted as ``%Y%m%dT%H%M%SZ`` (e.g. ``20260611T040000Z``),
    which is valid ISO-8601-basic and parseable by ``pd.Timestamp``.

    Args:
        path: File whose stem encodes the snapshot timestamp.

    Returns:
        A timezone-aware ``pd.Timestamp`` in UTC, or ``None`` when the stem
        cannot be parsed (a WARNING is logged naming the file).

    Examples:
        >>> from pathlib import Path
        >>> from spotforecast2_safe.utils.snapshot_store import parse_snapshot_timestamp
        >>> ts = parse_snapshot_timestamp(Path("20260611T040000Z.csv"))
        >>> ts
        Timestamp('2026-06-11 04:00:00+0000', tz='UTC')
        >>> parse_snapshot_timestamp(Path("bad_name.csv")) is None
        True
    """
    try:
        return pd.Timestamp(path.stem, tz="UTC")
    except Exception:  # noqa: BLE001
        logger.warning("unrecognised snapshot filename, skipping: %s", path)
        return None


@dataclass(frozen=True)
class SnapshotStore:
    """Generic TTL-aware atomic snapshot store.

    Manages timestamped CSV snapshots under ``root/<kind>/`` directories.
    Directories are created lazily.  All read operations respect a caller-supplied
    TTL; expired snapshots are invisible to `newest_valid` and deleted by `prune`.

    The store is a pure mechanism — it knows nothing about what the files
    represent or when restoring them is appropriate.  Restoring is always an
    explicit caller action via `restore`.

    Args:
        root: Parent directory under which per-kind subdirectories are created.
        ttl: Maximum age of a snapshot for it to be considered valid by
            `newest_valid` and `age_of`.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        import pandas as pd
        from spotforecast2_safe.utils.snapshot_store import SnapshotStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SnapshotStore(root=root, ttl=pd.Timedelta(hours=72))
            now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")

            source = root / "source.csv"
            source.write_text("x,y\\n1,2\\n")

            snap = store.write("mydata", source, now)
            print("snapshot path:", snap)

            valid = store.newest_valid("mydata", now)
            print("newest valid:", valid)

            dest = root / "restored.csv"
            store.restore(valid, dest)
            print("restored content:", dest.read_text())

            n = store.prune(now + pd.Timedelta(hours=73))
            print("pruned:", n)
        ```
    """

    root: Path
    ttl: pd.Timedelta

    def _kind_dir(self, kind: str) -> Path:
        return self.root / kind

    def write(self, kind: str, source: Path, ts: pd.Timestamp) -> Path | None:
        """Atomically copy ``source`` into the snapshot store for ``kind``.

        Writes to a ``.tmp`` file in the destination directory, then replaces
        the final target via ``os.replace``.  A crash between the two steps
        leaves any pre-existing snapshot intact; the ``.tmp`` is ignored by
        all read operations.

        Args:
            kind: Logical category name; controls the subdirectory used.
            source: File to snapshot.
            ts: Timestamp to embed in the snapshot filename (UTC recommended).

        Returns:
            Path of the written snapshot file, or ``None`` (with a WARNING) if
            ``source`` does not exist.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> from spotforecast2_safe.utils.snapshot_store import SnapshotStore
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     store = SnapshotStore(root=Path(tmp), ttl=pd.Timedelta(hours=24))
            ...     src = Path(tmp) / "data.csv"
            ...     src.write_text("a,b\\n1,2\\n")
            ...     now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")
            ...     snap = store.write("demo", src, now)
            ...     snap.name
            '20260611T040000Z.csv'
        """
        if not source.exists():
            logger.warning("snapshot source does not exist, skipping: %s", source)
            return None
        dest_dir = self._kind_dir(kind)
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = ts.strftime(_TS_FMT) + ".csv"
        dest = dest_dir / filename
        tmp = dest_dir / (filename + ".tmp")
        shutil.copy2(str(source), str(tmp))
        os.replace(str(tmp), str(dest))
        logger.debug("snapshot written: %s -> %s", source, dest)
        return dest

    def newest_valid(self, kind: str, now: pd.Timestamp) -> Path | None:
        """Return the newest valid snapshot for ``kind``, or ``None``.

        A snapshot is *valid* when its filename-encoded timestamp is parseable
        and within ``ttl`` of ``now``.  Files ending in ``.tmp`` and files whose
        stem cannot be parsed are silently skipped (with a WARNING for the
        latter).

        Args:
            kind: Logical category name.
            now: Reference time for the TTL check.

        Returns:
            Path of the newest valid snapshot, or ``None`` if none exists.

        Examples:
            >>> import tempfile, os
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> from spotforecast2_safe.utils.snapshot_store import SnapshotStore
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     store = SnapshotStore(root=Path(tmp), ttl=pd.Timedelta(hours=24))
            ...     src = Path(tmp) / "d.csv"; src.write_text("x\\n1\\n")
            ...     now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")
            ...     _ = store.write("demo", src, now)
            ...     result = store.newest_valid("demo", now)
            ...     result is not None
            True
        """
        kind_dir = self._kind_dir(kind)
        if not kind_dir.is_dir():
            return None
        cutoff = now - self.ttl
        best: tuple[pd.Timestamp, Path] | None = None
        for p in kind_dir.iterdir():
            if p.suffix != ".csv" or p.name.endswith(".tmp"):
                continue
            ts = parse_snapshot_timestamp(p)
            if ts is None:
                continue
            if ts < cutoff:
                continue
            if best is None or ts > best[0]:
                best = (ts, p)
        return best[1] if best is not None else None

    def restore(self, snapshot: Path, dest: Path) -> None:
        """Copy ``snapshot`` to ``dest`` atomically.

        Creates parent directories as needed.  The copy is atomic at the
        filesystem level: the file is written to a ``.tmp`` sibling in
        ``dest``'s directory, then renamed into place via ``os.replace``.

        Args:
            snapshot: Snapshot file to restore.
            dest: Destination path.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> from spotforecast2_safe.utils.snapshot_store import SnapshotStore
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     store = SnapshotStore(root=Path(tmp), ttl=pd.Timedelta(hours=24))
            ...     snap = Path(tmp) / "snap.csv"; snap.write_text("a\\n1\\n")
            ...     dest = Path(tmp) / "sub" / "out.csv"
            ...     store.restore(snap, dest)
            ...     dest.read_text()
            'a\\n1\\n'
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / (dest.name + ".tmp")
        shutil.copy2(str(snapshot), str(tmp))
        os.replace(str(tmp), str(dest))
        logger.debug("snapshot restored: %s -> %s", snapshot, dest)

    def prune(self, now: pd.Timestamp) -> int:
        """Delete expired snapshots and stale ``.tmp`` files under ``root``.

        Walks every kind subdirectory under ``root`` and removes:

        * ``.csv`` snapshots whose filename-encoded timestamp is older than
          ``now - ttl``.
        * Any ``.tmp`` files (left by interrupted writes).

        Unparseable filenames are logged as WARNING and skipped.

        Args:
            now: Reference time for the age calculation.

        Returns:
            Number of files deleted.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> from spotforecast2_safe.utils.snapshot_store import SnapshotStore
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     store = SnapshotStore(root=Path(tmp), ttl=pd.Timedelta(hours=24))
            ...     src = Path(tmp) / "d.csv"; src.write_text("x\\n1\\n")
            ...     old_ts = pd.Timestamp("2026-06-09 04:00:00", tz="UTC")
            ...     now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")
            ...     _ = store.write("demo", src, old_ts)
            ...     store.prune(now)
            1
        """
        if not self.root.is_dir():
            return 0
        cutoff = now - self.ttl
        removed = 0
        for kind_dir in self.root.iterdir():
            if not kind_dir.is_dir():
                continue
            for p in kind_dir.iterdir():
                # Always remove stale .tmp files
                if p.name.endswith(".tmp"):
                    try:
                        p.unlink()
                        removed += 1
                        logger.debug("pruned stale tmp: %s", p)
                    except OSError as exc:
                        logger.warning("could not prune tmp %s: %s", p, exc)
                    continue
                if p.suffix != ".csv":
                    continue
                ts = parse_snapshot_timestamp(p)
                if ts is None:
                    continue
                if ts < cutoff:
                    try:
                        p.unlink()
                        removed += 1
                        logger.debug("pruned expired snapshot: %s", p)
                    except OSError as exc:
                        logger.warning("could not prune snapshot %s: %s", p, exc)
        if removed:
            logger.info("pruned %d file(s) (TTL %s)", removed, self.ttl)
        return removed

    def seed_from_file(self, kind: str, source: Path, now: pd.Timestamp) -> Path | None:
        """Bootstrap: snapshot ``source`` using its mtime if no newer snapshot exists.

        Seeds the snapshot store from a pre-existing file.  Useful on the first
        run after the store is introduced so that history already on disk is
        preserved.  The seeded snapshot is stamped with the file's mtime (UTC),
        not with ``now``.

        Conditions for seeding (all must hold):

        * ``source`` exists.
        * The file's mtime converted to UTC is within ``ttl`` of ``now``.
        * No existing snapshot for ``kind`` has a timestamp >= the file's mtime.

        Args:
            kind: Logical category name.
            source: File to seed from.
            now: Reference time for the TTL check.

        Returns:
            The written snapshot path when seeding occurred, or ``None`` when
            any condition was not met.

        Examples:
            >>> import tempfile, os
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> from spotforecast2_safe.utils.snapshot_store import SnapshotStore
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     store = SnapshotStore(root=Path(tmp), ttl=pd.Timedelta(hours=72))
            ...     src = Path(tmp) / "src.csv"; src.write_text("v\\n1\\n")
            ...     # Back-date mtime to 10 h ago
            ...     now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")
            ...     mtime = (now - pd.Timedelta(hours=10)).timestamp()
            ...     os.utime(str(src), (mtime, mtime))
            ...     snap = store.seed_from_file("demo", src, now)
            ...     snap is not None
            True
        """
        if not source.exists():
            return None
        mtime_ts = pd.Timestamp(source.stat().st_mtime, unit="s", tz="UTC")
        cutoff = now - self.ttl
        if mtime_ts < cutoff:
            logger.debug(
                "source file %s mtime %s is older than TTL; not seeding snapshot",
                source,
                mtime_ts,
            )
            return None
        # Check whether a snapshot >= mtime already exists.
        kind_dir = self._kind_dir(kind)
        if kind_dir.is_dir():
            for p in kind_dir.iterdir():
                if p.suffix != ".csv" or p.name.endswith(".tmp"):
                    continue
                ts = parse_snapshot_timestamp(p)
                if ts is None:
                    continue
                if ts >= mtime_ts:
                    return None  # a snapshot as new or newer exists; skip
        logger.info(
            "seeding snapshot from existing file: %s (mtime=%s)", source, mtime_ts
        )
        return self.write(kind, source, mtime_ts)

    def age_of(self, snapshot: Path, now: pd.Timestamp) -> pd.Timedelta | None:
        """Return the age of ``snapshot`` relative to ``now``.

        Args:
            snapshot: Snapshot file whose stem encodes the timestamp.
            now: Reference time.

        Returns:
            ``now - snapshot_ts``, or ``None`` when the timestamp cannot be
            parsed.

        Examples:
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> from spotforecast2_safe.utils.snapshot_store import SnapshotStore
            >>> store = SnapshotStore(root=Path("/tmp"), ttl=pd.Timedelta(hours=24))
            >>> now = pd.Timestamp("2026-06-11 04:00:00", tz="UTC")
            >>> snap = Path("20260610T040000Z.csv")
            >>> store.age_of(snap, now)
            Timedelta('1 days 00:00:00')
        """
        ts = parse_snapshot_timestamp(snapshot)
        if ts is None:
            return None
        return now - ts
