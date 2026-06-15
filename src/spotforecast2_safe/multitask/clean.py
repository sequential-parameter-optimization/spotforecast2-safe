# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cache-cleaning task — removes all cached pipeline artefacts.

``CleanTask`` deletes the entire cache directory used by the pipeline,
including saved models, tuning results, and any other artefacts written by
``LazyTask``, ``DefaultsTask``, or intermediate pipeline helpers.  It does
not require pipeline data to be prepared first and can be used as a
standalone reset mechanism.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from spotforecast2_safe.data.fetch_data import get_cache_home
from spotforecast2_safe.multitask.base import BaseTask


def execute_clean(
    task: BaseTask,
    cache_home: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Remove all cached data from the pipeline cache directory.

    Deletes the entire target directory (models, tuning results, and any
    other artefacts).  When the directory does not exist the function
    returns without error.  When ``dry_run`` is ``True`` nothing is
    removed but details of what would be deleted are logged and returned.

    Args:
        task: A ``BaseTask`` (or subclass) instance.  Used to resolve the
            default cache directory when ``cache_home`` is ``None``.
        cache_home: Override the directory to clean.  ``None`` uses
            ``get_cache_home(task.config.cache_home)``.
        dry_run: If ``True``, report what would be deleted without
            actually removing anything.

    Returns:
        Dict with keys:
            status: ``"success"`` after successful removal, ``"dry_run"``
                when ``dry_run`` is ``True``, or ``"empty"`` when the
                directory did not exist.
            cache_dir: The Path of the directory targeted for cleaning.
            deleted_items: Sorted list of top-level item names that were
                (or would have been) removed.

    Raises:
        RuntimeError: If the directory exists but cannot be removed due
            to a permissions error or other OS-level failure.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.multitask import CleanTask
        from spotforecast2_safe.multitask.clean import execute_clean
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "sf2_cache"
            cache.mkdir()
            (cache / "models").mkdir()
            (cache / "tuning_results").mkdir()
            cfg = ConfigMulti(cache_home=cache)
            task = CleanTask(cfg)
            result = execute_clean(task, dry_run=True)
            print(result["status"])
            print(sorted(result["deleted_items"]))
        ```
    """
    target_dir: Path = (
        get_cache_home(cache_home, create_dir=False)
        if cache_home is not None
        else get_cache_home(task.config.cache_home, create_dir=False)
    )

    if not target_dir.exists():
        task.logger.info("[clean] Cache directory does not exist: %s", target_dir)
        return {
            "status": "empty",
            "cache_dir": target_dir,
            "deleted_items": [],
        }

    items: List[Path] = sorted(target_dir.iterdir())
    item_names: List[str] = [item.name for item in items]

    if dry_run:
        print(f"[clean] Dry run — would delete: {target_dir}")
        for item in items:
            print(f"  Would remove: {item.name}")
        task.logger.info("[clean] Dry run — would delete: %s", target_dir)
        for item in items:
            task.logger.info("  Would remove: %s", item.name)
        return {
            "status": "dry_run",
            "cache_dir": target_dir,
            "deleted_items": item_names,
        }

    try:
        shutil.rmtree(target_dir)
    except OSError as exc:
        raise RuntimeError(
            f"Could not clean cache directory '{target_dir}': {exc}.  "
            "Check that the directory is not in use and that you have "
            "write permission."
        ) from exc

    task.logger.info("[clean] Cache removed successfully: %s", target_dir)
    print(f"[clean] Cache removed successfully: {target_dir}")
    return {
        "status": "success",
        "cache_dir": target_dir,
        "deleted_items": item_names,
    }


class CleanTask(BaseTask):
    """Cache-cleaning task — removes all cached data from the pipeline cache.

    ``CleanTask`` deletes the entire cache directory configured for the
    pipeline, including saved models, tuning results, and any other
    cached artefacts written by training tasks.

    Unlike training or prediction tasks, ``CleanTask`` does not require
    ``prepare_data()`` to be called before ``run()``.  It operates purely on
    the file system and can be used as a standalone reset mechanism
    between experiments or deployments.

    Passing ``dry_run=True`` to ``run()`` reports what would be deleted
    without actually removing anything, which is useful for inspecting
    cache contents before committing to removal.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.multitask import CleanTask
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(data_frame_name="demo10", cache_home=Path(tmp))
            task = CleanTask(cfg)
            print(f"Task: {task.TASK}")
            print(f"Task name: {task._task_name}")
        ```
    """

    _task_name = "clean"

    def run(
        self,
        dry_run: bool = False,
        cache_home: Optional[Path] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Remove all cached data from the pipeline cache directory.

        Does not require ``prepare_data()`` to be called first.

        Args:
            dry_run: If ``True``, report what would be deleted without
                actually removing anything.  Useful for inspecting the
                cache before committing to removal.
            cache_home: Override the directory to clean.  ``None`` uses
                the directory configured on this instance via
                ``get_cache_home()``.
            show: Accepted for API consistency with other tasks.  Not used
                by the clean task.

        Returns:
            Dict with keys:
                status: ``"success"``, ``"dry_run"``, or ``"empty"``.
                cache_dir: The Path targeted for cleaning.
                deleted_items: Names of top-level items removed (or that
                    would have been removed in ``dry_run`` mode).

        Raises:
            RuntimeError: If the cache directory cannot be removed due to
                a permissions error or OS-level failure.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path
            from spotforecast2_safe.multitask import CleanTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(cache_home=Path(tmp) / "test_cache")
                task = CleanTask(cfg)
                result = task.run(dry_run=True)
                print(result["status"])
            ```
        """
        return execute_clean(self, cache_home=cache_home, dry_run=dry_run)
