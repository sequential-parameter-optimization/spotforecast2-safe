# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Architectural guard: spotforecast2-safe must never depend on challenge-leaderboard.

The dependency layering is strictly one-way. The higher-level
``challenge_leaderboard`` package may import ``spotforecast2_safe`` (it does, in
its scoring scripts) — but ``spotforecast2_safe``, the lower-level,
security-critical forecasting library, must never import ``challenge_leaderboard``
or declare it as a dependency. This test fails loudly if that invariant ever
regresses (e.g. a future submission helper reaches "up" into the leaderboard).
"""

from __future__ import annotations

import re
from pathlib import Path

import spotforecast2_safe

_PKG_ROOT = Path(spotforecast2_safe.__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent.parent  # src/spotforecast2_safe -> repo root
_FORBIDDEN = re.compile(r"challenge[_-]leaderboard")


def test_package_source_never_references_challenge_leaderboard() -> None:
    """No module under spotforecast2_safe may import/name challenge-leaderboard."""
    offenders: list[str] = []
    for py in _PKG_ROOT.rglob("*.py"):
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if _FORBIDDEN.search(line):
                offenders.append(f"{py.relative_to(_PKG_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "spotforecast2_safe must never reference challenge-leaderboard "
        "(one-way layering: challenge-leaderboard -> sf2-safe only). "
        "Offending lines:\n" + "\n".join(offenders)
    )


def test_pyproject_does_not_depend_on_challenge_leaderboard() -> None:
    """challenge-leaderboard must not appear as a declared dependency."""
    pyproject = _REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():  # installed wheel layout — nothing to check
        return
    offenders = [
        f"pyproject.toml:{lineno}: {line.strip()}"
        for lineno, line in enumerate(pyproject.read_text(encoding="utf-8").splitlines(), 1)
        if _FORBIDDEN.search(line)
    ]
    assert not offenders, (
        "challenge-leaderboard must not be a dependency of spotforecast2-safe.\n"
        + "\n".join(offenders)
    )
