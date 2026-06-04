# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Import-hygiene tests for spotforecast2_safe.multitask.

Scans every .py source file in the multitask package and asserts that none
of the prohibited imports appear in executable code.  The checks target
actual import statements (``import ...`` / ``from ... import``), not
docstring prose that may legitimately mention the sibling package by name
to guide users.
"""

import re
from pathlib import Path

import pytest

# All .py files in the multitask package (excludes __pycache__)
_MULTITASK_DIR = (
    Path(__file__).parent.parent.parent / "src" / "spotforecast2_safe" / "multitask"
)

_MULTITASK_SOURCES = sorted(_MULTITASK_DIR.glob("*.py"))


def _source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _import_lines(text: str) -> list[str]:
    """Return only lines that are actual import statements."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            lines.append(stripped)
    return lines


@pytest.mark.parametrize("src_path", _MULTITASK_SOURCES, ids=lambda p: p.name)
def test_no_sibling_import(src_path: Path) -> None:
    """Assert no ``from spotforecast2.`` or bare ``import spotforecast2``.

    Prose mentions of ``spotforecast2`` (sibling) in docstrings are fine and
    intentional.  This test only looks at actual import statement lines.
    """
    text = _source_text(src_path)
    import_lines = _import_lines(text)
    # Match any import referencing ``spotforecast2.`` (the sibling package root)
    # or the bare ``import spotforecast2`` (without _safe suffix).
    # ``spotforecast2_safe`` is allowed; only the un-suffixed form is prohibited.
    pattern = re.compile(
        r"(?:^from spotforecast2\.)|(?:^import spotforecast2\b(?!_safe))"
    )
    bad_lines = [ln for ln in import_lines if pattern.search(ln)]
    assert (
        not bad_lines
    ), f"{src_path.name} contains prohibited sibling-package import(s):\n" + "\n".join(
        bad_lines
    )


@pytest.mark.parametrize("src_path", _MULTITASK_SOURCES, ids=lambda p: p.name)
def test_no_optuna_import(src_path: Path) -> None:
    """Assert no ``import optuna`` in any multitask source."""
    import_lines = _import_lines(_source_text(src_path))
    bad = [ln for ln in import_lines if "optuna" in ln]
    assert not bad, f"{src_path.name} contains 'optuna' in import(s): {bad}"


@pytest.mark.parametrize("src_path", _MULTITASK_SOURCES, ids=lambda p: p.name)
def test_no_spotoptim_import(src_path: Path) -> None:
    """Assert no ``import spotoptim`` in any multitask source."""
    import_lines = _import_lines(_source_text(src_path))
    bad = [ln for ln in import_lines if "spotoptim" in ln]
    assert not bad, f"{src_path.name} contains 'spotoptim' in import(s): {bad}"


@pytest.mark.parametrize("src_path", _MULTITASK_SOURCES, ids=lambda p: p.name)
def test_no_plotly_import(src_path: Path) -> None:
    """Assert no ``import plotly`` in any multitask source."""
    import_lines = _import_lines(_source_text(src_path))
    bad = [ln for ln in import_lines if "plotly" in ln]
    assert not bad, f"{src_path.name} contains 'plotly' in import(s): {bad}"


@pytest.mark.parametrize("src_path", _MULTITASK_SOURCES, ids=lambda p: p.name)
def test_no_matplotlib_import(src_path: Path) -> None:
    """Assert no ``import matplotlib`` in any multitask source."""
    import_lines = _import_lines(_source_text(src_path))
    bad = [ln for ln in import_lines if "matplotlib" in ln]
    assert not bad, f"{src_path.name} contains 'matplotlib' in import(s): {bad}"
