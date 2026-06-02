# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``exceptions`` warning categories and the warning handlers.

Covers every registered warning category's message rendering, the box and rich
handlers (including the ``HAS_RICH=False`` fallback), and the
``set_warnings_style`` / ``set_skforecast_warnings`` toggles. The handlers
mutate ``warnings.showwarning`` globally, so an autouse fixture restores it.
"""

import warnings

import pytest

from spotforecast2_safe import exceptions
from spotforecast2_safe.exceptions import (
    MissingValuesWarning,
    format_warning_handler,
    rich_warning_handler,
    set_skforecast_warnings,
    set_warnings_style,
    warn_skforecast_categories,
)


@pytest.fixture(autouse=True)
def _restore_showwarning():
    original = warnings.showwarning
    yield
    warnings.showwarning = original


@pytest.mark.parametrize("category", warn_skforecast_categories)
def test_category_is_warning_and_renders_message(category):
    instance = category("a test message")
    assert issubclass(category, Warning)
    assert "a test message" in str(instance)


def test_format_warning_handler_boxes_skforecast_warning(capsys):
    format_warning_handler(
        MissingValuesWarning("boxed msg"), MissingValuesWarning, "f.py", 10
    )
    out = capsys.readouterr().out
    assert "MissingValuesWarning" in out
    assert "boxed msg" in out


def test_format_warning_handler_falls_back_for_plain_warning(capsys):
    # A non-skforecast warning takes the fallback branch: no box is rendered.
    format_warning_handler(UserWarning("plain"), UserWarning, "f.py", 10)
    assert "╭" not in capsys.readouterr().out


def test_rich_warning_handler_emits_for_skforecast_warning(capsys):
    rich_warning_handler(
        MissingValuesWarning("rich msg"), MissingValuesWarning, "f.py", 10
    )
    assert "MissingValuesWarning" in capsys.readouterr().out


def test_rich_warning_handler_fallback_without_rich(capsys, monkeypatch):
    monkeypatch.setattr(exceptions, "HAS_RICH", False)
    rich_warning_handler(
        MissingValuesWarning("norich"), MissingValuesWarning, "f.py", 10
    )
    # Falls back to the box handler.
    assert "MissingValuesWarning" in capsys.readouterr().out


def test_set_warnings_style_toggle():
    set_warnings_style("skforecast")
    assert warnings.showwarning in (rich_warning_handler, format_warning_handler)

    set_warnings_style("default")
    assert warnings.showwarning is warnings._original_showwarning


def test_set_skforecast_warnings_suppresses_categories():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        set_skforecast_warnings(suppress_warnings=True, action="ignore")
        warnings.warn("x", MissingValuesWarning)

    suppressed = [w for w in caught if issubclass(w.category, MissingValuesWarning)]
    assert len(suppressed) == 0
