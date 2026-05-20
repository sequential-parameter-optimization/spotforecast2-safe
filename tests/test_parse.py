# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the CLI boolean parser in `spotforecast2_safe.utils.parse`."""

import argparse

import pytest

from spotforecast2_safe.utils.parse import parse_bool


class TestParseBoolTrue:
    """Tokens that must parse as True."""

    @pytest.mark.parametrize(
        "value",
        ["true", "True", "TRUE", "t", "T", "yes", "YES", "Yes", "1"],
    )
    def test_true_variants(self, value):
        assert parse_bool(value) is True

    @pytest.mark.parametrize("value", ["  true  ", "\tyes\n", " 1 "])
    def test_true_with_whitespace(self, value):
        assert parse_bool(value) is True


class TestParseBoolFalse:
    """Tokens that must parse as False."""

    @pytest.mark.parametrize(
        "value",
        ["false", "False", "FALSE", "f", "F", "no", "NO", "No", "0"],
    )
    def test_false_variants(self, value):
        assert parse_bool(value) is False

    @pytest.mark.parametrize("value", ["  false  ", "\tno\n", " 0 "])
    def test_false_with_whitespace(self, value):
        assert parse_bool(value) is False


class TestParseBoolInvalid:
    """Unrecognised tokens must raise `argparse.ArgumentTypeError`."""

    @pytest.mark.parametrize("value", ["maybe", "2", "", "  ", "yesno", "tru"])
    def test_invalid_raises(self, value):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            parse_bool(value)
        assert "Expected a boolean value" in str(exc.value)


class TestParseBoolArgparseIntegration:
    """`parse_bool` is callable as `type=` in an `ArgumentParser`."""

    def test_used_as_type(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--enable", type=parse_bool, default=True)
        args = parser.parse_args(["--enable", "yes"])
        assert args.enable is True

    def test_invalid_through_argparse_exits(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--enable", type=parse_bool)
        with pytest.raises(SystemExit):
            parser.parse_args(["--enable", "maybe"])
