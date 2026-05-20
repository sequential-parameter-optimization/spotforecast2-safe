# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Utility helpers for command-line interface parsing.
"""

import argparse


def parse_bool(value: str) -> bool:
    """Parse case-insensitive boolean strings for CLI arguments.

    Args:
        value: String representation of a boolean value.

    Returns:
        bool: True for {'true', 't', 'yes', '1'}, False for {'false', 'f', 'no', '0'}.

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed as boolean.

    Examples:
        ```{python}
        import argparse

        from spotforecast2_safe.utils.parse import parse_bool

        # True values (case-insensitive, whitespace-tolerant)
        assert parse_bool("true") is True
        assert parse_bool("TRUE") is True
        assert parse_bool("t") is True
        assert parse_bool("yes") is True
        assert parse_bool("1") is True
        assert parse_bool("  true  ") is True

        # False values
        assert parse_bool("false") is False
        assert parse_bool("FALSE") is False
        assert parse_bool("f") is False
        assert parse_bool("no") is False
        assert parse_bool("0") is False
        assert parse_bool("  false  ") is False

        print("All true/false/whitespace cases pass.")
        ```

        ```{python}
        import argparse

        from spotforecast2_safe.utils.parse import parse_bool

        # Invalid values raise ArgumentTypeError
        try:
            parse_bool("invalid")
        except argparse.ArgumentTypeError:
            print("ArgumentTypeError raised as expected.")

        # Use as type= in an ArgumentParser
        parser = argparse.ArgumentParser()
        _ = parser.add_argument("--enable", type=parse_bool, default=True)
        args = parser.parse_args(["--enable", "yes"])
        assert args.enable is True
        print(f"Parsed --enable yes → {args.enable}")
        ```
    """
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "1"}:
        return True
    if normalized in {"false", "f", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")
