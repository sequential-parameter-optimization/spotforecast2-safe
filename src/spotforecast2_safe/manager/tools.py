# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Utility tools for command-line interface parsing.
"""

import argparse


def _parse_bool(value: str) -> bool:
    """Parse case-insensitive boolean strings for CLI arguments.

    Args:
        value: String representation of a boolean value.

    Returns:
        bool: True for {'true', 't', 'yes', '1'}, False for {'false', 'f', 'no', '0'}.

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed as boolean.

    Examples:
        >>> from spotforecast2_safe.manager.tools import _parse_bool
        >>>
        >>> # Example 1: Parse various true values
        >>> print(_parse_bool("true"))
        True
        >>> print(_parse_bool("TRUE"))
        True
        >>> print(_parse_bool("t"))
        True
        >>> print(_parse_bool("yes"))
        True
        >>> print(_parse_bool("1"))
        True
        >>>
        >>> # Example 2: Parse various false values
        >>> print(_parse_bool("false"))
        False
        >>> print(_parse_bool("FALSE"))
        False
        >>> print(_parse_bool("f"))
        False
        >>> print(_parse_bool("no"))
        False
        >>> print(_parse_bool("0"))
        False
        >>>
        >>> # Example 3: Handle whitespace
        >>> print(_parse_bool("  true  "))
        True
        >>> print(_parse_bool("  false  "))
        False
        >>>
        >>> # Example 4: Invalid values raise errors
        >>> try:
        ...     _parse_bool("invalid")
        ... except argparse.ArgumentTypeError as e:
        ...     print("Error raised as expected")
        Error raised as expected
        >>>
        >>> # Example 5: Use in CLI argument parser
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--enable', type=_parse_bool, default=True)
        >>> args = parser.parse_args(['--enable', 'yes'])
        >>> print(args.enable)
        True
    """
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "1"}:
        return True
    if normalized in {"false", "f", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")
