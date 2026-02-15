# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later


from spotforecast2_safe.data import Period
from spotforecast2_safe.preprocessing import (
    RepeatingBasisFunction,
    ExogBuilder,
    LinearlyInterpolateTS,
)
from spotforecast2_safe.manager.models import (
    ForecasterRecursiveXGB,
    ForecasterRecursiveLGBM,
    ForecasterRecursiveModel,
)

"""spotforecast2-safe: Safety-critical time series forecasting library.

Version management: The package version is defined in pyproject.toml and exposed
via __version__ for programmatic access and documentation generation.
"""

try:
    # Modern approach: importlib.metadata (Python 3.8+)
    from importlib.metadata import PackageNotFoundError, version as _get_version

    __version__ = _get_version("spotforecast2-safe")
except PackageNotFoundError:
    # Fallback if package is not installed (e.g., development environment)
    __version__ = "unknown"


def hello() -> str:
    return "Hello from spotforecast2-safe!"


__all__ = [
    "__version__",
    "hello",
    "Period",
    "RepeatingBasisFunction",
    "ExogBuilder",
    "LinearlyInterpolateTS",
    "ForecasterRecursiveModel",
    "ForecasterRecursiveLGBM",
    "ForecasterRecursiveXGB",
]
