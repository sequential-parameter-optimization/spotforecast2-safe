"""spotforecast2-safe: Safety-critical time series forecasting library.

Version management: The package version is defined in pyproject.toml and exposed
via __version__ for programmatic access and documentation generation.
"""

try:
    # Modern approach: importlib.metadata (Python 3.8+)
    from importlib.metadata import version as _get_version
    __version__ = _get_version("spotforecast2-safe")
except Exception:
    # Fallback for development environments
    __version__ = "0.0.5"


def hello() -> str:
    return "Hello from spotforecast2!"


__all__ = ["__version__", "hello"]
