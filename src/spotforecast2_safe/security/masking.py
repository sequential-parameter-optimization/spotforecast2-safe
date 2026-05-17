# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Masking helpers for safe logging of potentially sensitive values.

This module addresses CWE-532 (Insertion of Sensitive Information into Log File)
and CWE-312 (Cleartext Storage of Sensitive Information) by providing small,
deterministic helpers that reduce a value to a non-sensitive string suitable for
log output. The helpers are intended for use in pipeline code that writes to
shared log aggregators, crash dumps, or other long-lived sinks where raw
configuration details could leak to unauthorized parties.
"""

from typing import Any

__all__ = [
    "mask_estimator",
]


def mask_estimator(estimator: Any) -> str:
    """Return a non-sensitive string representation of an estimator.

    Logs the estimator *type* without exposing its hyperparameters or any other
    configuration. This is the safe form to emit at INFO level when the actual
    estimator object may carry tuning details that should not be persisted in
    log aggregation systems (CWE-532).

    Args:
        estimator: An estimator instance, an estimator name (string), or
            ``None``. When ``None``, the default project estimator label is
            returned.

    Returns:
        A short, safe string identifying the estimator family.

    Examples:
        ```{python}
        from spotforecast2_safe.security.masking import mask_estimator

        mask_estimator(None)
        ```

        ```{python}
        mask_estimator("ForecasterRecursive")
        ```

        ```{python}
        from sklearn.linear_model import Ridge
        mask_estimator(Ridge(alpha=42.0))
        ```
    """
    if estimator is None:
        return "LGBMRegressor (default)"
    if isinstance(estimator, str):
        return estimator
    return type(estimator).__name__
