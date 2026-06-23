# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data-governance guards for the multi-target forecasting pipeline.

Provides ``assert_no_leakage``, which ports the leakage-guard logic from the
operational script's ``assert_no_leakage`` (``team4_4zones_submit.py``,
lines ~968-1016) as a parameterised, raising function.

The mode-specific forbidden-set *construction* (which raw columns to ban per
pipeline variant) stays in the operator layer.  This function only enforces
the invariant: given a set of forbidden column names, none of them must appear
in any of the three model surfaces (training frame, selected exog names, fitted
feature list).

Protocol for ``mt`` (duck-typed, not type-checked at import time):

- ``mt.run_state.targets`` — list of target column names (``List[str]``).
- ``mt.results[task][target]["forecaster"]`` — fitted forecaster; its
  ``estimator.feature_name_`` attribute carries the list of fitted feature
  names.
- ``mt.data_with_exog`` — ``pd.DataFrame`` whose columns include both targets
  and all exogenous features used during training.
- ``mt.exog_feature_names`` — ``List[str]`` of the selected exogenous feature
  names.

Any ``BaseTask`` subclass (``MultiTask``, ``LazyTask``, etc.) satisfies this
protocol after a successful ``prepare_data`` + training run.
"""

from __future__ import annotations

from spotforecast2_safe.exceptions import LeakageError


def _fitted_feature_names(estimator: object) -> list[str] | None:
    """Return a fitted estimator's input feature names, or ``None``.

    Reads whichever attribute the estimator's backend populates after a fit on
    a named ``DataFrame``, trying the gradient-boosting-specific names first and
    the sklearn-standard one last:

    - ``feature_name_`` — LightGBM (``LGBMRegressor``)
    - ``feature_names_`` — CatBoost (``CatBoostRegressor``)
    - ``feature_names_in_`` — the sklearn convention (``XGBRegressor`` and
      ``LGBMRegressor`` set it too when fit on a ``DataFrame``)
    - ``get_booster().feature_names`` — XGBoost booster fallback

    Returns ``None`` only when none of these are available, which
    `assert_no_leakage` treats as a verifiability violation. Reading every
    supported backend (not just LightGBM's ``feature_name_``) lets the leakage
    guard run on XGBoost / CatBoost models instead of silently skipping them.
    """
    for attr in ("feature_name_", "feature_names_"):
        names = getattr(estimator, attr, None)
        if names is not None:
            return list(names)
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        return list(names)
    get_booster = getattr(estimator, "get_booster", None)
    if callable(get_booster):
        try:
            booster_names = get_booster().feature_names
        except Exception:  # noqa: BLE001 - booster may be unset / API drift
            booster_names = None
        if booster_names is not None:
            return list(booster_names)
    return None


def assert_no_leakage(
    mt: object,
    forbidden: set[str],
    *,
    task: str,
) -> None:
    """Raise `LeakageError` if any forbidden column reached the model.

    Checks three surfaces independently:

    1. **Training frame** — ``mt.data_with_exog.columns``
    2. **Selected exogenous features** — ``mt.exog_feature_names``
    3. **Fitted model features** — the fitted estimator's input feature names
       (LightGBM ``feature_name_``, CatBoost ``feature_names_``, or the
       sklearn-standard ``feature_names_in_`` / XGBoost booster) for every
       target in ``mt.run_state.targets``

    All three surfaces are checked so a single call reports every violation at once.
    If reading the fitted features fails (``estimator`` missing, not fitted,
    etc.) a ``RuntimeError`` is raised immediately rather than silently
    skipping: an unreadable feature list is itself a verifiability violation.

    Args:
        mt: A duck-typed pipeline object exposing the protocol described in
            the module docstring.  Any ``BaseTask`` subclass after training
            satisfies it.
        forbidden: Set of column names that must not appear in any model
            surface.  The operator constructs this set based on the pipeline
            mode (e.g. ``{"Forecasted Load", "Actual Load"}`` for a combined
            variant, plus per-zone forecast columns for the four-zone variant).
        task: Key identifying the task result to inspect inside
            ``mt.results`` (e.g. ``"spotoptim"`` or ``"defaults"``).

    Raises:
        LeakageError: When any forbidden column appears in the training frame,
            the selected exogenous feature names, or any fitted model's feature
            list.  The message names the surface and the offending columns.
        RuntimeError: When ``mt.results[task][target]["forecaster"].estimator``
            cannot be read or exposes none of the supported fitted-feature-name
            attributes (see `_fitted_feature_names`).  This signals a
            verifiability invariant violation rather than a leakage violation.

    Examples:
        ```{python}
        from types import SimpleNamespace
        from spotforecast2_safe.multitask.guards import assert_no_leakage
        from spotforecast2_safe.exceptions import LeakageError
        import pandas as pd

        # Build a minimal duck-typed stub that passes the guard.
        run_state = SimpleNamespace(targets=["load"])
        estimator = SimpleNamespace(feature_name_=["lag_1", "lag_24", "hour_sin"])
        forecaster = SimpleNamespace(estimator=estimator)
        results = {"defaults": {"load": {"forecaster": forecaster}}}
        idx = pd.date_range("2026-06-01", periods=3, freq="h", tz="UTC")
        data_with_exog = pd.DataFrame(
            {"load": [1.0, 2.0, 3.0], "lag_1": [0.0, 1.0, 2.0]}, index=idx
        )
        mt = SimpleNamespace(
            run_state=run_state,
            results=results,
            data_with_exog=data_with_exog,
            exog_feature_names=["lag_1"],
        )
        assert_no_leakage(mt, forbidden={"Forecasted Load"}, task="defaults")
        print("clean mt: leakage guard passed")

        # Inject a forbidden column into the training frame -> raises.
        mt_leak = SimpleNamespace(
            run_state=run_state,
            results=results,
            data_with_exog=pd.DataFrame(
                {"load": [1.0], "Forecasted Load": [50000.0]},
                index=idx[:1],
            ),
            exog_feature_names=["lag_1"],
        )
        try:
            assert_no_leakage(mt_leak, forbidden={"Forecasted Load"}, task="defaults")
        except LeakageError as exc:
            print(exc)
        ```
    """
    # Surface 1: fitted feature names (requires reading estimator).
    targets = list(mt.run_state.targets)
    fitted_features: set[str] = set()
    for target in targets:
        try:
            fc = mt.results[task][target]["forecaster"]
            names = _fitted_feature_names(fc.estimator)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read fitted feature names for target {target!r} "
                f"in task {task!r}: {exc}.  "
                f"The fitted feature list is required for leakage verification."
            ) from exc
        if names is None:
            raise RuntimeError(
                f"Cannot read fitted feature names for target {target!r} in "
                f"task {task!r}: estimator {type(fc.estimator).__name__!r} "
                f"exposes none of feature_name_ / feature_names_ / "
                f"feature_names_in_ / get_booster().feature_names.  "
                f"The fitted feature list is required for leakage verification."
            )
        fitted_features |= set(names)

    # Surface 2: training frame columns.
    dwe = getattr(mt, "data_with_exog", None)
    training_cols: set[str] = set(dwe.columns) if dwe is not None else set()

    # Surface 3: selected exogenous feature names.
    exog_names = getattr(mt, "exog_feature_names", None)
    exog_cols: set[str] = set(exog_names) if exog_names is not None else set()

    violations: list[str] = []
    checks = {
        "training frame": training_cols,
        "selected exogenous features": exog_cols,
        "fitted model features": fitted_features,
    }
    for surface, cols in checks.items():
        hit = sorted(forbidden & cols)
        if hit:
            violations.append(f"forbidden column(s) {hit} found in {surface}")

    if violations:
        detail = "; ".join(violations)
        raise LeakageError(
            f"Leakage detected in task {task!r}: {detail}. "
            f"Refusing to proceed (data-governance invariant)."
        )
