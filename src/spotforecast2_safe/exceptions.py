# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""Custom exceptions and warnings for spotforecast2.

This module contains all the custom warnings and error classes used
across spotforecast2.

Examples:
    ```{python}
    import warnings
    from spotforecast2_safe.exceptions import MissingValuesWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("Missing values detected in input data.", MissingValuesWarning)

    assert len(caught) == 1
    assert issubclass(caught[0].category, MissingValuesWarning)
    print(caught[0].category.__name__)
    ```
"""

import inspect
import textwrap
import warnings
from functools import wraps

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def runtime_deprecated(
    replacement: str = None,
    version: str = None,
    removal: str = None,
    category: type[Warning] = FutureWarning,
) -> object:
    """Decorator to mark functions or classes as deprecated.

    Works for both function and class targets, and ensures warnings are visible
    even inside Jupyter notebooks.

    Args:
        replacement: Name of the replacement function/class to use instead.
        version: Version in which the function/class was deprecated.
        removal: Version in which the function/class will be removed.
        category: Warning category to use. Default is FutureWarning.

    Returns:
        Decorator function.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import runtime_deprecated

        @runtime_deprecated(replacement='new_function', version='0.5', removal='1.0')
        def old_function():
            return 42

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = old_function()

        assert result == 42
        assert len(caught) == 1
        assert issubclass(caught[0].category, FutureWarning)
        assert "0.5" in str(caught[0].message)
        print(type(caught[0].category).__name__, str(caught[0].message)[:60])
        ```
    """

    def decorator(obj):
        is_function = inspect.isfunction(obj) or inspect.ismethod(obj)
        is_class = inspect.isclass(obj)

        if not (is_function or is_class):
            raise TypeError(
                "@runtime_deprecated can only be used on functions or classes"
            )

        # ----- Build warning message -----
        name = obj.__name__
        message = (
            f"{name}() is deprecated" if is_function else f"{name} class is deprecated"
        )
        if version:
            message += f" since version {version}"
        if replacement:
            message += f"; use {replacement} instead"
        if removal:
            message += f". It will be removed in version {removal}."
        else:
            message += "."

        def issue_warning():
            """Emit warning in a way that always shows in notebooks."""
            with warnings.catch_warnings():
                warnings.simplefilter("always", category)
                warnings.warn(message, category, stacklevel=3)

        # ----- Case 1: decorating a function -----
        if is_function:

            @wraps(obj)
            def wrapper(*args, **kwargs):
                issue_warning()
                return obj(*args, **kwargs)

            # Add metadata
            wrapper.__deprecated__ = True
            wrapper.__replacement__ = replacement
            wrapper.__version__ = version
            wrapper.__removal__ = removal
            return wrapper

        # ----- Case 2: decorating a class -----
        else:  # is_class must be True due to earlier check
            orig_init = getattr(obj, "__init__", None)
            orig_new = getattr(obj, "__new__", None)

            # Only wrap whichever exists (some classes use __new__, others __init__)
            if orig_new and (orig_new is not object.__new__):

                @wraps(orig_new)
                def wrapped_new(cls, *args, **kwargs):
                    issue_warning()
                    return orig_new(cls, *args, **kwargs)

                obj.__new__ = staticmethod(wrapped_new)

            elif orig_init:

                @wraps(orig_init)
                def wrapped_init(self, *args, **kwargs):
                    issue_warning()
                    return orig_init(self, *args, **kwargs)

                obj.__init__ = wrapped_init

            # Add metadata
            obj.__deprecated__ = True
            obj.__replacement__ = replacement
            obj.__version__ = version
            obj.__removal__ = removal

            return obj

    return decorator


class DataTypeWarning(UserWarning):
    """Warning for incompatible data types in exogenous data.

    Used to notify there are dtypes in the exogenous data that are not
    'int', 'float', 'bool' or 'category'. Most machine learning models do not
    accept other data types, therefore the forecaster `fit` and `predict` may fail.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import DataTypeWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Exogenous data contains unsupported dtypes.", DataTypeWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, DataTypeWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTypeWarning)"
        )
        return self.message + "\n" + extra_message


class DataTransformationWarning(UserWarning):
    """Warning for output data in transformed space.

    Used to notify that the output data is in the transformed space.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import DataTransformationWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Output is in transformed space.", DataTransformationWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, DataTransformationWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTransformationWarning)"
        )
        return self.message + "\n" + extra_message


class ExogenousInterpretationWarning(UserWarning):
    """Warning about implications when using exogenous variables.

    Used to notify about important implications when using exogenous
    variables with models that use a two-step approach (e.g., regression + ARAR).

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import ExogenousInterpretationWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(
                "Exogenous variables may not be used as expected.",
                ExogenousInterpretationWarning,
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, ExogenousInterpretationWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ExogenousInterpretationWarning)"
        )
        return self.message + "\n" + extra_message


class FeatureOutOfRangeWarning(UserWarning):
    """Warning for features out of training range.

    Used to notify that a feature is out of the range seen during training.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import FeatureOutOfRangeWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Feature value exceeds training range.", FeatureOutOfRangeWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, FeatureOutOfRangeWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=FeatureOutOfRangeWarning)"
        )
        return self.message + "\n" + extra_message


class IgnoredArgumentWarning(UserWarning):
    """Warning for ignored arguments.

    Used to notify that an argument is ignored when using a method
    or a function.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import IgnoredArgumentWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Argument 'x' is ignored in this context.", IgnoredArgumentWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, IgnoredArgumentWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=IgnoredArgumentWarning)"
        )
        return self.message + "\n" + extra_message


class InputTypeWarning(UserWarning):
    """Warning for inefficient input format.

    Used to notify that input format is not the most efficient or
    recommended for the forecaster.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import InputTypeWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(
                "Input format is not optimal for this forecaster.", InputTypeWarning
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, InputTypeWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=InputTypeWarning)"
        )
        return self.message + "\n" + extra_message


class LongTrainingWarning(UserWarning):
    """Warning for potentially long training processes.

    Used to notify that a large number of models will be trained and the
    the process may take a while to run.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import LongTrainingWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Training may take a long time.", LongTrainingWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, LongTrainingWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=LongTrainingWarning)"
        )
        return self.message + "\n" + extra_message


class MissingExogWarning(UserWarning):
    """Warning for missing exogenous variables.

    Used to indicate that there are missing exogenous variables in the
    data. Most machine learning models do not accept missing values, so the
    Forecaster's `fit' and `predict' methods may fail.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import MissingExogWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Missing exogenous variables detected.", MissingExogWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, MissingExogWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingExogWarning)"
        )
        return self.message + "\n" + extra_message


class MissingValuesWarning(UserWarning):
    """Warning for missing values in data.

    Used to indicate that there are missing values in the data. This
    warning occurs when the input data contains missing values, or the training
    matrix generates missing values. Most machine learning models do not accept
    missing values, so the Forecaster's `fit' and `predict' methods may fail.

    Args:
        message (str): The message to display.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import MissingValuesWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Missing values detected in input data.", MissingValuesWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, MissingValuesWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingValuesWarning)"
        )
        return self.message + "\n" + extra_message


class OneStepAheadValidationWarning(UserWarning):
    """Warning for one-step-ahead validation usage.

    Used to notify that the one-step-ahead validation is being used.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import OneStepAheadValidationWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(
                "Using one-step-ahead validation.", OneStepAheadValidationWarning
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, OneStepAheadValidationWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)"
        )
        return self.message + "\n" + extra_message


class ResidualsUsageWarning(UserWarning):
    """Warning for incorrect residuals usage.

    Used to notify that a residual are not correctly used in the
    probabilistic forecasting process.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import ResidualsUsageWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Residuals are not properly used.", ResidualsUsageWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, ResidualsUsageWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ResidualsUsageWarning)"
        )
        return self.message + "\n" + extra_message


class UnknownLevelWarning(UserWarning):
    """Warning for unknown levels in prediction.

    Used to notify that a level being predicted was not part of the
    training data.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import UnknownLevelWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Predicting for an unknown level.", UnknownLevelWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, UnknownLevelWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=UnknownLevelWarning)"
        )
        return self.message + "\n" + extra_message


class SaveLoadSkforecastWarning(UserWarning):
    """Warning for save/load operations.

    Used to notify any issues that may arise when saving or loading
    a forecaster.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import SaveLoadSkforecastWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(
                "Issues detected when saving forecaster.", SaveLoadSkforecastWarning
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, SaveLoadSkforecastWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SaveLoadSkforecastWarning)"
        )
        return self.message + "\n" + extra_message


class SpotforecastVersionWarning(UserWarning):
    """Warning for version mismatch.

    Used to notify that the version installed in the
    environment differs from the version used to initialize the forecaster.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import SpotforecastVersionWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Version mismatch detected.", SpotforecastVersionWarning)

        assert len(caught) == 1
        assert issubclass(caught[0].category, SpotforecastVersionWarning)
        print(caught[0].category.__name__)
        ```
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SpotforecastVersionWarning)"
        )
        return self.message + "\n" + extra_message


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples:
        ```{python}
        from spotforecast2_safe.exceptions import NotFittedError

        try:
            raise NotFittedError("Forecaster not fitted")
        except NotFittedError as e:
            print(type(e).__name__, str(e))

        assert issubclass(NotFittedError, ValueError)
        assert issubclass(NotFittedError, AttributeError)
        ```
    """


class PredictionPackageError(RuntimeError):
    """Exception raised when building a prediction package fails.

    Raised by `ForecasterRecursiveModel.package_prediction()` (and propagated
    by callers such as `manager.predictor.get_model_prediction()`) when the
    underlying prediction pipeline cannot produce a complete result.
    Inherits from `RuntimeError` so callers that catch `RuntimeError` keep
    working; the dedicated class lets safety-critical callers distinguish a
    prediction-pipeline failure from generic runtime errors.

    Examples:
        ```{python}
        from spotforecast2_safe.exceptions import PredictionPackageError

        try:
            raise PredictionPackageError("Predict step returned no rows")
        except PredictionPackageError as e:
            print(type(e).__name__, str(e))

        assert issubclass(PredictionPackageError, RuntimeError)
        ```
    """


warn_skforecast_categories = [
    DataTypeWarning,
    DataTransformationWarning,
    ExogenousInterpretationWarning,
    FeatureOutOfRangeWarning,
    IgnoredArgumentWarning,
    InputTypeWarning,
    LongTrainingWarning,
    MissingExogWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning,
    ResidualsUsageWarning,
    UnknownLevelWarning,
    SaveLoadSkforecastWarning,
    SpotforecastVersionWarning,
]


def format_warning_handler(
    message: str,
    category: str,
    filename: str,
    lineno: str,
    file: object = None,
    line: str = None,
) -> None:
    """Custom warning handler to format warnings in a box.

    Args:
        message: Warning message.
        category: Warning category.
        filename: Filename where the warning was raised.
        lineno: Line number where the warning was raised.
        file: File where the warning was raised.
        line: Line where the warning was raised.

    Returns:
        None

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import (
            MissingValuesWarning,
            set_warnings_style,
        )

        # Activate the custom box-formatted handler, then emit one warning.
        set_warnings_style("skforecast")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Missing values in test data.", MissingValuesWarning)

        # Restore default style so subsequent examples are unaffected.
        set_warnings_style("default")
        assert len(caught) == 1
        print("handler demo: warning category =", caught[0].category.__name__)
        ```
    """

    if isinstance(message, tuple(warn_skforecast_categories)):
        width = 88
        title = type(message).__name__
        output_text = ["\n"]

        wrapped_message = textwrap.fill(
            str(message), width=width - 2, expand_tabs=True, replace_whitespace=True
        )
        title_top_border = f"╭{'─' * ((width - len(title) - 2) // 2)} {title} {'─' * ((width - len(title) - 2) // 2)}╮"
        if len(title) % 2 != 0:
            title_top_border = title_top_border[:-1] + "─" + "╮"
        bottom_border = f"╰{'─' * width}╯"
        output_text.append(title_top_border)

        for line in wrapped_message.split("\n"):
            output_text.append(f"│ {line.ljust(width - 2)} │")

        output_text.append(bottom_border)
        output_text = "\n".join(output_text)
        color = "\033[38;5;208m"
        reset = "\033[0m"
        output_text = f"{color}{output_text}{reset}"
        print(output_text)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


def rich_warning_handler(
    message: str,
    category: str,
    filename: str,
    lineno: str,
    file: object = None,
    line: str = None,
) -> None:
    """Custom handler for warnings that uses rich to display formatted panels.

    Args:
        message: Warning message.
        category: Warning category.
        filename: Filename where the warning was raised.
        lineno: Line number where the warning was raised.
        file: File where the warning was raised.
        line: Line where the warning was raised.

    Returns:
        None

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import (
            InputTypeWarning,
            set_warnings_style,
        )

        # Activate the rich (or box-formatted fallback) handler, emit one warning.
        set_warnings_style("skforecast")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Input format is suboptimal.", InputTypeWarning)

        # Restore default style so subsequent examples are unaffected.
        set_warnings_style("default")
        assert len(caught) == 1
        print("rich handler demo: warning category =", caught[0].category.__name__)
        ```
    """

    if isinstance(message, tuple(warn_skforecast_categories)):
        if not HAS_RICH:
            # Fallback to format_warning_handler if rich is not available
            format_warning_handler(message, category, filename, lineno, file, line)
            return

        console = Console()

        category_name = category.__name__
        text = (
            f"{message.message}\n\n"
            f"Category : spotforecast2.exceptions.{category_name}\n"
            f"Location : {filename}:{lineno}\n"
            f"Suppress : warnings.simplefilter('ignore', category={category_name})"
        )

        panel = Panel(
            Text(text, justify="left"),
            title=category_name,
            title_align="center",
            border_style="color(214)",
            width=88,
        )

        console.print(panel)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


def set_warnings_style(style: str = "skforecast") -> None:
    """Set the warning handler based on the provided style.

    Args:
        style: The style of the warning handler. Either 'skforecast' or 'default'.

    Returns:
        None

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import (
            IgnoredArgumentWarning,
            set_warnings_style,
        )

        # Switch to skforecast box-formatted style and emit a warning.
        set_warnings_style("skforecast")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("Argument 'verbose' is ignored here.", IgnoredArgumentWarning)
        assert len(caught) == 1

        # Switch back to Python default style.
        set_warnings_style("default")
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            warnings.warn("Argument 'verbose' is ignored here.", IgnoredArgumentWarning)
        assert len(caught2) == 1
        print("style toggle: both modes captured 1 warning each")
        ```
    """
    if style == "skforecast":
        if not hasattr(warnings, "_original_showwarning"):
            warnings._original_showwarning = warnings.showwarning
        if HAS_RICH:
            warnings.showwarning = rich_warning_handler
        else:
            warnings.showwarning = format_warning_handler
    else:
        if hasattr(warnings, "_original_showwarning"):
            warnings.showwarning = warnings._original_showwarning


set_warnings_style(style="skforecast")


def set_skforecast_warnings(suppress_warnings: bool, action: str = "ignore") -> None:
    """Suppress or configure spotforecast warning filters.

    Iterates over all spotforecast warning categories and registers
    a `warnings.simplefilter` for each one.

    Args:
        suppress_warnings (bool): If True, apply `action` to all spotforecast
            warning categories.
        action (str): Filter action passed to `warnings.simplefilter`.
            Common values are ``'ignore'``, ``'always'``, and ``'once'``.
            Defaults to ``'ignore'``.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.exceptions import (
            MissingValuesWarning,
            ResidualsUsageWarning,
            set_skforecast_warnings,
        )

        # Suppress all spotforecast warnings.
        with warnings.catch_warnings(record=True) as caught_suppressed:
            warnings.simplefilter("always")   # baseline: catch everything first
            set_skforecast_warnings(suppress_warnings=True, action="ignore")
            warnings.warn("Missing values.", MissingValuesWarning)
            warnings.warn("Bad residuals.", ResidualsUsageWarning)

        suppressed = [w for w in caught_suppressed
                      if issubclass(w.category, (MissingValuesWarning, ResidualsUsageWarning))]
        assert len(suppressed) == 0, f"Expected 0 suppressed warnings, got {len(suppressed)}"
        print(f"suppressed warnings: {len(suppressed)}")

        # Re-enable all spotforecast warnings with 'always'.
        with warnings.catch_warnings(record=True) as caught_enabled:
            set_skforecast_warnings(suppress_warnings=True, action="always")
            warnings.warn("Missing values.", MissingValuesWarning)
            warnings.warn("Bad residuals.", ResidualsUsageWarning)

        enabled = [w for w in caught_enabled
                   if issubclass(w.category, (MissingValuesWarning, ResidualsUsageWarning))]
        assert len(enabled) == 2, f"Expected 2 enabled warnings, got {len(enabled)}"
        print(f"enabled warnings: {len(enabled)}")

        # suppress_warnings=False is a no-op.
        with warnings.catch_warnings(record=True) as caught_noop:
            warnings.simplefilter("always")
            set_skforecast_warnings(suppress_warnings=False)
            warnings.warn("Missing values.", MissingValuesWarning)

        noop = [w for w in caught_noop if issubclass(w.category, MissingValuesWarning)]
        assert len(noop) == 1
        print(f"no-op (False) warnings: {len(noop)}")
        ```
    """
    if suppress_warnings:
        for category in warn_skforecast_categories:
            warnings.simplefilter(action, category=category)
