# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from typing import Dict, List, Tuple, Union

from joblib import dump, load

# ============================================================================
# Model Persistence Functions
# ============================================================================


def ensure_model_dir(model_dir: Union[str, Path]) -> Path:
    """Ensure model directory exists.

    Args:
        model_dir: Directory path for model storage.

    Returns:
        Path: Validated Path object.

    Raises:
        OSError: If directory cannot be created.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.manager.persistence import ensure_model_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = ensure_model_dir(Path(tmpdir) / "models")
            assert model_path.exists()
            print(model_path.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = ensure_model_dir(Path(tmpdir) / "a" / "b" / "c")
            assert nested.is_dir()
            print(nested.is_dir())
        ```
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path


def get_model_filepath(model_dir: Path, target: str) -> Path:
    """Get filepath for a single model.

    Args:
        model_dir: Directory containing models.
        target: Target variable name.

    Returns:
        Path: Full filepath for the model.

    Examples:
        ```{python}
        from pathlib import Path
        from spotforecast2_safe.manager.persistence import get_model_filepath

        path = get_model_filepath(Path("/models"), "power")
        assert path.name == "forecaster_power.joblib"
        assert path.suffix == ".joblib"
        print(path.name)
        print(path.suffix)

        path2 = get_model_filepath(Path("/models"), "energy")
        assert path2.name == "forecaster_energy.joblib"
        print(path2.name)
        ```
    """
    return model_dir / f"forecaster_{target}.joblib"


def save_forecasters(
    forecasters: Dict[str, object],
    model_dir: Union[str, Path],
    verbose: bool = False,
) -> Dict[str, Path]:
    """Save trained forecasters to disk using joblib.

    Follows scikit-learn persistence conventions using joblib for efficient
    serialization of sklearn-compatible estimators.

    Args:
        forecasters: Dictionary mapping target names to trained ForecasterEquivalentDate objects.
        model_dir: Directory to save models. Created if it doesn't exist.
        verbose: Print progress messages. Default: False.

    Returns:
        Dict[str, Path]: Dictionary mapping target names to saved model filepaths.

    Raises:
        OSError: If models cannot be written to disk.
        TypeError: If forecasters contain non-serializable objects.

    Examples:
        ```{python}
        import tempfile
        import warnings
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMRegressor
        from spotforecast2_safe.forecaster.recursive._forecaster_recursive import (
            ForecasterRecursive,
        )
        from spotforecast2_safe.manager.persistence import save_forecasters

        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(size=50), name="target")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = ForecasterRecursive(
                estimator=LGBMRegressor(
                    n_estimators=10, random_state=1234, verbose=-1
                ),
                lags=3,
            )
            fc.fit(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_forecasters({"power": fc, "energy": fc}, tmpdir, verbose=False)
            assert "power" in paths
            assert paths["power"].exists()
            assert paths["power"].suffix == ".joblib"
            print("power" in paths)
            print(paths["power"].exists())
            print(paths["power"].suffix)
        ```
    """
    model_path = ensure_model_dir(model_dir)
    saved_paths = {}

    for target, forecaster in forecasters.items():
        filepath = get_model_filepath(model_path, target)
        try:
            dump(forecaster, filepath, compress=3)
            saved_paths[target] = filepath
            if verbose:
                print(f"  ✓ Saved forecaster for {target} to {filepath}")
        except Exception as e:
            raise OSError(f"Failed to save model for {target}: {e}")

    return saved_paths


def load_forecasters(
    target_columns: List[str],
    model_dir: Union[str, Path],
    verbose: bool = False,
) -> Tuple[Dict[str, object], List[str]]:
    """Load trained forecasters from disk using joblib.

    Attempts to load all forecasters for given targets. Missing models
    are indicated in the return value for selective retraining.

    Args:
        target_columns: List of target variable names to load.
        model_dir: Directory containing saved models.
        verbose: Print progress messages. Default: False.

    Returns:
        Tuple[Dict[str, object], List[str]]:
        - forecasters: Dictionary of successfully loaded ForecasterEquivalentDate objects.
        - missing_targets: List of target names without saved models.

    Examples:
        ```{python}
        import tempfile
        import warnings
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMRegressor
        from spotforecast2_safe.forecaster.recursive._forecaster_recursive import (
            ForecasterRecursive,
        )
        from spotforecast2_safe.manager.persistence import load_forecasters, save_forecasters

        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(size=50), name="target")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = ForecasterRecursive(
                estimator=LGBMRegressor(
                    n_estimators=10, random_state=1234, verbose=-1
                ),
                lags=3,
            )
            fc.fit(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            _ = save_forecasters({"power": fc}, tmpdir)
            forecasters, missing = load_forecasters(
                ["power", "energy"], tmpdir, verbose=False
            )
            assert "power" in forecasters
            assert "energy" in missing
            assert isinstance(forecasters["power"], ForecasterRecursive)
            print("power" in forecasters)
            print("energy" in missing)
            print(type(forecasters["power"]).__name__)

        with tempfile.TemporaryDirectory() as tmpdir:
            forecasters, missing = load_forecasters(["nonexistent"], tmpdir)
            assert len(forecasters) == 0
            assert len(missing) == 1
            print(len(forecasters), len(missing))
        ```
    """
    model_path = Path(model_dir)
    forecasters = {}
    missing_targets = []

    for target in target_columns:
        filepath = get_model_filepath(model_path, target)
        if filepath.exists():
            try:
                forecasters[target] = load(filepath)
                if verbose:
                    print(f"  ✓ Loaded forecaster for {target} from {filepath}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to load {target}: {e}")
                missing_targets.append(target)
        else:
            missing_targets.append(target)

    return forecasters, missing_targets


def model_directory_exists(model_dir: Union[str, Path]) -> bool:
    """Check if model directory exists.

    Args:
        model_dir: Directory path to check.

    Returns:
        bool: True if directory exists, False otherwise.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from spotforecast2_safe.manager.persistence import model_directory_exists

        with tempfile.TemporaryDirectory() as tmpdir:
            assert model_directory_exists(tmpdir)
            print(model_directory_exists(tmpdir))

        assert not model_directory_exists("/nonexistent/path/to/models")
        print(model_directory_exists("/nonexistent/path/to/models"))
        ```
    """
    return Path(model_dir).exists()


def save_forecaster(
    forecaster: object,
    model_dir: Union[str, Path],
    target: str,
    task_name: str = "",
    verbose: bool = False,
) -> Path:
    """Save a single trained forecaster to disk using joblib.

    Public single-model counterpart to `save_forecasters()`. When
    `task_name` is provided the file is named
    `{task_name}_{target}.joblib`; otherwise the standard convention
    `forecaster_{target}.joblib` (identical to `get_model_filepath()`)
    is used.

    Args:
        forecaster: Trained forecaster object (any joblib-serialisable model).
        model_dir: Directory to save the model.  Created if it doesn't exist.
        target: Target variable name used in the filename.
        task_name: Optional task identifier prepended to the filename (e.g.
            ``"task_1_lazy"``).  Defaults to ``""`` (standard naming).
        verbose: Print a confirmation message.  Default: ``False``.

    Returns:
        Path: Full filepath of the saved model.

    Raises:
        OSError: If the model cannot be written to disk.

    Examples:
        ```{python}
        import tempfile
        from pathlib import Path
        from sklearn.linear_model import LinearRegression
        from spotforecast2_safe.manager.persistence import save_forecaster

        model = LinearRegression()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_forecaster(model, tmpdir, "power")
            assert path.name == "forecaster_power.joblib"
            print(path.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_forecaster(model, tmpdir, "power", task_name="task_1")
            assert path.name == "task_1_power.joblib"
            print(path.name)
        ```
    """
    model_path = ensure_model_dir(model_dir)
    if task_name:
        filepath = model_path / f"{task_name}_{target}.joblib"
    else:
        filepath = get_model_filepath(model_path, target)
    try:
        dump(forecaster, filepath, compress=3)
        if verbose:
            print(f"  ✓ Saved forecaster for {target} to {filepath}")
    except Exception as e:
        raise OSError(f"Failed to save model for {target}: {e}")
    return filepath
