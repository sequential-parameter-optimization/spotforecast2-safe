# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from typing import Dict, List, Tuple, Union

from joblib import dump, load

# ============================================================================
# Model Persistence Functions
# ============================================================================


def _ensure_model_dir(model_dir: Union[str, Path]) -> Path:
    """Ensure model directory exists.

    Args:
        model_dir: Directory path for model storage.

    Returns:
        Path: Validated Path object.

    Raises:
        OSError: If directory cannot be created.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.persistence import _ensure_model_dir
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     model_path = _ensure_model_dir(Path(tmpdir) / "models")
        ...     print(model_path.exists())
        True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     nested = _ensure_model_dir(Path(tmpdir) / "a" / "b" / "c")
        ...     print(nested.is_dir())
        True
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path


def _get_model_filepath(model_dir: Path, target: str) -> Path:
    """Get filepath for a single model.

    Args:
        model_dir: Directory containing models.
        target: Target variable name.

    Returns:
        Path: Full filepath for the model.

    Examples:
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.persistence import _get_model_filepath
        >>> path = _get_model_filepath(Path("./models"), "power")
        >>> str(path)
        'models/forecaster_power.joblib'
        >>> path = _get_model_filepath(Path("/tmp/models"), "energy")
        >>> path.name
        'forecaster_energy.joblib'
        >>> path.suffix
        '.joblib'
    """
    return model_dir / f"forecaster_{target}.joblib"


def _save_forecasters(
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
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.persistence import _save_forecasters
        >>> from sklearn.linear_model import LinearRegression
        >>> mock_model = LinearRegression()
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     forecasters = {"power": mock_model, "energy": mock_model}
        ...     paths = _save_forecasters(forecasters, tmpdir, verbose=False)
        ...     print("power" in paths)
        ...     print(paths["power"].exists())
        True
        True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     paths = _save_forecasters({"demand": mock_model}, tmpdir)
        ...     print(paths["demand"].suffix)
        .joblib
    """
    model_path = _ensure_model_dir(model_dir)
    saved_paths = {}

    for target, forecaster in forecasters.items():
        filepath = _get_model_filepath(model_path, target)
        try:
            dump(forecaster, filepath, compress=3)
            saved_paths[target] = filepath
            if verbose:
                print(f"  ✓ Saved forecaster for {target} to {filepath}")
        except Exception as e:
            raise OSError(f"Failed to save model for {target}: {e}")

    return saved_paths


def _load_forecasters(
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
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.persistence import (
        ...     _save_forecasters,
        ...     _load_forecasters,
        ... )
        >>> from sklearn.linear_model import LinearRegression
        >>> mock_model = LinearRegression()
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _ = _save_forecasters({"power": mock_model}, tmpdir)
        ...     forecasters, missing = _load_forecasters(
        ...         ["power", "energy"],
        ...         tmpdir,
        ...         verbose=False
        ...     )
        ...     print("power" in forecasters)
        ...     print("energy" in missing)
        True
        True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     forecasters, missing = _load_forecasters(["nonexistent"], tmpdir)
        ...     print(len(forecasters), len(missing))
        0 1
    """
    model_path = Path(model_dir)
    forecasters = {}
    missing_targets = []

    for target in target_columns:
        filepath = _get_model_filepath(model_path, target)
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


def _model_directory_exists(model_dir: Union[str, Path]) -> bool:
    """Check if model directory exists.

    Args:
        model_dir: Directory path to check.

    Returns:
        bool: True if directory exists, False otherwise.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.persistence import _model_directory_exists
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     print(_model_directory_exists(tmpdir))
        True
        >>> print(_model_directory_exists("/nonexistent/path/to/models"))
        False
    """
    return Path(model_dir).exists()
