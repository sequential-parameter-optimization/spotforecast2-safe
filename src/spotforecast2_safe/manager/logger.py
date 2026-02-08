# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def setup_logging(
    level: int = logging.INFO, log_dir: Optional[Path] = None
) -> Tuple[logging.Logger, Optional[Path]]:
    """
    Configure robust logging for safety-critical execution.

    Sets up both a stream (stdout) and an optional informative file handler.
    Always logs INFO level or higher to the file, regardless of console level.

    Args:
        level: Logging level for console output. Default: logging.INFO
        log_dir: Optional directory for log files. If provided, creates timestamped log files.

    Returns:
        Tuple[logging.Logger, Optional[Path]]: Logger instance and optional log file path.

    Raises:
        None: Warnings are logged if file handler creation fails.

    Examples:
        >>> import logging
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.logger import setup_logging
        >>>
        >>> # Example 1: Basic console-only logging
        >>> logger, log_path = setup_logging(level=logging.INFO)  # doctest: +SKIP
        >>> print(f"Logger name: {logger.name}")  # doctest: +SKIP
        Logger name: task_safe_n_to_1
        >>> print(f"Log file path: {log_path}")  # doctest: +SKIP
        Log file path: None
        >>>
        >>> # Example 2: File and console logging for audit trail
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     log_dir = Path(tmpdir)
        ...     logger, log_path = setup_logging(level=logging.DEBUG, log_dir=log_dir)
        ...     logger.info("Model training started")
        ...     logger.debug("Hyperparameter: learning_rate=0.01")
        ...     print(f"Log file created: {log_path.exists()}")
        ...     print(f"Log file suffix: {log_path.suffix}")
        Log file created: True
        Log file suffix: .log
        >>>
        >>> # Example 3: Verify log levels and file creation
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     logger, log_path = setup_logging(log_dir=Path(tmpdir))
        ...     # Verify logger is configured
        ...     assert logger.name == "task_safe_n_to_1"
        ...     assert log_path is not None or len(logger.handlers) > 0
        ...     print("Logger configured successfully")
        Logger configured successfully
        >>>
        >>> # Example 4: Safety-critical scenario - file path verification
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     _, log_path = setup_logging(log_dir=Path(tmpdir))
        ...     # In actual usage, log_path may be None if called multiple times
        ...     # First call creates file, subsequent calls reuse handlers
        ...     print("Logging system ready for audit trail")
        Logging system ready for audit trail
    """
    logger = logging.getLogger("task_safe_n_to_1")
    logger.setLevel(logging.DEBUG)  # Root level allows handlers to filter

    # Avoid duplicate handlers if main is called multiple times
    if logger.handlers:
        existing_path = None
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                existing_path = Path(h.baseFilename)
        return logger, existing_path

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 1. Console Handler (Respects the requested level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # 2. File Handler (Always INFO+ for audit durability)
    log_file_path = None
    if log_dir:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = log_dir / f"task_safe_n_to_1_{timestamp}.log"

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.info(f"Persistent logging initialized at: {log_file_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize file logging in {log_dir}: {e}")

    return logger, log_file_path
