# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""MkDocs hooks for dynamic version injection from pyproject.toml."""

import tomllib
from pathlib import Path


def define_env(env):
    """Define custom MkDocs macros variables.

    This hook is called by mkdocs-macros-plugin to define variables
    that can be used in markdown files with {{ variable_name }} syntax.
    """
    # Find pyproject.toml relative to the docs directory
    docs_dir = Path(env.conf["docs_dir"])
    project_root = docs_dir.parent
    pyproject_path = project_root / "pyproject.toml"

    version = "unknown"
    project_name = "spotforecast2-safe"

    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
            version = pyproject.get("project", {}).get("version", "unknown")
            project_name = pyproject.get("project", {}).get("name", project_name)

    # Make variables available in markdown files
    env.variables["version"] = version
    env.variables["project_name"] = project_name

    # URL-encoded version for badges
    env.variables["version_badge"] = version.replace("-", "--")
