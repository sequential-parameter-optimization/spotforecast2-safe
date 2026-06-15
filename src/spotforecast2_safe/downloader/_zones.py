# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Single source of truth for the German TSO zone constants.

This module is kept import-free (zero intra-package imports) so that both
``downloader.resilience`` and ``downloader.entsoe`` can depend on it without
introducing a cycle.  It forms the leaf of the layering DAG:

    _zones  <-  data.fetch_data  <-  downloader.entsoe  <-  downloader.resilience

No other module in ``spotforecast2_safe`` should define these constants
independently.
"""

ZONE_COLUMNS: list[str] = [
    "load_amprion",
    "load_tennet",
    "load_transnetbw",
    "load_50hertz",
]

ZONE_MODEL_FILENAME = "energy_load_zones.csv"
