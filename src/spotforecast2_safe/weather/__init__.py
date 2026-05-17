# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weather data fetching and processing using Open-Meteo API.

Threat model (STRIDE).
    This module crosses a network boundary and therefore carries its own
    STRIDE table. The table enumerates, for every data flow, which of the six
    categories apply, which countermeasure is in force, and where that
    countermeasure is implemented. Contributors who change the surface listed
    below MUST update this table in the same pull request; the rule is
    anchored in CONTRIBUTING.md ("Threat-model update rule") and gated by the
    checklist in .github/pull_request_template.md.

    Data flow 1: outbound HTTPS request to the Open-Meteo API
    (``ARCHIVE_BASE_URL`` and ``FORECAST_BASE_URL`` below).

    - Spoofing: a forged endpoint could serve crafted weather series that
      then drive a downstream forecast. Mitigated by default TLS certificate
      verification in ``requests`` and by the pinned host constants; do not
      pass ``verify=False``.
    - Tampering: an on-path attacker could modify the JSON payload.
      Mitigated by TLS integrity and by the typed pandas coercion below that
      rejects any column whose dtype diverges from the declared schema.
    - Repudiation: Open-Meteo does not require authentication, so there is
      no caller identity at this module boundary. Downstream audit is
      provided by ``manager/logger.py``, which records the fetch URL, the
      request UTC timestamp, and the response-byte hash.
    - Information Disclosure: the request URL contains latitude and
      longitude, which reveal the deployment location. Mitigated by the
      logger's URL filter, which records only the hostname and path (not
      the query string) at the default log level.
    - Denial of Service: upstream outages or rate limiting could stall the
      pipeline. Mitigated by the ``Retry`` adapter configured below, which
      caps retries and raises an explicit exception after the budget is
      exhausted rather than returning an empty DataFrame.
    - Elevation of Privilege: the client runs with the host-process
      privileges; it opens no setuid boundary. Not applicable at module
      scope.

    Data flow 2: on-disk parquet cache under the operator-supplied cache
    directory.

    - Tampering: an attacker with write access to the cache could plant a
      malformed parquet file. Mitigated by the round-trip schema contract
      enforced on read (covered by
      ``tests/test_weather_client.py::TestWeatherServiceCache``), which
      raises rather than silently accepting a shape-mismatched file.
    - Information Disclosure: cached files contain weather series indexed
      by the operator's coordinates. Not sensitive in isolation, but
      operators running this library against colocation-sensitive
      deployments must configure cache-directory permissions; this module
      does not set permissions on behalf of the caller.
    - Other STRIDE categories: not applicable for a local filesystem flow
      whose threat model is owned by the host operating system.
"""

from .client import WeatherClient, WeatherService
from .features import get_weather_features

__all__ = ["WeatherClient", "WeatherService", "get_weather_features"]
