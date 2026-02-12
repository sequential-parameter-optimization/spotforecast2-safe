# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later


def get_cpe_identifier(version: str = "*") -> str:
    """Generates the CPE 2.3 identifier for the spotforecast2-safe project.

    This function constructs a Common Platform Enumeration (CPE) 2.3 formatted
    string that uniquely identifies the spotforecast2-safe software. CPE
    identifiers are standardized by NIST and are essential for vulnerability
    tracking, software inventory management, and compliance documentation
    (e.g., EU AI Act, SBOM generation).

    The returned CPE follows the format:
        cpe:2.3:part:vendor:product:version:update:edition:language:sw_edition:target_sw:target_hw:other

    Args:
        version: The specific version of the software. Use wildcard "*" to match
            all versions, or provide a semantic version string (e.g., "0.8.0",
            "0.8.0-rc.1"). Defaults to "*".

    Returns:
        str: The formatted Common Platform Enumeration 2.3 string.

    Raises:
        TypeError: If version is not a string.

    Examples:
        Generate a CPE identifier for all versions of spotforecast2-safe:

        >>> get_cpe_identifier()
        'cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:*:*:*:*:*:*:*:*'

        Generate a CPE identifier for a specific release version:

        >>> get_cpe_identifier("0.8.0")
        'cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:0.8.0:*:*:*:*:*:*:*'

        Generate a CPE identifier for a release candidate version:

        >>> get_cpe_identifier("0.8.0-rc.1")
        'cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:0.8.0-rc.1:*:*:*:*:*:*:*'

    Note:
        This function is used in compliance documentation, SBOM (Software Bill of
        Materials) generation, and vulnerability tracking. The CPE identifier
        should be included in release notes and security advisories.

    See Also:
        For more information on CPE 2.3 specification, visit:
        https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-188.pdf
    """
    if not isinstance(version, str):
        raise TypeError(f"version must be a string, got {type(version).__name__}")

    vendor = "sequential_parameter_optimization"
    product = "spotforecast2_safe"
    return f"cpe:2.3:a:{vendor}:{product}:{version}:*:*:*:*:*:*:*"
