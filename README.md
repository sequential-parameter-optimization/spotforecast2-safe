<div align="left">
  <img src="logo/spotlogo.png" alt="spotforecast2-safe Logo" width="300">
</div>

# spotforecast2-safe (Core)

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Ready-success)](MODEL_CARD.md)
[![Dependencies](https://img.shields.io/badge/dependencies-minimal-blue)](pyproject.toml)
[![Audit](https://img.shields.io/badge/audit-whitebox-brightgreen)](MODEL_CARD.md)
[![License](https://img.shields.io/github/license/sequential-parameter-optimization/spotforecast2-safe)](LICENSE)

**Testing & Quality**

[![Build Status](https://img.shields.io/github/actions/workflow/status/sequential-parameter-optimization/spotforecast2-safe/ci.yml?branch=main&label=Tests)](https://github.com/sequential-parameter-optimization/spotforecast2-safe/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://sequential-parameter-optimization.github.io/spotforecast2-safe/)
[![Reliability](https://img.shields.io/badge/robustness-fail--safe-orange)](MODEL_CARD.md)

**Status**

[![Maintenance](https://img.shields.io/badge/maintenance-active-green)](https://github.com/sequential-parameter-optimization/spotforecast2-safe)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Safety-Critical Design

`spotforecast2-safe` is a specialized Python library designed for time series forecasting in **safety-critical production environments** and **embedded systems**. 

Unlike standard ML libraries, it follows a strict **"Safety-First"** architecture:
- **Zero Dead Code**: The package contains no visualization (Plotly), optimization (Optuna), or training logic. This significantly simplifies code audits and minimizes the cybersecurity attack surface (CVE reduction).
- **Deterministic Logic**: Transformations are purely mathematical and deterministic. Identical inputs always yield identical, bit-perfect outputs.
- **Fail-Safe Operation**: Explicit rejection of incomplete or "dirty" data (NaNs/Infs). The system favors a controlled crash over a silent failure.
- **EU AI Act Compliance**: Engineered for transparency and data governance as required for high-risk AI components.

For a detailed technical audit of our safety mechanisms, see our **[MODEL_CARD.md](MODEL_CARD.md)**.


Parts of the code are ported from `skforecast` to reduce external dependencies.
Many thanks to the [skforecast team](https://skforecast.org/0.20.0/more/about-skforecast.html) for their great work!

## Documentation

 Documentation (API) is available at: [https://sequential-parameter-optimization.github.io/spotforecast2-safe/](https://sequential-parameter-optimization.github.io/spotforecast2-safe/)

## License

`spotforecast2-safe` software: [BSD-3-Clause License](LICENSE)


# References

## skforecast: 

* Amat Rodrigo, J., & Escobar Ortiz, J. (2026). skforecast (Version 0.20.0) [Computer software]. https://doi.org/10.5281/zenodo.8382788 

## spotoptim:

* [spotoptim documentation](https://sequential-parameter-optimization.github.io/spotoptim/)