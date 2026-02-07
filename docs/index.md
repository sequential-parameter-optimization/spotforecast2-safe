# Welcome to spotforecast2-safe (Core)

[![Version](https://img.shields.io/badge/version-0.0.6-blue.svg)](https://github.com/sequential-parameter-optimization/spotforecast2-safe/releases)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Ready-success)](safe/spotforecast2-safe.md)
[![Audit](https://img.shields.io/badge/audit-whitebox-brightgreen)](safe/MODEL_CARD.md)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

**spotforecast2-safe** is a specialized, hardened Python package for time series forecasting in safety-critical production environments. It provides a minimal, auditable core for feature engineering and recursive forecasting.

## Quick Links

- 📦 [GitHub Repository](https://github.com/sequential-parameter-optimization/spotforecast2-safe)
- 📚 [API Reference](api/data.md)
- �️ [Safety & Compliance](safe/spotforecast2-safe.md)
- 📊 [Model/Method Card](safe/MODEL_CARD.md)
- �🚀 Current Version: **0.0.6**

## Installation

```bash
git clone https://github.com/sequential-parameter-optimization/spotforecast2-safe.git
cd spotforecast2-safe
uv sync
```

## Safety-Critical Features

- **Zero Dead Code**: No GUI, plotting, or AutoML dependencies (No Plotly, No Optuna).
- **Deterministic Transformations**: Mathematical logic that ensures bit-level reproducibility.
- **Fail-Safe Processing**: Explicit failure on dirty or incomplete data (NaNs/Infs) instead of silent imputation.
- **Minimal Footprint**: Reduced attack surface for high-security deployment targets.

## Core Capabilities

- **Data Service**: Robust fetching of time series, weather, and holiday data.
- **Preprocessing**: Hardened tools for data curation, resampling, and temporal splitting.
- **Forecasting Engine**: Simplified recursive forecasting and seasonal baselines.

## ⚠️ Disclaimer & Liability

**IMPORTANT**: This software is provided "as is" and any express or implied warranties are disclaimed. The use of this software in safety-critical systems is at the sole risk of the user. For full details, see the [Disclaimer in the Model Card](safe/MODEL_CARD.md#8-disclaimer-liability).

## Attributions

Parts of the code are ported from `skforecast` to reduce external dependencies. Many thanks to the [skforecast team](https://skforecast.org/0.20.0/more/about-skforecast.htm) for their great work!