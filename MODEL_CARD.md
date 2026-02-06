# Model/Method Card: spotforecast2-safe

## 1. System Details
- **Name**: spotforecast2-safe
- **Version**: 0.0.1 (Initial Safety Release)
- **Type**: Deterministic library for time series transformation and feature generation (Preprocessing).
- **License**: BSD-3-Clause
- **Developers**: bartzbeielstein
- **Repository**: [https://github.com/sequential-parameter-optimization/spotforecast2-safe](https://github.com/sequential-parameter-optimization/spotforecast2-safe)
- **Core Dependencies**: `numpy`, `pandas` (Minimal Dependency Footprint).
- **Prohibited Dependencies**: `plotly`, `matplotlib`, `spotoptim`, `optuna`, `torch`, `tensorflow`.

## 2. Intended Use
### Primary Use Cases
- **Safety-Critical Forecasting Systems**: Preparation of time series data for regression models in environments requiring auditability (e.g., energy supply, finance).
- **Embedded Systems / Edge AI**: Use in resource-constrained environments where large ML frameworks cannot be installed.
- **Reproducible Research**: Ensuring exact mathematical reproducibility of N-to-1 transformations without hidden stochastics.

### Out-of-Scope
- **Interactive Visualization**: The package deliberately contains no plotting functions to minimize the attack surface.
- **Automated Hyperparameter Tuning**: Optimization (e.g., via spotoptim or Optuna) must take place outside the "Safe Environment".
- **Silent Data Cleaning**: The package does not perform "silent" data imputation. Missing values (NaNs) lead to explicit errors (Fail-Safe), not estimations.

## 3. Algorithm & Logic
The core module `task_n_to_1` implements a deterministic sliding-window transformation.

### Mathematical Description
Given a univariate time series $X = \{x_1, x_2, ..., x_T\}$, the system transforms this into a feature matrix $X_{feat}$ and a target vector $y$ based on the window size $w$ (lags):

$$X_{row, t} = [x_{t-w}, x_{t-w+1}, ..., x_{t-1}] \rightarrow y_t = x_t$$

### Guarantees
- **Deterministic**: The same input always generates the exact same bit-level output.
- **Leakage-Free**: The implementation guarantees that the target value $y_t$ is never contained within the input vector $X_{row, t}$.

## 4. Performance & Robustness (Safety Features)
In the absence of "Accuracy" (as no model is trained), software metrics according to IEC 61508 / EU AI Act are listed here:

### Fail-Safe Behavior
- **Input**: DataFrame with `NaN` or `Inf`.
- **Behavior**: Throws an explicit `ValueError`. No silent processing (Silent Failure).

### Input Validation
- **Strict Checks**: Type hinting and runtime checks for `pd.DataFrame` and `np.ndarray`.

### Cybersecurity Footprint
- **Minimal CVE Surface**: By avoiding complex dependencies (like PyTorch or web server components), the Common Vulnerabilities and Exposures (CVE) attack surface is minimized.

## 5. Compliance & EU AI Act Reference
This package serves as a base component for high-risk AI systems according to the EU AI Act.

- **Transparency (Art. 13)**: The code is fully transparent ("White Box"). There are no compiled binaries or obfuscated models.
- **Accuracy & Robustness (Art. 15)**: The transformations are mathematically provably correct. There are no stochastic elements (random numbers) that jeopardize reproducibility.
- **Data Governance (Art. 10)**: The package enforces clean data formats by rejecting "dirty" data (wrong types, gaps) instead of guessing.

## 6. Caveats & Limitations
- **No Extrapolation**: The package prepares data; it does not predict by itself. The quality of the forecast depends on the downstream regressor (e.g., `scikit-learn` LinearRegression).
- **Memory Requirements**: Creating the Lag matrix (N-to-1) can be memory-intensive for extremely large time series ($T > 10^7$) as data is duplicated.

## 7. How to Audit
For auditors who need to validate this package:
1. Check `pyproject.toml` to confirm the absence of unsafe libraries.
2. Run `pytest tests/` to verify the functional correctness of the matrix transformation.
3. Check the hash values of input and output data to prove determinism.

## 8. Disclaimer & Liability
**LIMITATION OF LIABILITY**: While this library is designed with safety principles and deterministic logic in mind, it is provided "AS IS" without any warranties. The developers and contributors assume **NO LIABILITY** for any direct or indirect damages, system failures, or financial losses resulting from the use of this software. 

It is the sole responsibility of the system integrator to perform a full system-level safety validation (e.g., as per ISO 26262, IEC 61508, or the EU AI Act) before deploying this software in a production or safety-critical environment.

