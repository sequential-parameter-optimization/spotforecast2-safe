# Security Policy

## Reporting a Vulnerability

**Please do NOT create public GitHub issues for security vulnerabilities.** This puts all users at risk.

Instead, please use GitHub's **Private Security Advisory** feature:

1. Go to: https://github.com/sequential-parameter-optimization/spotforecast2-safe/security/advisories
2. Click **"Report a vulnerability"**
3. Provide details about the vulnerability

Alternatively, email your findings to:
```
bartzbeielstein@users.noreply.github.com
```

with the subject line: `[SECURITY] spotforecast2-safe Vulnerability Report`

Include:

- **Description** of the vulnerability
- **Affected version(s)**
- **Steps to reproduce** (if applicable)
- **Potential impact**
- **Suggested fix** (if available)

## Response Timeline

- **Acknowledgment**: Within 24 hours
- **Initial assessment**: Within 3 business days
- **Fix and patch**: Varies based on severity
- **Public disclosure**: Coordinated after patch is available

## Security Advisories

Published security advisories can be found in the [GitHub Security Advisories](https://github.com/sequential-parameter-optimization/spotforecast2-safe/security/advisories) section.

## Supported Versions

| Version | Status | End of Support |
|---------|--------|----------------|
| 0.3.x   | ✅ Supported | Oct 2027 |
| 0.2.x   | ⚠️ Limited   | Feb 2026 |
| < 0.2.0 | ❌ Unsupported | N/A |

## Compliance & Standards

- **REUSE Compliant**: All code contains SPDX license headers
- **SPDX**: Files use `SPDX-License-Identifier` headers
- **EU AI Act**: Support for compliance via [MODEL_CARD.md](../MODEL_CARD.md)
- **Python**: Requires Python 3.13+
- **OpenSSF**: Scorecard monitoring enabled

## Security Best Practices for Users

### Current Version

1. Always use the latest available version from [PyPI](https://pypi.org/project/spotforecast2-safe/)
2. Review [CHANGELOG.md](../CHANGELOG.md) for security patches
3. Monitor [GitHub Releases](https://github.com/sequential-parameter-optimization/spotforecast2-safe/releases) for updates

### Production Deployment

1. Pin exact versions in `requirements.txt` or `pyproject.toml`
2. Use virtual environments (`venv` or `conda`)
3. Keep dependencies updated via your dependency management tool
4. Review [MODEL_CARD.md](../MODEL_CARD.md) for safety-critical considerations

### Development

1. Use `pre-commit` hooks for code quality
2. Enable local GPG commit signing
3. Follow the contribution guidelines
4. Run full test suite before submitting PRs

## Dependencies & Supply Chain Security

This project maintains minimal dependencies to reduce the attack surface:

```python
dependencies = [
    "astral",           # Solar position calculations
    "feature-engine",   # Feature preprocessing
    "flake8",          # Code linting  
    "holidays",        # Holiday calendars
    "lightgbm",        # Gradient boosting
    "numba",           # JIT compilation
    "pandas",          # Data handling
    "pyarrow",         # Parquet/Arrow support
    "requests",        # HTTP client
    "scikit-learn",    # ML utilities
    "tqdm",            # Progress bars
]
```

**Supply Chain Measures:**
- ✅ All dependencies pinned with compatible release specifiers
- ✅ Dependabot enabled for automated dependency updates
- ✅ GitHub Actions pinned to specific commit hashes
- ✅ REUSE compliance for license tracking
- ✅ Regular security scanning via bandit and Safety

## Development Setup

For contributors, ensure security best practices:

```bash
# Clone repository
git clone https://github.com/sequential-parameter-optimization/spotforecast2-safe.git
cd spotforecast2-safe

# Set up GPG signing (optional but recommended)
git config --local commit.gpgsign true
git config --local tag.gpgsign true

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run security checks locally
bandit -r src/spotforecast2_safe/
safety check
```

## Continuous Integration Security

All commits to `main` and `develop` branches undergo:

1. **REUSE Compliance Check**: License header verification
2. **Code Quality**: Black, isort, ruff, mypy
3. **Security Scanning**: bandit, Safety
4. **Test Coverage**: pytest with coverage reporting
5. **Dependency Analysis**: Dependabot automated updates

## Contact

For general security inquiries: https://github.com/sequential-parameter-optimization

For vulnerability reports: Use private advisory feature or email above

---

**Last Updated**: February 2026
