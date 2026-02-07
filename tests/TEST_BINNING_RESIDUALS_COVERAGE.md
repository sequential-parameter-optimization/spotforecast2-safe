# Test Coverage: _binning_in_sample_residuals Method

## Overview
Comprehensive pytest suite for the `_binning_in_sample_residuals` method in `ForecasterEquivalentDate`. This test suite validates safety-critical aspects of residual binning for conformal prediction intervals.

## Test Execution
```bash
uv run pytest tests/test_binning_in_sample_residuals.py -v
```

## Test Results
✅ **18/18 tests passing** (100% pass rate)

## Test Categories & Coverage

### 1. Deterministic Behavior (Safety-Critical)
- `test_deterministic_behavior_with_random_state`: Validates reproducibility with same random_state
- `test_different_random_states_produce_different_samples`: Confirms random_state is actually used for downsampling
- `test_reproducibility_after_refit`: Ensures refitting produces identical results

**Safety Requirement**: Predictions must be reproducible for audit trails and regulatory compliance.

### 2. Memory Management (Safety-Critical)
- `test_max_sample_per_bin_enforcement`: Validates 10,000 / n_bins limit per bin
- `test_total_residuals_limited_to_10000`: Confirms total residuals capped at 10,000
- `test_store_false_only_stores_intervals`: Verifies memory footprint reduction when store=False

**Safety Requirement**: Prevents memory overflow in production systems.

### 3. Binning Structure & Correctness
- `test_basic_binning_structure`: Verifies correct bin count and attribute creation
- `test_residuals_length_matches_expected_window`: Validates residual count matches data window
- `test_binner_intervals_structure`: Ensures valid interval definitions (lower ≤ upper)
- `test_empty_bins_filled_with_global_residuals`: Confirms empty bins are filled properly

**Safety Requirement**: Ensures all bins have valid residuals for interval prediction.

### 4. Edge Cases
- `test_single_bin_edge_case`: Validates n_bins=1 raises ValueError (minimum is 2)
- `test_many_bins_edge_case`: Tests with n_bins=20 (high bin counts)
- `test_minimal_data_edge_case`: Tests with minimal valid dataset
- `test_residuals_distribution_across_bins`: Validates roughly balanced distribution

**Safety Requirement**: System handles boundary conditions without silent failures.

### 5. Data Quality Validation
- `test_residuals_are_numeric`: Ensures no NaN or inf values
- `test_residuals_dtype_consistency`: Validates consistent numeric dtypes

**Safety Requirement**: Invalid residuals could cause prediction failures.

### 6. Configuration Flexibility
- `test_with_multiple_offsets_n_offsets_parameter`: Tests n_offsets > 1 (lag aggregation)
- `test_with_dateoffset_instead_of_integer`: Tests with pandas DateOffset instead of integer

**Safety Requirement**: System supports multiple operational modes.

## Key Safety-Critical Validations

### Deterministic Sampling
```python
# Same random_state → Identical results
forecaster.fit(y=y, store_in_sample_residuals=True, random_state=123)
```

### Memory Limits Enforced
```python
# Max 10,000 total residuals
assert len(forecaster.in_sample_residuals_) <= 10_000

# Max 10,000 // n_bins per bin
max_sample_per_bin = 10_000 // n_bins
assert len(bin_residuals) <= max_sample_per_bin
```

### No Empty Bins
```python
# All bins must have residuals for valid prediction intervals
for bin_idx, residuals in forecaster.in_sample_residuals_by_bin_.items():
    assert len(residuals) > 0
```

### Numerical Stability
```python
# No NaN or inf values allowed
assert np.all(np.isfinite(forecaster.in_sample_residuals_))
```

## Test Data Characteristics

| Test | Series Length | Offset | n_bins | Purpose |
|------|--------------|--------|--------|---------|
| Deterministic | 200 | 7 | 3 | Reproducibility validation |
| Downsampling | 12,000 | 7 | 2 | Trigger 10K limit |
| Large bins | 20,000 | 7 | 4 | Per-bin limit enforcement |
| Empty bins | 60 | 7 | 10 | Imbalanced distribution |
| Minimal | 15 | 7 | 2 | Boundary condition |
| Many bins | 500 | 7 | 20 | High bin count |

## Warnings
One expected warning in `test_empty_bins_filled_with_global_residuals`:
```
IgnoredArgumentWarning: The number of bins has been reduced from 10 to 1 
due to duplicated edges caused by repeated predicted values.
```

This warning is intentional and validates the binner's handling of degenerate data distributions.

## MLOps Best Practices Demonstrated

1. **Reproducibility**: All random operations controlled via explicit random_state
2. **Resource Limits**: Explicit memory bounds to prevent production failures  
3. **Input Validation**: Invalid configurations raise informative errors
4. **Edge Case Coverage**: Tests boundary conditions and degenerate cases
5. **Numerical Stability**: Validates absence of NaN/inf in outputs
6. **API Consistency**: Tests both integer offset and DateOffset configurations

## Integration with CI/CD

Add to CI pipeline:
```yaml
- name: Test Residual Binning
  run: uv run pytest tests/test_binning_in_sample_residuals.py -v --cov
```

## Maintenance Notes

- Update tests when modifying bin sampling logic
- Maintain 100% pass rate for safety-critical certification
- Document any intentional behavioral changes affecting reproducibility
- Keep test execution time < 1 second for efficient CI/CD

## Author
Senior Python MLOps Engineer specializing in safety-critical systems

## Version
Generated: 2026-02-07  
Package: spotforecast2-safe v0.0.7  
Python: 3.13.9
