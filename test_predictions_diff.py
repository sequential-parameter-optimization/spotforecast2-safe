"""Quick test to compare baseline vs covariate predictions."""

import warnings
warnings.simplefilter("ignore")

from spotforecast2_safe.processing.n2n_predict import n2n_predict
from spotforecast2_safe.processing.n2n_predict_with_covariates import n2n_predict_with_covariates
from spotforecast2_safe.processing.agg_predict import agg_predict

FORECAST_HORIZON = 24
WEIGHTS = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]

print("Running baseline predictions...")
baseline_preds, _ = n2n_predict(
    forecast_horizon=FORECAST_HORIZON,
    force_train=True,  # Force retrain to use new model directory
    verbose=True,
    show_progress=True
)
baseline_combined = agg_predict(baseline_preds, weights=WEIGHTS)

print("\n" + "="*80)
print("Running covariate predictions...")
print("="*80)
cov_preds, _, _ = n2n_predict_with_covariates(
    forecast_horizon=FORECAST_HORIZON,
    lags=24,
    train_ratio=0.8,
    force_train=False,  # Persistence disabled, force_train doesn't matter
    verbose=True,
    show_progress=True
)
cov_combined = agg_predict(cov_preds, weights=WEIGHTS)

print("\n=== COMPARISON ===")
print(f"Baseline combined shape: {baseline_combined.shape}")
print(f"Baseline index: {baseline_combined.index[0]} to {baseline_combined.index[-1]}")
print(f"Baseline values (first 5):")
print(baseline_combined.head())

print(f"\nCovariates combined shape: {cov_combined.shape}")
print(f"Covariates index: {cov_combined.index[0]} to {cov_combined.index[-1]}")
print(f"Covariates values (first 5):")
print(cov_combined.head())

print(f"\nAre indices aligned? {(baseline_combined.index == cov_combined.index).all()}")
print(f"Are values identical? {(baseline_combined.values == cov_combined.values).all()}")

if not (baseline_combined.values == cov_combined.values).all():
    diff = baseline_combined - cov_combined
    print(f"\nDifference stats:")
    print(diff.describe())
    print(f"\nMax absolute difference: {diff.abs().max()}")
    print(f"Mean absolute difference: {diff.abs().mean()}")
else:
    print("\n⚠️  WARNING: Predictions are IDENTICAL!")
    print("This suggests both models are producing the same output.")
