import os
print("Checking for new model directories:")
if os.path.exists("models_baseline"):
    print("✓ models_baseline exists")
    print(f"  Files: {os.listdir('models_baseline')}")
else:
    print("✗ models_baseline does not exist")

if os.path.exists("models_covariates"):
    print("✓ models_covariates exists")
    print(f"  Files: {os.listdir('models_covariates')}")
else:
    print("✗ models_covariates does not exist")

if os.path.exists("forecaster_models"):
    print("⚠ forecaster_models (old directory) still exists")
    print(f"  Files: {os.listdir('forecaster_models')}")
