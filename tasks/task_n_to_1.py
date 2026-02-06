import warnings
from spotforecast2_safe.processing.n2n_predict import n2n_predict
from spotforecast2_safe.processing.agg_predict import agg_predict

warnings.simplefilter("ignore")


def main():
    FORECAST_HORIZON = 24
    CONTAMINATION = 0.01
    WINDOW_SIZE = 72
    VERBOSE = True
    SHOW_PROGRESS = True
    WEIGHTS = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]

    print("--- Starting n_to_1_task using modular functions ---")

    # --- Prediction ---
    # Fetch, Preprocess, Train, Evaluate, Predict
    predictions, _ = n2n_predict(
        columns=None,
        forecast_horizon=FORECAST_HORIZON,
        contamination=CONTAMINATION,
        window_size=WINDOW_SIZE,
        verbose=VERBOSE,
        show_progress=SHOW_PROGRESS,
    )

    print("\nMulti-output predictions head:")
    print(predictions)

    # --- Aggregation ---
    print("Calculating combined prediction...")
    combined_prediction = agg_predict(predictions, weights=WEIGHTS)

    print("Combined Prediction:")
    print(combined_prediction)


if __name__ == "__main__":
    main()
