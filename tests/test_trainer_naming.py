import tempfile
from pathlib import Path
import pandas as pd
from spotforecast2_safe.manager.trainer import handle_training, get_last_model

from unittest.mock import patch, MagicMock

class MockModel:
    """Mock model class for testing"""
    def __init__(self, iteration, end_dev, train_size=None, **kwargs):
        self.iteration = iteration
        self.end_dev = end_dev
        self.train_size = train_size
        self.name = 'mock'
        self.is_tuned = False
    def tune(self):
        self.is_tuned = True

@patch("spotforecast2_safe.manager.trainer.fetch_data")
def test_handle_training_with_custom_name(mock_fetch):
    """Verify that handle_training passes and uses custom model_name correctly."""
    # Setup mock data for fetch_data
    dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    mock_fetch.return_value = pd.DataFrame({"value": range(10)}, index=dates)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        model_name = "custom_lgbm"
        
        # 1. First training (iteration 0)
        handle_training(
            MockModel,
            model_name=model_name,
            model_dir=model_dir,
            end_dev="2024-01-01"
        )
        
        # Verify the file was saved with the custom name
        expected_file = model_dir / f"{model_name}_forecaster_0.joblib"
        assert expected_file.exists(), f"File {expected_file} was not created"
        
        # 2. Verify get_last_model finds it with the custom name
        iteration, model = get_last_model(model_name, model_dir=model_dir)
        assert iteration == 0
        assert model is not None
        assert model.name == model_name # Internal name should also be updated
        
        # 3. Trigger retraining and verify next iteration
        handle_training(
            MockModel,
            model_name=model_name,
            model_dir=model_dir,
            force=True,
            end_dev="2024-01-02"
        )
        
        expected_file_1 = model_dir / f"{model_name}_forecaster_1.joblib"
        assert expected_file_1.exists()
        
        iteration_1, model_1 = get_last_model(model_name, model_dir=model_dir)
        assert iteration_1 == 1
        assert model_1.name == model_name
