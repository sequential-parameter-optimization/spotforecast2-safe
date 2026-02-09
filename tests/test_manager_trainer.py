# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from spotforecast2_safe.manager.trainer import train_new_model, get_last_model, handle_training

class TestTrainer(unittest.TestCase):
    """Tests for the trainer manager."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.manager.trainer.fetch_data")
    def test_train_new_model(self, mock_fetch_data):
        """Test train_new_model function."""
        # Setup mock data
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        mock_df = pd.DataFrame({"value": range(10)}, index=dates)
        mock_fetch_data.return_value = mock_df

        # Setup mock model class
        mock_model_class = MagicMock()
        mock_model_class.__name__ = "MyLGBM"
        mock_model_instance = mock_model_class.return_value
        mock_model_instance.name = "mylgbm"
        
        n_iteration = 5
        train_size = pd.Timedelta(days=365)
        
        # Execute
        result = train_new_model(
            mock_model_class, n_iteration, train_size=train_size, model_dir=self.test_dir
        )
        
        # Verify
        mock_fetch_data.assert_called_once()
        expected_cutoff = dates[-1] - pd.Timedelta(days=1)
        
        mock_model_class.assert_called_once_with(
            iteration=n_iteration,
            end_dev=expected_cutoff,
            train_size=train_size
        )
        mock_model_instance.tune.assert_called_once()
        self.assertEqual(result, mock_model_instance)
        
        # Verify file creation
        expected_file = self.test_dir / f"mylgbm_forecaster_{n_iteration}.joblib"
        self.assertTrue(expected_file.exists())

    @patch("spotforecast2_safe.manager.trainer.fetch_data")
    def test_train_new_model_empty_data(self, mock_fetch_data):
        """Test train_new_model handles empty data."""
        mock_fetch_data.return_value = pd.DataFrame()
        
        mock_model_class = MagicMock()
        
        result = train_new_model(mock_model_class, 1)
        
        self.assertIsNone(result)
        mock_model_class.assert_not_called()

    @patch("spotforecast2_safe.manager.trainer.load")
    def test_get_last_model(self, mock_load):
        """Test get_last_model retrieval logic."""
        model_name = "testmodel"
        # Create dummy files
        (self.test_dir / f"{model_name}_forecaster_0.joblib").touch()
        (self.test_dir / f"{model_name}_forecaster_5.joblib").touch()
        (self.test_dir / f"{model_name}_forecaster_2.joblib").touch()
        
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        iteration, model = get_last_model(model_name, self.test_dir)
        
        self.assertEqual(iteration, 5)
        self.assertEqual(model, mock_model)
        mock_load.assert_called_once_with(self.test_dir / f"{model_name}_forecaster_5.joblib")

    def test_get_last_model_not_found(self):
        """Test get_last_model when no files exist."""
        iteration, model = get_last_model("nonexistent", self.test_dir)
        self.assertEqual(iteration, -1)
        self.assertIsNone(model)

    @patch("spotforecast2_safe.manager.trainer.train_new_model")
    @patch("spotforecast2_safe.manager.trainer.get_last_model")
    def test_handle_training_new_model(self, mock_get_last, mock_train):
        """Test handle_training trains iteration 0 if no model exists."""
        mock_get_last.return_value = (-1, None)
        mock_model_class = MagicMock()
        
        handle_training(mock_model_class, "lgbm", model_dir=self.test_dir)
        
        mock_train.assert_called_once_with(
            mock_model_class, 0, train_size=None, model_dir=self.test_dir, end_dev=None
        )

    @patch("spotforecast2_safe.manager.trainer.train_new_model")
    @patch("spotforecast2_safe.manager.trainer.get_last_model")
    def test_handle_training_recent_model(self, mock_get_last, mock_train):
        """Test handle_training skips training if model is recent."""
        mock_model = MagicMock()
        # Set end_dev to 24 hours ago
        mock_model.end_dev = pd.Timestamp.now("UTC") - pd.Timedelta(hours=24)
        mock_get_last.return_value = (5, mock_model)
        
        mock_model_class = MagicMock()
        handle_training(mock_model_class, "lgbm", model_dir=self.test_dir)
        
        mock_train.assert_not_called()

    @patch("spotforecast2_safe.manager.trainer.train_new_model")
    @patch("spotforecast2_safe.manager.trainer.get_last_model")
    def test_handle_training_old_model(self, mock_get_last, mock_train):
        """Test handle_training retrains if model is old (e.g. 8 days)."""
        mock_model = MagicMock()
        # Set end_dev to 8 days ago
        mock_model.end_dev = pd.Timestamp.now("UTC") - pd.Timedelta(days=8)
        mock_get_last.return_value = (5, mock_model)
        
        mock_model_class = MagicMock()
        handle_training(mock_model_class, "lgbm", model_dir=self.test_dir)
        
        mock_train.assert_called_once_with(
            mock_model_class, 6, train_size=None, model_dir=self.test_dir, end_dev=None
        )

    @patch("spotforecast2_safe.manager.trainer.train_new_model")
    @patch("spotforecast2_safe.manager.trainer.get_last_model")
    def test_handle_training_force(self, mock_get_last, mock_train):
        """Test handle_training retrains if force=True."""
        mock_model = MagicMock()
        mock_model.end_dev = pd.Timestamp.now("UTC") - pd.Timedelta(hours=1)
        mock_get_last.return_value = (5, mock_model)
        
        mock_model_class = MagicMock()
        handle_training(mock_model_class, "lgbm", force=True, model_dir=self.test_dir)
        
        mock_train.assert_called_once_with(
            mock_model_class, 6, train_size=None, model_dir=self.test_dir, end_dev=None
        )

if __name__ == "__main__":
    unittest.main()

