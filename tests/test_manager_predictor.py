# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from unittest.mock import MagicMock, patch
from spotforecast2_safe.manager.predictor import get_model_prediction


class TestPredictor(unittest.TestCase):
    """Tests for the predictor manager."""

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_success(self, mock_get_last):
        """Test successful prediction retrieval."""
        mock_model = MagicMock()
        mock_model.package_prediction.return_value = {"key": "value"}
        mock_get_last.return_value = (5, mock_model)

        result = get_model_prediction("lgbm")

        self.assertEqual(result, {"key": "value"})
        mock_get_last.assert_called_once_with("lgbm", None)
        mock_model.package_prediction.assert_called_once()

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_no_model(self, mock_get_last):
        """Test handling when no model is found."""
        mock_get_last.return_value = (-1, None)

        result = get_model_prediction("lgbm")

        self.assertIsNone(result)

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_missing_method(self, mock_get_last):
        """Test handling when model lacks package_prediction method."""
        mock_model = MagicMock(spec=[])  # No methods
        mock_get_last.return_value = (5, mock_model)

        result = get_model_prediction("lgbm")

        self.assertIsNone(result)

    @patch("spotforecast2_safe.manager.predictor.get_last_model")
    def test_get_model_prediction_exception(self, mock_get_last):
        """Test handling when package_prediction raises an exception."""
        mock_model = MagicMock()
        mock_model.package_prediction.side_effect = Exception("Prediction failed")
        mock_get_last.return_value = (5, mock_model)

        result = get_model_prediction("lgbm")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
