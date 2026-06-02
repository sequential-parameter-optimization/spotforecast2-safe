# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from spotforecast2_safe.manager.trainer import get_last_model, should_retrain


class TestTrainer(unittest.TestCase):
    """Tests for the trainer manager."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

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
        mock_load.assert_called_once_with(
            self.test_dir / f"{model_name}_forecaster_5.joblib"
        )

    def test_get_last_model_not_found(self):
        """Test get_last_model when no files exist."""
        iteration, model = get_last_model("nonexistent", self.test_dir)
        self.assertEqual(iteration, -1)
        self.assertIsNone(model)


class TestShouldRetrain(unittest.TestCase):
    """Tests for the retraining cadence gate."""

    def setUp(self):
        self.max_age = pd.Timedelta(days=7)
        self.now = pd.Timestamp("2026-05-24 12:00", tz="UTC")

    def test_should_retrain_returns_true_when_no_previous_model(self):
        self.assertTrue(should_retrain(None, max_age=self.max_age, now=self.now))

    def test_should_retrain_returns_false_when_less_than_max_age(self):
        last = self.now - pd.Timedelta(days=2)
        self.assertFalse(should_retrain(last, max_age=self.max_age, now=self.now))

    def test_should_retrain_returns_true_when_at_or_above_max_age(self):
        last = self.now - pd.Timedelta(days=7)
        self.assertTrue(should_retrain(last, max_age=self.max_age, now=self.now))

    def test_should_retrain_returns_true_with_force(self):
        last = self.now - pd.Timedelta(hours=1)
        self.assertTrue(
            should_retrain(last, max_age=self.max_age, force=True, now=self.now)
        )

    def test_should_retrain_uses_injected_now(self):
        last = pd.Timestamp("2026-05-01 00:00", tz="UTC")
        # 23 days ago — well over max_age
        self.assertTrue(should_retrain(last, max_age=self.max_age, now=self.now))

    def test_should_retrain_handles_naive_timestamps(self):
        last = pd.Timestamp("2026-05-22 12:00")  # tz-naive
        now = pd.Timestamp("2026-05-24 12:00")  # tz-naive
        self.assertFalse(should_retrain(last, max_age=self.max_age, now=now))


if __name__ == "__main__":
    unittest.main()
