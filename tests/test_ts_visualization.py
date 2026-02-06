"""Tests for time series visualization functions."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from spotforecast2_safe.preprocessing.time_series_visualization import (
    visualize_ts_plotly,
    visualize_ts_comparison,
)


class TestVisualizeTsPlotly:
    """Test suite for visualize_ts_plotly function."""

    def setup_method(self):
        """Create sample data for each test."""
        np.random.seed(42)
        self.dates_train = pd.date_range('2024-01-01', periods=100, freq='h')
        self.dates_val = pd.date_range('2024-05-11', periods=50, freq='h')
        self.dates_test = pd.date_range('2024-07-01', periods=30, freq='h')

        self.data_train = pd.DataFrame({
            'temperature': np.random.normal(20, 5, 100),
            'humidity': np.random.normal(60, 10, 100)
        }, index=self.dates_train)

        self.data_val = pd.DataFrame({
            'temperature': np.random.normal(22, 5, 50),
            'humidity': np.random.normal(55, 10, 50)
        }, index=self.dates_val)

        self.data_test = pd.DataFrame({
            'temperature': np.random.normal(25, 5, 30),
            'humidity': np.random.normal(50, 10, 30)
        }, index=self.dates_test)

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_basic(self, mock_go):
        """Test basic visualization with three datasets."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {
            'Train': self.data_train,
            'Validation': self.data_val,
            'Test': self.data_test
        }

        visualize_ts_plotly(dataframes)

        # Verify Figure was created for each column
        assert mock_go.Figure.call_count >= 2

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_single_dataset(self, mock_go):
        """Test visualization with single dataset."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Data': self.data_train}

        visualize_ts_plotly(dataframes)

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_specific_columns(self, mock_go):
        """Test visualization with specific columns."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {
            'Train': self.data_train,
            'Validation': self.data_val,
        }

        visualize_ts_plotly(dataframes, columns=['temperature'])

        # Should create one figure for temperature
        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_custom_colors(self, mock_go):
        """Test visualization with custom colors."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {
            'Train': self.data_train,
            'Validation': self.data_val,
        }

        colors = {'Train': 'blue', 'Validation': 'red'}

        visualize_ts_plotly(dataframes, colors=colors)

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_custom_template(self, mock_go):
        """Test visualization with custom template."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Train': self.data_train}

        visualize_ts_plotly(dataframes, template='plotly_dark')

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_custom_figsize(self, mock_go):
        """Test visualization with custom figure size."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Train': self.data_train}

        visualize_ts_plotly(dataframes, figsize=(1500, 600))

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_with_kwargs(self, mock_go):
        """Test visualization with additional kwargs."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Train': self.data_train}

        visualize_ts_plotly(
            dataframes,
            fill='tozeroy'
        )

        assert mock_go.Figure.called

    def test_visualize_ts_plotly_empty_dict(self):
        """Test that function raises ValueError with empty dataframes dict."""
        with pytest.raises(ValueError, match="dataframes dictionary is empty"):
            visualize_ts_plotly({})

    def test_visualize_ts_plotly_empty_dataframe(self):
        """Test that function raises ValueError with empty DataFrame."""
        dataframes = {'Empty': pd.DataFrame()}

        with pytest.raises(ValueError, match="DataFrame 'Empty' is empty"):
            visualize_ts_plotly(dataframes)

    def test_visualize_ts_plotly_no_columns(self):
        """Test that function raises ValueError with DataFrame with no columns."""
        dataframes = {'NoColumns': pd.DataFrame(index=[1, 2, 3])}

        # DataFrame with no columns is considered empty by pandas
        with pytest.raises(ValueError, match="DataFrame 'NoColumns' is empty"):
            visualize_ts_plotly(dataframes)

    def test_visualize_ts_plotly_not_dict(self):
        """Test that function raises TypeError if dataframes is not a dict."""
        with pytest.raises(TypeError, match="dataframes parameter must be a dictionary"):
            visualize_ts_plotly([self.data_train])

    def test_visualize_ts_plotly_missing_column(self):
        """Test that function raises ValueError if column not in all dataframes."""
        dataframes = {
            'Train': self.data_train,
            'Validation': self.data_val.drop('humidity', axis=1)
        }

        with pytest.raises(ValueError, match="Column 'humidity' not found"):
            visualize_ts_plotly(dataframes)

    def test_visualize_ts_plotly_no_plotly(self):
        """Test that ImportError is raised when plotly is not available."""
        dataframes = {'Train': self.data_train}

        with patch('spotforecast2_safe.preprocessing.time_series_visualization.go', None):
            with pytest.raises(ImportError, match="plotly is required"):
                visualize_ts_plotly(dataframes)

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_many_datasets(self, mock_go):
        """Test visualization with many datasets (>10)."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {
            f'Dataset_{i}': pd.DataFrame(
                {'value': np.random.normal(0, 1, 50)},
                index=pd.date_range(f'2024-{i+1:02d}-01', periods=50, freq='h')
            )
            for i in range(12)
        }

        visualize_ts_plotly(dataframes)

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_plotly_title_suffix(self, mock_go):
        """Test visualization with title suffix."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Train': self.data_train}

        visualize_ts_plotly(dataframes, title_suffix='[°C]')

        # Check that update_layout was called
        mock_fig.update_layout.assert_called()


class TestVisualizeTsComparison:
    """Test suite for visualize_ts_comparison function."""

    def setup_method(self):
        """Create sample data for each test."""
        np.random.seed(42)
        self.dates1 = pd.date_range('2024-01-01', periods=100, freq='h')
        self.dates2 = pd.date_range('2024-05-11', periods=100, freq='h')

        self.df1 = pd.DataFrame({
            'temperature': np.random.normal(20, 5, 100)
        }, index=self.dates1)

        self.df2 = pd.DataFrame({
            'temperature': np.random.normal(22, 5, 100)
        }, index=self.dates2)

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_comparison_basic(self, mock_go):
        """Test basic comparison visualization."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Dataset1': self.df1, 'Dataset2': self.df2}

        visualize_ts_comparison(dataframes)

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_comparison_with_mean(self, mock_go):
        """Test comparison visualization with mean overlay."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Dataset1': self.df1, 'Dataset2': self.df2}

        visualize_ts_comparison(dataframes, show_mean=True)

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_visualize_ts_comparison_custom_colors(self, mock_go):
        """Test comparison with custom colors."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        dataframes = {'Dataset1': self.df1, 'Dataset2': self.df2}
        colors = {'Dataset1': 'blue', 'Dataset2': 'red'}

        visualize_ts_comparison(dataframes, colors=colors)

        assert mock_go.Figure.called

    def test_visualize_ts_comparison_empty_dict(self):
        """Test that function raises ValueError with empty dataframes dict."""
        with pytest.raises(ValueError, match="dataframes dictionary is empty"):
            visualize_ts_comparison({})

    def test_visualize_ts_comparison_no_plotly(self):
        """Test that ImportError is raised when plotly is not available."""
        dataframes = {'Data': self.df1}

        with patch('spotforecast2_safe.preprocessing.time_series_visualization.go', None):
            with pytest.raises(ImportError, match="plotly is required"):
                visualize_ts_comparison(dataframes)


class TestIntegration:
    """Integration tests for time series visualization."""

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_workflow_train_val_test(self, mock_go):
        """Test complete workflow with train/val/test split."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        np.random.seed(42)
        dates_train = pd.date_range('2024-01-01', periods=100, freq='h')
        dates_val = pd.date_range('2024-05-11', periods=50, freq='h')
        dates_test = pd.date_range('2024-07-01', periods=30, freq='h')

        data_train = pd.DataFrame({
            'temperature': np.random.normal(20, 5, 100),
            'humidity': np.random.normal(60, 10, 100)
        }, index=dates_train)

        data_val = pd.DataFrame({
            'temperature': np.random.normal(22, 5, 50),
            'humidity': np.random.normal(55, 10, 50)
        }, index=dates_val)

        data_test = pd.DataFrame({
            'temperature': np.random.normal(25, 5, 30),
            'humidity': np.random.normal(50, 10, 30)
        }, index=dates_test)

        dataframes = {
            'Train': data_train,
            'Validation': data_val,
            'Test': data_test
        }

        visualize_ts_plotly(
            dataframes,
            template='plotly_white',
            figsize=(1200, 600)
        )

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_workflow_single_dataset(self, mock_go):
        """Test workflow with single dataset."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        data = pd.DataFrame({
            'value': np.random.normal(0, 1, 100)
        }, index=dates)

        dataframes = {'Data': data}

        visualize_ts_plotly(dataframes)

        assert mock_go.Figure.called

    @patch('spotforecast2_safe.preprocessing.time_series_visualization.go')
    def test_workflow_many_columns(self, mock_go):
        """Test workflow with many columns."""
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        
        data = pd.DataFrame({
            f'col_{i}': np.random.normal(0, 1, 50)
            for i in range(10)
        }, index=dates)

        dataframes = {'Data': data}

        visualize_ts_plotly(dataframes, columns=['col_0', 'col_5'])

        # Should create 2 figures (one per column)
        assert mock_go.Figure.call_count >= 2
