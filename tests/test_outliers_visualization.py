"""Tests for outlier detection and visualization functions."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from spotforecast2_safe.preprocessing.outlier import (
    get_outliers,
    visualize_outliers_hist,
    visualize_outliers_plotly_scatter,
)


class TestGetOutliers:
    """Test suite for get_outliers function."""

    def test_get_outliers_basic(self):
        """Test basic outlier detection with synthetic data."""
        np.random.seed(42)
        data = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
            "B": np.concatenate([np.random.normal(5, 2, 100), [50, 60, 70]]),
        })

        outliers = get_outliers(data, contamination=0.03)

        assert isinstance(outliers, dict)
        assert "A" in outliers
        assert "B" in outliers
        assert len(outliers["A"]) > 0
        assert len(outliers["B"]) > 0

    def test_get_outliers_with_original_data(self):
        """Test outlier detection when original data is provided separately."""
        np.random.seed(42)
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })
        data_cleaned = data_original.copy()

        outliers = get_outliers(
            data_original,
            data_original=data_original,
            contamination=0.03
        )

        assert isinstance(outliers, dict)
        assert "A" in outliers
        assert len(outliers["A"]) > 0

    def test_get_outliers_empty_data(self):
        """Test that get_outliers raises ValueError on empty DataFrame."""
        data = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            get_outliers(data)

    def test_get_outliers_no_columns(self):
        """Test that get_outliers raises ValueError on DataFrame with no columns."""
        data = pd.DataFrame(index=[1, 2, 3])

        # DataFrame with no columns is considered empty by pandas
        with pytest.raises(ValueError, match="Input data is empty"):
            get_outliers(data)

    def test_get_outliers_returns_dict(self):
        """Test that get_outliers returns a dictionary with all columns."""
        np.random.seed(42)
        data = pd.DataFrame({
            "col1": np.random.normal(0, 1, 50),
            "col2": np.random.normal(5, 2, 50),
            "col3": np.random.normal(-5, 0.5, 50),
        })

        outliers = get_outliers(data)

        assert isinstance(outliers, dict)
        assert set(outliers.keys()) == {"col1", "col2", "col3"}
        for col, outlier_vals in outliers.items():
            assert isinstance(outlier_vals, pd.Series)

    def test_get_outliers_random_state(self):
        """Test that same random_state produces consistent results."""
        np.random.seed(42)
        data = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })

        outliers1 = get_outliers(data, random_state=42)
        outliers2 = get_outliers(data, random_state=42)

        # Same random state should produce same results
        assert len(outliers1["A"]) == len(outliers2["A"])

    def test_get_outliers_contamination_parameter(self):
        """Test that contamination parameter affects number of outliers detected."""
        np.random.seed(42)
        data = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })

        outliers_low = get_outliers(data, contamination=0.01)
        outliers_high = get_outliers(data, contamination=0.05)

        # Higher contamination should detect more outliers
        assert len(outliers_high["A"]) >= len(outliers_low["A"])

    def test_get_outliers_series_type(self):
        """Test that returned outliers are pandas Series."""
        np.random.seed(42)
        data = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })

        outliers = get_outliers(data)

        for col, outlier_vals in outliers.items():
            assert isinstance(outlier_vals, pd.Series)


class TestVisualizeOutliersHist:
    """Test suite for visualize_outliers_hist function."""

    @patch('matplotlib.pyplot.show')
    def test_visualize_outliers_hist_basic(self, mock_show):
        """Test basic histogram visualization."""
        np.random.seed(42)
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })
        data_cleaned = data_original.copy()

        # Should not raise
        visualize_outliers_hist(
            data_cleaned,
            data_original,
            contamination=0.03
        )

        # Check that show was called
        assert mock_show.called

    @patch('matplotlib.pyplot.show')
    def test_visualize_outliers_hist_with_columns(self, mock_show):
        """Test histogram visualization with specific columns."""
        np.random.seed(42)
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
            "B": np.concatenate([np.random.normal(5, 2, 100), [50, 60, 70]]),
        })
        data_cleaned = data_original.copy()

        visualize_outliers_hist(
            data_cleaned,
            data_original,
            columns=["A"],
            contamination=0.03
        )

        assert mock_show.called

    @patch('matplotlib.pyplot.show')
    def test_visualize_outliers_hist_custom_figsize(self, mock_show):
        """Test histogram with custom figure size."""
        np.random.seed(42)
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })
        data_cleaned = data_original.copy()

        visualize_outliers_hist(
            data_cleaned,
            data_original,
            figsize=(15, 8),
            contamination=0.03
        )

        assert mock_show.called

    @patch('matplotlib.pyplot.show')
    def test_visualize_outliers_hist_custom_bins(self, mock_show):
        """Test histogram with custom number of bins."""
        np.random.seed(42)
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })
        data_cleaned = data_original.copy()

        visualize_outliers_hist(
            data_cleaned,
            data_original,
            bins=100,
            contamination=0.03
        )

        assert mock_show.called

    def test_visualize_outliers_hist_empty_data(self):
        """Test that function raises ValueError on empty data."""
        data = pd.DataFrame()
        data_original = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            visualize_outliers_hist(data, data_original)

    def test_visualize_outliers_hist_missing_column(self):
        """Test that function raises ValueError on missing column."""
        np.random.seed(42)
        data_original = pd.DataFrame({
            "A": np.random.normal(0, 1, 100)
        })
        data_cleaned = data_original.copy()

        with pytest.raises(ValueError, match="Columns not found"):
            visualize_outliers_hist(
                data_cleaned,
                data_original,
                columns=["NonExistent"]
            )

    @patch('matplotlib.pyplot.show')
    def test_visualize_outliers_hist_with_kwargs(self, mock_show):
        """Test histogram with additional kwargs."""
        np.random.seed(42)
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })
        data_cleaned = data_original.copy()

        visualize_outliers_hist(
            data_cleaned,
            data_original,
            alpha=0.7,
            edgecolor='black',
            contamination=0.03
        )

        assert mock_show.called


class TestVisualizeOutliersPlotlyScatter:
    """Test suite for visualize_outliers_plotly_scatter function."""

    @patch('spotforecast2_safe.preprocessing.outlier.go')
    def test_visualize_outliers_plotly_scatter_basic(self, mock_go):
        """Test basic Plotly scatter visualization."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=103, freq='h')
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        }, index=dates)
        data_cleaned = data_original.copy()

        # Mock the Figure class
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        visualize_outliers_plotly_scatter(
            data_cleaned,
            data_original,
            contamination=0.03
        )

        # Check that Figure was instantiated
        mock_go.Figure.assert_called()

    def test_visualize_outliers_plotly_scatter_no_plotly(self):
        """Test that ImportError is raised when plotly is not available."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=103, freq='h')
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        }, index=dates)
        data_cleaned = data_original.copy()

        # Patch go to be None to simulate missing plotly
        with patch('spotforecast2_safe.preprocessing.outlier.go', None):
            with pytest.raises(ImportError, match="plotly is required"):
                visualize_outliers_plotly_scatter(
                    data_cleaned,
                    data_original
                )

    @patch('spotforecast2_safe.preprocessing.outlier.go')
    def test_visualize_outliers_plotly_scatter_with_columns(self, mock_go):
        """Test Plotly scatter with specific columns."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=103, freq='h')
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
            "B": np.concatenate([np.random.normal(5, 2, 100), [50, 60, 70]]),
        }, index=dates)
        data_cleaned = data_original.copy()

        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        visualize_outliers_plotly_scatter(
            data_cleaned,
            data_original,
            columns=["A"],
            contamination=0.03
        )

        mock_go.Figure.assert_called()

    def test_visualize_outliers_plotly_scatter_empty_data(self):
        """Test that function raises ValueError on empty data."""
        data = pd.DataFrame()
        data_original = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            visualize_outliers_plotly_scatter(data, data_original)

    def test_visualize_outliers_plotly_scatter_missing_column(self):
        """Test that function raises ValueError on missing column."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        data_original = pd.DataFrame({
            "A": np.random.normal(0, 1, 100)
        }, index=dates)
        data_cleaned = data_original.copy()

        with pytest.raises(ValueError, match="Columns not found"):
            visualize_outliers_plotly_scatter(
                data_cleaned,
                data_original,
                columns=["NonExistent"]
            )

    @patch('spotforecast2_safe.preprocessing.outlier.go')
    def test_visualize_outliers_plotly_scatter_with_kwargs(self, mock_go):
        """Test Plotly scatter with additional kwargs."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=103, freq='h')
        data_original = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        }, index=dates)
        data_cleaned = data_original.copy()

        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig

        visualize_outliers_plotly_scatter(
            data_cleaned,
            data_original,
            template='plotly_dark',
            height=600,
            contamination=0.03
        )

        mock_go.Figure.assert_called()
        # Check that update_layout was called with custom kwargs
        mock_fig.update_layout.assert_called()


class TestIntegration:
    """Integration tests for outlier detection and visualization."""

    @patch('matplotlib.pyplot.show')
    @patch('spotforecast2_safe.preprocessing.outlier.go')
    def test_outlier_workflow(self, mock_go, mock_show):
        """Test complete outlier detection and visualization workflow."""
        np.random.seed(42)
        
        # Create realistic time series data with outliers
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        data_original = pd.DataFrame({
            'temperature': np.concatenate([
                np.random.normal(20, 5, 197),
                [50, 60, 70]  # outliers
            ]),
            'humidity': np.concatenate([
                np.random.normal(60, 10, 197),
                [95, 98, 99]  # outliers
            ])
        }, index=dates)
        
        # Step 1: Detect outliers
        outliers = get_outliers(
            data_original,
            contamination=0.015
        )
        
        assert len(outliers) == 2
        assert all(isinstance(v, pd.Series) for v in outliers.values())
        
        # Step 2: Visualize with histogram
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig
        
        visualize_outliers_hist(
            data_original,
            data_original,
            contamination=0.015
        )
        
        assert mock_show.called
        
        # Step 3: Visualize with Plotly
        visualize_outliers_plotly_scatter(
            data_original,
            data_original,
            contamination=0.015
        )
        
        mock_go.Figure.assert_called()

    def test_outlier_detection_reproducibility(self):
        """Test that outlier detection is reproducible with same random state."""
        np.random.seed(42)
        data = pd.DataFrame({
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])
        })
        
        # Detect outliers twice with same random state
        outliers1 = get_outliers(data, random_state=1234)
        outliers2 = get_outliers(data, random_state=1234)
        
        # Should be identical
        pd.testing.assert_series_equal(
            outliers1["A"].sort_index(),
            outliers2["A"].sort_index()
        )
