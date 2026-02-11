import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
from os import environ

from spotforecast2_safe.data.fetch_data import (
    get_data_home,
    fetch_data,
    fetch_holiday_data,
    fetch_weather_data,
)


class TestGetDataHome:
    """Test suite for the get_data_home function."""

    def test_get_data_home_default(self):
        """Test that get_data_home returns default path when no argument is provided."""
        result = get_data_home()
        assert isinstance(result, Path)
        assert "spotforecast2_data" in str(result)
        assert result.is_absolute()

    def test_get_data_home_with_string_path(self):
        """Test that get_data_home accepts a string path and returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_data_home(tmpdir)
            assert isinstance(result, Path)
            assert str(result) == str(Path(tmpdir).absolute())
            assert result.is_dir()

    def test_get_data_home_with_path_object(self):
        """Test that get_data_home accepts a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            result = get_data_home(tmppath)
            assert isinstance(result, Path)
            assert result == tmppath.absolute()
            assert result.is_dir()

    def test_get_data_home_creates_directory(self):
        """Test that get_data_home creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_spotforecast2_data"
            assert not new_dir.exists()

            result = get_data_home(new_dir)

            assert result.exists()
            assert result.is_dir()

    def test_get_data_home_with_tilde_expansion(self):
        """Test that get_data_home expands ~ to home directory."""
        result = get_data_home("~/test_spotforecast2_data")
        assert isinstance(result, Path)
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_get_data_home_environment_variable(self):
        """Test that get_data_home respects SPOTFORECAST2_DATA environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(environ, {"SPOTFORECAST2_DATA": tmpdir}):
                result = get_data_home()
                assert str(result) == str(Path(tmpdir).absolute())

    def test_get_data_home_path_precedence(self):
        """Test that explicit path argument takes precedence over environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                with patch.dict(environ, {"SPOTFORECAST2_DATA": tmpdir1}):
                    result = get_data_home(tmpdir2)
                    assert str(result) == str(Path(tmpdir2).absolute())

    def test_get_data_home_returns_absolute_path(self):
        """Test that get_data_home always returns an absolute path."""
        result = get_data_home("./relative/path")
        assert result.is_absolute()

    def test_get_data_home_nested_directory_creation(self):
        """Test that get_data_home creates nested directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "level1" / "level2" / "level3"
            assert not nested_path.exists()

            result = get_data_home(nested_path)

            assert result.exists()
            assert result.is_dir()


class TestFetchData:
    """Test suite for the fetch_data function."""

    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"

            # Create sample data
            data = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                    "FS_Sum_DEA_Herdecke": [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                    ],
                    "FS_Sum_DEA_Höchsten": [
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                    ],
                    "FM_Sum_Höchsten": [
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                    ],
                }
            )
            data.set_index("date", inplace=True)
            data.to_csv(csv_path)

            yield csv_path

    def test_fetch_data_columns_none(self, sample_csv):
        """Test that fetch_data returns all columns when columns is None."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(filename=sample_csv.name, columns=None)

                # verification that columns=None is passed to Data.from_csv
                call_kwargs = mock_from_csv.call_args[1]
                assert call_kwargs["columns"] is None

    def test_fetch_data_columns_empty_list(self):
        """Test that fetch_data raises ValueError when columns is empty list."""
        with pytest.raises(ValueError, match="columns must be specified"):
            fetch_data(filename="test.csv", columns=[])

    def test_fetch_data_file_not_found(self):
        """Test that fetch_data raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            fetch_data(filename="nonexistent_file.csv", columns=["col1"])

    def test_fetch_data_default_parameters(self, sample_csv):
        """Test fetch_data with default parameters."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            # Mock the Data.from_csv to return sample data
            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame(
                    {
                        "FS_Sum_DEA_Herdecke": [1.0, 2.0, 3.0],
                        "FS_Sum_DEA_Höchsten": [2.0, 3.0, 4.0],
                    }
                )
                mock_from_csv.return_value = mock_data

                result = fetch_data(
                    filename=sample_csv.name,
                    columns=["FS_Sum_DEA_Herdecke", "FS_Sum_DEA_Höchsten"],
                )

                assert isinstance(result, pd.DataFrame)
                mock_from_csv.assert_called_once()

    def test_fetch_data_custom_filename(self, sample_csv):
        """Test fetch_data with custom filename."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(filename=sample_csv.name, columns=["col1"])

                call_args = mock_from_csv.call_args
                assert sample_csv.name in str(call_args)

    def test_fetch_data_custom_columns(self, sample_csv):
        """Test fetch_data with custom columns parameter."""
        custom_cols = ["FS_Sum_DEA_Herdecke", "FM_Sum_Höchsten"]

        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(filename=sample_csv.name, columns=custom_cols)

                call_kwargs = mock_from_csv.call_args[1]
                assert call_kwargs["columns"] == custom_cols

    def test_fetch_data_custom_index_col(self, sample_csv):
        """Test fetch_data with custom index_col parameter."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(filename=sample_csv.name, columns=["col1"], index_col=1)

                call_kwargs = mock_from_csv.call_args[1]
                assert call_kwargs["index_col"] == 1

    def test_fetch_data_parse_dates(self, sample_csv):
        """Test fetch_data with parse_dates parameter."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(filename=sample_csv.name, columns=["col1"], parse_dates=True)

                call_kwargs = mock_from_csv.call_args[1]
                assert call_kwargs["parse_dates"] is True

    def test_fetch_data_dayfirst(self, sample_csv):
        """Test fetch_data with dayfirst parameter."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(filename=sample_csv.name, columns=["col1"], dayfirst=True)

                call_kwargs = mock_from_csv.call_args[1]
                assert call_kwargs["dayfirst"] is True

    def test_fetch_data_timezone(self, sample_csv):
        """Test fetch_data with custom timezone parameter."""
        custom_tz = "Europe/Berlin"

        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(
                    filename=sample_csv.name, columns=["col1"], timezone=custom_tz
                )

                call_kwargs = mock_from_csv.call_args[1]
                assert call_kwargs["timezone"] == custom_tz

    def test_fetch_data_all_parameters(self, sample_csv):
        """Test fetch_data with all parameters customized."""
        custom_cols = ["FS_Sum_DEA_Herdecke", "FM_Sum_Höchsten"]
        custom_tz = "Europe/Amsterdam"

        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(
                    filename=sample_csv.name,
                    columns=custom_cols,
                    index_col=2,
                    parse_dates=True,
                    dayfirst=True,
                    timezone=custom_tz,
                )

                call_kwargs = mock_from_csv.call_args[1]
                assert call_kwargs["columns"] == custom_cols
                assert call_kwargs["index_col"] == 2
                assert call_kwargs["parse_dates"] is True
                assert call_kwargs["dayfirst"] is True
                assert call_kwargs["timezone"] == custom_tz

    def test_fetch_data_returns_dataframe(self, sample_csv):
        """Test that fetch_data returns a pandas DataFrame."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                expected_df = pd.DataFrame({"col1": [1, 2, 3]})
                mock_data.data = expected_df
                mock_from_csv.return_value = mock_data

                result = fetch_data(filename=sample_csv.name, columns=["col1"])

                assert isinstance(result, pd.DataFrame)
                pd.testing.assert_frame_equal(result, expected_df)

    def test_fetch_data_calls_data_from_csv(self, sample_csv):
        """Test that fetch_data calls Data.from_csv with correct path."""
        with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
            mock_home.return_value = sample_csv.parent

            with patch(
                "spotforecast2_safe.data.fetch_data.Data.from_csv"
            ) as mock_from_csv:
                mock_data = MagicMock()
                mock_data.data = pd.DataFrame()
                mock_from_csv.return_value = mock_data

                fetch_data(filename=sample_csv.name, columns=["col1"])

                assert mock_from_csv.called
                call_kwargs = mock_from_csv.call_args[1]
                assert "csv_path" in call_kwargs


class TestIntegration:
    """Integration tests for get_data_home and fetch_data."""

    def test_get_data_home_fetch_data_integration(self):
        """Test that get_data_home directory can be used by fetch_data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a data directory
            data_dir = get_data_home(tmpdir)
            assert data_dir.exists()

            # Create a test CSV file in the directory
            csv_path = data_dir / "test_data.csv"
            test_data = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=5),
                    "value": [1, 2, 3, 4, 5],
                }
            )
            test_data.set_index("date").to_csv(csv_path)

            # Verify the file exists and can be found
            assert csv_path.exists()
            assert csv_path.is_file()

    def test_get_data_home_environment_integration(self):
        """Test get_data_home with environment variable integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(environ, {"SPOTFORECAST2_DATA": tmpdir}):
                home = get_data_home()
                assert str(home) == str(Path(tmpdir).absolute())
                assert home.is_dir()


class TestFetchHolidayData:
    """Test suite for the fetch_holiday_data function."""

    def test_fetch_holiday_data_returns_dataframe(self):
        """Test that fetch_holiday_data returns a pandas DataFrame."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_df = pd.DataFrame({"is_holiday": [0, 1, 0]})
            mock_create.return_value = mock_df

            result = fetch_holiday_data(
                start="2023-01-01T00:00", end="2023-01-10T00:00"
            )

            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, mock_df)

    def test_fetch_holiday_data_default_parameters(self):
        """Test fetch_holiday_data with default parameters."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            fetch_holiday_data(start="2023-01-01T00:00", end="2023-01-10T00:00")

            mock_create.assert_called_once_with(
                start="2023-01-01T00:00",
                end="2023-01-10T00:00",
                tz="UTC",
                freq="h",
                country_code="DE",
                state="NW",
            )

    def test_fetch_holiday_data_custom_timezone(self):
        """Test fetch_holiday_data with custom timezone."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            fetch_holiday_data(
                start="2023-01-01T00:00", end="2023-01-10T00:00", tz="Europe/Berlin"
            )

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["tz"] == "Europe/Berlin"

    def test_fetch_holiday_data_custom_frequency(self):
        """Test fetch_holiday_data with custom frequency."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            fetch_holiday_data(
                start="2023-01-01T00:00", end="2023-01-10T00:00", freq="D"
            )

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["freq"] == "D"

    def test_fetch_holiday_data_custom_country_code(self):
        """Test fetch_holiday_data with custom country code."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            fetch_holiday_data(
                start="2023-01-01T00:00", end="2023-01-10T00:00", country_code="US"
            )

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["country_code"] == "US"

    def test_fetch_holiday_data_custom_state(self):
        """Test fetch_holiday_data with custom state."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            fetch_holiday_data(
                start="2023-01-01T00:00", end="2023-01-10T00:00", state="BY"
            )

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["state"] == "BY"

    def test_fetch_holiday_data_all_custom_parameters(self):
        """Test fetch_holiday_data with all custom parameters."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            fetch_holiday_data(
                start="2023-01-01T00:00",
                end="2023-12-31T23:59",
                tz="Europe/Amsterdam",
                freq="30min",
                country_code="NL",
                state="NH",
            )

            mock_create.assert_called_once_with(
                start="2023-01-01T00:00",
                end="2023-12-31T23:59",
                tz="Europe/Amsterdam",
                freq="30min",
                country_code="NL",
                state="NH",
            )

    def test_fetch_holiday_data_with_timestamp_objects(self):
        """Test fetch_holiday_data with pd.Timestamp objects."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            start_ts = pd.Timestamp("2023-01-01")
            end_ts = pd.Timestamp("2023-01-10")

            fetch_holiday_data(start=start_ts, end=end_ts)

            call_args = mock_create.call_args
            assert call_args[1]["start"] == start_ts
            assert call_args[1]["end"] == end_ts

    def test_fetch_holiday_data_with_string_dates(self):
        """Test fetch_holiday_data with string dates."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            mock_create.return_value = pd.DataFrame()

            fetch_holiday_data(start="2023-01-01T00:00", end="2023-01-10T00:00")

            call_args = mock_create.call_args[1]
            assert call_args["start"] == "2023-01-01T00:00"
            assert call_args["end"] == "2023-01-10T00:00"

    def test_fetch_holiday_data_returns_correct_structure(self):
        """Test that fetch_holiday_data returns correctly structured data."""
        with patch(
            "spotforecast2_safe.data.fetch_data.create_holiday_df"
        ) as mock_create:
            expected_df = pd.DataFrame({"is_holiday": [0, 1, 0, 1, 0]})
            mock_create.return_value = expected_df

            result = fetch_holiday_data(
                start="2023-01-01T00:00", end="2023-01-05T00:00"
            )

            pd.testing.assert_frame_equal(result, expected_df)


class TestFetchWeatherData:
    """Test suite for the fetch_weather_data function."""

    def test_fetch_weather_data_returns_dataframe(self):
        """Test that fetch_weather_data returns a pandas DataFrame."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_df = pd.DataFrame({"temperature": [20.0, 21.0, 22.0]})
            mock_service.get_dataframe.return_value = mock_df
            mock_service_class.return_value = mock_service

            result = fetch_weather_data(
                cov_start="2023-01-01T00:00", cov_end="2023-01-11T00:00"
            )

            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, mock_df)

    def test_fetch_weather_data_default_parameters(self):
        """Test fetch_weather_data with default parameters."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
                mock_home.return_value = Path("/tmp/spotforecast2_data")
                mock_service = MagicMock()
                mock_service.get_dataframe.return_value = pd.DataFrame()
                mock_service_class.return_value = mock_service

                fetch_weather_data(
                    cov_start="2023-01-01T00:00", cov_end="2023-01-11T00:00"
                )

                # Check WeatherService initialization
                mock_service_class.assert_called_once()
                init_kwargs = mock_service_class.call_args[1]
                assert init_kwargs["latitude"] == 51.5136
                assert init_kwargs["longitude"] == 7.4653
                assert "cache_path" in init_kwargs

                # Check get_dataframe call
                mock_service.get_dataframe.assert_called_once_with(
                    start="2023-01-01T00:00",
                    end="2023-01-11T00:00",
                    timezone="UTC",
                    freq="h",
                    fallback_on_failure=True,
                )

    def test_fetch_weather_data_custom_coordinates(self):
        """Test fetch_weather_data with custom latitude and longitude."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataframe.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service

            fetch_weather_data(
                cov_start="2023-01-01T00:00",
                cov_end="2023-01-11T00:00",
                latitude=52.52,
                longitude=13.405,
            )

            init_kwargs = mock_service_class.call_args[1]
            assert init_kwargs["latitude"] == 52.52
            assert init_kwargs["longitude"] == 13.405

    def test_fetch_weather_data_custom_timezone(self):
        """Test fetch_weather_data with custom timezone."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataframe.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service

            fetch_weather_data(
                cov_start="2023-01-01T00:00",
                cov_end="2023-01-11T00:00",
                timezone="Europe/Berlin",
            )

            call_kwargs = mock_service.get_dataframe.call_args[1]
            assert call_kwargs["timezone"] == "Europe/Berlin"

    def test_fetch_weather_data_custom_frequency(self):
        """Test fetch_weather_data with custom frequency."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataframe.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service

            fetch_weather_data(
                cov_start="2023-01-01T00:00", cov_end="2023-01-11T00:00", freq="30min"
            )

            call_kwargs = mock_service.get_dataframe.call_args[1]
            assert call_kwargs["freq"] == "30min"

    def test_fetch_weather_data_fallback_on_failure_false(self):
        """Test fetch_weather_data with fallback_on_failure=False."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataframe.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service

            fetch_weather_data(
                cov_start="2023-01-01T00:00",
                cov_end="2023-01-11T00:00",
                fallback_on_failure=False,
            )

            call_kwargs = mock_service.get_dataframe.call_args[1]
            assert call_kwargs["fallback_on_failure"] is False

    def test_fetch_weather_data_cached_true(self):
        """Test fetch_weather_data with caching enabled."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
                mock_home.return_value = Path("/tmp/spotforecast2_data")
                mock_service = MagicMock()
                mock_service.get_dataframe.return_value = pd.DataFrame()
                mock_service_class.return_value = mock_service

                fetch_weather_data(
                    cov_start="2023-01-01T00:00",
                    cov_end="2023-01-11T00:00",
                    cached=True,
                )

                init_kwargs = mock_service_class.call_args[1]
                assert init_kwargs["cache_path"] == Path(
                    "/tmp/spotforecast2_data/weather_cache.parquet"
                )

    def test_fetch_weather_data_cached_false(self):
        """Test fetch_weather_data with caching disabled."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataframe.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service

            fetch_weather_data(
                cov_start="2023-01-01T00:00", cov_end="2023-01-11T00:00", cached=False
            )

            init_kwargs = mock_service_class.call_args[1]
            assert init_kwargs["cache_path"] is None

    def test_fetch_weather_data_all_custom_parameters(self):
        """Test fetch_weather_data with all custom parameters."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
                mock_home.return_value = Path("/custom/path")
                mock_service = MagicMock()
                mock_service.get_dataframe.return_value = pd.DataFrame()
                mock_service_class.return_value = mock_service

                fetch_weather_data(
                    cov_start="2023-06-01T00:00",
                    cov_end="2023-06-30T23:59",
                    latitude=48.8566,
                    longitude=2.3522,
                    timezone="Europe/Paris",
                    freq="15min",
                    fallback_on_failure=False,
                    cached=True,
                )

                # Check service initialization
                init_kwargs = mock_service_class.call_args[1]
                assert init_kwargs["latitude"] == 48.8566
                assert init_kwargs["longitude"] == 2.3522
                assert init_kwargs["cache_path"] == Path(
                    "/custom/path/weather_cache.parquet"
                )

                # Check get_dataframe call
                call_kwargs = mock_service.get_dataframe.call_args[1]
                assert call_kwargs["start"] == "2023-06-01T00:00"
                assert call_kwargs["end"] == "2023-06-30T23:59"
                assert call_kwargs["timezone"] == "Europe/Paris"
                assert call_kwargs["freq"] == "15min"
                assert call_kwargs["fallback_on_failure"] is False

    def test_fetch_weather_data_cache_path_creation(self):
        """Test that fetch_weather_data creates correct cache path."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            with patch("spotforecast2_safe.data.fetch_data.get_data_home") as mock_home:
                test_path = Path("/test/data/home")
                mock_home.return_value = test_path
                mock_service = MagicMock()
                mock_service.get_dataframe.return_value = pd.DataFrame()
                mock_service_class.return_value = mock_service

                fetch_weather_data(
                    cov_start="2023-01-01T00:00",
                    cov_end="2023-01-11T00:00",
                    cached=True,
                )

                expected_cache_path = test_path / "weather_cache.parquet"
                init_kwargs = mock_service_class.call_args[1]
                assert init_kwargs["cache_path"] == expected_cache_path

    def test_fetch_weather_data_returns_service_dataframe(self):
        """Test that fetch_weather_data returns the dataframe from WeatherService."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            expected_df = pd.DataFrame(
                {
                    "temperature": [18.5, 19.0, 20.5],
                    "humidity": [65, 70, 68],
                    "wind_speed": [5.2, 4.8, 6.1],
                }
            )
            mock_service.get_dataframe.return_value = expected_df
            mock_service_class.return_value = mock_service

            result = fetch_weather_data(
                cov_start="2023-01-01T00:00", cov_end="2023-01-01T03:00"
            )

            pd.testing.assert_frame_equal(result, expected_df)

    def test_fetch_weather_data_date_range_propagation(self):
        """Test that date range is correctly propagated to WeatherService."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataframe.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service

            start_date = "2023-03-15T12:30"
            end_date = "2023-04-20T18:45"

            fetch_weather_data(cov_start=start_date, cov_end=end_date)

            call_kwargs = mock_service.get_dataframe.call_args[1]
            assert call_kwargs["start"] == start_date
            assert call_kwargs["end"] == end_date

    def test_fetch_weather_data_default_dortmund_coordinates(self):
        """Test that default coordinates are for Dortmund."""
        with patch(
            "spotforecast2_safe.data.fetch_data.WeatherService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.get_dataframe.return_value = pd.DataFrame()
            mock_service_class.return_value = mock_service

            fetch_weather_data(cov_start="2023-01-01T00:00", cov_end="2023-01-11T00:00")

            init_kwargs = mock_service_class.call_args[1]
            # Dortmund coordinates
            assert init_kwargs["latitude"] == 51.5136
            assert init_kwargs["longitude"] == 7.4653
