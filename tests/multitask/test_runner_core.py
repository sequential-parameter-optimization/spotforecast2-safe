# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for run_with — the parameterized pipeline orchestration core.

Covers:
- Unknown task with default message raises ValueError mentioning spotforecast2.
- Unknown task with injected message callable produces exact custom text.
- multitask_cls is honored: MagicMock class is constructed with expected kwargs
  and overrides pass through verbatim.
- all_tasks is honored: injected set permits a custom task name; absence raises.
- "clean" short-circuit: only .run() called, prepare_data never called, returns
  empty DataFrame.
- Pipeline path order: prepare_data -> detect_outliers -> impute ->
  build_exogenous_features -> run(show=...); plot_with_outliers=True gate calls
  mt.plot_with_outliers(); result shaped to single "forecast" column.
- cache_home=None resolves via get_cache_home() (monkeypatched, assert called).
- SAFE_PIPELINE_TASKS == frozenset({"lazy", "defaults", "predict"}).
- run_with and SAFE_PIPELINE_TASKS importable from spotforecast2_safe.multitask.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spotforecast2_safe.multitask import SAFE_PIPELINE_TASKS, run_with
from spotforecast2_safe.multitask.runner import _ALL_TASKS, _PIPELINE_TASKS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_safe_pipeline_tasks_value(self):
        assert SAFE_PIPELINE_TASKS == frozenset({"lazy", "defaults", "predict"})

    def test_pipeline_tasks_alias(self):
        assert _PIPELINE_TASKS is SAFE_PIPELINE_TASKS

    def test_all_tasks_includes_clean(self):
        assert "clean" in _ALL_TASKS
        assert _PIPELINE_TASKS <= _ALL_TASKS

    def test_importable_from_multitask(self):
        from spotforecast2_safe.multitask import SAFE_PIPELINE_TASKS as spt
        from spotforecast2_safe.multitask import run_with as rw

        assert spt == frozenset({"lazy", "defaults", "predict"})
        assert callable(rw)


# ---------------------------------------------------------------------------
# Unknown task — default message
# ---------------------------------------------------------------------------


class TestUnknownTaskDefaultMessage:
    def test_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            run_with(
                multitask_cls=MagicMock(),
                all_tasks=_ALL_TASKS,
                task="optuna",
                cache_home=str(tmp_path),
            )

    def test_error_mentions_spotforecast2(self, tmp_path):
        with pytest.raises(ValueError, match="spotforecast2"):
            run_with(
                multitask_cls=MagicMock(),
                all_tasks=_ALL_TASKS,
                task="optuna",
                cache_home=str(tmp_path),
            )

    def test_error_mentions_unknown_task_name(self, tmp_path):
        with pytest.raises(ValueError, match="banana"):
            run_with(
                multitask_cls=MagicMock(),
                all_tasks=_ALL_TASKS,
                task="banana",
                cache_home=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# Unknown task — injected message callable
# ---------------------------------------------------------------------------


class TestUnknownTaskInjectedMessage:
    def test_custom_message_used(self, tmp_path):
        def my_msg(task, sorted_tasks):
            return f"CUSTOM: {task} not in {sorted_tasks}"

        with pytest.raises(ValueError, match="CUSTOM: nope"):
            run_with(
                multitask_cls=MagicMock(),
                all_tasks=_ALL_TASKS,
                unknown_task_message=my_msg,
                task="nope",
                cache_home=str(tmp_path),
            )

    def test_callable_receives_sorted_tasks(self, tmp_path):
        received: dict = {}

        def capture(task, sorted_tasks):
            received["task"] = task
            received["sorted_tasks"] = sorted_tasks
            return "fail"

        with pytest.raises(ValueError, match="fail"):
            run_with(
                multitask_cls=MagicMock(),
                all_tasks=frozenset({"a", "b", "c"}),
                unknown_task_message=capture,
                task="z",
                cache_home=str(tmp_path),
            )

        assert received["task"] == "z"
        assert received["sorted_tasks"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# all_tasks honored
# ---------------------------------------------------------------------------


class TestAllTasksHonored:
    def _make_pipeline_mock(self):
        """Return a mock class whose instances support the pipeline protocol."""
        mt = MagicMock()
        mt.run.return_value = {"future_pred": pd.Series([1.0, 2.0])}
        cls = MagicMock(return_value=mt)
        return cls, mt

    def test_custom_task_name_permitted(self, tmp_path):
        extended = frozenset({"lazy", "defaults", "predict", "clean", "optuna"})
        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=extended,
            task="optuna",
            cache_home=str(tmp_path),
        )
        # Should not raise and should have called prepare_data (pipeline path)
        mt.prepare_data.assert_called_once()

    def test_task_absent_from_injected_set_raises(self, tmp_path):
        restricted = frozenset({"lazy", "clean"})
        with pytest.raises(ValueError, match="defaults"):
            run_with(
                multitask_cls=MagicMock(),
                all_tasks=restricted,
                task="defaults",
                cache_home=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# multitask_cls honored
# ---------------------------------------------------------------------------


class TestMultitaskClsHonored:
    def _make_pipeline_mock(self):
        mt = MagicMock()
        mt.run.return_value = {"future_pred": pd.Series([10.0, 20.0, 30.0])}
        cls = MagicMock(return_value=mt)
        return cls, mt

    def test_cls_is_called_with_config(self, tmp_path):
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        cfg = ConfigMulti(predict_size=3)
        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            config=cfg,
            task="lazy",
            cache_home=str(tmp_path),
        )
        # First positional arg to cls() must be the config
        args, _ = cls.call_args
        assert args[0] is cfg

    def test_overrides_passed_through(self, tmp_path):
        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
            my_custom_param=99,
        )
        _, kwargs = cls.call_args
        assert kwargs.get("my_custom_param") == 99

    def test_result_has_forecast_column(self, tmp_path):
        cls, mt = self._make_pipeline_mock()

        result = run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
        )
        assert "forecast" in result.columns
        assert list(result["forecast"]) == [10.0, 20.0, 30.0]

    def test_config_none_defaults_to_config_multi(self, tmp_path):
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
        )
        args, _ = cls.call_args
        assert isinstance(args[0], ConfigMulti)

    def test_project_name_sets_data_frame_name(self, tmp_path):
        from spotforecast2_safe.configurator.config_multi import ConfigMulti

        cfg = ConfigMulti(predict_size=3)
        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            config=cfg,
            task="lazy",
            project_name="my_project",
            cache_home=str(tmp_path),
        )
        assert cfg.data_frame_name == "my_project"


# ---------------------------------------------------------------------------
# "clean" short-circuit
# ---------------------------------------------------------------------------


class TestCleanShortCircuit:
    def test_returns_empty_dataframe(self, tmp_path):
        mt = MagicMock()
        cls = MagicMock(return_value=mt)

        result = run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="clean",
            cache_home=str(tmp_path),
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_only_run_called_not_prepare_data(self, tmp_path):
        mt = MagicMock()
        cls = MagicMock(return_value=mt)

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="clean",
            cache_home=str(tmp_path),
        )
        mt.run.assert_called_once()
        mt.prepare_data.assert_not_called()

    def test_clean_cls_called_with_task_clean(self, tmp_path):
        mt = MagicMock()
        cls = MagicMock(return_value=mt)

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="clean",
            cache_home=str(tmp_path),
        )
        _, kwargs = cls.call_args
        assert kwargs.get("task") == "clean"
        # The clean constructor must not receive pipeline-only kwargs.
        assert "show_progress" not in kwargs
        assert "dataframe" not in kwargs
        assert "data_test" not in kwargs


# ---------------------------------------------------------------------------
# Pipeline path order and plot_with_outliers gate
# ---------------------------------------------------------------------------


class TestPipelinePath:
    def _make_pipeline_mock(self, series=None):
        if series is None:
            series = pd.Series([1.0])
        mt = MagicMock()
        mt.prepare_data.return_value = mt
        mt.detect_outliers.return_value = mt
        mt.impute.return_value = mt
        mt.build_exogenous_features.return_value = mt
        mt.run.return_value = {"future_pred": series}
        cls = MagicMock(return_value=mt)
        return cls, mt

    def test_pipeline_stage_order(self, tmp_path):
        cls, mt = self._make_pipeline_mock()
        manager = MagicMock()
        manager.attach_mock(mt.prepare_data, "prepare_data")
        manager.attach_mock(mt.detect_outliers, "detect_outliers")
        manager.attach_mock(mt.impute, "impute")
        manager.attach_mock(mt.build_exogenous_features, "build_exogenous_features")
        manager.attach_mock(mt.run, "run")

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
            show=False,
        )
        method_names = [c[0] for c in manager.mock_calls]
        assert method_names == [
            "prepare_data",
            "detect_outliers",
            "impute",
            "build_exogenous_features",
            "run",
        ]

    def test_show_forwarded_to_run(self, tmp_path):
        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
            show=True,
        )
        mt.run.assert_called_once_with(show=True)

    def test_plot_with_outliers_true_calls_method(self, tmp_path):
        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
            plot_with_outliers=True,
        )
        mt.plot_with_outliers.assert_called_once()

    def test_plot_with_outliers_false_does_not_call(self, tmp_path):
        cls, mt = self._make_pipeline_mock()

        run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
            plot_with_outliers=False,
        )
        mt.plot_with_outliers.assert_not_called()

    def test_result_shaped_to_forecast_column(self, tmp_path):
        series = pd.Series([5.0, 6.0, 7.0], name="future_pred")
        cls, mt = self._make_pipeline_mock(series=series)

        result = run_with(
            multitask_cls=cls,
            all_tasks=_ALL_TASKS,
            task="lazy",
            cache_home=str(tmp_path),
        )
        assert list(result.columns) == ["forecast"]
        assert list(result["forecast"]) == [5.0, 6.0, 7.0]


# ---------------------------------------------------------------------------
# cache_home=None resolves via get_cache_home()
# ---------------------------------------------------------------------------


class TestCacheHomeResolution:
    def test_get_cache_home_called_when_none(self, tmp_path):
        mt = MagicMock()
        mt.run.return_value = {"future_pred": pd.Series([1.0])}
        cls = MagicMock(return_value=mt)

        with patch(
            "spotforecast2_safe.multitask.runner.get_cache_home",
            return_value=str(tmp_path),
        ) as mock_gch:
            run_with(
                multitask_cls=cls,
                all_tasks=_ALL_TASKS,
                task="lazy",
                cache_home=None,
            )
            mock_gch.assert_called_once()

    def test_get_cache_home_not_called_when_provided(self, tmp_path):
        mt = MagicMock()
        mt.run.return_value = {"future_pred": pd.Series([1.0])}
        cls = MagicMock(return_value=mt)

        with patch(
            "spotforecast2_safe.multitask.runner.get_cache_home"
        ) as mock_gch:
            run_with(
                multitask_cls=cls,
                all_tasks=_ALL_TASKS,
                task="lazy",
                cache_home=str(tmp_path),
            )
            mock_gch.assert_not_called()
