# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for stats.comparison (paired/panel model-comparison statistics)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from spotforecast2_safe.stats.comparison import (
    PairedComparison,
    RepeatedMeasuresAnova,
    holm_adjust,
    paired_comparison,
    pairwise_paired_t,
    rank_concordance,
    rank_table,
    repeated_measures_anova,
)

DAYS = pd.date_range("2026-06-10", periods=8, freq="D")


def _panel(seed=0, n_days=8, entries=("alpha", "beta", "gamma")):
    rng = np.random.default_rng(seed)
    offsets = {name: 40.0 * i for i, name in enumerate(entries)}
    idx = pd.date_range("2026-06-10", periods=n_days, freq="D")
    return pd.DataFrame(
        {name: 500.0 + offsets[name] + rng.normal(0, 25, n_days) for name in entries},
        index=idx,
    )


class TestPairedComparison:
    def test_constant_difference(self):
        a = pd.Series([500.0, 520.0, 480.0, 510.0, 490.0], index=DAYS[:5])
        b = a + 40.0
        result = paired_comparison(a, b)
        assert isinstance(result, PairedComparison)
        assert result.n == 5
        assert result.wins_a == 5
        assert result.mean_a == a.mean()
        assert result.mean_b == b.mean()

    def test_p_value_matches_scipy(self):
        panel = _panel()
        result = paired_comparison(panel["alpha"], panel["beta"])
        direct = scipy_stats.ttest_rel(panel["alpha"], panel["beta"])
        assert result.p_value == float(direct.pvalue)
        assert result.statistic == float(direct.statistic)

    def test_matches_manuscript_paired_function(self):
        """Pin equality with the inline paired() of the bart26o manuscript."""
        panel = _panel(seed=5)
        piv = panel.copy()
        piv.loc[piv.index[:2], "alpha"] = np.nan  # unequal coverage

        def manuscript_paired(a, b):
            j = piv[[a, b]].dropna()
            return dict(
                n=len(j),
                mean_a=j[a].mean(),
                mean_b=j[b].mean(),
                wins=int((j[a] < j[b]).sum()),
                p=scipy_stats.ttest_rel(j[a], j[b]).pvalue,
            )

        expected = manuscript_paired("alpha", "beta")
        result = paired_comparison(piv["alpha"], piv["beta"])
        assert result.n == expected["n"]
        assert result.mean_a == expected["mean_a"]
        assert result.mean_b == expected["mean_b"]
        assert result.wins_a == expected["wins"]
        assert result.p_value == float(expected["p"])

    def test_alignment_drops_missing_periods(self):
        a = pd.Series([1.0, 2.0, 3.0], index=DAYS[:3])
        b = pd.Series([2.0, 1.0, 4.0, 5.0], index=DAYS[1:5])
        result = paired_comparison(a, b)
        assert result.n == 2  # only days 2 and 3 are shared

    def test_higher_is_better(self):
        a = pd.Series([3.0, 4.0, 5.0], index=DAYS[:3])
        b = pd.Series([1.0, 2.0, 3.0], index=DAYS[:3])
        result = paired_comparison(a, b, lower_is_better=False)
        assert result.wins_a == 3

    def test_too_few_shared_periods_raises(self):
        a = pd.Series([1.0], index=DAYS[:1])
        with pytest.raises(ValueError, match="at least 2 shared"):
            paired_comparison(a, a)

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pd.Series"):
            paired_comparison([1.0, 2.0], pd.Series([1.0, 2.0]))


class TestHolmAdjust:
    def test_textbook_example(self):
        raw = pd.Series([0.01, 0.02, 0.03, 0.04], index=list("abcd"))
        adjusted = holm_adjust(raw)
        assert list(adjusted) == [0.04, 0.06, 0.06, 0.06]

    def test_cap_at_one(self):
        raw = pd.Series([0.5, 0.6, 0.9])
        assert (holm_adjust(raw) <= 1.0).all()

    def test_index_and_order_preserved(self):
        raw = pd.Series([0.04, 0.01], index=["late", "early"])
        adjusted = holm_adjust(raw)
        assert list(adjusted.index) == ["late", "early"]
        assert adjusted["early"] == 0.02
        assert adjusted["late"] == pytest.approx(0.04)

    def test_tie_order_independence(self):
        raw = pd.Series([0.02, 0.02, 0.01], index=list("abc"))
        permuted = raw.iloc[[2, 0, 1]]
        adjusted = holm_adjust(raw)
        adjusted_permuted = holm_adjust(permuted)
        for key in "abc":
            assert adjusted[key] == adjusted_permuted[key]

    def test_matches_manuscript_loop(self):
        """Pin equality with the inline Holm loop of the bart26o manuscript."""
        rng = np.random.default_rng(9)
        raw_p = {
            (f"e{i}", f"e{j}"): float(rng.uniform(0.001, 0.9))
            for i in range(5)
            for j in range(i + 1, 5)
        }
        prev = 0.0
        expected = {}
        for step, pair in enumerate(sorted(raw_p, key=raw_p.get)):
            prev = min(1.0, max(prev, (len(raw_p) - step) * raw_p[pair]))
            expected[pair] = prev
        adjusted = holm_adjust(pd.Series(raw_p))
        for pair, value in expected.items():
            assert adjusted[pair] == value

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            holm_adjust(pd.Series(dtype=float))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            holm_adjust(pd.Series([0.01, np.nan]))


class TestPairwisePairedT:
    def test_symmetric_with_unit_diagonal(self):
        p = pairwise_paired_t(_panel())
        assert (p.values == p.values.T).all()
        assert (np.diag(p.values) == 1.0).all()

    def test_none_correction_matches_scipy(self):
        panel = _panel()
        p = pairwise_paired_t(panel, correction="none")
        direct = scipy_stats.ttest_rel(panel["alpha"], panel["gamma"]).pvalue
        assert p.loc["alpha", "gamma"] == float(direct)

    def test_holm_never_below_raw(self):
        panel = _panel()
        raw = pairwise_paired_t(panel, correction="none")
        holm = pairwise_paired_t(panel, correction="holm")
        off_diag = ~np.eye(len(raw), dtype=bool)
        assert (holm.values[off_diag] >= raw.values[off_diag]).all()

    def test_matches_manuscript_matrix(self):
        """Pin equality with the inline raw_p + Holm matrix of the manuscript."""
        panel = _panel(seed=21, n_days=12, entries=("a", "b", "c", "d"))
        cols = list(panel.columns)
        raw_p = {
            (a, b): scipy_stats.ttest_rel(panel[a], panel[b]).pvalue
            for i, a in enumerate(cols)
            for b in cols[i + 1 :]
        }
        expected = pd.DataFrame(1.0, index=cols, columns=cols)
        prev = 0.0
        for step, pair in enumerate(sorted(raw_p, key=raw_p.get)):
            a, b = pair
            prev = min(1.0, max(prev, (len(raw_p) - step) * raw_p[pair]))
            expected.loc[a, b] = expected.loc[b, a] = prev
        result = pairwise_paired_t(panel)
        assert (result.values == expected.values).all()

    def test_nan_raises(self):
        panel = _panel()
        panel.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            pairwise_paired_t(panel)

    def test_single_column_raises(self):
        with pytest.raises(ValueError, match="two columns"):
            pairwise_paired_t(_panel()[["alpha"]])

    def test_unknown_correction_raises(self):
        with pytest.raises(ValueError, match="correction"):
            pairwise_paired_t(_panel(), correction="bonferroni")


class TestRepeatedMeasuresAnova:
    def test_matches_manuscript_formulation(self):
        """Pin equality with the inline blocked ANOVA of the manuscript."""
        panel = _panel(seed=13, n_days=10, entries=("a", "b", "c", "d"))
        X = panel.values
        n_panel, k_panel = X.shape
        grand = X.mean()
        ss_treat = n_panel * ((X.mean(axis=0) - grand) ** 2).sum()
        ss_block = k_panel * ((X.mean(axis=1) - grand) ** 2).sum()
        ss_err = ((X - grand) ** 2).sum() - ss_treat - ss_block
        df_treat, df_err = k_panel - 1, (k_panel - 1) * (n_panel - 1)
        expected_f = (ss_treat / df_treat) / (ss_err / df_err)
        expected_p = scipy_stats.f.sf(expected_f, df_treat, df_err)

        anova = repeated_measures_anova(panel)
        assert anova.f_statistic == float(expected_f)
        assert anova.p_value == float(expected_p)
        assert anova.ss_treatment == float(ss_treat)
        assert anova.ss_block == float(ss_block)
        assert anova.ss_error == float(ss_err)
        assert anova.df_treatment == df_treat
        assert anova.df_error == df_err
        assert anova.n_blocks == n_panel
        assert anova.n_treatments == k_panel

    def test_hand_computed_small_panel(self):
        panel = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 5.0]},
            index=pd.date_range("2026-06-10", periods=3, freq="D"),
        )
        # grand = 8/3; col means (2, 10/3), each 2/3 from the grand mean:
        # ss_treatment = 3 * ((2/3)^2 + (2/3)^2) = 8/3.
        anova = repeated_measures_anova(panel)
        assert isinstance(anova, RepeatedMeasuresAnova)
        assert anova.ss_treatment == pytest.approx(8 / 3)
        assert anova.df_treatment == 1
        assert anova.df_error == 2

    def test_strong_effect_is_significant(self):
        anova = repeated_measures_anova(_panel(n_days=20))
        assert anova.p_value < 1e-6

    def test_nan_raises(self):
        panel = _panel()
        panel.iloc[2, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            repeated_measures_anova(panel)

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="two columns"):
            repeated_measures_anova(_panel()[["alpha"]])
        with pytest.raises(ValueError, match="two rows"):
            repeated_measures_anova(_panel().iloc[:1])


class TestRankTable:
    def test_hand_example(self):
        scores = pd.DataFrame(
            {"mae": [510.0, 462.0, 891.0], "rmse": [640.0, 655.0, 1024.0]},
            index=["alpha", "beta", "gamma"],
        )
        ranks = rank_table(scores)
        assert list(ranks["mae"]) == [2, 1, 3]
        assert list(ranks["rmse"]) == [1, 2, 3]
        assert ranks.dtypes.eq(int).all() or ranks.dtypes.eq("int64").all()

    def test_matches_manuscript_ranking(self):
        rng = np.random.default_rng(2)
        scores = pd.DataFrame(
            {m: rng.uniform(400, 900, 6) for m in ["mae", "rmse", "mape", "mase"]},
            index=[f"team_{i}" for i in range(6)],
        )
        ranks = rank_table(scores)
        expected = pd.DataFrame(
            {m: scores[m].rank(method="min") for m in scores.columns}
        ).astype(int)
        assert ranks.equals(expected)

    def test_min_method_on_ties(self):
        scores = pd.DataFrame({"m": [1.0, 1.0, 2.0]}, index=list("abc"))
        assert list(rank_table(scores)["m"]) == [1, 1, 3]

    def test_descending(self):
        scores = pd.DataFrame({"m": [1.0, 3.0, 2.0]}, index=list("abc"))
        assert list(rank_table(scores, ascending=False)["m"]) == [3, 1, 2]

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            rank_table(pd.DataFrame({"m": [1.0, np.nan]}))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            rank_table(pd.DataFrame())


class TestRankConcordance:
    def test_identical_and_reversed(self):
        ranks = pd.DataFrame(
            {"mae": [1, 2, 3, 4], "rmse": [1, 2, 3, 4], "mape": [4, 3, 2, 1]}
        )
        tau = rank_concordance(ranks, reference="mae")
        assert tau["rmse"] == 1.0
        assert tau["mape"] == -1.0
        assert list(tau.index) == ["rmse", "mape"]

    def test_default_reference_is_first_column(self):
        ranks = pd.DataFrame({"mae": [1, 2, 3], "rmse": [2, 1, 3]})
        tau = rank_concordance(ranks)
        assert list(tau.index) == ["rmse"]
        expected = scipy_stats.kendalltau(ranks["mae"], ranks["rmse"]).statistic
        assert tau["rmse"] == float(expected)

    def test_matches_manuscript_tau(self):
        rng = np.random.default_rng(17)
        ranks = pd.DataFrame(
            {
                m: rng.permutation(np.arange(1, 9))
                for m in ["mae", "rmse", "mape", "mase"]
            }
        )
        tau = rank_concordance(ranks, reference="mae")
        for m in ["rmse", "mape", "mase"]:
            expected = scipy_stats.kendalltau(ranks["mae"], ranks[m]).statistic
            assert tau[m] == float(expected)

    def test_missing_reference_raises(self):
        ranks = pd.DataFrame({"mae": [1, 2], "rmse": [2, 1]})
        with pytest.raises(ValueError, match="reference"):
            rank_concordance(ranks, reference="mase")

    def test_single_column_raises(self):
        with pytest.raises(ValueError, match="two columns"):
            rank_concordance(pd.DataFrame({"mae": [1, 2]}))
