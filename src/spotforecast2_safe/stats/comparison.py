# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Paired and panel model-comparison statistics — pure computation.

Statistical comparison of forecasting entries scored per period (per
day, typically): paired two-sided t-tests on shared periods, a
Holm-corrected pairwise p-value matrix, a repeated-measures ANOVA
blocked by period, and rank-concordance across metrics.  The rendering
counterparts (critical-difference diagram, rank-stability chart) live in
`spotforecast2.plots.comparison`.

All p-values are two-sided.  Inputs follow the panel convention of
`spotforecast2_safe.processing.forecast_scoring.score_forecasts_by_period`
pivoted wide: one row per period (block), one column per entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PairedComparison:
    """Result of `paired_comparison()` — two entries on shared periods.

    Attributes:
        n: Number of shared (paired) periods.
        mean_a: Mean of the first entry over the shared periods.
        mean_b: Mean of the second entry over the shared periods.
        wins_a: Number of shared periods on which the first entry was
            strictly better.
        statistic: Paired t-test statistic.
        p_value: Two-sided paired t-test p-value.
    """

    n: int
    mean_a: float
    mean_b: float
    wins_a: int
    statistic: float
    p_value: float


def paired_comparison(
    a: pd.Series,
    b: pd.Series,
    *,
    lower_is_better: bool = True,
) -> PairedComparison:
    """Compare two entries on their shared periods with a paired t-test.

    Aligns the two per-period score series on their common index, drops
    periods where either is missing, and reports the per-side means, the
    win count of the first entry, and a two-sided paired t-test — the
    shared-days comparison protocol for leaderboard entries with
    different coverage.

    Args:
        a: Per-period scores of the first entry (e.g. daily MAE).
        b: Per-period scores of the second entry.
        lower_is_better: Whether a strictly smaller score is a win for
            the first entry. Defaults to True.

    Returns:
        A `PairedComparison` over the shared periods.

    Raises:
        TypeError: When ``a`` or ``b`` is not a ``pd.Series``.
        ValueError: When fewer than two shared periods remain after
            alignment.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.stats.comparison import paired_comparison

        days = pd.date_range("2026-06-10", periods=6, freq="D")
        a = pd.Series([500.0, 520.0, 480.0, 510.0, 490.0, 505.0], index=days)
        b = a + 40.0  # consistently worse
        result = paired_comparison(a, b)
        print(result.n, result.wins_a, round(result.p_value, 6))
        assert result.wins_a == 6
        ```
    """
    for name, series in (("a", a), ("b", b)):
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"{name} must be a pd.Series, got {type(series).__name__!r}."
            )
    joined = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(joined) < 2:
        raise ValueError(f"need at least 2 shared periods, got {len(joined)}.")
    if lower_is_better:
        wins = int((joined["a"] < joined["b"]).sum())
    else:
        wins = int((joined["a"] > joined["b"]).sum())
    test = stats.ttest_rel(joined["a"], joined["b"])
    return PairedComparison(
        n=len(joined),
        mean_a=float(joined["a"].mean()),
        mean_b=float(joined["b"].mean()),
        wins_a=wins,
        statistic=float(test.statistic),
        p_value=float(test.pvalue),
    )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Adjust p-values for multiple testing with the Holm step-down method.

    Sorts the raw p-values ascending (stable sort), multiplies the
    ``i``-th smallest by ``m - i`` remaining hypotheses, enforces
    monotonicity with a running maximum, and caps at 1.0.

    Args:
        p_values: Raw p-values; the index identifies the hypotheses and
            is preserved.

    Returns:
        pd.Series: Adjusted p-values in the original index order.

    Raises:
        ValueError: When ``p_values`` is empty or contains NaN.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.stats.comparison import holm_adjust

        raw = pd.Series([0.01, 0.02, 0.03, 0.04], index=list("abcd"))
        adjusted = holm_adjust(raw)
        print(adjusted.to_string())
        assert list(adjusted) == [0.04, 0.06, 0.06, 0.06]
        ```
    """
    if len(p_values) == 0:
        raise ValueError("p_values is empty; nothing to adjust.")
    if p_values.isna().any():
        raise ValueError("p_values contains NaN.")

    m = len(p_values)
    order = np.argsort(p_values.to_numpy(), kind="stable")
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for step, position in enumerate(order):
        running = min(1.0, max(running, (m - step) * float(p_values.iloc[position])))
        adjusted[position] = running
    return pd.Series(adjusted, index=p_values.index, name="p_holm")


def pairwise_paired_t(
    panel: pd.DataFrame,
    *,
    correction: str = "holm",
) -> pd.DataFrame:
    """Pairwise two-sided paired t-tests over the columns of a panel.

    Runs a paired t-test for every pair of entries (columns) over the
    shared blocks (rows) and returns the p-values as a square symmetric
    matrix — the pairwise stage of the blocked comparison protocol,
    rendered with
    `spotforecast2.plots.comparison.plot_critical_difference`.

    Args:
        panel: Complete block-by-entry frame (e.g. one row per target
            day, one column per entry, daily MAE values).  Must contain
            no NaN — subset to shared blocks first.
        correction: Multiple-testing correction applied across the
            pairs. Options:
            - ``"holm"``: Holm step-down adjustment (`holm_adjust`).
            - ``"none"``: raw p-values.
            Defaults to ``"holm"``.

    Returns:
        pd.DataFrame: Square symmetric p-value matrix over
        ``panel.columns`` with 1.0 on the diagonal.

    Raises:
        ValueError: When ``panel`` has fewer than two columns or rows,
            contains NaN, or ``correction`` is unknown.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.stats.comparison import pairwise_paired_t

        days = pd.date_range("2026-06-10", periods=8, freq="D")
        panel = pd.DataFrame(
            {
                "alpha": [500.0, 520, 480, 510, 490, 505, 515, 495],
                "beta": [540.0, 565, 515, 555, 530, 545, 560, 535],
                "gamma": [502.0, 519, 483, 508, 493, 504, 517, 496],
            },
            index=days,
        )
        p = pairwise_paired_t(panel)
        print(p.round(4).to_string())
        assert (p.values == p.values.T).all()
        ```
    """
    if panel.shape[1] < 2:
        raise ValueError("panel needs at least two columns (entries).")
    if panel.shape[0] < 2:
        raise ValueError("panel needs at least two rows (blocks).")
    if panel.isna().any().any():
        raise ValueError("panel contains NaN; subset to shared blocks first.")
    if correction not in ("holm", "none"):
        raise ValueError(f"correction must be 'holm' or 'none', got {correction!r}.")

    columns = list(panel.columns)
    pairs = [(a, b) for i, a in enumerate(columns) for b in columns[i + 1 :]]
    raw = pd.Series(
        [float(stats.ttest_rel(panel[a], panel[b]).pvalue) for a, b in pairs],
        index=pd.MultiIndex.from_tuples(pairs),
    )
    values = holm_adjust(raw) if correction == "holm" else raw

    matrix = pd.DataFrame(1.0, index=columns, columns=columns)
    for (a, b), p in values.items():
        matrix.loc[a, b] = matrix.loc[b, a] = p
    return matrix


@dataclass(frozen=True)
class RepeatedMeasuresAnova:
    """Result of `repeated_measures_anova()` — one-way RM-ANOVA.

    Attributes:
        f_statistic: F statistic of the treatment effect.
        p_value: Right-tail p-value of ``f_statistic`` under
            ``F(df_treatment, df_error)``.
        df_treatment: Treatment degrees of freedom (entries - 1).
        df_error: Error degrees of freedom
            ((entries - 1) * (blocks - 1)).
        n_blocks: Number of blocks (rows; e.g. target days).
        n_treatments: Number of treatments (columns; e.g. entries).
        ss_treatment: Treatment sum of squares.
        ss_block: Block sum of squares.
        ss_error: Error sum of squares.
    """

    f_statistic: float
    p_value: float
    df_treatment: int
    df_error: int
    n_blocks: int
    n_treatments: int
    ss_treatment: float
    ss_block: float
    ss_error: float


def repeated_measures_anova(panel: pd.DataFrame) -> RepeatedMeasuresAnova:
    """Run a one-way repeated-measures ANOVA on a complete panel.

    Tests equality of the column (entry) means of a complete
    block-by-entry matrix, treating the rows (e.g. target days) as
    blocks — the parametric omnibus test run before pairwise comparison
    with `pairwise_paired_t`.

    Args:
        panel: Complete block-by-entry frame (one row per block, one
            column per treatment).  Must contain no NaN.

    Returns:
        A `RepeatedMeasuresAnova` with the F test and the full
        sum-of-squares decomposition.

    Raises:
        ValueError: When ``panel`` has fewer than two rows or columns,
            or contains NaN.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.stats.comparison import repeated_measures_anova

        days = pd.date_range("2026-06-10", periods=5, freq="D")
        panel = pd.DataFrame(
            {
                "alpha": [500.0, 700, 400, 650, 550],
                "beta": [540.0, 745, 435, 690, 595],
                "gamma": [505.0, 702, 405, 653, 552],
            },
            index=days,
        )
        anova = repeated_measures_anova(panel)
        print(f"F({anova.df_treatment}, {anova.df_error}) = "
              f"{anova.f_statistic:.2f}, p = {anova.p_value:.4f}")
        assert anova.n_blocks == 5 and anova.n_treatments == 3
        ```
    """
    if panel.shape[1] < 2:
        raise ValueError("panel needs at least two columns (treatments).")
    if panel.shape[0] < 2:
        raise ValueError("panel needs at least two rows (blocks).")
    if panel.isna().any().any():
        raise ValueError("panel contains NaN; subset to complete blocks first.")

    X = panel.values
    n_blocks, n_treatments = X.shape
    grand = X.mean()
    ss_treatment = n_blocks * ((X.mean(axis=0) - grand) ** 2).sum()
    ss_block = n_treatments * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_error = ((X - grand) ** 2).sum() - ss_treatment - ss_block
    df_treatment = n_treatments - 1
    df_error = (n_treatments - 1) * (n_blocks - 1)
    f_statistic = (ss_treatment / df_treatment) / (ss_error / df_error)
    p_value = stats.f.sf(f_statistic, df_treatment, df_error)
    return RepeatedMeasuresAnova(
        f_statistic=float(f_statistic),
        p_value=float(p_value),
        df_treatment=df_treatment,
        df_error=df_error,
        n_blocks=n_blocks,
        n_treatments=n_treatments,
        ss_treatment=float(ss_treatment),
        ss_block=float(ss_block),
        ss_error=float(ss_error),
    )


def rank_table(
    scores: pd.DataFrame,
    *,
    method: str = "min",
    ascending: bool = True,
) -> pd.DataFrame:
    """Rank entries under each metric column.

    Converts a per-entry score table (e.g. mean MAE / RMSE / MAPE / MASE
    per entry) into integer ranks per metric, for concordance analysis
    (`rank_concordance`) or a rank-stability chart
    (`spotforecast2.plots.comparison.plot_rank_stability`).

    Args:
        scores: Frame indexed by entry, one numeric column per metric.
        method: Tie-breaking method forwarded to ``pd.DataFrame.rank``
            (e.g. ``"min"``, ``"average"``, ``"dense"``). Defaults to
            ``"min"``.
        ascending: Whether lower scores rank first. Defaults to True.

    Returns:
        pd.DataFrame: Integer ranks, same shape and labels as
        ``scores``.

    Raises:
        ValueError: When ``scores`` is empty or contains NaN.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.stats.comparison import rank_table

        scores = pd.DataFrame(
            {"mae": [510.0, 462.0, 891.0], "rmse": [640.0, 655.0, 1024.0]},
            index=["alpha", "beta", "gamma"],
        )
        ranks = rank_table(scores)
        print(ranks.to_string())
        assert list(ranks["mae"]) == [2, 1, 3]
        ```
    """
    if scores.empty:
        raise ValueError("scores is empty; nothing to rank.")
    if scores.isna().any().any():
        raise ValueError("scores contains NaN; drop incomplete entries first.")
    return scores.rank(method=method, ascending=ascending).astype(int)


def rank_concordance(
    ranks: pd.DataFrame,
    *,
    reference: str | None = None,
) -> pd.Series:
    """Kendall's tau between a reference ranking and every other column.

    Quantifies how stable a leaderboard is across metrics: a tau of 1.0
    means a metric reproduces the reference ranking exactly.

    Args:
        ranks: Frame of ranks (or scores), one column per metric —
            typically the output of `rank_table`.
        reference: Column the others are compared against.  Defaults to
            None, which uses the first column.

    Returns:
        pd.Series: Kendall's tau per non-reference column, in column
        order, name ``"kendall_tau"``.

    Raises:
        ValueError: When ``ranks`` has fewer than two columns or
            ``reference`` is not a column.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2_safe.stats.comparison import rank_concordance

        ranks = pd.DataFrame(
            {"mae": [1, 2, 3, 4], "rmse": [1, 2, 3, 4], "mape": [4, 3, 2, 1]},
            index=["a", "b", "c", "d"],
        )
        tau = rank_concordance(ranks, reference="mae")
        print(tau.to_string())
        assert tau["rmse"] == 1.0 and tau["mape"] == -1.0
        ```
    """
    if ranks.shape[1] < 2:
        raise ValueError("ranks needs at least two columns.")
    if reference is None:
        reference = str(ranks.columns[0])
    if reference not in ranks.columns:
        raise ValueError(f"reference {reference!r} is not a column of ranks.")

    others = [c for c in ranks.columns if c != reference]
    values = [
        float(stats.kendalltau(ranks[reference], ranks[c]).statistic) for c in others
    ]
    return pd.Series(values, index=pd.Index(others), name="kendall_tau")
