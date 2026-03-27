"""
Tests for survey_toolkit.stats module.
"""

import pytest
import pandas as pd
import numpy as np


class TestCronbachsAlpha:
    """Tests for Cronbach's alpha calculation."""

    def test_basic_alpha(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.cronbachs_alpha(likert_columns)

        assert "alpha" in result
        assert "n_items" in result
        assert "n_valid" in result
        assert "interpretation" in result
        assert "item_total_correlations" in result
        assert "alpha_if_deleted" in result
        assert result["n_items"] == len(likert_columns)
        assert 0 <= result["alpha"] <= 1

    def test_high_reliability(self):
        from survey_toolkit.stats import SurveyStats

        # Create highly correlated items
        rng = np.random.RandomState(42)
        base = rng.normal(3, 1, size=100)
        data = {
            f"q{i}": np.clip(np.round(base + rng.normal(0, 0.2, 100)), 1, 5)
            for i in range(1, 6)
        }
        df = pd.DataFrame(data)

        stats = SurveyStats(df)
        result = stats.cronbachs_alpha([f"q{i}" for i in range(1, 6)])
        assert result["alpha"] > 0.8

    def test_low_reliability(self):
        from survey_toolkit.stats import SurveyStats

        # Create uncorrelated items
        rng = np.random.RandomState(42)
        data = {
            f"q{i}": rng.randint(1, 6, size=100) for i in range(1, 6)
        }
        df = pd.DataFrame(data)

        stats = SurveyStats(df)
        result = stats.cronbachs_alpha([f"q{i}" for i in range(1, 6)])
        assert result["alpha"] < 0.5

    def test_too_few_items(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        with pytest.raises(ValueError, match="at least 2 items"):
            stats.cronbachs_alpha(["q1"])

    def test_item_total_correlations(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.cronbachs_alpha(likert_columns)

        itc = result["item_total_correlations"]
        assert len(itc) == len(likert_columns)
        for col, corr in itc.items():
            assert -1 <= corr <= 1

    def test_alpha_if_deleted(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.cronbachs_alpha(likert_columns)

        aid = result["alpha_if_deleted"]
        assert len(aid) == len(likert_columns)

    def test_interpretation_categories(self):
        from survey_toolkit.stats import SurveyStats

        rng = np.random.RandomState(42)
        base = rng.normal(3, 1, size=200)
        data = {
            f"q{i}": np.clip(np.round(base + rng.normal(0, 0.1, 200)), 1, 5)
            for i in range(1, 6)
        }
        df = pd.DataFrame(data)

        stats = SurveyStats(df)
        result = stats.cronbachs_alpha([f"q{i}" for i in range(1, 6)])
        assert result["interpretation"] in [
            "Excellent", "Good", "Acceptable",
            "Questionable", "Poor", "Unacceptable",
        ]


class TestCompareGroups:
    """Tests for group comparison."""

    def test_two_group_comparison(self, small_likert_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(small_likert_df)
        result = stats.compare_groups("q1", "group")

        assert "test" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "effect_size" in result
        assert "significant" in result
        assert "group_stats" in result
        assert len(result["groups"]) == 2

    def test_auto_selects_test(self, small_likert_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(small_likert_df)
        result = stats.compare_groups("q1", "group", test="auto")
        assert result["test"] in [
            "Independent t-test",
            "Independent t-test (Welch's)",
            "Mann-Whitney U",
        ]

    def test_forced_ttest(self, small_likert_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(small_likert_df)
        result = stats.compare_groups("q1", "group", test="t-test")
        assert "t-test" in result["test"]

    def test_forced_mann_whitney(self, small_likert_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(small_likert_df)
        result = stats.compare_groups("q1", "group", test="mann-whitney")
        assert result["test"] == "Mann-Whitney U"

    def test_three_group_comparison(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.compare_groups("q1", "age_group")
        assert result["test"] in ["One-way ANOVA", "Kruskal-Wallis H"]
        assert len(result["groups"]) >= 3

    def test_forced_anova(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.compare_groups("q1", "age_group", test="anova")
        assert result["test"] == "One-way ANOVA"

    def test_forced_kruskal(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.compare_groups("q1", "age_group", test="kruskal")
        assert result["test"] == "Kruskal-Wallis H"

    def test_group_stats_structure(self, small_likert_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(small_likert_df)
        result = stats.compare_groups("q1", "group")

        for name, gs in result["group_stats"].items():
            assert "mean" in gs
            assert "median" in gs
            assert "std" in gs
            assert "n" in gs

    def test_invalid_test(self, small_likert_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(small_likert_df)
        with pytest.raises(ValueError, match="Unknown test"):
            stats.compare_groups("q1", "group", test="invalid_test")

    def test_single_group_raises(self):
        from survey_toolkit.stats import SurveyStats

        df = pd.DataFrame({"q1": [1, 2, 3], "group": ["A", "A", "A"]})
        stats = SurveyStats(df)
        with pytest.raises(ValueError, match="at least 2 groups"):
            stats.compare_groups("q1", "group")


class TestCorrelationMatrix:
    """Tests for correlation analysis."""

    def test_pearson_correlation(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        cols = likert_columns[:5]
        stats = SurveyStats(clean_survey_df)
        result = stats.correlation_matrix(cols, method="pearson")

        assert "correlation_matrix" in result
        assert "p_values" in result
        assert "significant_pairs" in result
        assert result["method"] == "pearson"
        assert result["correlation_matrix"].shape == (5, 5)

    def test_spearman_correlation(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        cols = likert_columns[:3]
        stats = SurveyStats(clean_survey_df)
        result = stats.correlation_matrix(cols, method="spearman")
        assert result["method"] == "spearman"

    def test_kendall_correlation(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        cols = likert_columns[:3]
        stats = SurveyStats(clean_survey_df)
        result = stats.correlation_matrix(cols, method="kendall")
        assert result["method"] == "kendall"

    def test_diagonal_is_one(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        cols = likert_columns[:4]
        stats = SurveyStats(clean_survey_df)
        result = stats.correlation_matrix(cols)
        corr = result["correlation_matrix"]

        for col in cols:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10

    def test_correlations_in_range(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        cols = likert_columns[:5]
        stats = SurveyStats(clean_survey_df)
        result = stats.correlation_matrix(cols)
        corr = result["correlation_matrix"]

        assert (corr >= -1).all().all()
        assert (corr <= 1).all().all()

    def test_significant_pairs_structure(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        cols = likert_columns[:5]
        stats = SurveyStats(clean_survey_df)
        result = stats.correlation_matrix(cols)

        for pair in result["significant_pairs"]:
            assert "var1" in pair
            assert "var2" in pair
            assert "correlation" in pair
            assert "p_value" in pair
            assert pair["p_value"] < 0.05

    def test_invalid_method(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        with pytest.raises(ValueError, match="Unknown method"):
            stats.correlation_matrix(likert_columns[:3], method="invalid")


class TestChiSquare:
    """Tests for chi-square test of independence."""

    def test_basic_chi_square(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.chi_square_test("age_group", "gender")

        assert "chi2" in result
        assert "p_value" in result
        assert "dof" in result
        assert "cramers_v" in result
        assert "significant" in result
        assert "contingency_table" in result
        assert "expected_frequencies" in result

    def test_cramers_v_range(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.chi_square_test("age_group", "gender")
        assert 0 <= result["cramers_v"] <= 1

    def test_effect_size_interpretation(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.chi_square_test("age_group", "gender")
        assert result["effect_size_interpretation"] in [
            "Large", "Medium", "Small", "Negligible"
        ]

    def test_contingency_table_shape(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.chi_square_test("age_group", "gender")
        ct = result["contingency_table"]
        ef = result["expected_frequencies"]
        assert ct.shape == ef.shape


class TestFactorAnalysis:
    """Tests for exploratory factor analysis."""

    def test_basic_factor_analysis(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.factor_analysis(likert_columns)

        assert "bartlett_chi_sq" in result
        assert "bartlett_p" in result
        assert "kmo" in result
        assert "n_factors" in result
        assert "loadings" in result
        assert "communalities" in result
        assert "variance_explained" in result
        assert "item_assignments" in result

    def test_specified_n_factors(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.factor_analysis(likert_columns, n_factors=2)
        assert result["n_factors"] == 2
        assert result["loadings"].shape[1] == 2

    def test_loadings_shape(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.factor_analysis(likert_columns, n_factors=3)
        loadings = result["loadings"]
        assert loadings.shape[0] == len(likert_columns)
        assert loadings.shape[1] == 3

    def test_kmo_range(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.factor_analysis(likert_columns)
        assert 0 <= result["kmo"] <= 1

    def test_kmo_interpretation(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.factor_analysis(likert_columns)
        assert result["kmo_interpretation"] in [
            "Marvelous", "Meritorious", "Middling",
            "Mediocre", "Miserable", "Unacceptable",
        ]

    def test_variance_explained_structure(
        self, clean_survey_df, likert_columns
    ):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.factor_analysis(likert_columns, n_factors=2)
        ve = result["variance_explained"]

        assert "per_factor" in ve
        assert "cumulative" in ve
        assert "eigenvalues" in ve
        assert len(ve["per_factor"]) == 2
        assert ve["cumulative"][-1] >= ve["cumulative"][0]


class TestProportionTest:
    """Tests for proportion tests."""

    def test_one_sample_proportion(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.proportion_test("gender", "Male")

        assert "proportion" in result
        assert "p_value" in result
        assert "statistic" in result
        assert 0 <= result["proportion"] <= 1

    def test_two_sample_proportion(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.proportion_test(
            "satisfaction_group", "Satisfied", group_col="gender"
        )

        assert "proportions" in result
        assert "p_value" in result
        assert len(result["groups"]) == 2


class TestDescriptivesByGroup:
    """Tests for grouped descriptive statistics."""

    def test_descriptives(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.descriptives_by_group(
            likert_columns[:3], "age_group"
        )

        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "age_group"


class TestSurveyStatsGeneral:
    """General tests for SurveyStats."""

    def test_get_all_results(self, clean_survey_df, likert_columns):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        stats.cronbachs_alpha(likert_columns)
        results = stats.get_all_results()
        assert "cronbachs_alpha" in results

    def test_repr(self, clean_survey_df):
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        repr_str = repr(stats)
        assert "SurveyStats" in repr_str