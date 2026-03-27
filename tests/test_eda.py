"""
Tests for survey_toolkit.eda module.
"""

import pytest
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests


class TestSurveyEDA:
    """Tests for the SurveyEDA class."""

    def test_init(self, clean_survey_df, tmp_dir):
        from survey_toolkit.eda import SurveyEDA

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        assert len(eda.data) == len(clean_survey_df)
        assert eda.output_dir.exists()

    def test_response_summary(self, clean_survey_df, tmp_dir):
        from survey_toolkit.eda import SurveyEDA

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        summary = eda.response_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "column" in summary.columns
        assert "dtype" in summary.columns
        assert "n_valid" in summary.columns
        assert "pct_missing" in summary.columns
        assert len(summary) == len(clean_survey_df.columns)

    def test_response_summary_numeric_stats(self, clean_survey_df, tmp_dir):
        from survey_toolkit.eda import SurveyEDA

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        summary = eda.response_summary()

        # Check that numeric columns have mean/std
        q1_row = summary[summary["column"] == "q1"].iloc[0]
        assert pd.notna(q1_row.get("mean"))
        assert pd.notna(q1_row.get("std"))

    def test_plot_likert_distribution(
        self, clean_survey_df, likert_columns, tmp_dir
    ):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        eda.plot_likert_distribution(likert_columns[:3], save=True)
        plt.close("all")

        assert (tmp_dir / "likert_distribution.png").exists()

    def test_plot_correlation_heatmap(
        self, clean_survey_df, likert_columns, tmp_dir
    ):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        eda.plot_correlation_heatmap(likert_columns[:5], save=True)
        plt.close("all")

        assert (tmp_dir / "correlation_heatmap.png").exists()

    def test_plot_demographic_breakdown_bar(
        self, clean_survey_df, tmp_dir
    ):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        eda.plot_demographic_breakdown("age_group", plot_type="bar", save=True)
        plt.close("all")

        assert (tmp_dir / "demographic_age_group.png").exists()

    def test_plot_demographic_breakdown_pie(
        self, clean_survey_df, tmp_dir
    ):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        eda.plot_demographic_breakdown("gender", plot_type="pie", save=True)
        plt.close("all")

        assert (tmp_dir / "demographic_gender.png").exists()

    def test_plot_response_by_group(self, clean_survey_df, tmp_dir):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        eda.plot_response_by_group("q1", "age_group", plot_type="box", save=True)
        plt.close("all")

        assert (tmp_dir / "q1_by_age_group.png").exists()

    def test_missing_data_report_with_missing(
        self, sample_survey_df, tmp_dir
    ):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(sample_survey_df, output_dir=str(tmp_dir))
        eda.missing_data_report(save=True)
        plt.close("all")

        assert (tmp_dir / "missing_data.png").exists()

    def test_missing_data_report_no_missing(self, tmp_dir):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        eda = SurveyEDA(df, output_dir=str(tmp_dir))
        eda.missing_data_report(save=True)
        plt.close("all")

        # No file should be created when no missing data
        assert not (tmp_dir / "missing_data.png").exists()

    def test_full_eda_report(
        self, clean_survey_df, likert_columns, demographic_columns, tmp_dir
    ):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        summary = eda.full_eda_report(
            likert_cols=likert_columns[:3],
            demographic_cols=demographic_columns[:1],
        )
        plt.close("all")

        assert isinstance(summary, pd.DataFrame)
        assert len(eda.figures) > 0

    def test_figures_tracked(self, clean_survey_df, likert_columns, tmp_dir):
        from survey_toolkit.eda import SurveyEDA
        import matplotlib.pyplot as plt

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        eda.plot_likert_distribution(likert_columns[:2], save=True)
        eda.plot_correlation_heatmap(likert_columns[:3], save=True)
        plt.close("all")

        assert len(eda.figures) == 2

    def test_repr(self, clean_survey_df, tmp_dir):
        from survey_toolkit.eda import SurveyEDA

        eda = SurveyEDA(clean_survey_df, output_dir=str(tmp_dir))
        repr_str = repr(eda)
        assert "SurveyEDA" in repr_str