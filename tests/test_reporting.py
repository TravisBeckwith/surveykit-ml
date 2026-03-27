"""
Tests for survey_toolkit.reporting module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestReportGenerator:
    """Tests for the ReportGenerator class."""

    def test_init(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        assert len(report.sections) == 0
        assert report.title == "Survey Analysis Report"

    def test_set_metadata(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.set_metadata(
            title="Test Report",
            author="Test Author",
            description="Test description",
        )
        assert report.title == "Test Report"
        assert report.author == "Test Author"
        assert report.description == "Test description"

    def test_set_metadata_chaining(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        result = report.set_metadata(title="Test")
        assert isinstance(result, ReportGenerator)

    def test_add_section(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.add_section("Test Section", "<p>Hello</p>", "text")
        assert len(report.sections) == 1
        assert report.sections[0]["title"] == "Test Section"
        assert report.sections[0]["type"] == "text"

    def test_add_section_chaining(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        result = report.add_section("Test", "content")
        assert isinstance(result, ReportGenerator)

    def test_add_dataframe(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report.add_dataframe("Test Table", df, description="A table")

        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "table"
        assert "table" in report.sections[0]["content"].lower()
        assert "A table" in report.sections[0]["content"]

    def test_add_dataframe_truncation(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        df = pd.DataFrame({"a": range(100)})
        report.add_dataframe("Big Table", df, max_rows=10)

        assert "Showing 10 of 100" in report.sections[0]["content"]

    def test_add_stats_result(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        result = {
            "test": "Independent t-test",
            "statistic": 2.45,
            "p_value": 0.015,
            "significant": True,
            "effect_size": 0.35,
        }
        report.add_stats_result("Group Comparison", result)

        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "stats"
        content = report.sections[0]["content"]
        assert "2.45" in content
        assert "0.015" in content

    def test_add_stats_result_significance_colors(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)

        # Significant result
        report.add_stats_result("Sig Test", {
            "p_value": 0.01,
            "significant": True,
        })
        content = report.sections[0]["content"]
        assert "green" in content

    def test_add_stats_result_not_significant(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.add_stats_result("NS Test", {
            "p_value": 0.50,
            "significant": False,
        })
        content = report.sections[0]["content"]
        assert "red" in content

    def test_add_figure(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        # Create a dummy figure
        fig_path = tmp_dir / "test_fig.png"
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        fig.savefig(fig_path)
        plt.close("all")

        report = ReportGenerator(clean_survey_df)
        report.add_figure(
            "Test Figure", str(fig_path), description="A test figure"
        )

        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "figure"
        assert "base64" in report.sections[0]["content"]
        assert "A test figure" in report.sections[0]["content"]

    def test_add_figure_missing_file(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.add_figure("Missing", "nonexistent.png")
        assert "not found" in report.sections[0]["content"]

    def test_add_summary_statistics(self, clean_survey_df, likert_columns):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.add_summary_statistics(likert_columns[:3])
        assert len(report.sections) == 1
        assert "Summary Statistics" in report.sections[0]["title"]

    def test_generate_html(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.set_metadata(title="Test Report", author="Tester")
        report.add_section("Intro", "<p>Introduction text</p>")
        report.add_summary_statistics()

        output_path = str(tmp_dir / "test_report.html")
        result_path = report.generate(output_path=output_path)

        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "Test Report" in content
        assert "Tester" in content
        assert "<!DOCTYPE html>" in content

    def test_generate_auto_sections(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        output_path = str(tmp_dir / "auto_report.html")
        report.generate(output_path=output_path, auto_sections=True)

        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert "Summary Statistics" in content

    def test_generate_table_of_contents(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        # Add 4+ sections to trigger TOC
        for i in range(5):
            report.add_section(f"Section {i}", f"<p>Content {i}</p>")

        output_path = str(tmp_dir / "toc_report.html")
        report.generate(output_path=output_path, auto_sections=False)

        content = Path(output_path).read_text()
        assert "Table of Contents" in content

    def test_generate_no_toc_few_sections(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.add_section("Only Section", "<p>Hello</p>")

        output_path = str(tmp_dir / "no_toc.html")
        report.generate(output_path=output_path, auto_sections=False)

        content = Path(output_path).read_text()
        assert "Table of Contents" not in content

    def test_generate_creates_directories(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        deep_path = str(tmp_dir / "a" / "b" / "c" / "report.html")
        report.generate(output_path=deep_path)
        assert Path(deep_path).exists()

    def test_generate_with_title_override(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.set_metadata(title="Original Title")

        output_path = str(tmp_dir / "override.html")
        report.generate(output_path=output_path, title="Override Title")

        content = Path(output_path).read_text()
        assert "Override Title" in content

    def test_generate_css_included(self, clean_survey_df, tmp_dir):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        output_path = str(tmp_dir / "styled.html")
        report.generate(output_path=output_path)

        content = Path(output_path).read_text()
        assert "<style>" in content
        assert "font-family" in content

    def test_full_report_pipeline(
        self, clean_survey_df, likert_columns, tmp_dir
    ):
        from survey_toolkit.reporting import ReportGenerator
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        alpha = stats.cronbachs_alpha(likert_columns)
        corr = stats.correlation_matrix(likert_columns[:5])

        report = ReportGenerator(clean_survey_df)
        report.set_metadata(title="Full Pipeline Test")
        report.add_summary_statistics(likert_columns)
        report.add_stats_result("Cronbach's Alpha", alpha)
        report.add_dataframe(
            "Correlations",
            corr["correlation_matrix"],
        )

        output_path = str(tmp_dir / "full_pipeline.html")
        result = report.generate(output_path=output_path, auto_sections=False)

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "Cronbach" in content
        assert "Correlations" in content
        assert len(report.sections) == 3

    def test_repr(self, clean_survey_df):
        from survey_toolkit.reporting import ReportGenerator

        report = ReportGenerator(clean_survey_df)
        report.add_section("Test", "Content")
        repr_str = repr(report)
        assert "ReportGenerator" in repr_str
        assert "sections=1" in repr_str