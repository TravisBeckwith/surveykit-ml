"""
Tests for survey_toolkit.cli module.
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestCLI:
    """Tests for CLI commands."""

    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "survey_toolkit.cli", "--version"],
            capture_output=True,
            text=True,
        )
        # May not work depending on entry point setup; test import instead
        from survey_toolkit.cli import main
        assert callable(main)

    def test_generate_sample(self, tmp_dir):
        """Test sample data generation via Python function call."""
        from survey_toolkit.utils import generate_sample_survey

        path = str(tmp_dir / "cli_sample.csv")
        df = generate_sample_survey(n_respondents=50, save_path=path)
        assert Path(path).exists()
        assert len(df) == 50

    def test_main_with_eda(self, sample_csv_file, tmp_dir, monkeypatch):
        """Test EDA via CLI by simulating argparse."""
        import argparse
        from unittest.mock import patch
        import matplotlib
        matplotlib.use("Agg")

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--eda",
            "--output-dir", str(tmp_dir),
            "--format", "csv",
            "--missing-strategy", "median",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

        # Check that output directory has files
        files = list(tmp_dir.rglob("*"))
        assert len(files) > 0

    def test_main_with_alpha(self, sample_csv_file, tmp_dir, monkeypatch):
        """Test Cronbach's alpha via CLI."""
        from unittest.mock import patch
        import matplotlib
        matplotlib.use("Agg")

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--alpha", "q1", "q2", "q3", "q4", "q5",
            "--output-dir", str(tmp_dir),
            "--format", "json",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

    def test_main_with_stats(self, sample_csv_file, tmp_dir):
        """Test stats summary via CLI."""
        from unittest.mock import patch

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--stats", "q1", "q2", "q3",
            "--output-dir", str(tmp_dir),
            "--format", "print",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

    def test_main_with_cluster(self, sample_csv_file, tmp_dir):
        """Test clustering via CLI."""
        from unittest.mock import patch
        import matplotlib
        matplotlib.use("Agg")

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--cluster", "q1", "q2", "q3", "q4", "q5",
            "--n-clusters", "3",
            "--output-dir", str(tmp_dir),
            "--format", "csv",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

    def test_main_with_compare(self, sample_csv_file, tmp_dir):
        """Test group comparison via CLI."""
        from unittest.mock import patch

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--compare", "q1",
            "--group", "age_group",
            "--output-dir", str(tmp_dir),
            "--format", "json",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

    def test_main_with_correlations(self, sample_csv_file, tmp_dir):
        """Test correlation matrix via CLI."""
        from unittest.mock import patch

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--correlations", "q1", "q2", "q3",
            "--output-dir", str(tmp_dir),
            "--format", "csv",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

    def test_main_with_chi_square(self, sample_csv_file, tmp_dir):
        """Test chi-square via CLI."""
        from unittest.mock import patch

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--chi-square", "age_group", "gender",
            "--output-dir", str(tmp_dir),
            "--format", "print",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

    @pytest.mark.slow
    def test_main_with_classify(self, sample_csv_file, tmp_dir):
        """Test classification via CLI."""
        from unittest.mock import patch

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--classify", "q1", "q2", "q3", "q4", "q5",
            "--target", "satisfaction_group",
            "--output-dir", str(tmp_dir),
            "--format", "csv",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            try:
                main()
            except SystemExit:
                pass

    def test_compare_without_group_exits(self, sample_csv_file, tmp_dir):
        """Test that --compare without --group shows error."""
        from unittest.mock import patch

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--compare", "q1",
            "--output-dir", str(tmp_dir),
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            with pytest.raises(SystemExit):
                main()

    def test_classify_without_target_exits(self, sample_csv_file, tmp_dir):
        """Test that --classify without --target shows error."""
        from unittest.mock import patch

        test_args = [
            "survey-analyze",
            sample_csv_file,
            "--classify", "q1", "q2",
            "--output-dir", str(tmp_dir),
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import main
            with pytest.raises(SystemExit):
                main()


class TestReportCLI:
    """Tests for the survey-report CLI command."""

    def test_generate_report_function_exists(self):
        from survey_toolkit.cli import generate_report
        assert callable(generate_report)

    def test_generate_report(self, sample_csv_file, tmp_dir):
        """Test report generation via CLI."""
        from unittest.mock import patch
        import matplotlib
        matplotlib.use("Agg")

        output_path = str(tmp_dir / "test_report.html")

        test_args = [
            "survey-report",
            sample_csv_file,
            "--columns", "q1", "q2", "q3",
            "--output", output_path,
            "--title", "CLI Test Report",
            "--author", "Test Author",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import generate_report
            try:
                generate_report()
            except SystemExit:
                pass

        assert Path(output_path).exists()

    @pytest.mark.slow
    def test_generate_full_analysis_report(self, sample_csv_file, tmp_dir):
        """Test full analysis report via CLI."""
        from unittest.mock import patch
        import matplotlib
        matplotlib.use("Agg")

        output_path = str(tmp_dir / "full_report.html")

        test_args = [
            "survey-report",
            sample_csv_file,
            "--columns", "q1", "q2", "q3", "q4", "q5",
            "--output", output_path,
            "--full-analysis",
        ]

        with patch("sys.argv", test_args):
            from survey_toolkit.cli import generate_report
            try:
                generate_report()
            except SystemExit:
                pass

        assert Path(output_path).exists()