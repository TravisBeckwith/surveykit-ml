"""
Tests for survey_toolkit.utils module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestGenerateSampleSurvey:
    """Tests for the sample survey generator."""

    def test_default_generation(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500
        assert "respondent_id" in df.columns

    def test_custom_size(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey(n_respondents=100)
        assert len(df) == 100

    def test_custom_likert_items(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey(n_likert_items=5)
        likert_cols = [c for c in df.columns if c.startswith("q")]
        assert len(likert_cols) == 5

    def test_likert_range(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey(n_likert_items=5)
        for i in range(1, 6):
            col = f"q{i}"
            valid = df[col].dropna()
            assert valid.min() >= 1
            assert valid.max() <= 5

    def test_has_speeders(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey(n_respondents=500)
        speeders = df[df["duration_seconds"] < 60]
        assert len(speeders) > 0

    def test_has_missing_values(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey()
        assert df.isnull().any().any()

    def test_has_satisfaction_group(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey()
        assert "satisfaction_group" in df.columns
        assert set(df["satisfaction_group"].unique()).issubset(
            {"Dissatisfied", "Neutral", "Satisfied"}
        )

    def test_open_ended_included(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey(include_open_ended=True)
        assert "open_ended_feedback" in df.columns

    def test_open_ended_excluded(self):
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey(include_open_ended=False)
        assert "open_ended_feedback" not in df.columns

    def test_save_to_file(self, tmp_dir):
        from survey_toolkit.utils import generate_sample_survey

        path = str(tmp_dir / "sample.csv")
        df = generate_sample_survey(save_path=path)
        assert Path(path).exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == len(df)

    def test_reproducibility(self):
        from survey_toolkit.utils import generate_sample_survey

        df1 = generate_sample_survey(random_state=42)
        df2 = generate_sample_survey(random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        from survey_toolkit.utils import generate_sample_survey

        df1 = generate_sample_survey(random_state=42)
        df2 = generate_sample_survey(random_state=99)
        assert not df1.equals(df2)


class TestDetectColumnTypes:
    """Tests for automatic column type detection."""

    def test_detects_likert(self, sample_survey_df):
        from survey_toolkit.utils import detect_column_types

        types = detect_column_types(sample_survey_df)
        assert "likert" in types
        # q1-q10 should be detected as Likert
        for i in range(1, 11):
            assert f"q{i}" in types["likert"], f"q{i} not detected as Likert"

    def test_detects_categorical(self, sample_survey_df):
        from survey_toolkit.utils import detect_column_types

        types = detect_column_types(sample_survey_df)
        assert "categorical" in types
        assert "age_group" in types["categorical"]
        assert "gender" in types["categorical"]

    def test_detects_id(self, sample_survey_df):
        from survey_toolkit.utils import detect_column_types

        types = detect_column_types(sample_survey_df)
        assert "respondent_id" in types["id"]

    def test_detects_text(self, sample_survey_df):
        from survey_toolkit.utils import detect_column_types

        types = detect_column_types(sample_survey_df)
        # open_ended could be text or categorical depending on uniqueness
        all_cols = []
        for col_list in types.values():
            all_cols.extend(col_list)
        assert "open_ended" in all_cols

    def test_all_columns_assigned(self, sample_survey_df):
        from survey_toolkit.utils import detect_column_types

        types = detect_column_types(sample_survey_df)
        all_detected = []
        for col_list in types.values():
            all_detected.extend(col_list)
        for col in sample_survey_df.columns:
            assert col in all_detected, f"Column {col} not assigned a type"


class TestValidateSurveyData:
    """Tests for data validation."""

    def test_valid_data(self, sample_survey_df):
        from survey_toolkit.utils import validate_survey_data

        result = validate_survey_data(sample_survey_df)
        assert result["valid"] is True
        assert result["n_respondents"] == len(sample_survey_df)

    def test_too_few_respondents(self):
        from survey_toolkit.utils import validate_survey_data

        df = pd.DataFrame({"q1": [1, 2, 3]})
        result = validate_survey_data(df, min_respondents=30)
        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_missing_required_columns(self, sample_survey_df):
        from survey_toolkit.utils import validate_survey_data

        result = validate_survey_data(
            sample_survey_df,
            required_columns=["nonexistent_column"],
        )
        assert result["valid"] is False

    def test_high_missing_warning(self):
        from survey_toolkit.utils import validate_survey_data

        df = pd.DataFrame({
            "q1": [1, 2, np.nan, np.nan, np.nan] * 20,
            "q2": [1, 2, 3, 4, 5] * 20,
        })
        result = validate_survey_data(df, max_missing_pct=50)
        assert len(result["warnings"]) > 0


class TestComputeScaleScores:
    """Tests for composite scale score computation."""

    def test_mean_scores(self, sample_survey_df):
        from survey_toolkit.utils import compute_scale_scores

        construct_map = {
            "satisfaction": ["q1", "q2", "q3"],
            "usability": ["q4", "q5", "q6"],
        }
        scores = compute_scale_scores(
            sample_survey_df, construct_map, method="mean"
        )
        assert "satisfaction" in scores.columns
        assert "usability" in scores.columns
        assert len(scores) == len(sample_survey_df)

    def test_sum_scores(self, sample_survey_df):
        from survey_toolkit.utils import compute_scale_scores

        construct_map = {"total": ["q1", "q2", "q3"]}
        scores = compute_scale_scores(
            sample_survey_df, construct_map, method="sum"
        )
        assert "total" in scores.columns

    def test_invalid_method(self, sample_survey_df):
        from survey_toolkit.utils import compute_scale_scores

        with pytest.raises(ValueError, match="Unknown method"):
            compute_scale_scores(
                sample_survey_df,
                {"x": ["q1"]},
                method="invalid",
            )


class TestExportResults:
    """Tests for export functionality."""

    def test_export_csv(self, tmp_dir):
        from survey_toolkit.utils import export_results

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        saved = export_results(
            {"test_data": df},
            output_dir=str(tmp_dir),
            formats=["csv"],
        )
        assert len(saved) == 1
        assert Path(saved[0]).exists()

    def test_export_json(self, tmp_dir):
        from survey_toolkit.utils import export_results

        result = {"key": "value", "number": 42}
        saved = export_results(
            {"test_result": result},
            output_dir=str(tmp_dir),
            formats=["json"],
        )
        assert len(saved) == 1
        assert Path(saved[0]).suffix == ".json"

    def test_export_multiple_formats(self, tmp_dir):
        from survey_toolkit.utils import export_results

        df = pd.DataFrame({"a": [1, 2, 3]})
        saved = export_results(
            {"data": df},
            output_dir=str(tmp_dir),
            formats=["csv", "json"],
        )
        assert len(saved) == 2


class TestTimer:
    """Tests for the timer decorator."""

    def test_timer_returns_result(self):
        from survey_toolkit.utils import timer

        @timer
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_timer_preserves_name(self):
        from survey_toolkit.utils import timer

        @timer
        def my_function():
            pass

        assert my_function.__name__ == "my_function"