"""
Tests for survey_toolkit.cleaner module.
"""

import pytest
import pandas as pd
import numpy as np


class TestSurveyCleaner:
    """Tests for the SurveyCleaner class."""

    def test_init(self, sample_survey_df):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        assert cleaner.data.shape == sample_survey_df.shape
        assert len(cleaner.cleaning_log) == 0

    def test_does_not_modify_original(self, sample_survey_df):
        from survey_toolkit.cleaner import SurveyCleaner

        original_shape = sample_survey_df.shape
        cleaner = SurveyCleaner(sample_survey_df)
        cleaner.handle_missing(strategy="drop")
        assert sample_survey_df.shape == original_shape

    def test_remove_speeders(self, sample_survey_df):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        result = cleaner.remove_speeders("duration_seconds", min_seconds=60)
        clean = result.get_clean_data()

        assert len(clean) < len(sample_survey_df)
        assert (clean["duration_seconds"] >= 60).all()
        assert isinstance(result, type(cleaner))  # Method chaining

    def test_remove_straightliners(self, sample_survey_df, likert_columns):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        result = cleaner.remove_straightliners(
            likert_columns, threshold=0.95
        )
        clean = result.get_clean_data()
        assert len(clean) <= len(sample_survey_df)

    def test_remove_duplicates(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({
            "q1": [1, 2, 3, 3],
            "q2": [4, 5, 6, 6],
        })
        cleaner = SurveyCleaner(df)
        clean = cleaner.remove_duplicates().get_clean_data()
        assert len(clean) == 3

    def test_handle_missing_drop(self, sample_survey_df):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        clean = cleaner.handle_missing(strategy="drop").get_clean_data()
        assert len(clean) <= len(sample_survey_df)

    def test_handle_missing_median(self, sample_survey_df, likert_columns):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        clean = cleaner.handle_missing(
            strategy="median", columns=likert_columns
        ).get_clean_data()

        for col in likert_columns:
            assert clean[col].isna().sum() == 0

    def test_handle_missing_mode(self, sample_survey_df, likert_columns):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        clean = cleaner.handle_missing(
            strategy="mode", columns=likert_columns
        ).get_clean_data()

        for col in likert_columns:
            assert clean[col].isna().sum() == 0

    def test_handle_missing_fill(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({"q1": [1, np.nan, 3], "q2": [np.nan, 2, 3]})
        cleaner = SurveyCleaner(df)
        clean = cleaner.handle_missing(
            strategy="fill", fill_value=0
        ).get_clean_data()
        assert clean.isna().sum().sum() == 0
        assert clean.loc[1, "q1"] == 0

    def test_handle_missing_drop_cols(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({
            "good_col": [1, 2, 3, 4, 5],
            "bad_col": [np.nan, np.nan, np.nan, np.nan, 1],
        })
        cleaner = SurveyCleaner(df)
        clean = cleaner.handle_missing(
            strategy="drop_cols", threshold=0.5
        ).get_clean_data()
        assert "bad_col" not in clean.columns
        assert "good_col" in clean.columns

    def test_handle_missing_interpolate(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({"q1": [1.0, np.nan, 3.0, np.nan, 5.0]})
        cleaner = SurveyCleaner(df)
        clean = cleaner.handle_missing(
            strategy="interpolate"
        ).get_clean_data()
        assert clean["q1"].isna().sum() == 0

    def test_handle_missing_invalid_strategy(self, sample_survey_df):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        with pytest.raises(ValueError, match="Unknown strategy"):
            cleaner.handle_missing(strategy="invalid")

    def test_encode_likert(self, text_likert_df):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(text_likert_df)
        clean = cleaner.encode_likert(["q1", "q2", "q3"]).get_clean_data()

        assert clean["q1"].dtype in [np.int64, np.float64]
        assert clean["q1"].min() >= 1
        assert clean["q1"].max() <= 5

    def test_encode_likert_custom_mapping(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({"q1": ["Bad", "OK", "Good"]})
        mapping = {"Bad": 1, "OK": 2, "Good": 3}
        cleaner = SurveyCleaner(df)
        clean = cleaner.encode_likert(["q1"], mapping=mapping).get_clean_data()
        assert list(clean["q1"]) == [1, 2, 3]

    def test_recode_reverse_scored(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({"q1": [1, 2, 3, 4, 5]})
        cleaner = SurveyCleaner(df)
        clean = cleaner.recode_reverse_scored(
            ["q1"], scale_max=5
        ).get_clean_data()
        assert list(clean["q1"]) == [5, 4, 3, 2, 1]

    def test_rename_columns(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({"old_name": [1, 2, 3]})
        cleaner = SurveyCleaner(df)
        clean = cleaner.rename_columns(
            {"old_name": "new_name"}
        ).get_clean_data()
        assert "new_name" in clean.columns
        assert "old_name" not in clean.columns

    def test_filter_respondents_keep(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({
            "group": ["A", "B", "C", "A", "B"],
            "q1": [1, 2, 3, 4, 5],
        })
        cleaner = SurveyCleaner(df)
        clean = cleaner.filter_respondents(
            "group", ["A", "B"]
        ).get_clean_data()
        assert len(clean) == 4
        assert set(clean["group"].unique()) == {"A", "B"}

    def test_filter_respondents_exclude(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({
            "group": ["A", "B", "C", "A", "B"],
            "q1": [1, 2, 3, 4, 5],
        })
        cleaner = SurveyCleaner(df)
        clean = cleaner.filter_respondents(
            "group", ["C"], exclude=True
        ).get_clean_data()
        assert len(clean) == 4
        assert "C" not in clean["group"].values

    def test_add_computed_column(self):
        from survey_toolkit.cleaner import SurveyCleaner

        df = pd.DataFrame({"q1": [1, 2, 3], "q2": [4, 5, 6]})
        cleaner = SurveyCleaner(df)
        clean = cleaner.add_computed_column(
            "total", lambda row: row["q1"] + row["q2"]
        ).get_clean_data()
        assert "total" in clean.columns
        assert list(clean["total"]) == [5, 7, 9]

    def test_method_chaining(self, sample_survey_df, likert_columns):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        clean = (
            cleaner
            .remove_speeders("duration_seconds", min_seconds=60)
            .remove_straightliners(likert_columns, threshold=0.95)
            .handle_missing(strategy="median")
            .get_clean_data()
        )
        assert isinstance(clean, pd.DataFrame)
        assert len(clean) > 0

    def test_cleaning_log(self, sample_survey_df, likert_columns):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        (
            cleaner
            .remove_speeders("duration_seconds")
            .handle_missing(strategy="median")
        )
        log = cleaner.get_log()
        assert len(log) == 2
        assert log[0]["action"] == "remove_speeders"
        assert log[1]["action"] == "handle_missing"

    def test_repr(self, sample_survey_df):
        from survey_toolkit.cleaner import SurveyCleaner

        cleaner = SurveyCleaner(sample_survey_df)
        repr_str = repr(cleaner)
        assert "SurveyCleaner" in repr_str
        assert "steps=0" in repr_str