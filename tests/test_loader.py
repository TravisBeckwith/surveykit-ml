"""
Tests for survey_toolkit.loader module.
"""

import pytest
import pandas as pd
from pathlib import Path


class TestSurveyLoader:
    """Tests for the SurveyLoader class."""

    def test_load_csv(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_excel(self, sample_excel_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_excel_file)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_json(self, sample_json_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_json_file)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_file_not_found(self):
        from survey_toolkit.loader import SurveyLoader

        with pytest.raises(FileNotFoundError):
            SurveyLoader("nonexistent_file.csv")

    def test_unsupported_format(self, tmp_dir):
        from survey_toolkit.loader import SurveyLoader

        # Create a dummy file with unsupported extension
        path = tmp_dir / "data.xyz"
        path.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported format"):
            SurveyLoader(str(path))

    def test_metadata_collected(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        loader.load()
        meta = loader.summary()

        assert "n_respondents" in meta
        assert "n_questions" in meta
        assert "columns" in meta
        assert "dtypes" in meta
        assert "missing_pct" in meta
        assert "memory_usage_mb" in meta
        assert meta["n_respondents"] > 0

    def test_summary_before_load_raises(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        with pytest.raises(ValueError, match="No data loaded"):
            loader.summary()

    def test_head(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        loader.load()
        head = loader.head(3)
        assert len(head) == 3

    def test_head_before_load_raises(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        with pytest.raises(ValueError, match="No data loaded"):
            loader.head()

    def test_repr_before_load(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        repr_str = repr(loader)
        assert "loaded=False" in repr_str

    def test_repr_after_load(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        loader.load()
        repr_str = repr(loader)
        assert "respondents=" in repr_str

    def test_load_with_kwargs(self, tmp_dir):
        from survey_toolkit.loader import SurveyLoader

        # CSV with custom separator
        path = tmp_dir / "semicolon.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(
            path, index=False, sep=";"
        )
        loader = SurveyLoader(str(path))
        df = loader.load(sep=";")
        assert list(df.columns) == ["a", "b"]

    def test_missing_pct_in_metadata(self, sample_csv_file):
        from survey_toolkit.loader import SurveyLoader

        loader = SurveyLoader(sample_csv_file)
        loader.load()
        meta = loader.summary()
        assert isinstance(meta["missing_pct"], dict)