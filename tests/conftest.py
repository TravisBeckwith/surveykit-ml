"""
Shared test fixtures for the Survey ML Toolkit test suite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


# ============================================================
# Directories
# ============================================================

@pytest.fixture(scope="session")
def test_output_dir():
    """Create a temporary output directory for test artifacts."""
    tmpdir = Path(tempfile.mkdtemp(prefix="survey_test_"))
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def tmp_dir(tmp_path):
    """Per-test temporary directory (built-in pytest fixture)."""
    return tmp_path


# ============================================================
# Sample DataFrames
# ============================================================

@pytest.fixture
def sample_survey_df():
    """Create a basic sample survey DataFrame."""
    rng = np.random.RandomState(42)
    n = 200

    data = {
        "respondent_id": [f"R{str(i).zfill(4)}" for i in range(1, n + 1)],
        "age_group": rng.choice(
            ["18-24", "25-34", "35-44", "45-54", "55+"],
            size=n,
            p=[0.15, 0.25, 0.30, 0.20, 0.10],
        ),
        "gender": rng.choice(
            ["Male", "Female", "Non-binary"],
            size=n,
            p=[0.48, 0.48, 0.04],
        ),
        "education": rng.choice(
            ["High School", "Bachelor's", "Master's", "Doctorate"],
            size=n,
            p=[0.20, 0.40, 0.30, 0.10],
        ),
        "duration_seconds": np.clip(
            rng.normal(600, 200, size=n).astype(int), 30, 3600
        ),
    }

    # Likert items with some correlation structure
    base = rng.normal(3.2, 0.8, size=n)
    for i in range(1, 11):
        noise = rng.normal(0, 0.5, size=n)
        data[f"q{i}"] = np.clip(np.round(base + noise), 1, 5).astype(int)

    # NPS score
    data["nps_score"] = np.clip(
        (base * 2 + rng.normal(0, 0.5, size=n)).round(), 0, 10
    ).astype(int)

    # Satisfaction group (target)
    likert_mean = np.mean(
        [data[f"q{i}"] for i in range(1, 11)], axis=0
    )
    data["satisfaction_group"] = pd.cut(
        likert_mean,
        bins=[0, 2.5, 3.5, 5.1],
        labels=["Dissatisfied", "Neutral", "Satisfied"],
    ).astype(str)

    # Open-ended
    data["open_ended"] = rng.choice(
        [
            "Great product!",
            "Could be better.",
            "Very disappointed.",
            "Average experience.",
            np.nan,
        ],
        size=n,
    )

    df = pd.DataFrame(data)

    # Inject some missing values in Likert items (~3%)
    for col in [f"q{i}" for i in range(1, 11)]:
        missing_idx = rng.choice(n, size=int(n * 0.03), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Inject speeders (~5%)
    speeder_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[speeder_idx, "duration_seconds"] = rng.randint(15, 59, size=len(speeder_idx))

    # Inject straightliners (~4%)
    sl_idx = rng.choice(n, size=int(n * 0.04), replace=False)
    for idx in sl_idx:
        val = rng.randint(1, 6)
        for col in [f"q{i}" for i in range(1, 11)]:
            df.loc[idx, col] = val

    return df


@pytest.fixture
def likert_columns():
    """Return standard Likert column names."""
    return [f"q{i}" for i in range(1, 11)]


@pytest.fixture
def demographic_columns():
    """Return demographic column names."""
    return ["age_group", "gender", "education"]


@pytest.fixture
def small_likert_df():
    """Small DataFrame for quick tests."""
    rng = np.random.RandomState(99)
    n = 50
    data = {}
    base = rng.normal(3.0, 1.0, size=n)
    for i in range(1, 6):
        noise = rng.normal(0, 0.4, size=n)
        data[f"q{i}"] = np.clip(np.round(base + noise), 1, 5).astype(int)
    data["group"] = rng.choice(["A", "B"], size=n)
    return pd.DataFrame(data)


@pytest.fixture
def text_likert_df():
    """DataFrame with text-based Likert responses."""
    rng = np.random.RandomState(123)
    n = 100
    labels = [
        "Strongly Disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly Agree",
    ]
    data = {}
    for i in range(1, 4):
        data[f"q{i}"] = rng.choice(labels, size=n, p=[0.05, 0.15, 0.30, 0.35, 0.15])
    return pd.DataFrame(data)


# ============================================================
# Sample Files
# ============================================================

@pytest.fixture
def sample_csv_file(tmp_dir, sample_survey_df):
    """Save sample survey to a CSV file and return path."""
    path = tmp_dir / "test_survey.csv"
    sample_survey_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_excel_file(tmp_dir, sample_survey_df):
    """Save sample survey to an Excel file and return path."""
    path = tmp_dir / "test_survey.xlsx"
    sample_survey_df.to_excel(path, index=False, engine="openpyxl")
    return str(path)


@pytest.fixture
def sample_json_file(tmp_dir, sample_survey_df):
    """Save sample survey to a JSON file and return path."""
    path = tmp_dir / "test_survey.json"
    sample_survey_df.to_json(path)
    return str(path)


# ============================================================
# Clean DataFrames (pre-processed)
# ============================================================

@pytest.fixture
def clean_survey_df(sample_survey_df, likert_columns):
    """Return a pre-cleaned survey DataFrame (no NaN in Likert cols)."""
    from survey_toolkit.cleaner import SurveyCleaner

    cleaner = SurveyCleaner(sample_survey_df)
    return (
        cleaner
        .remove_speeders("duration_seconds", min_seconds=60)
        .remove_straightliners(likert_columns, threshold=0.95)
        .handle_missing(strategy="median")
        .get_clean_data()
    )


# ============================================================
# Helper Assertions
# ============================================================

@pytest.fixture
def assert_valid_result():
    """Factory fixture for validating result dicts."""

    def _assert(result, required_keys):
        assert isinstance(result, dict), "Result should be a dict"
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        return True

    return _assert