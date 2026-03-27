"""
Utility functions and helpers for the Survey ML Toolkit.
"""

import pandas as pd
import numpy as np
import logging
import time
import functools
from typing import Optional
from pathlib import Path


# ============================================================
# Logging
# ============================================================

def get_logger(
    name: str = "survey_toolkit",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    log_file : str, optional
        Path to a log file. If None, logs only to console.

    Returns
    -------
    logging.Logger
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    # Avoid adding duplicate handlers
    if not log.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)

    return log


# Default package logger
logger = get_logger()


# ============================================================
# Decorators
# ============================================================

def timer(func):
    """Decorator that logs the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapper


def validate_dataframe(func):
    """Decorator that validates the first argument is a non-empty DataFrame."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check positional args and 'data' keyword
        df = None
        if len(args) > 1 and isinstance(args[1], pd.DataFrame):
            df = args[1]
        elif "data" in kwargs and isinstance(kwargs["data"], pd.DataFrame):
            df = kwargs["data"]
        elif len(args) > 0 and isinstance(args[0], pd.DataFrame):
            df = args[0]

        if df is not None and df.empty:
            raise ValueError("DataFrame is empty.")
        return func(*args, **kwargs)

    return wrapper


# ============================================================
# Sample Data Generator
# ============================================================

def generate_sample_survey(
    n_respondents: int = 500,
    n_likert_items: int = 10,
    n_demographic_cols: int = 3,
    include_open_ended: bool = True,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a realistic sample survey dataset for testing.

    Parameters
    ----------
    n_respondents : int
        Number of survey respondents.
    n_likert_items : int
        Number of Likert-scale questions (1–5).
    n_demographic_cols : int
        Number of demographic columns to include.
    include_open_ended : bool
        Whether to include a sample open-ended text column.
    random_state : int
        Seed for reproducibility.
    save_path : str, optional
        File path to save the generated CSV.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.RandomState(random_state)
    data = {}

    # ---- Respondent ID ----
    data["respondent_id"] = [f"R{str(i).zfill(4)}" for i in range(1, n_respondents + 1)]

    # ---- Demographics ----
    demographic_options = {
        "age_group": {
            "values": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            "weights": [0.12, 0.22, 0.25, 0.20, 0.13, 0.08],
        },
        "gender": {
            "values": ["Male", "Female", "Non-binary", "Prefer not to say"],
            "weights": [0.47, 0.48, 0.03, 0.02],
        },
        "education": {
            "values": [
                "High School",
                "Some College",
                "Bachelor's",
                "Master's",
                "Doctorate",
            ],
            "weights": [0.15, 0.20, 0.35, 0.22, 0.08],
        },
        "income_bracket": {
            "values": [
                "<\$25K", "\$25K-\$50K", "\$50K-\$75K",
                "\$75K-\$100K", "\$100K-\$150K", "\$150K+",
            ],
            "weights": [0.10, 0.18, 0.25, 0.22, 0.15, 0.10],
        },
        "region": {
            "values": ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
            "weights": [0.20, 0.22, 0.18, 0.15, 0.25],
        },
        "employment": {
            "values": [
                "Full-time", "Part-time", "Self-employed",
                "Unemployed", "Student", "Retired",
            ],
            "weights": [0.45, 0.12, 0.10, 0.08, 0.15, 0.10],
        },
    }

    demo_keys = list(demographic_options.keys())[:n_demographic_cols]
    for key in demo_keys:
        opts = demographic_options[key]
        data[key] = rng.choice(
            opts["values"],
            size=n_respondents,
            p=opts["weights"],
        )

    # ---- Likert-Scale Items ----
    # Create correlated Likert items grouped into constructs
    n_constructs = max(1, n_likert_items // 3)
    items_per_construct = n_likert_items // n_constructs
    remainder = n_likert_items % n_constructs

    construct_names = [
        "satisfaction", "usability", "trust",
        "loyalty", "engagement", "value",
    ]

    item_idx = 1
    construct_map = {}

    for c in range(n_constructs):
        construct_name = construct_names[c % len(construct_names)]
        n_items = items_per_construct + (1 if c < remainder else 0)

        # Generate correlated responses
        base = rng.normal(3.2, 0.8, size=n_respondents)
        for j in range(n_items):
            col_name = f"q{item_idx}"
            noise = rng.normal(0, 0.5, size=n_respondents)
            raw = base + noise
            # Clip and round to Likert scale
            likert = np.clip(np.round(raw), 1, 5).astype(int)
            data[col_name] = likert
            construct_map[col_name] = construct_name
            item_idx += 1

    # ---- Duration (seconds) ----
    data["duration_seconds"] = np.clip(
        rng.normal(600, 200, size=n_respondents).astype(int),
        30,
        3600,
    )
    # Inject some speeders
    n_speeders = int(n_respondents * 0.05)
    speeder_idx = rng.choice(n_respondents, size=n_speeders, replace=False)
    for idx in speeder_idx:
        data["duration_seconds"][idx] = rng.randint(15, 59)

    # ---- Satisfaction Group (target variable) ----
    likert_cols = [f"q{i}" for i in range(1, n_likert_items + 1)]
    likert_df = pd.DataFrame({col: data[col] for col in likert_cols})
    mean_score = likert_df.mean(axis=1)
    data["satisfaction_group"] = pd.cut(
        mean_score,
        bins=[0, 2.5, 3.5, 5.1],
        labels=["Dissatisfied", "Neutral", "Satisfied"],
    ).astype(str)

    # ---- NPS Score ----
    data["nps_score"] = np.clip(
        (mean_score * 2 + rng.normal(0, 0.5, size=n_respondents)).round(),
        0,
        10,
    ).astype(int)

    # ---- Open-Ended Response ----
    if include_open_ended:
        positive_responses = [
            "Great experience overall, very satisfied with the service.",
            "The product exceeded my expectations. Would recommend.",
            "Easy to use and intuitive. Love it!",
            "Customer support was excellent and responsive.",
            "Best purchase I've made this year.",
            "Very professional and well-designed.",
            "Impressed with the quality and attention to detail.",
        ]
        neutral_responses = [
            "It was okay, nothing special.",
            "Average experience. Some room for improvement.",
            "Decent product but could be better.",
            "Met basic expectations but nothing more.",
            "No strong feelings either way.",
        ]
        negative_responses = [
            "Very disappointed with the quality.",
            "Would not recommend. Poor experience overall.",
            "Difficult to use and frustrating.",
            "Customer service was unhelpful.",
            "Did not meet my expectations at all.",
            "Needs significant improvement.",
        ]

        open_ended = []
        for score in mean_score:
            if score >= 3.8:
                open_ended.append(rng.choice(positive_responses))
            elif score >= 2.5:
                open_ended.append(rng.choice(neutral_responses))
            else:
                open_ended.append(rng.choice(negative_responses))

        # Inject some missing open-ended responses (~15%)
        n_missing = int(n_respondents * 0.15)
        missing_idx = rng.choice(n_respondents, size=n_missing, replace=False)
        for idx in missing_idx:
            open_ended[idx] = np.nan

        data["open_ended_feedback"] = open_ended

    # ---- Inject Missing Values (~3% across Likert items) ----
    df = pd.DataFrame(data)
    for col in likert_cols:
        n_missing = int(n_respondents * 0.03)
        missing_idx = rng.choice(n_respondents, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    # ---- Inject Straightliners (~4%) ----
    n_straightliners = int(n_respondents * 0.04)
    sl_idx = rng.choice(n_respondents, size=n_straightliners, replace=False)
    for idx in sl_idx:
        straight_val = rng.randint(1, 6)
        for col in likert_cols:
            df.loc[idx, col] = straight_val

    # ---- Save if requested ----
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Sample survey saved to {path}")

    # ---- Store construct map as attribute ----
    df.attrs["construct_map"] = construct_map

    return df


# ============================================================
# Column Type Detection
# ============================================================

def detect_column_types(
    data: pd.DataFrame,
    likert_range: tuple[int, int] = (1, 5),
    categorical_threshold: int = 15,
) -> dict:
    """
    Automatically detect and categorize survey column types.

    Parameters
    ----------
    data : pd.DataFrame
        Survey data.
    likert_range : tuple
        Expected (min, max) range for Likert-scale items.
    categorical_threshold : int
        Max unique values to classify as categorical.

    Returns
    -------
    dict
        Mapping of column type to list of column names.
    """
    column_types = {
        "id": [],
        "likert": [],
        "numeric_continuous": [],
        "categorical": [],
        "binary": [],
        "text": [],
        "datetime": [],
        "unknown": [],
    }

    for col in data.columns:
        series = data[col].dropna()

        if series.empty:
            column_types["unknown"].append(col)
            continue

        # Check for ID columns
        if (
            col.lower() in ("id", "respondent_id", "response_id", "uid")
            or (series.nunique() == len(series) and series.dtype == "object")
        ):
            column_types["id"].append(col)
            continue

        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            column_types["datetime"].append(col)
            continue

        # Check for text (long strings)
        if series.dtype == "object":
            avg_len = series.astype(str).str.len().mean()
            n_unique = series.nunique()
            if avg_len > 50 or n_unique > len(series) * 0.5:
                column_types["text"].append(col)
            elif n_unique == 2:
                column_types["binary"].append(col)
            elif n_unique <= categorical_threshold:
                column_types["categorical"].append(col)
            else:
                column_types["text"].append(col)
            continue

        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = sorted(series.unique())
            min_val, max_val = likert_range

            # Check if Likert
            if (
                all(isinstance(v, (int, float, np.integer)) for v in unique_vals)
                and min(unique_vals) >= min_val
                and max(unique_vals) <= max_val
                and len(unique_vals) <= (max_val - min_val + 1)
                and all(v == int(v) for v in unique_vals)
            ):
                column_types["likert"].append(col)
            elif series.nunique() == 2:
                column_types["binary"].append(col)
            elif series.nunique() <= categorical_threshold:
                column_types["categorical"].append(col)
            else:
                column_types["numeric_continuous"].append(col)
            continue

        column_types["unknown"].append(col)

    return column_types


# ============================================================
# Data Validation
# ============================================================

def validate_survey_data(
    data: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    max_missing_pct: float = 50.0,
    min_respondents: int = 30,
) -> dict:
    """
    Validate survey data quality and return a diagnostic report.

    Parameters
    ----------
    data : pd.DataFrame
        Survey data to validate.
    required_columns : list, optional
        Columns that must be present.
    max_missing_pct : float
        Max allowable missing percentage per column.
    min_respondents : int
        Minimum number of respondents required.

    Returns
    -------
    dict
        Validation results with pass/fail status and details.
    """
    issues = []
    warnings = []

    # Check minimum respondents
    if len(data) < min_respondents:
        issues.append(
            f"Only {len(data)} respondents (minimum: {min_respondents})"
        )

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

    # Check missing data
    high_missing = []
    for col in data.columns:
        pct = data[col].isna().mean() * 100
        if pct > max_missing_pct:
            high_missing.append(f"{col} ({pct:.1f}%)")
    if high_missing:
        warnings.append(
            f"Columns with >{max_missing_pct}% missing: {high_missing}"
        )

    # Check for zero-variance columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    zero_var = [col for col in numeric_cols if data[col].std() == 0]
    if zero_var:
        warnings.append(f"Zero-variance columns: {zero_var}")

    # Check for duplicates
    n_dupes = data.duplicated().sum()
    if n_dupes > 0:
        warnings.append(f"Found {n_dupes} duplicate rows")

    return {
        "valid": len(issues) == 0,
        "n_respondents": len(data),
        "n_columns": len(data.columns),
        "issues": issues,
        "warnings": warnings,
        "overall_missing_pct": round(data.isna().mean().mean() * 100, 2),
    }


# ============================================================
# Scale Scoring
# ============================================================

def compute_scale_scores(
    data: pd.DataFrame,
    construct_map: dict[str, list[str]],
    method: str = "mean",
) -> pd.DataFrame:
    """
    Compute composite scale scores from item groupings.

    Parameters
    ----------
    data : pd.DataFrame
        Survey data with individual items.
    construct_map : dict
        Mapping of construct names to lists of column names.
        Example: {"satisfaction": ["q1", "q2", "q3"]}
    method : str
        Aggregation method: 'mean' or 'sum'.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed scale scores.
    """
    scores = {}
    for construct, items in construct_map.items():
        valid_items = [col for col in items if col in data.columns]
        if not valid_items:
            logger.warning(
                f"No valid items found for construct '{construct}'"
            )
            continue
        if method == "mean":
            scores[construct] = data[valid_items].mean(axis=1)
        elif method == "sum":
            scores[construct] = data[valid_items].sum(axis=1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean' or 'sum'.")

    return pd.DataFrame(scores, index=data.index)


# ============================================================
# Export Helpers
# ============================================================

def export_results(
    results: dict,
    output_dir: str = "outputs",
    formats: list[str] = None,
) -> list[str]:
    """
    Export analysis results to multiple formats.

    Parameters
    ----------
    results : dict
        Dictionary of named results (DataFrames, dicts, etc.).
    output_dir : str
        Output directory path.
    formats : list
        List of formats: 'csv', 'json', 'excel'. Default: ['csv'].

    Returns
    -------
    list[str]
        Paths of saved files.
    """
    formats = formats or ["csv"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for name, result in results.items():
        for fmt in formats:
            if isinstance(result, pd.DataFrame):
                if fmt == "csv":
                    fpath = output_path / f"{name}.csv"
                    result.to_csv(fpath)
                elif fmt == "json":
                    fpath = output_path / f"{name}.json"
                    result.to_json(fpath, indent=2)
                elif fmt == "excel":
                    fpath = output_path / f"{name}.xlsx"
                    result.to_excel(fpath, engine="openpyxl")
                else:
                    logger.warning(f"Unknown format: {fmt}")
                    continue
                saved_files.append(str(fpath))
                logger.info(f"Saved {name} to {fpath}")

            elif isinstance(result, dict):
                import json
                fpath = output_path / f"{name}.json"
                with open(fpath, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                saved_files.append(str(fpath))
                logger.info(f"Saved {name} to {fpath}")

    return saved_files