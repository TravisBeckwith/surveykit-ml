"""
Module for cleaning and preprocessing survey data.
"""

import pandas as pd
import numpy as np
from typing import Optional
from survey_toolkit.utils import logger


class SurveyCleaner:
    """Handle common survey data cleaning tasks with method chaining."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.original_shape = data.shape
        self.cleaning_log = []

    def _log(self, action: str, details: str):
        """Log a cleaning action."""
        entry = {"action": action, "details": details}
        self.cleaning_log.append(entry)
        logger.info(f"[Clean] {action}: {details}")

    def remove_speeders(
        self,
        duration_col: str,
        min_seconds: int = 60,
    ) -> "SurveyCleaner":
        """Remove respondents who completed the survey too quickly."""
        n_before = len(self.data)
        self.data = self.data[self.data[duration_col] >= min_seconds]
        n_removed = n_before - len(self.data)
        self._log("remove_speeders", f"Removed {n_removed} speeders (< {min_seconds}s)")
        return self

    def remove_straightliners(
        self,
        likert_cols: list[str],
        threshold: float = 0.95,
    ) -> "SurveyCleaner":
        """
        Remove respondents who gave the same answer
        for >= threshold of Likert-scale questions.
        """
        n_before = len(self.data)
        subset = self.data[likert_cols]
        mode_pct = subset.apply(
            lambda row: (
                row.value_counts().max() / row.notna().sum()
                if row.notna().sum() > 0
                else 0
            ),
            axis=1,
        )
        self.data = self.data[mode_pct < threshold]
        n_removed = n_before - len(self.data)
        self._log(
            "remove_straightliners",
            f"Removed {n_removed} straightliners (threshold={threshold})",
        )
        return self

    def remove_duplicates(
        self,
        subset: Optional[list[str]] = None,
        keep: str = "first",
    ) -> "SurveyCleaner":
        """Remove duplicate rows."""
        n_before = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        n_removed = n_before - len(self.data)
        self._log("remove_duplicates", f"Removed {n_removed} duplicate rows")
        return self

    def handle_missing(
        self,
        strategy: str = "drop",
        threshold: float = 0.5,
        fill_value: Optional[any] = None,
        columns: Optional[list[str]] = None,
    ) -> "SurveyCleaner":
        """
        Handle missing data.

        Strategies:
            - 'drop': Drop rows with too many missing values.
            - 'fill': Fill with a specific value.
            - 'median': Fill numeric columns with median.
            - 'mode': Fill all columns with mode.
            - 'drop_cols': Drop columns with too many missing values.
            - 'interpolate': Interpolate numeric columns.
        """
        target_cols = columns or self.data.columns.tolist()

        if strategy == "drop":
            self.data = self.data.dropna(
                subset=target_cols,
                thresh=int(len(target_cols) * threshold),
            )

        elif strategy == "fill":
            self.data[target_cols] = self.data[target_cols].fillna(fill_value)

        elif strategy == "median":
            numeric_cols = self.data[target_cols].select_dtypes(
                include=[np.number]
            ).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(
                self.data[numeric_cols].median()
            )

        elif strategy == "mode":
            for col in target_cols:
                if self.data[col].isnull().any():
                    mode_val = self.data[col].mode()
                    if not mode_val.empty:
                        self.data[col] = self.data[col].fillna(mode_val.iloc[0])

        elif strategy == "drop_cols":
            cols_to_drop = [
                col for col in target_cols
                if self.data[col].isnull().mean() > threshold
            ]
            self.data = self.data.drop(columns=cols_to_drop)
            self._log(
                "handle_missing",
                f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}",
            )
            return self

        elif strategy == "interpolate":
            numeric_cols = self.data[target_cols].select_dtypes(
                include=[np.number]
            ).columns
            self.data[numeric_cols] = self.data[numeric_cols].interpolate()

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                "Choose from: drop, fill, median, mode, drop_cols, interpolate"
            )

        self._log("handle_missing", f"Strategy: {strategy}")
        return self

    def encode_likert(
        self,
        columns: list[str],
        mapping: Optional[dict] = None,
    ) -> "SurveyCleaner":
        """Encode Likert scale text responses to numeric."""
        default_mapping = {
            "Strongly Disagree": 1,
            "Disagree": 2,
            "Neutral": 3,
            "Agree": 4,
            "Strongly Agree": 5,
        }
        mapping = mapping or default_mapping
        for col in columns:
            self.data[col] = self.data[col].map(mapping)
        self._log("encode_likert", f"Encoded {len(columns)} columns")
        return self

    def recode_reverse_scored(
        self,
        columns: list[str],
        scale_max: int = 5,
    ) -> "SurveyCleaner":
        """Reverse-code negatively worded items."""
        for col in columns:
            self.data[col] = (scale_max + 1) - self.data[col]
        self._log("recode_reverse", f"Reverse-coded {len(columns)} columns")
        return self

    def rename_columns(
        self,
        mapping: dict[str, str],
    ) -> "SurveyCleaner":
        """Rename columns using a mapping dictionary."""
        self.data = self.data.rename(columns=mapping)
        self._log("rename_columns", f"Renamed {len(mapping)} columns")
        return self

    def filter_respondents(
        self,
        column: str,
        values: list,
        exclude: bool = False,
    ) -> "SurveyCleaner":
        """Filter respondents by column values."""
        n_before = len(self.data)
        if exclude:
            self.data = self.data[~self.data[column].isin(values)]
        else:
            self.data = self.data[self.data[column].isin(values)]
        n_after = len(self.data)
        action = "excluded" if exclude else "kept"
        self._log(
            "filter_respondents",
            f"{action} {n_after}/{n_before} rows on {column}",
        )
        return self

    def add_computed_column(
        self,
        name: str,
        func: callable,
    ) -> "SurveyCleaner":
        """Add a new column based on a function applied to each row."""
        self.data[name] = self.data.apply(func, axis=1)
        self._log("add_computed_column", f"Added column: {name}")
        return self

    def get_clean_data(self) -> pd.DataFrame:
        """Return the cleaned dataframe."""
        logger.info(
            f"Cleaning complete: {self.original_shape} -> {self.data.shape}"
        )
        return self.data

    def get_log(self) -> list[dict]:
        """Return the cleaning log."""
        return self.cleaning_log

    def __repr__(self) -> str:
        return (
            f"SurveyCleaner(original={self.original_shape}, "
            f"current={self.data.shape}, "
            f"steps={len(self.cleaning_log)})"
        )