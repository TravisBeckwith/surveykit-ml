"""
Module for loading and validating survey data from multiple formats.
"""

import pandas as pd
from pathlib import Path
from survey_toolkit.utils import logger, timer


class SurveyLoader:
    """Load survey data from various file formats with validation."""

    SUPPORTED_FORMATS = [".csv", ".xlsx", ".json", ".sav", ".dta"]

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._validate_file()
        self.data = None
        self.metadata = {}

    def _validate_file(self):
        """Check that the file exists and is a supported format."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        if self.filepath.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {self.filepath.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

    @timer
    def load(self, **kwargs) -> pd.DataFrame:
        """Load data based on file extension."""
        loaders = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".json": pd.read_json,
            ".sav": self._load_spss,
            ".dta": pd.read_stata,
        }
        loader_func = loaders[self.filepath.suffix]
        self.data = loader_func(self.filepath, **kwargs)
        self._collect_metadata()
        logger.info(
            f"Loaded {self.metadata['n_respondents']} respondents, "
            f"{self.metadata['n_questions']} columns from {self.filepath.name}"
        )
        return self.data

    @staticmethod
    def _load_spss(filepath, **kwargs):
        """Load SPSS .sav files."""
        try:
            import pyreadstat
        except ImportError:
            raise ImportError(
                "pyreadstat is required for SPSS files. "
                "Install with: pip install survey-ml-toolkit[io]"
            )
        df, meta = pyreadstat.read_sav(str(filepath), **kwargs)
        return df

    def _collect_metadata(self):
        """Collect basic metadata about the loaded dataset."""
        self.metadata = {
            "n_respondents": len(self.data),
            "n_questions": len(self.data.columns),
            "columns": list(self.data.columns),
            "dtypes": {
                col: str(dtype)
                for col, dtype in self.data.dtypes.to_dict().items()
            },
            "missing_pct": {
                col: round(pct, 2)
                for col, pct in (
                    self.data.isnull().sum() / len(self.data) * 100
                ).to_dict().items()
            },
            "memory_usage_mb": round(
                self.data.memory_usage(deep=True).sum() / 1e6, 2
            ),
        }

    def summary(self) -> dict:
        """Return dataset metadata summary."""
        if not self.metadata:
            raise ValueError("No data loaded. Call .load() first.")
        return self.metadata

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first n rows of loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call .load() first.")
        return self.data.head(n)

    def __repr__(self) -> str:
        if self.data is not None:
            return (
                f"SurveyLoader(file='{self.filepath.name}', "
                f"respondents={self.metadata['n_respondents']}, "
                f"columns={self.metadata['n_questions']})"
            )
        return f"SurveyLoader(file='{self.filepath.name}', loaded=False)"