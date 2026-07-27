# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-27

### Changed
- First stable release. Bumped version 0.1.0 → 1.0.0 and package
  classifier from Alpha to Production/Stable.
- Folded the orphaned `Config/Custom_Config`, `Config/Key Config Values`,
  `Config/Config in Modules`, `Config/Runtime Override`, and
  `Config/Example_Default_Config` snippet files into a proper
  "Configuration" section in `README.md`, and removed the loose files.

### Fixed
- **Broken package install**: `pyproject.toml` declared a non-existent
  `build-backend` (`setuptools.backends._legacy:_Backend`), causing
  `pip install .` / `pip install -e .` to fail for everyone. Corrected to
  `setuptools.build_meta`.
- **Missing `pyyaml` dependency**: `survey_toolkit/config.py` imports `yaml`,
  but PyYAML wasn't declared in `requirements.txt`, `setup.py`, or
  `pyproject.toml`, so config loading (and anything depending on it, via
  `utils.py`) crashed with `ModuleNotFoundError` on a clean install. Added
  `pyyaml>=6.0` everywhere.
- **pandas 3.0 compatibility**: `detect_column_types()` and
  `SurveyClassifier.prepare_data()` checked `series.dtype == "object"` to
  detect string columns. Pandas 3.0 (allowed by the unpinned
  `pandas>=2.0` constraint) defaults string columns to a new `str` dtype
  instead of `object`, so every categorical/text/ID/target-encoding branch
  silently misclassified data. Switched to
  `pd.api.types.is_string_dtype()`, which handles both dtypes.
- **scikit-learn / factor-analyzer incompatibility**: unpinned
  `scikit-learn>=1.3` resolves to a version that removed the
  `force_all_finite` parameter `factor-analyzer==0.5.1` still relies on,
  breaking every factor-analysis call with a `TypeError`. Constrained to
  `scikit-learn>=1.3,<1.6`.
- **SHAP multiclass feature importance**: `SurveyClassifier.feature_importance()`
  only handled the legacy list-of-per-class-arrays SHAP output. Newer SHAP
  versions return a single 3D `(samples, features, classes)` ndarray for
  multiclass models instead, which broke the importance `DataFrame`
  construction. Added handling for the 3D ndarray case.
- **Dead validation code in `SurveyStats.correlation_matrix()`**: an invalid
  `method` argument raised pandas' own (differently worded) `ValueError`
  before the function's own `"Unknown method"` check could ever run.
  Validation now happens up front.
- Invalid `\$` escape sequences in `utils.py` sample-data generator
  (`SyntaxWarning` today, will be a `SyntaxError` in a future Python).
- Added the `.secrets.baseline` file required by the `detect-secrets`
  pre-commit hook (was missing, breaking `pre-commit run` for anyone who
  installed the hooks).
- **Missing `openpyxl` dependency**: `SurveyLoader` treats `.xlsx` as a
  first-class, always-available format (no optional-import guard), but
  `openpyxl` was only declared under the `reporting`/`all`/`full` extras —
  so Excel loading broke on a plain `pip install` of the base package.
  Moved `openpyxl>=3.1` into the core dependencies (removed from the now-
  redundant extras entries).

## [0.1.0] - 2024-XX-XX

### Added
- `SurveyLoader` — Multi-format data loading (CSV, Excel, SPSS, Stata, JSON)
- `SurveyCleaner` — Survey-specific data cleaning with method chaining
  - Speeder removal
  - Straightliner detection
  - Missing data handling (6 strategies)
  - Likert encoding and reverse coding
- `SurveyEDA` — Automated exploratory data analysis
  - Likert distribution plots
  - Correlation heatmaps
  - Demographic breakdowns
  - Missing data visualization
- `SurveyStats` — Statistical analysis suite
  - Cronbach's alpha with item diagnostics
  - Group comparisons (auto-selecting t-test/ANOVA/non-parametric)
  - Correlation matrix with p-values
  - Chi-square test of independence
  - Exploratory Factor Analysis (EFA)
  - Proportion tests
- `SurveyClassifier` — ML classification pipeline
  - Multi-model comparison (LR, RF, GB, XGBoost)
  - SHAP feature importance
  - Hyperparameter tuning
  - Prediction on new data
- `SurveySegmentation` — Respondent clustering
  - Optimal k detection (silhouette + elbow)
  - Cluster profiling
  - PCA visualization
  - Demographic cross-tabulation
- `ReportGenerator` — Automated HTML/PDF reports
- CLI tools (`survey-analyze`, `survey-report`)
- Utility functions (sample data generator, column type detection, validation)
- Full test suite (~164 tests)
- Jupyter notebooks (EDA, stats, ML, reporting)
- CI/CD pipeline (GitHub Actions)

[Unreleased]: https://github.com/yourusername/survey-ml-toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/survey-ml-toolkit/releases/tag/v0.1.0