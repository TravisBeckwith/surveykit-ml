# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure

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