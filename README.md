#  Survey ML Toolkit

<div align="center">
  <p>
    <strong>A Python toolkit for statistical analysis and machine learning on survey data.</strong>
  </p>
  <p>Built for research analysts who need reproducible, rigorous analysis pipelines.</p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/TravisBeckwith/surveykit-ml/actions/workflows/ci.yml">
      <img src="https://github.com/TravisBeckwith/surveykit-ml/actions/workflows/ci.yml/badge.svg" alt="CI">
    </a>
    <a href="https://codecov.io/gh/TravisBeckwith/surveykit-ml">
      <img src="https://codecov.io/gh/TravisBeckwith/surveykit-ml/branch/main/graph/badge.svg" alt="Coverage">
    </a>
    <a href="https://pypi.org/project/survey-ml-toolkit/">
      <img src="https://badge.fury.io/py/survey-ml-toolkit.svg" alt="PyPI version">
    </a>
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    </a>
  </p>

  <!-- Quick Links -->
  <p>
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-installation">Installation</a> •
    <a href="#-documentation">Documentation</a> •
    <a href="#-examples">Examples</a> •
    <a href="#-contributing">Contributing</a>
  </p>
</div>

---

##  Overview

**Survey ML Toolkit** is an end-to-end Python library designed specifically for analyzing survey and questionnaire data. It combines traditional statistical methods used in social science research with modern machine learning techniques, all wrapped in a clean, chainable API.

Whether you're a market researcher analyzing customer satisfaction, an academic studying survey instruments, or a data scientist building predictive models from questionnaire data — this toolkit provides the building blocks you need.

### Why This Toolkit?

| Challenge | Solution |
|-----------|----------|
| Survey data has unique quality issues (speeders, straightliners) | **Built-in survey-specific cleaning** |
| Need both traditional stats AND modern ML | **Unified API for both paradigms** |
| Results need to be explainable to stakeholders | **SHAP values + auto-generated reports** |
| Repetitive boilerplate for every survey project | **One-line commands for common analyses** |
| Factor analysis, reliability testing are tedious to set up | **Pre-configured with best practices** |

---

##  Features

###  Data Loading
- Multi-format support: **CSV, Excel, SPSS (.sav), Stata (.dta), JSON**
- Automatic metadata collection (respondent count, missing data, dtypes)
- Lazy loading for large datasets

###  Survey-Specific Cleaning
- **Speeder detection** — remove respondents who finished too fast
- **Straightliner detection** — remove respondents giving identical answers
- **Missing data handling** — 6 strategies (drop, fill, median, mode, interpolate, drop columns)
- **Likert encoding** — convert text responses ("Strongly Agree") to numeric (5)
- **Reverse coding** — recode negatively worded items
- **Method chaining** — fluent API for clean, readable pipelines

###  Exploratory Data Analysis
- Automated response summaries with skewness and kurtosis
- **Likert distribution plots** (stacked horizontal bars)
- **Correlation heatmaps** (Pearson, Spearman, Kendall)
- **Demographic breakdowns** (bar charts, pie charts)
- **Response-by-group plots** (box, violin, strip)
- **Missing data visualization**
- Publication-ready figure export (PNG, 150 DPI)

###  Statistical Analysis
- **Cronbach's alpha** with item-total correlations and "alpha if deleted" diagnostics
- **Group comparisons** — auto-selects the right test:
  - 2 groups, normal: Independent t-test (or Welch's)
  - 2 groups, non-normal: Mann-Whitney U
  - 3+ groups, normal: One-way ANOVA + Tukey HSD post-hoc
  - 3+ groups, non-normal: Kruskal-Wallis H
- **Correlation matrices** with p-values and significant pair detection
- **Chi-square test** of independence with Cramér's V
- **Exploratory Factor Analysis (EFA)** — Bartlett's test, KMO, varimax/promax rotation
- **Proportion tests** — one-sample and two-sample z-tests
- **Effect sizes** — Cohen's d, eta-squared, epsilon-squared, rank-biserial r

###  Machine Learning
- **Model comparison** — Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Stratified k-fold cross-validation**
- **SHAP feature importance** — identify key survey drivers
- **Hyperparameter tuning** — grid search with sensible defaults
- **Prediction** — classify new respondents with probability estimates

###  Respondent Segmentation
- **K-Means clustering** with automatic optimal k detection
- **Silhouette analysis** + **elbow method** + **Calinski-Harabasz index**
- **Cluster profiling** — mean scores, standard deviations per segment
- **PCA visualization** — 2D scatter plots of segments
- **Demographic cross-tabulation** — understand who's in each segment

###  Automated Reporting
- **HTML reports** with embedded figures and styled tables
- **PDF export** (via WeasyPrint)
- Table of contents, executive summaries, methodology sections
- Significance highlighting (green/red)
- Customizable sections — mix and match analysis results
- Multiple report templates (executive, technical, full)

###  Command-Line Interface
- `survey-analyze` — run analyses from the terminal
- `survey-report` — generate reports from the command line
- JSON, CSV, and console output formats

---

##  Installation

### Basic Install

```bash
pip install survey-ml-toolkit
```

### With ML Features (XGBoost, SHAP)

```bash
pip install "survey-ml-toolkit[ml]"
```

### With Reporting (PDF export)

```bash
pip install "survey-ml-toolkit[reporting]"
```

### With SPSS/Stata File Support

```bash
pip install "survey-ml-toolkit[io]"
```

### With Interactive Dashboards

```bash
pip install "survey-ml-toolkit[dashboard]"
```

### Everything

```bash
pip install "survey-ml-toolkit[all]"
```

### Development Install

```bash
git clone https://github.com/TravisBeckwith/surveykit-ml.git
cd survey-ml-toolkit
pip install -e ".[full]"
pre-commit install
```

### Requirements
- Python ≥ 3.10
- Core dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `factor-analyzer`

---

##  Quick Start

### 30-Second Example

```python
from survey_toolkit import SurveyLoader, SurveyCleaner, SurveyStats

# Load
df = SurveyLoader("data/survey.csv").load()

# Clean
clean_df = (
    SurveyCleaner(df)
    .remove_speeders("duration_seconds", min_seconds=60)
    .remove_straightliners(["q1", "q2", "q3", "q4", "q5"])
    .handle_missing(strategy="median")
    .get_clean_data()
)

# Analyze
stats = SurveyStats(clean_df)
print(stats.cronbachs_alpha(["q1", "q2", "q3", "q4", "q5"]))
```

### Generate Sample Data

```python
from survey_toolkit import generate_sample_survey

df = generate_sample_survey(
    n_respondents=500,
    n_likert_items=10,
    save_path="data/sample_survey.csv",
)
print(f"Generated: {df.shape}")
```

---

##  Examples

### 1. Full EDA Pipeline

```python
from survey_toolkit import (
    SurveyLoader,
    SurveyCleaner,
    SurveyEDA,
    detect_column_types,
    validate_survey_data,
)

# Load & validate
df = SurveyLoader("data/survey.csv").load()
validation = validate_survey_data(df, min_respondents=30)
print(f"Valid: {validation['valid']}")

# Auto-detect column types
types = detect_column_types(df)
likert_cols = types["likert"]
demo_cols = types["categorical"]

# Clean
clean_df = (
    SurveyCleaner(df)
    .remove_speeders("duration_seconds", min_seconds=60)
    .remove_straightliners(likert_cols, threshold=0.95)
    .handle_missing(strategy="median", columns=likert_cols)
    .get_clean_data()
)

# EDA
eda = SurveyEDA(clean_df, output_dir="outputs/figures")
summary = eda.response_summary()
eda.plot_likert_distribution(likert_cols)
eda.plot_correlation_heatmap(likert_cols)
eda.plot_demographic_breakdown("age_group")
eda.missing_data_report()
```

### 2. Statistical Analysis

```python
from survey_toolkit import SurveyStats

stats = SurveyStats(clean_df)

# Reliability
alpha = stats.cronbachs_alpha(["q1", "q2", "q3", "q4", "q5"])
print(f"Cronbach's α = {alpha['alpha']} ({alpha['interpretation']})")

# Group comparison (auto-selects test)
result = stats.compare_groups("q1", "age_group")
print(f"{result['test']}: p = {result['p_value']}, {result['effect_size_name']} = {result['effect_size']}")

# Correlation matrix with p-values
corr = stats.correlation_matrix(["q1", "q2", "q3"], method="spearman")
print(f"Significant pairs: {len(corr['significant_pairs'])}")

# Chi-square test
chi2 = stats.chi_square_test("age_group", "gender")
print(f"χ² = {chi2['chi2']}, Cramér's V = {chi2['cramers_v']}")

# Factor analysis
fa = stats.factor_analysis(["q1", "q2", "q3", "q4", "q5"], rotation="varimax")
print(f"KMO = {fa['kmo']} ({fa['kmo_interpretation']})")
print(fa["loadings"])
```

### 3. Machine Learning Classification

```python
from survey_toolkit import SurveyClassifier

classifier = SurveyClassifier(clean_df)

# Prepare data
X, y = classifier.prepare_data(
    feature_cols=["q1", "q2", "q3", "q4", "q5"],
    target_col="satisfaction_group",
)

# Compare models
results = classifier.run_model_comparison(cv_folds=5)
print(results)

# Feature importance (SHAP)
importance = classifier.feature_importance()
print(importance)

# Hyperparameter tuning
tuned = classifier.hyperparameter_tune(
    model_name="Random Forest",
    param_grid={
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
    },
)
print(f"Best accuracy: {tuned['best_score']}")

# Predict
predictions = classifier.predict(new_survey_data)
probabilities = classifier.predict(new_survey_data, return_proba=True)
```

### 4. Respondent Segmentation

```python
from survey_toolkit import SurveySegmentation

segmenter = SurveySegmentation(clean_df)
segmenter.prepare_data(["q1", "q2", "q3", "q4", "q5"])

# Find optimal k
optimal = segmenter.find_optimal_k(k_range=range(2, 8))
print(f"Optimal clusters: {optimal['optimal_k']}")

# Fit and profile
profiles = segmenter.fit_clusters(n_clusters=optimal["optimal_k"])
print(profiles)

# Visualize (PCA)
viz = segmenter.visualize_clusters()

# Demographic breakdown
demo_profiles = segmenter.profile_clusters_by_demographics(["age_group", "gender"])

# Get labels for original data
labels = segmenter.get_cluster_labels()
```

### 5. Automated Report Generation

```python
from survey_toolkit import ReportGenerator

report = ReportGenerator(clean_df)
report.set_metadata(
    title="Q4 2024 Customer Satisfaction Report",
    author="Research Team",
    description="Analysis of 500 survey respondents.",
)

# Add sections
report.add_summary_statistics(["q1", "q2", "q3", "q4", "q5"])
report.add_stats_result("Reliability", alpha)
report.add_dataframe("Correlations", corr["correlation_matrix"])
report.add_figure("Distribution", "outputs/figures/likert_distribution.png")
report.add_stats_result("Group Comparison", result)
report.add_dataframe("Model Performance", results)

# Generate
report.generate(output_path="outputs/reports/q4_report.html")
report.generate_pdf(output_path="outputs/reports/q4_report.pdf")
```

### 6. Command-Line Interface

```bash
# Generate sample data
survey-analyze --generate-sample --sample-size 500 --output-dir data/

# Run EDA
survey-analyze data/survey.csv --eda --output-dir results/

# Cronbach's alpha
survey-analyze data/survey.csv --alpha q1 q2 q3 q4 q5 --format json

# Group comparison
survey-analyze data/survey.csv --compare q1 --group age_group

# Clustering
survey-analyze data/survey.csv --cluster q1 q2 q3 q4 q5 --format csv

# Classification
survey-analyze data/survey.csv --classify q1 q2 q3 q4 q5 --target satisfaction

# Generate report
survey-report data/survey.csv --columns q1 q2 q3 q4 q5 \
    --title "Q4 Survey" --full-analysis --pdf
```

### 7. Scale Score Computation

```python
from survey_toolkit import compute_scale_scores

# Define constructs from factor analysis
construct_map = {
    "satisfaction": ["q1", "q2", "q3"],
    "usability": ["q4", "q5", "q6"],
    "trust": ["q7", "q8", "q9", "q10"],
}

# Compute composite scores
scores = compute_scale_scores(clean_df, construct_map, method="mean")
print(scores.describe())
```

---

##  Project Structure

```
survey-ml-toolkit/
│
├── survey_toolkit/              # Main package
│   ├── __init__.py              # Lazy imports & public API
│   ├── loader.py                # Multi-format data loading
│   ├── cleaner.py               # Survey-specific data cleaning
│   ├── eda.py                   # Exploratory data analysis
│   ├── stats.py                 # Statistical tests & factor analysis
│   ├── ml_models.py             # Classification & clustering
│   ├── reporting.py             # HTML/PDF report generation
│   ├── cli.py                   # Command-line interface
│   └── utils.py                 # Helpers, sample data, validation
│
├── tests/                       # Test suite (~164 tests)
│   ├── conftest.py              # Shared fixtures
│   ├── test_loader.py
│   ├── test_cleaner.py
│   ├── test_eda.py
│   ├── test_stats.py
│   ├── test_ml_models.py
│   ├── test_reporting.py
│   ├── test_cli.py
│   ├── test_utils.py
│   └── test_integration.py
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_statistical_tests.ipynb
│   ├── 03_ml_models.ipynb
│   └── 04_reporting.ipynb
│
├── data/                        # Data directory
│   ├── raw/
│   ├── processed/
│   └── sample_survey.csv
│
├── outputs/                     # Analysis outputs
│   ├── figures/
│   └── reports/
│
├── .github/                     # CI/CD
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── docs.yml
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── dependabot.yml
│   └── CODEOWNERS
│
├── setup.py                     # Package configuration
├── pyproject.toml               # Build config & tool settings
├── requirements.txt             # Pinned dependencies
├── Makefile                     # Developer commands
├── LICENSE                      # MIT License
├── CHANGELOG.md                 # Version history
├── MANIFEST.in                  # Distribution manifest
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

##  API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `SurveyLoader` | `loader.py` | Load survey data from CSV, Excel, SPSS, Stata, JSON |
| `SurveyCleaner` | `cleaner.py` | Chain-able survey data cleaning |
| `SurveyEDA` | `eda.py` | Automated exploratory data analysis |
| `SurveyStats` | `stats.py` | Statistical tests for survey data |
| `SurveyClassifier` | `ml_models.py` | ML classification pipeline |
| `SurveySegmentation` | `ml_models.py` | K-Means respondent clustering |
| `ReportGenerator` | `reporting.py` | HTML/PDF report builder |

### Utility Functions

| Function | Description |
|----------|-------------|
| `generate_sample_survey()` | Generate realistic sample survey data |
| `detect_column_types()` | Auto-detect Likert, categorical, ID, text columns |
| `validate_survey_data()` | Check data quality and return diagnostics |
| `compute_scale_scores()` | Compute composite scores from item groupings |
| `export_results()` | Export results to CSV, JSON, or Excel |

### Statistical Tests Available

| Test | Use Case | Auto-Selected When |
|------|----------|--------------------|
| Independent t-test | 2-group mean comparison | 2 groups, normal data |
| Welch's t-test | 2-group comparison, unequal variance | 2 groups, normal, unequal variance |
| Mann-Whitney U | 2-group comparison, non-parametric | 2 groups, non-normal data |
| One-way ANOVA | 3+ group comparison | 3+ groups, normal data |
| Kruskal-Wallis H | 3+ group comparison, non-parametric | 3+ groups, non-normal data |
| Tukey HSD | Post-hoc pairwise comparisons | Significant ANOVA result |
| Chi-square | Categorical independence | Two categorical variables |
| Pearson/Spearman/Kendall | Correlation | Numeric variables |
| Cronbach's alpha | Scale reliability | Likert-scale items |
| Exploratory Factor Analysis | Latent construct identification | Multiple Likert items |
| Proportion z-test | Compare proportions | Categorical outcomes |

---

##  Testing

```bash
# Run all tests
make test

# Run fast tests only (skip ML training)
make test-fast

# Run with coverage
make coverage

# Run specific test file
pytest tests/test_stats.py -v

# Run specific test class
pytest tests/test_stats.py::TestCronbachsAlpha -v

# Run specific test
pytest tests/test_stats.py::TestCronbachsAlpha::test_basic_alpha -v
```

### Test Coverage

| Module | Coverage |
|--------|----------|
| `loader.py` | ~95% |
| `cleaner.py` | ~95% |
| `eda.py` | ~90% |
| `stats.py` | ~90% |
| `ml_models.py` | ~85% |
| `reporting.py` | ~85% |
| `utils.py` | ~90% |
| `cli.py` | ~80% |
| **Overall** | **~88%** |

---

##  Development

### Setup

```bash
git clone https://github.com/TravisBeckwith/surveykit-ml.git
cd survey-ml-toolkit
make dev  # Installs all deps + pre-commit hooks
```

### Common Commands

```bash
make help          # Show all available commands
make test-fast     # Quick test run
make lint          # Check code quality
make format        # Auto-format code
make coverage      # Run tests with coverage
make build         # Build package
make sample        # Generate sample data
make clean         # Clean build artifacts
```

### Code Quality

This project uses:
- **Black** for code formatting (line length: 88)
- **Ruff** for linting (pyflakes, pycodestyle, isort, and more)
- **mypy** for type checking
- **pre-commit** hooks for automated quality checks
- **pytest** with coverage reporting

### Branch Strategy

```
main        ← stable releases
develop     ← integration branch
feature/*   ← new features
bugfix/*    ← bug fixes
release/*   ← release preparation
```

---

##  Roadmap

### v0.2.0 (Planned)
- [ ] NLP module for open-ended response analysis
  - Sentiment analysis
  - Topic modeling (LDA)
  - Word clouds
  - Text classification
- [ ] Weighted analysis for stratified sampling
- [ ] Streamlit dashboard for interactive exploration

### v0.3.0 (Planned)
- [ ] Structural Equation Modeling (SEM) integration
- [ ] Conjoint analysis module
- [ ] MaxDiff analysis module
- [ ] Multi-level modeling for nested survey designs

### v0.4.0 (Planned)
- [ ] PowerPoint export for reports
- [ ] Automated insight generation using LLMs
- [ ] Survey design validation tools
- [ ] Real-time dashboard with Plotly Dash

### Future Ideas
- [ ] SPSS syntax export
- [ ] R integration (rpy2)
- [ ] Bayesian analysis module
- [ ] Panel data / longitudinal survey support
- [ ] Multi-language survey support
- [ ] API endpoint for survey analysis as a service

---

##  Contributing

Contributions are welcome! Here's how to get started:

### Quick Contribution Guide

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/survey-ml-toolkit.git
   ```
3. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install dev dependencies:
   ```bash
   make dev
   ```
5. Make changes and add tests
6. Run quality checks:
   ```bash
   make quality
   make test
   ```
7. Commit (using conventional commits):
   ```bash
   git commit -m "feat: add new analysis method"
   ```
8. Push and create a Pull Request

### Commit Convention

| Prefix | Purpose |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `test:` | Adding/updating tests |
| `refactor:` | Code refactoring |
| `ci:` | CI/CD changes |
| `deps:` | Dependency updates |
| `release:` | Version release |

### Areas Where Help Is Needed
-  **More test cases** — edge cases, large datasets
-  **Documentation** — docstrings, examples, tutorials
-  **Internationalization** — multi-language support
-  **New statistical methods** — SEM, Bayesian, multilevel
-  **Visualization themes** — APA style, custom palettes
-  **Bug reports** — especially on Windows/macOS

---

##  Documentation

| Resource | Link |
|----------|------|
| API Reference | [Wiki](https://github.com/TravisBeckwith/surveykit-ml/wiki) |
| Tutorials | [Notebooks](notebooks/) |
| Examples | [Examples](examples/) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Issue Tracker | [Issues](https://github.com/TravisBeckwith/surveykit-ml/issues) |
| Discussions | [GitHub Discussions](https://github.com/TravisBeckwith/surveykit-ml/discussions) |

---

##  Dependencies

### Core

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scipy` | Statistical tests |
| `scikit-learn` | ML models, clustering, preprocessing |
| `statsmodels` | Advanced statistics, post-hoc tests |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `factor-analyzer` | Factor analysis |

### Optional

| Package | Extra | Purpose |
|---------|-------|---------|
| `xgboost` | `ml` | Gradient boosting |
| `shap` | `ml` | Feature importance |
| `imbalanced-learn` | `ml` | Handling class imbalance |
| `pyreadstat` | `io` | SPSS/Stata file reading |
| `weasyprint` | `reporting` | PDF generation |
| `jinja2` | `reporting` | Report templating |
| `streamlit` | `dashboard` | Interactive dashboards |
| `plotly` | `dashboard` | Interactive charts |

---

##  Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{surveykit_ml,
  title = {Survey ML Toolkit: Statistical Analysis and Machine Learning for Survey Data},
  author = Travis Beckwith
  year = {2026},
  url = {https://github.com/TravisBeckwith/surveykit-ml},
  version = {0.1.0},
  license = {MIT}
}
```

---

##  Acknowledgments

This project builds on the excellent work of:

- [scikit-learn](https://scikit-learn.org/) — ML framework
- [scipy](https://scipy.org/) & [statsmodels](https://www.statsmodels.org/) — statistical methods
- [SHAP](https://shap.readthedocs.io/) — explainable AI
- [factor-analyzer](https://factor-analyzer.readthedocs.io/) — EFA implementation
- [pandas](https://pandas.pydata.org/) — data wrangling
- [seaborn](https://seaborn.pydata.org/) & [matplotlib](https://matplotlib.org/) — visualization

Inspired by survey analysis tools in R (`psych`, `lavaan`, `likert`) and the need for a comprehensive Python equivalent.

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">
  Built for the research community<br><br>
   <a href="https://github.com/TravisBeckwith/surveykit-ml">Star this repo</a> •
   <a href="https://github.com/TravisBeckwith/surveykit-ml/issues">Report bug</a> •
   <a href="https://github.com/TravisBeckwith/surveykit-ml/discussions">Discuss</a>
</div>
