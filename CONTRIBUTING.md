# Contributing to Survey ML Toolkit

First off, **thank you** for considering contributing to Survey ML Toolkit! 🎉

Every contribution matters — whether it's fixing a typo, reporting a bug,
suggesting a feature, or writing code. This guide will help you get started.

---

## 📑 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Coding Standards](#-coding-standards)
- [Testing Guidelines](#-testing-guidelines)
- [Documentation Guidelines](#-documentation-guidelines)
- [Commit Convention](#-commit-convention)
- [Pull Request Process](#-pull-request-process)
- [Release Process](#-release-process)
- [Getting Help](#-getting-help)

---

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold a welcoming, inclusive, and harassment-free
environment for everyone.

**In short:**
- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what's best for the community
- Accept constructive criticism gracefully

---

## 🤝 How Can I Contribute?

### 🐛 Report Bugs

Found a bug? Please [open an issue](https://github.com/yourusername/survey-ml-toolkit/issues/new?template=bug_report.md) with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, package version)
- Error messages or tracebacks
- A minimal code example if possible

```python
# Example bug report code
from survey_toolkit import SurveyStats
import pandas as pd

df = pd.DataFrame({"q1": [1, 2, 3], "q2": [4, 5, 6]})
stats = SurveyStats(df)
# This raises an unexpected error:
stats.cronbachs_alpha(["q1", "q2"])
```

### ✨ Suggest Features

Have an idea? Open a feature request with:

- The problem or use case it solves
- Your proposed solution
- Example of how the API might look
- Any relevant academic references or tools

### 📝 Improve Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples to docstrings
- Write tutorials or how-to guides
- Improve README sections
- Add notebook examples

### 🧪 Write Tests

Help us improve coverage:

- Add tests for edge cases
- Test with different data types and sizes
- Add integration tests for common workflows
- Test error handling and validation

### 🔧 Submit Code Changes

Ready to code? See the sections below for our development workflow and coding standards.

### 🌍 Translations & Accessibility

- Help make the toolkit accessible to non-English speakers
- Improve error messages and documentation clarity
- Add support for international survey formats

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git for version control
- A GitHub account

### Fork & Clone

```bash
# 1. Fork the repository on GitHub (click the "Fork" button)

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/survey-ml-toolkit.git
cd survey-ml-toolkit

# 3. Add the upstream remote
git remote add upstream https://github.com/yourusername/survey-ml-toolkit.git

# 4. Verify remotes
git remote -v
# origin    https://github.com/YOUR_USERNAME/survey-ml-toolkit.git (fetch)
# origin    https://github.com/YOUR_USERNAME/survey-ml-toolkit.git (push)
# upstream  https://github.com/yourusername/survey-ml-toolkit.git (fetch)
# upstream  https://github.com/yourusername/survey-ml-toolkit.git (push)
```

### Install Development Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install with all development dependencies
pip install -e ".[full]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import survey_toolkit; print(f'v{survey_toolkit.__version__} ready!')"
pytest tests/ -v -m "not slow" --timeout=60
```

### Using Make (Optional)

If you have `make` installed:

```bash
make dev         # Install everything + pre-commit hooks
make test-fast   # Quick test run
make lint        # Check code quality
make format      # Auto-format code
make help        # See all available commands
```

---

## 🔄 Development Workflow

### 1. Sync Your Fork

Always start from the latest upstream code:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

### 2. Create a Feature Branch

```bash
# Branch naming convention:
# feature/description   — new features
# bugfix/description    — bug fixes
# docs/description      — documentation
# test/description      — test improvements
# refactor/description  — code refactoring

git checkout -b feature/add-weighted-analysis
```

### 3. Make Your Changes

- Write code following our coding standards
- Add or update tests for any new functionality
- Update documentation and docstrings
- Keep changes focused — one feature/fix per PR

### 4. Test Your Changes

```bash
# Run the fast test suite
make test-fast
# or
pytest tests/ -v -m "not slow" --timeout=60

# Run full test suite (includes ML model training)
make test
# or
pytest tests/ -v --timeout=300

# Check test coverage
make coverage
# or
pytest tests/ --cov=survey_toolkit --cov-report=term-missing

# Run only your new tests
pytest tests/test_your_module.py -v
```

### 5. Check Code Quality

```bash
# Auto-format
make format
# or
black survey_toolkit/ tests/
ruff check --fix survey_toolkit/ tests/

# Verify formatting
make lint
# or
black --check survey_toolkit/ tests/
ruff check survey_toolkit/ tests/

# Type checking (non-blocking)
mypy survey_toolkit/ --ignore-missing-imports
```

### 6. Commit Your Changes

```bash
# Stage changes
git add -A

# Commit with conventional commit message
git commit -m "feat: add weighted survey analysis support"

# Push to your fork
git push origin feature/add-weighted-analysis
```

### 7. Open a Pull Request

1. Go to [Pull Requests](https://github.com/yourusername/survey-ml-toolkit/pulls)
2. Click "New Pull Request"
3. Select your branch → `main`
4. Fill in the PR template
5. Request review

---

## 📐 Coding Standards

### Python Style

We follow PEP 8 with these specifics:

| Tool | Setting |
|------|---------|
| Formatter | Black (line length: 88) |
| Linter | Ruff |
| Type Checker | mypy (non-blocking) |
| Python | 3.10+ (use modern syntax) |

### Naming Conventions

```python
# Classes: PascalCase
class SurveyAnalyzer:
    pass

# Functions/methods: snake_case
def compute_alpha(items):
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_SIGNIFICANCE = 0.05

# Private methods: leading underscore
def _validate_input(self, data):
    pass

# Module-level variables: snake_case
logger = get_logger()
```

### Docstring Format

We use NumPy-style docstrings:

```python
def compare_groups(
    self,
    variable: str,
    group_col: str,
    test: str = "auto",
) -> dict:
    """
    Compare a numeric variable across categorical groups.

    Automatically selects the appropriate statistical test based on
    the number of groups and data distribution (normality).

    Parameters
    ----------
    variable : str
        Name of the numeric variable to compare.
    group_col : str
        Name of the categorical grouping variable.
    test : str, optional
        Test to use. Options: 'auto', 't-test', 'mann-whitney',
        'anova', 'kruskal'. Default is 'auto'.

    Returns
    -------
    dict
        Dictionary containing:
        - 'test' (str): Name of the test used
        - 'statistic' (float): Test statistic
        - 'p_value' (float): p-value
        - 'effect_size' (float): Effect size measure
        - 'significant' (bool): Whether p < 0.05

    Raises
    ------
    ValueError
        If fewer than 2 groups are found.

    Examples
    --------
    >>> stats = SurveyStats(df)
    >>> result = stats.compare_groups("satisfaction", "age_group")
    >>> print(f"p = {result['p_value']:.4f}")
    p = 0.0312

    Notes
    -----
    For 2 groups, the method chooses between:
    - Independent t-test (normal data)
    - Mann-Whitney U (non-normal data)

    For 3+ groups:
    - One-way ANOVA (normal data) with Tukey HSD post-hoc
    - Kruskal-Wallis H (non-normal data)

    Normality is assessed using the Shapiro-Wilk test on each group.

    See Also
    --------
    chi_square_test : For categorical × categorical comparisons.
    correlation_matrix : For numeric × numeric associations.
    """
    # Implementation...
```

### Type Hints

Use type hints for all public functions and methods:

```python
from typing import Optional
import pandas as pd
import numpy as np


def prepare_data(
    self,
    feature_cols: list[str],
    target_col: str,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target."""
    ...


def find_optimal_k(
    self,
    k_range: range = range(2, 11),
) -> dict[str, any]:
    """Find optimal number of clusters."""
    ...
```

### Import Order

Imports are sorted by isort (via Ruff):

```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Optional

# 2. Third-party
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score

# 3. Local
from survey_toolkit.utils import logger, timer
from survey_toolkit.config import get
```

### Code Organization

```python
class SurveyStats:
    """Statistical analysis methods for survey data."""

    # 1. Constructor
    def __init__(self, data: pd.DataFrame):
        ...

    # 2. Public methods (grouped by functionality)
    def cronbachs_alpha(self, columns: list[str]) -> dict:
        ...

    def compare_groups(self, variable: str, group_col: str) -> dict:
        ...

    # 3. Private/helper methods
    def _check_normality(self, data: np.ndarray) -> bool:
        ...

    def _calculate_effect_size(self, groups: list) -> float:
        ...

    # 4. Dunder methods
    def __repr__(self) -> str:
        ...
```

### Error Handling

```python
# Use specific exceptions with helpful messages
def cronbachs_alpha(self, columns: list[str]) -> dict:
    if len(columns) < 2:
        raise ValueError(
            f"Need at least 2 items for Cronbach's alpha, "
            f"got {len(columns)}. Provide more column names."
        )

    missing_cols = set(columns) - set(self.data.columns)
    if missing_cols:
        raise KeyError(
            f"Columns not found in data: {missing_cols}. "
            f"Available: {list(self.data.columns)}"
        )
```

### Logging

```python
from survey_toolkit.utils import logger

# Use appropriate log levels
logger.debug("Detailed debugging info")
logger.info("General information (default)")
logger.warning("Something unexpected but non-fatal")
logger.error("Something failed")

# Include context in log messages
logger.info(
    f"Loaded {len(df)} respondents, {len(df.columns)} columns "
    f"from {filepath.name}"
)
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures (data generators, helpers)
├── test_loader.py        # Tests mirror source module structure
├── test_cleaner.py
├── test_eda.py
├── test_stats.py
├── test_ml_models.py
├── test_reporting.py
├── test_cli.py
├── test_utils.py
├── test_config.py
└── test_integration.py   # End-to-end workflows
```

### Writing Tests

Use pytest with class-based organization:

```python
"""
Tests for survey_toolkit.stats module.
"""

import pytest
import pandas as pd
import numpy as np


class TestCronbachsAlpha:
    """Tests for Cronbach's alpha calculation."""

    def test_basic_alpha(self, clean_survey_df, likert_columns):
        """Test that alpha is computed correctly for correlated items."""
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        result = stats.cronbachs_alpha(likert_columns)

        assert "alpha" in result
        assert 0 <= result["alpha"] <= 1
        assert result["n_items"] == len(likert_columns)

    def test_too_few_items_raises(self, clean_survey_df):
        """Test that single-item scale raises ValueError."""
        from survey_toolkit.stats import SurveyStats

        stats = SurveyStats(clean_survey_df)
        with pytest.raises(ValueError, match="at least 2 items"):
            stats.cronbachs_alpha(["q1"])

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test alpha computation on a large dataset."""
        from survey_toolkit.stats import SurveyStats
        from survey_toolkit.utils import generate_sample_survey

        df = generate_sample_survey(n_respondents=10000, n_likert_items=50)
        stats = SurveyStats(df)
        cols = [f"q{i}" for i in range(1, 51)]
        result = stats.cronbachs_alpha(cols)
        assert result["alpha"] > 0
```

### Test Naming

```python
# Pattern: test_[what]_[condition/scenario]
def test_basic_alpha(self):                    # Happy path
def test_too_few_items_raises(self):           # Error case
def test_high_reliability_items(self):         # Specific scenario
def test_handles_missing_values(self):         # Edge case
def test_alpha_with_reverse_coded_items(self): # Feature-specific
```

### Using Fixtures

Use fixtures from `conftest.py` for shared test data:

```python
# Available fixtures (see tests/conftest.py):
# - sample_survey_df      → Raw survey data with quality issues
# - clean_survey_df       → Pre-cleaned data
# - likert_columns        → ["q1", "q2", ..., "q10"]
# - demographic_columns   → ["age_group", "gender", "education"]
# - small_likert_df       → Small dataset for quick tests
# - text_likert_df        → Text-based Likert responses
# - sample_csv_file       → Path to saved CSV
# - tmp_dir               → Temporary directory for outputs

def test_load_csv(self, sample_csv_file):
    loader = SurveyLoader(sample_csv_file)
    df = loader.load()
    assert len(df) > 0
```

### Test Markers

```python
# Mark slow tests (ML training, large datasets)
@pytest.mark.slow
def test_model_comparison(self):
    ...

# Mark integration tests
@pytest.mark.integration
def test_full_pipeline(self):
    ...
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only
pytest tests/ -v -m "not slow"

# Specific file
pytest tests/test_stats.py -v

# Specific test
pytest tests/test_stats.py::TestCronbachsAlpha::test_basic_alpha -v

# With coverage
pytest tests/ --cov=survey_toolkit --cov-report=term-missing

# Stop on first failure
pytest tests/ -x

# Show print output
pytest tests/ -s
```

### What to Test

| Category | Examples |
|----------|----------|
| Happy path | Normal inputs produce expected outputs |
| Edge cases | Empty data, single row, all missing values |
| Error handling | Invalid inputs raise appropriate exceptions |
| Types | Return types match expectations |
| Ranges | Values within expected bounds (0 ≤ alpha ≤ 1) |
| Side effects | Files created, data not mutated |
| Integration | Multiple modules work together |

### Coverage Goals

| Module | Minimum | Target |
|--------|---------|--------|
| `loader.py` | 85% | 95% |
| `cleaner.py` | 85% | 95% |
| `stats.py` | 80% | 90% |
| `ml_models.py` | 75% | 85% |
| `eda.py` | 80% | 90% |
| `reporting.py` | 75% | 85% |
| `utils.py` | 85% | 90% |
| `config.py` | 90% | 95% |
| **Overall** | **80%** | **88%** |

---

## 📝 Documentation Guidelines

### Docstrings

Every public class, method, and function must have a docstring:

```python
def my_function(param1: str, param2: int = 5) -> dict:
    """
    Brief one-line summary.

    Longer description if needed, explaining the purpose,
    methodology, or important details.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, optional
        Description of param2. Default is 5.

    Returns
    -------
    dict
        Description of what's returned.

    Raises
    ------
    ValueError
        When param2 is negative.

    Examples
    --------
    >>> result = my_function("hello", param2=10)
    >>> print(result)
    {'status': 'ok'}

    Notes
    -----
    Any additional notes about methodology, references, etc.

    See Also
    --------
    related_function : Description of related function.
    """
```

### Inline Comments

```python
# Good: explains WHY
# Use Welch's t-test when variances are unequal (Levene's p < 0.05)
stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)

# Bad: explains WHAT (obvious from code)
# Calculate the mean
mean = np.mean(data)
```

### Notebook Documentation

Each notebook should include:

- Title and objectives at the top
- Markdown cells explaining each step
- Interpretation of results (not just code)
- Summary at the bottom

---

## 💬 Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Purpose | Example |
|------|---------|---------|
| `feat` | New feature | `feat(stats): add Bayesian t-test` |
| `fix` | Bug fix | `fix(cleaner): handle empty dataframe` |
| `docs` | Documentation | `docs: update API reference` |
| `test` | Tests | `test(stats): add edge case for alpha` |
| `refactor` | Code restructuring | `refactor(ml): simplify pipeline` |
| `perf` | Performance | `perf(eda): vectorize likert plot` |
| `style` | Formatting | `style: apply black formatting` |
| `ci` | CI/CD | `ci: add Python 3.12 to matrix` |
| `deps` | Dependencies | `deps: upgrade scikit-learn to 1.4` |
| `chore` | Maintenance | `chore: clean up unused imports` |
| `release` | Version release | `release: v0.2.0` |

### Scopes (Optional)

| Scope | Module |
|-------|--------|
| `loader` | `loader.py` |
| `cleaner` | `cleaner.py` |
| `eda` | `eda.py` |
| `stats` | `stats.py` |
| `ml` | `ml_models.py` |
| `report` | `reporting.py` |
| `cli` | `cli.py` |
| `utils` | `utils.py` |
| `config` | `config.py` |

### Examples

```bash
# Feature
git commit -m "feat(stats): add structural equation modeling support"

# Bug fix
git commit -m "fix(cleaner): prevent NaN in reverse coding when scale_max is None"

# Documentation
git commit -m "docs(stats): add examples to compare_groups docstring"

# Test
git commit -m "test(ml): add edge case for single-class target"

# Multi-line commit
git commit -m "feat(report): add PowerPoint export

- Add python-pptx integration
- Support slide layouts for tables and figures
- Add template customization options

Closes #42"
```

---

## 🔀 Pull Request Process

### Before Submitting

- [ ] Code follows coding standards
- [ ] All tests pass: `make test` or `pytest tests/ -v`
- [ ] Linting passes: `make lint` or `ruff check .`
- [ ] Formatting passes: `make format-check` or `black --check .`
- [ ] New features have tests
- [ ] Docstrings added/updated for public API changes
- [ ] `CHANGELOG.md` updated (for user-facing changes)
- [ ] `README.md` updated (if adding new features/API)
- [ ] No merge conflicts with `main`

### PR Title Format

Follow the same commit convention:

```
feat(stats): add weighted survey analysis
fix(cleaner): handle empty DataFrame in remove_speeders
docs: add SEM tutorial notebook
```

### PR Size Guidelines

| Size | Lines Changed | Review Time |
|------|--------------|-------------|
| 🟢 Small | < 100 | Quick review |
| 🟡 Medium | 100–500 | Standard review |
| 🔴 Large | > 500 | Consider splitting |

Prefer smaller, focused PRs over large ones. If your change is big, consider splitting it into:
1. Refactoring PR (no new features)
2. Feature PR (builds on the refactor)
3. Test PR (additional test coverage)

### Review Process

1. Automated checks run (CI, linting, tests)
2. Maintainer reviews code, tests, and documentation
3. Feedback is provided via inline comments
4. Iterate on feedback
5. Approval and merge

### After Merge

```bash
# Sync your fork
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Clean up feature branch
git branch -d feature/your-feature
git push origin --delete feature/your-feature
```

---

## 🏷️ Release Process

*(For maintainers)*

### Versioning

We follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

0.1.0 → 0.1.1  (patch: bug fix)
0.1.1 → 0.2.0  (minor: new feature, backward compatible)
0.2.0 → 1.0.0  (major: breaking changes)
```

### Pre-release Versions

```
0.2.0-alpha.1   # Early testing
0.2.0-beta.1    # Feature complete, testing
0.2.0-rc.1      # Release candidate
0.2.0           # Stable release
```

### Release Checklist

1. Update version in:
   - `survey_toolkit/__init__.py`
   - `setup.py`
   - `pyproject.toml`

2. Update `CHANGELOG.md` with all changes since last release

3. Run full test suite:
   ```bash
   make test
   make lint
   ```

4. Commit and tag:
   ```bash
   git add -A
   git commit -m "release: v0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```

5. GitHub Actions automatically:
   - Validates version matches tag
   - Runs full test suite
   - Builds package
   - Publishes to PyPI
   - Creates GitHub Release

---

## 🆘 Getting Help

### Resources

| Resource | Link |
|----------|------|
| Documentation | [Wiki](https://github.com/yourusername/survey-ml-toolkit/wiki) |
| Discussions | [GitHub Discussions](https://github.com/yourusername/survey-ml-toolkit/discussions) |
| Issue Tracker | [Issues](https://github.com/yourusername/survey-ml-toolkit/issues) |
| Example Notebooks | [notebooks/](notebooks/) |

### Where to Ask Questions

- **Usage questions** → GitHub Discussions (Q&A category)
- **Bug reports** → Issues
- **Feature ideas** → Issues
- **Development questions** → GitHub Discussions (Development category)

### First-Time Contributors

Look for issues labeled:

- `good first issue` — Simple, well-defined tasks
- `help wanted` — We'd love help with these
- `documentation` — Improve docs and examples

### Suggested First Contributions

| Contribution | Difficulty | Files |
|--------------|------------|-------|
| Fix a typo in docs | 🟢 Easy | `README.md`, docstrings |
| Add a test case | 🟢 Easy | `tests/` |
| Improve an error message | 🟢 Easy | Any module |
| Add a notebook example | 🟡 Medium | `notebooks/` |
| Add a new plot type to EDA | 🟡 Medium | `eda.py`, `test_eda.py` |
| Add a new statistical test | 🟡 Medium | `stats.py`, `test_stats.py` |
| Add NLP module for open-ended responses | 🔴 Hard | New module |
| Add weighted analysis support | 🔴 Hard | Multiple modules |

---

## 🏆 Recognition

All contributors are recognized in:

- **CONTRIBUTORS.md** — Listed by name
- **GitHub Release notes** — Mentioned in changelogs
- **README.md** — Acknowledged in the project

### Adding Yourself

After your first PR is merged, add yourself to `CONTRIBUTORS.md`:

```markdown
## Contributors

| Name | GitHub | Contributions |
|------|--------|--------------|
| Your Name | [@yourusername](https://github.com/yourusername) | Creator & maintainer |
| New Contributor | [@contributor](https://github.com/contributor) | Added weighted analysis |
```

---

## ❓ FAQ

**Q: Do I need to sign a CLA?**
A: No, we don't require a Contributor License Agreement. By submitting a PR, you agree that your contribution is licensed under the project's MIT License.

**Q: Can I work on multiple issues at once?**
A: Yes! Just create separate branches for each issue to keep changes isolated.

**Q: How long does a PR review take?**
A: We aim to provide initial feedback within 3–5 business days. Complex changes may take longer.

**Q: What if my PR conflicts with another?**
A: Rebase your branch on the latest `main`:

```bash
git fetch upstream
git rebase upstream/main
# Resolve conflicts if any
git push --force-with-lease origin your-branch
```

**Q: Can I add a new dependency?**
A: Maybe — please discuss in your PR. We try to minimize dependencies. If the dependency is large (e.g., TensorFlow), consider making it optional (an "extra" in `setup.py`).

**Q: My tests pass locally but fail in CI?**
A: Common causes:
- Missing `MPLBACKEND=Agg` for matplotlib tests
- OS-specific path separators
- Floating-point precision differences
- Missing optional dependencies

---

Thank you for helping make Survey ML Toolkit better! 🎉

*Every contribution, no matter how small, makes a difference.*
