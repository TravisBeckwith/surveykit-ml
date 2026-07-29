# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Docs workflow failure: fallback never ran, upload step failed on
  missing `docs/_build/html/`.** `docs.yml`'s "Build Sphinx docs" step
  had `continue-on-error: true`, which means the step is reported as
  successful even when `cd docs` fails (there's no `docs/` directory in
  this repo) — so the old fallback step, gated on `if: failure()`, never
  triggered, and `actions/upload-pages-artifact` then failed outright on
  the missing directory. Changed the fallback to run unconditionally
  (`if: always()`) with its own directory-existence check instead,
  so it fires whether Sphinx fails, `docs/` is missing, or the build
  step is skipped for any other reason. Added `id: build_sphinx` to the
  build step for clearer log attribution. Verified locally: simulated
  the exact failure (`cd docs` erroring in the absence of a `docs/`
  directory) and confirmed the fallback step now produces a valid
  `docs/_build/html/index.html` every time.

- **`pyproject.toml` `full` extras missing `pytest-timeout`**: present in
  `dev` but not in the separately-maintained `full` array, so
  `pip install -e ".[full]"` didn't actually pull in the timeout plugin
  even though `pip install -e ".[dev]"` did. Added it to `full` too.
- **`setup.py` dependency floors drifted out of sync with
  `pyproject.toml`** after several Dependabot version-bump PRs merged.
  Dependabot only patches `pyproject.toml` (the PEP 621 file it scans),
  so `setup.py`'s duplicate `INSTALL_REQUIRES`/`EXTRAS_REQUIRE` lists were
  left pointing at older floors: `seaborn>=0.12` (should be `>=0.13.2`),
  `shap>=0.42` (should be `>=0.49.1`), `pytest>=7.4` (should be
  `>=9.0.3`), `black>=23.7` (should be `>=26.3.1`), `mypy>=1.4` (should be
  `>=1.20.1`), `sphinx>=7.0` (should be `>=8.1.3`), `myst-parser>=2.0`
  (should be `>=4.0.1`). Synced all of them to match. Note: `setup.py`'s
  `all`/`full` extras are computed dynamically from its own dict, so they
  didn't have the same static-duplication bug as `pyproject.toml`'s did.

  Longer-term, maintaining dependency floors in two separate files
  (`pyproject.toml` and `setup.py`) is what let this drift happen
  silently — worth considering trimming `setup.py` down to a minimal
  shim (or removing it) now that `pyproject.toml`'s `[project]` table is
  the complete, working PEP 621 metadata source and `setuptools.build_meta`
  doesn't require a `setup.py` to function.

### Changed
- **Distribution: GitHub-only for now, not PyPI.** Updated `README.md`
  install instructions to use `pip install git+https://...` (with `@tag`
  pinning and `[extra]` support via PEP 508 direct references) instead of
  `pip install survey-ml-toolkit`, since the package isn't published to
  PyPI yet. Replaced the (previously broken/misleading) PyPI version
  badge with a GitHub release badge. Also fixed a pre-existing typo in
  the Development Install instructions (`cd survey-ml-toolkit` →
  `cd surveykit-ml`, matching the actual `git clone` target directory).
- **`release.yml` PyPI publishing disabled.** The workflow previously
  attempted to publish to PyPI via Trusted Publishing on every version
  tag — not appropriate while distribution is GitHub-only, and would
  have failed outright since no Trusted Publisher is configured. The
  `publish`/`test-publish` jobs are preserved as commented-out code with
  setup instructions for when that's ready, rather than deleted.
- **Fixed a separate, pre-existing `release.yml` bug** while restructuring
  it: the old `publish` job never actually uploaded a `dist` artifact,
  but `github-release` tried to download one anyway with
  `continue-on-error: true` — meaning every GitHub Release ever created
  by this workflow would have silently shipped with no wheel/sdist
  attached. Added a dedicated `build` job that builds and uploads the
  `dist` artifact, and pointed `github-release` at it. Also updated the
  auto-generated release notes' install snippet to match the new
  git-based install command.

### Fixed

- **Windows CI failure: `UnicodeDecodeError` in report tests** (`test_generate_html`,
  `test_generate_table_of_contents`, `test_generate_no_toc_few_sections`, and
  the equivalent read in `test_integration.py`). `reporting.py` correctly
  writes generated HTML reports with `encoding="utf-8"` (the templates
  include emoji, e.g. 📊/📋 in headers), but the tests read them back with
  `Path(...).read_text()` and no explicit encoding — which uses the
  platform's default locale encoding. On Linux/macOS runners that
  defaults to UTF-8 (so it never failed there), but on Windows it
  defaults to `cp1252`, which can't decode the multi-byte UTF-8 emoji
  sequences and raises `UnicodeDecodeError`. Added `encoding="utf-8"` to
  all 8 affected `read_text()` calls so the read side always matches the
  write side regardless of platform.

- **Redundant/inconsistent coverage thresholds**: the coverage job enforced
  80% via pytest-cov (`pyproject.toml`'s `[tool.coverage.report] fail_under`)
  and then, in a separate later step, a redundant 70% via a bare
  `coverage report --fail-under=70`. Both currently pass against the
  actual ~92% coverage, but two different numbers enforced by two
  different mechanisms was confusing and not a single source of truth.
  Aligned the second check to 80% to match.

  Note: a report attributing a 64.52%-vs-80%-required coverage failure to
  this job was investigated and doesn't match — running the job's actual
  steps against the current code reproduces 92.10% coverage, passing
  both gates. That number matches the "Run slow tests" step in the
  separate `test` job instead, which was already fixed (see the
  `--no-cov` entry below) by not coverage-gating an intentionally partial
  test subset. Lowering the coverage threshold, as that report
  recommended, would have masked real regressions on the actual
  full-suite check rather than fixing anything.

- **CI test job failure: `unrecognized arguments: --timeout=300`**. The
  workflow passes `--timeout=<n>` to `pytest` in four places (fast tests,
  slow tests, full-suite coverage, integration tests), but the
  `pytest-timeout` plugin that provides that flag was never installed —
  it wasn't in `requirements.txt`, `setup.py`, or the `dev` extra in
  `pyproject.toml`. Added `pytest-timeout>=2.1` to the `dev` extras in
  both `pyproject.toml` and `setup.py`.
- **Coverage gate failing on intentionally-partial test runs**. The
  project sets `[tool.coverage.report] fail_under = 80`, and pytest's
  `addopts` unconditionally enables `--cov=survey_toolkit` for every
  invocation. That means the "Run fast tests" (`-m "not slow"`), "Run
  slow tests" (`-m "slow"`), and "Run integration tests"
  (`tests/test_integration.py` only) CI steps were each independently
  enforcing an 80% coverage threshold against a deliberately small
  subset of the suite — the slow-tests subset only reaches ~65%
  coverage and the integration-only subset ~67%, so those steps would
  fail purely from running fewer tests, regardless of whether the tests
  themselves pass. Added `--no-cov` to those three CI steps; coverage
  enforcement now applies only where it belongs — the dedicated
  `coverage` job, which runs the full suite.

- **CI lint job failures**: `black --check` was failing on all 22 source/test
  files (never actually run against the codebase before). Ran `black` to
  reformat everything; `black --check` now passes.
- **Ruff lint errors** (52 found once `black` was fixed, since CI stops at
  the first failing step in the job): moved the deprecated top-level
  `[tool.ruff]` `select`/`ignore`/`isort` settings to `[tool.ruff.lint]`;
  added `N806` to the ignore list (the codebase's `X`/`y` scikit-learn
  variable naming is intentional, not a naming error); fixed ambiguous
  Unicode characters, a mid-file import, a mutable class-attribute default,
  an implicit-`Optional` hint, a list-concatenation, unused
  unpacked/loop variables, missing `raise ... from err` exception chaining,
  missing `zip(..., strict=True)`, and `try/except/pass` blocks replaced
  with `contextlib.suppress`. `ruff check` now passes with zero errors.
- **Runtime `TypeError` on import** (introduced and caught during the ruff
  fix above, verified via a full test-suite run before considering it
  done): `cleaner.py` and `stats.py` used the builtin `any`/`callable` as
  type hints instead of `typing.Any`/`typing.Callable`. This was
  harmless as `Optional[any]`, but ruff's `--fix` auto-modernized it to
  `any | None`, which raises `TypeError` at class-definition time (i.e.
  on `import survey_toolkit`) since the builtin `any` function doesn't
  support the `|` operator. Fixed by importing and using the correct
  `Any`/`Callable` types from `typing`/`collections.abc`.
- `mypy` (informational only — already `continue-on-error: true` in CI,
  so non-blocking) still reports 29 pre-existing type-annotation gaps
  (mostly untyped empty containers and `self.x = None` attributes later
  reassigned to a real type) across `cleaner.py`, `eda.py`, `loader.py`,
  `ml_models.py`, `reporting.py`, `stats.py`, and `utils.py`. Left as-is
  since fixing them is a larger, separate effort outside the scope of the
  failing lint job.

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