"""
Setup configuration for the Survey ML Toolkit package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


# Core dependencies
INSTALL_REQUIRES = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "statsmodels>=0.14",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "factor-analyzer>=0.5",
]

# Optional / extra dependency groups
EXTRAS_REQUIRE = {
    # ML-specific extras
    "ml": [
        "xgboost>=1.7",
        "imbalanced-learn>=0.11",
        "shap>=0.42",
    ],
    # NLP for open-ended survey responses
    "nlp": [
        "nltk>=3.8",
        "spacy>=3.5",
        "transformers>=4.30",
        "wordcloud>=1.9",
    ],
    # Reporting & export
    "reporting": [
        "jinja2>=3.1",
        "weasyprint>=59.0",
        "openpyxl>=3.1",
        "python-pptx>=0.6",
    ],
    # SPSS / Stata file support
    "io": [
        "pyreadstat>=1.2",
    ],
    # Interactive dashboards
    "dashboard": [
        "streamlit>=1.24",
        "plotly>=5.15",
    ],
    # Development & testing
    "dev": [
        "pytest>=7.4",
        "pytest-cov>=4.1",
        "pytest-mock>=3.11",
        "black>=23.7",
        "ruff>=0.0.280",
        "mypy>=1.4",
        "pre-commit>=3.3",
        "ipykernel>=6.25",
        "nbstripout>=0.6",
    ],
    # Documentation
    "docs": [
        "sphinx>=7.0",
        "sphinx-rtd-theme>=1.2",
        "myst-parser>=2.0",
        "nbsphinx>=0.9",
    ],
}

# 'all' installs everything except dev and docs
EXTRAS_REQUIRE["all"] = list(set(
    dep
    for key, deps in EXTRAS_REQUIRE.items()
    if key not in ("dev", "docs")
    for dep in deps
))

# 'full' installs absolutely everything
EXTRAS_REQUIRE["full"] = list(set(
    dep
    for deps in EXTRAS_REQUIRE.values()
    for dep in deps
))


setup(
    # ---- Package Metadata ----
    name="survey-ml-toolkit",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description=(
        "A Python toolkit for statistical analysis and machine learning "
        "on survey data. Built for research analysts."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/survey-ml-toolkit",
    project_urls={
        "Bug Tracker": (
            "https://github.com/yourusername/survey-ml-toolkit/issues"
        ),
        "Documentation": (
            "https://github.com/yourusername/survey-ml-toolkit/wiki"
        ),
        "Source Code": (
            "https://github.com/yourusername/survey-ml-toolkit"
        ),
        "Changelog": (
            "https://github.com/yourusername/"
            "survey-ml-toolkit/blob/main/CHANGELOG.md"
        ),
    },
    license="MIT",
    keywords=[
        "survey",
        "research",
        "statistics",
        "machine-learning",
        "factor-analysis",
        "data-analysis",
        "likert",
        "questionnaire",
        "social-science",
        "market-research",
    ],

    # ---- Classifiers (PyPI metadata) ----
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Typing :: Typed",
    ],

    # ---- Package Discovery ----
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "notebooks",
            "notebooks.*",
            "data",
            "data.*",
            "outputs",
            "outputs.*",
        ]
    ),
    include_package_data=True,
    package_data={
        "survey_toolkit": [
            "config.yml",
            "templates/*.html",
            "templates/*.jinja2",
            "data/sample_*.csv",
        ],
    },

    # ---- Dependencies ----
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # ---- Entry Points (CLI commands) ----
    entry_points={
        "console_scripts": [
            "survey-analyze=survey_toolkit.cli:main",
            "survey-report=survey_toolkit.cli:generate_report",
        ],
    },

    # ---- Additional Options ----
    zip_safe=False,
)