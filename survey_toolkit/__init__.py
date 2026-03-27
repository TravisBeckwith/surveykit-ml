"""
Survey ML Toolkit
~~~~~~~~~~~~~~~~~

A Python toolkit for statistical analysis and machine learning
on survey data. Built for research analysts.
"""

__title__ = "survey-ml-toolkit"
__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Your Name"

# Mapping of public names to their source modules
_LAZY_IMPORTS = {
    # Core classes
    "SurveyLoader": "survey_toolkit.loader",
    "SurveyCleaner": "survey_toolkit.cleaner",
    "SurveyEDA": "survey_toolkit.eda",
    "SurveyStats": "survey_toolkit.stats",
    "SurveyClassifier": "survey_toolkit.ml_models",
    "SurveySegmentation": "survey_toolkit.ml_models",
    "ReportGenerator": "survey_toolkit.reporting",
    # Utility functions
    "generate_sample_survey": "survey_toolkit.utils",
    "detect_column_types": "survey_toolkit.utils",
    "validate_survey_data": "survey_toolkit.utils",
    "compute_scale_scores": "survey_toolkit.utils",
    "export_results": "survey_toolkit.utils",
    "timer": "survey_toolkit.utils",
    "logger": "survey_toolkit.utils",
    # Configuration
    "load_config": "survey_toolkit.config",
    "get_config": "survey_toolkit.config",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    """Lazy-load public API objects on first access."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr

    raise AttributeError(
        f"module 'survey_toolkit' has no attribute '{name}'"
    )


def __dir__():
    """Show available public API in autocomplete."""
    public = list(_LAZY_IMPORTS.keys())
    public += [
        "__title__",
        "__version__",
        "__author__",
        "__license__",
        "__copyright__",
    ]
    return public