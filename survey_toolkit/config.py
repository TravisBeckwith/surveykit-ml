"""
Configuration management for the Survey ML Toolkit.

Loads default settings from config.yml and allows user overrides.
"""

import yaml
from pathlib import Path
from typing import Any, Optional
from copy import deepcopy


# Path to default config
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yml"

# Global config cache
_config = None


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Override values take precedence.

    Parameters
    ----------
    base : dict
        Base configuration.
    override : dict
        Override values.

    Returns
    -------
    dict
        Merged configuration.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """
    Load toolkit configuration.

    Priority (highest to lowest):
    1. Explicit overrides dict
    2. User config file
    3. Default config.yml

    Parameters
    ----------
    config_path : str, optional
        Path to a custom YAML config file.
    overrides : dict, optional
        Dictionary of override values.

    Returns
    -------
    dict
        Complete configuration dictionary.
    """
    global _config

    # Load defaults
    if _DEFAULT_CONFIG_PATH.exists():
        with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = _get_hardcoded_defaults()

    # Merge user config file
    if config_path:
        user_path = Path(config_path)
        if user_path.exists():
            with open(user_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_config)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    # Merge explicit overrides
    if overrides:
        config = _deep_merge(config, overrides)

    _config = config
    return config


def get_config(section: Optional[str] = None) -> dict:
    """
    Get the current configuration (or a section of it).

    Parameters
    ----------
    section : str, optional
        Config section name (e.g., 'stats', 'ml', 'eda').
        Returns full config if None.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    global _config

    if _config is None:
        _config = load_config()

    if section:
        if section not in _config:
            raise KeyError(
                f"Unknown config section: '{section}'. "
                f"Available: {list(_config.keys())}"
            )
        return deepcopy(_config[section])

    return deepcopy(_config)


def get(key: str, default: Any = None) -> Any:
    """
    Get a specific config value using dot notation.

    Parameters
    ----------
    key : str
        Dot-separated key path (e.g., 'stats.significance_level').
    default : Any
        Default value if key not found.

    Returns
    -------
    Any
        Configuration value.

    Examples
    --------
    >>> get('stats.significance_level')
    0.05
    >>> get('ml.cv_folds')
    5
    >>> get('nonexistent.key', 'fallback')
    'fallback'
    """
    config = get_config()
    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def reset_config():
    """Reset configuration to defaults."""
    global _config
    _config = None


def update_config(overrides: dict):
    """
    Update the current configuration with new values.

    Parameters
    ----------
    overrides : dict
        Override values to merge into current config.
    """
    global _config

    if _config is None:
        _config = load_config()

    _config = _deep_merge(_config, overrides)


def _get_hardcoded_defaults() -> dict:
    """
    Hardcoded fallback defaults if config.yml is missing.

    Returns
    -------
    dict
        Minimal default configuration.
    """
    return {
        "general": {
            "random_state": 42,
            "verbose": True,
            "log_level": "INFO",
            "output_dir": "outputs",
            "figure_format": "png",
            "figure_dpi": 150,
            "float_precision": 4,
        },
        "cleaner": {
            "speeder_threshold_seconds": 60,
            "straightliner_threshold": 0.95,
            "missing_strategy": "median",
            "missing_drop_threshold": 0.5,
        },
        "likert": {
            "default_range": [1, 5],
            "text_to_numeric": {
                "Strongly Disagree": 1,
                "Disagree": 2,
                "Neutral": 3,
                "Agree": 4,
                "Strongly Agree": 5,
            },
        },
        "detection": {
            "categorical_threshold": 15,
            "likert_range": [1, 5],
            "text_min_avg_length": 50,
        },
        "eda": {
            "figure_size": {"default": [10, 6]},
            "seaborn_style": "whitegrid",
            "save_figures": True,
            "color_palette": {
                "likert_5": [
                    "#d73027",
                    "#fc8d59",
                    "#fee08b",
                    "#91cf60",
                    "#1a9850",
                ],
            },
        },
        "stats": {
            "significance_level": 0.05,
            "normality_test": "shapiro",
            "normality_min_n": 8,
            "posthoc_method": "tukey",
            "alpha_thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7,
                "questionable": 0.6,
                "poor": 0.5,
            },
            "factor_analysis": {
                "default_rotation": "varimax",
                "default_method": "ml",
                "kmo_threshold": 0.6,
                "loading_threshold": 0.4,
            },
        },
        "ml": {
            "cv_folds": 5,
            "scoring": "accuracy",
            "scale_features": True,
        },
        "clustering": {
            "default_method": "kmeans",
            "k_range": [2, 10],
            "n_init": 10,
        },
        "reporting": {
            "default_format": "html",
            "default_title": "Survey Analysis Report",
            "include_toc": True,
            "toc_min_sections": 4,
            "max_table_rows": 50,
            "embed_figures": True,
        },
        "cli": {
            "default_output_dir": "outputs",
            "default_format": "print",
            "default_missing_strategy": "median",
        },
        "sample_data": {
            "default_respondents": 500,
            "default_likert_items": 10,
        },
    }