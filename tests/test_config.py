"""
Tests for survey_toolkit.config module.
"""

import pytest
from pathlib import Path


class TestConfig:
    """Tests for configuration management."""

    def setup_method(self):
        """Reset config before each test."""
        from survey_toolkit.config import reset_config
        reset_config()

    def test_load_default_config(self):
        from survey_toolkit.config import load_config

        config = load_config()
        assert isinstance(config, dict)
        assert "general" in config
        assert "stats" in config
        assert "ml" in config
        assert "eda" in config

    def test_get_config_full(self):
        from survey_toolkit.config import get_config

        config = get_config()
        assert "general" in config
        assert "stats" in config

    def test_get_config_section(self):
        from survey_toolkit.config import get_config

        stats = get_config("stats")
        assert isinstance(stats, dict)
        assert "significance_level" in stats

    def test_get_config_invalid_section(self):
        from survey_toolkit.config import get_config

        with pytest.raises(KeyError, match="Unknown config section"):
            get_config("nonexistent_section")

    def test_get_dot_notation(self):
        from survey_toolkit.config import get

        alpha = get("stats.significance_level")
        assert alpha == 0.05

    def test_get_nested_dot_notation(self):
        from survey_toolkit.config import get

        rotation = get("stats.factor_analysis.default_rotation")
        assert rotation == "varimax"

    def test_get_with_default(self):
        from survey_toolkit.config import get

        value = get("nonexistent.key", "fallback")
        assert value == "fallback"

    def test_get_missing_key_returns_default(self):
        from survey_toolkit.config import get

        value = get("stats.nonexistent", 999)
        assert value == 999

    def test_default_values(self):
        from survey_toolkit.config import get

        assert get("general.random_state") == 42
        assert get("ml.cv_folds") == 5
        assert get("stats.significance_level") == 0.05
        assert get("cleaner.speeder_threshold_seconds") == 60
        assert get("cleaner.straightliner_threshold") == 0.95

    def test_load_with_overrides(self):
        from survey_toolkit.config import load_config, get

        load_config(overrides={
            "stats": {"significance_level": 0.01},
            "ml": {"cv_folds": 10},
        })

        assert get("stats.significance_level") == 0.01
        assert get("ml.cv_folds") == 10
        # Defaults should still be preserved
        assert get("general.random_state") == 42

    def test_update_config(self):
        from survey_toolkit.config import load_config, update_config, get

        load_config()
        update_config({"ml": {"cv_folds": 20}})

        assert get("ml.cv_folds") == 20
        # Other values unchanged
        assert get("stats.significance_level") == 0.05

    def test_reset_config(self):
        from survey_toolkit.config import load_config, update_config, reset_config, get

        load_config()
        update_config({"ml": {"cv_folds": 99}})
        assert get("ml.cv_folds") == 99

        reset_config()
        assert get("ml.cv_folds") == 5  # Back to default

    def test_load_custom_config_file(self, tmp_path):
        from survey_toolkit.config import load_config, get

        # Create custom config
        custom_config = tmp_path / "custom.yml"
        custom_config.write_text(
            "stats:\n  significance_level: 0.001\n"
            "ml:\n  cv_folds: 15\n"
        )

        load_config(config_path=str(custom_config))

        assert get("stats.significance_level") == 0.001
        assert get("ml.cv_folds") == 15
        # Defaults preserved
        assert get("general.random_state") == 42

    def test_load_missing_config_file_raises(self):
        from survey_toolkit.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config(config_path="nonexistent.yml")

    def test_deep_merge(self):
        from survey_toolkit.config import _deep_merge

        base = {
            "a": {"b": 1, "c": 2},
            "d": 3,
        }
        override = {
            "a": {"b": 99},
            "e": 4,
        }
        result = _deep_merge(base, override)

        assert result["a"]["b"] == 99   # Overridden
        assert result["a"]["c"] == 2    # Preserved
        assert result["d"] == 3         # Preserved
        assert result["e"] == 4         # Added

    def test_deep_merge_does_not_mutate(self):
        from survey_toolkit.config import _deep_merge

        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)

        assert base["a"]["b"] == 1  # Original unchanged

    def test_config_immutability(self):
        from survey_toolkit.config import get_config

        config1 = get_config()
        config1["stats"]["significance_level"] = 999

        config2 = get_config()
        assert config2["stats"]["significance_level"] == 0.05  # Unchanged

    def test_color_palette_config(self):
        from survey_toolkit.config import get

        colors = get("eda.color_palette.likert_5")
        assert isinstance(colors, list)
        assert len(colors) == 5
        assert colors[0].startswith("#")

    def test_likert_config(self):
        from survey_toolkit.config import get

        likert_range = get("likert.default_range")
        assert likert_range == [1, 5]

        mapping = get("likert.text_to_numeric")
        assert mapping["Strongly Agree"] == 5
        assert mapping["Strongly Disagree"] == 1

    def test_effect_size_thresholds(self):
        from survey_toolkit.config import get

        cohens_d = get("stats.effect_sizes.cohens_d")
        assert cohens_d["small"] == 0.2
        assert cohens_d["medium"] == 0.5
        assert cohens_d["large"] == 0.8

    def test_ml_model_defaults(self):
        from survey_toolkit.config import get

        rf = get("ml.models.random_forest")
        assert rf["enabled"] is True
        assert rf["params"]["n_estimators"] == 100

    def test_tuning_grids(self):
        from survey_toolkit.config import get

        rf_grid = get("ml.tuning_grids.random_forest")
        assert "model__n_estimators" in rf_grid
        assert "model__max_depth" in rf_grid

    def test_hardcoded_defaults_fallback(self):
        from survey_toolkit.config import _get_hardcoded_defaults

        defaults = _get_hardcoded_defaults()
        assert isinstance(defaults, dict)
        assert defaults["general"]["random_state"] == 42
        assert defaults["stats"]["significance_level"] == 0.05


class TestConfigIntegration:
    """Integration tests for config with other modules."""

    def setup_method(self):
        from survey_toolkit.config import reset_config
        reset_config()

    def test_config_with_stats(self):
        """Verify stats module can read config values."""
        from survey_toolkit.config import get

        sig = get("stats.significance_level")
        assert isinstance(sig, float)
        assert 0 < sig < 1

    def test_config_with_clustering(self):
        """Verify clustering config values."""
        from survey_toolkit.config import get

        k_range = get("clustering.k_range")
        assert isinstance(k_range, list)
        assert len(k_range) == 2
        assert k_range[0] < k_range[1]