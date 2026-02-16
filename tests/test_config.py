"""Tests for claw_review.config module."""

from __future__ import annotations

import pytest

from claw_review.config import Config, DEFAULT_MODELS, PRESETS, _parse_models, get_preset, list_presets


class TestParseModels:
    """Tests for _parse_models() function."""

    def test_returns_defaults_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODELS", raising=False)
        result = _parse_models()
        assert result == DEFAULT_MODELS
        # Ensure it returns a copy, not the same list
        assert result is not DEFAULT_MODELS

    def test_parses_comma_separated_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "model/a,model/b,model/c")
        result = _parse_models()
        assert result == ["model/a", "model/b", "model/c"]

    def test_strips_whitespace_from_model_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "  model/a , model/b ,  model/c  ")
        result = _parse_models()
        assert result == ["model/a", "model/b", "model/c"]

    def test_filters_empty_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "model/a,,model/b,,,model/c,")
        result = _parse_models()
        assert result == ["model/a", "model/b", "model/c"]

    def test_empty_string_returns_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "")
        result = _parse_models()
        assert result == DEFAULT_MODELS

    def test_whitespace_only_returns_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "   ")
        result = _parse_models()
        assert result == DEFAULT_MODELS

    def test_single_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "anthropic/claude-sonnet-4")
        result = _parse_models()
        assert result == ["anthropic/claude-sonnet-4"]


class TestConfigCreation:
    """Tests for Config dataclass creation."""

    def test_config_with_all_valid_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test456")
        monkeypatch.setenv("TARGET_REPO", "owner/repo")
        monkeypatch.setenv("MAX_PRS", "50")
        monkeypatch.setenv("MODELS", "model/a,model/b,model/c")
        monkeypatch.setenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")

        config = Config()

        assert config.github_token == "ghp_test123"
        assert config.openrouter_api_key == "sk-or-test456"
        assert config.target_repo == "owner/repo"
        assert config.max_prs == 50
        assert config.models == ["model/a", "model/b", "model/c"]
        assert config.embedding_model == "openai/text-embedding-3-large"

    def test_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("TARGET_REPO", raising=False)
        monkeypatch.delenv("MAX_PRS", raising=False)
        monkeypatch.delenv("MODELS", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

        config = Config()

        assert config.github_token == ""
        assert config.openrouter_api_key == ""
        assert config.target_repo == "openclaw/openclaw"
        assert config.max_prs == 100
        assert config.models == DEFAULT_MODELS
        assert config.embedding_model == "openai/text-embedding-3-small"

    def test_max_prs_env_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_PRS", "25")
        config = Config()
        assert config.max_prs == 25

    def test_max_prs_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_PRS", "not_a_number")
        with pytest.raises(ValueError):
            Config()


class TestConfigThresholdDefaults:
    """Tests for Config threshold default values."""

    def test_similarity_threshold_default(self) -> None:
        config = Config()
        assert config.similarity_threshold == 0.82

    def test_quality_disagreement_threshold_default(self) -> None:
        config = Config()
        assert config.quality_disagreement_threshold == 3.0

    def test_alignment_reject_threshold_default(self) -> None:
        config = Config()
        assert config.alignment_reject_threshold == 4.0


class TestModelCount:
    """Tests for Config.model_count property."""

    def test_model_count_with_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODELS", raising=False)
        config = Config()
        assert config.model_count == len(DEFAULT_MODELS)
        assert config.model_count == 3

    def test_model_count_with_custom_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "model/a,model/b")
        config = Config()
        assert config.model_count == 2

    def test_model_count_single_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODELS", "model/only")
        config = Config()
        assert config.model_count == 1


class TestConfigValidation:
    """Tests for Config.validate() method."""

    def test_validate_all_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid")
        monkeypatch.delenv("MODELS", raising=False)
        config = Config()
        issues = config.validate()
        assert issues == []

    def test_validate_missing_github_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid")
        monkeypatch.delenv("MODELS", raising=False)
        config = Config()
        issues = config.validate()
        assert any("GITHUB_TOKEN" in issue for issue in issues)

    def test_validate_missing_openrouter_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("MODELS", raising=False)
        config = Config()
        issues = config.validate()
        assert any("OPENROUTER_API_KEY" in issue for issue in issues)

    def test_validate_both_keys_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("MODELS", raising=False)
        config = Config()
        issues = config.validate()
        assert len(issues) >= 2
        assert any("GITHUB_TOKEN" in i for i in issues)
        assert any("OPENROUTER_API_KEY" in i for i in issues)

    def test_validate_single_model_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid")
        monkeypatch.setenv("MODELS", "model/only")
        config = Config()
        issues = config.validate()
        assert any("at least 2 models" in issue for issue in issues)

    def test_validate_zero_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid")
        # Create config then manually clear models to simulate 0 models
        config = Config()
        config.models = []
        issues = config.validate()
        assert any("at least 2 models" in issue for issue in issues)
        assert any("0" in issue for issue in issues)

    def test_validate_two_models_no_model_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid")
        monkeypatch.setenv("MODELS", "model/a,model/b")
        config = Config()
        issues = config.validate()
        assert not any("models" in issue.lower() for issue in issues)

    def test_validate_empty_string_github_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid")
        monkeypatch.delenv("MODELS", raising=False)
        config = Config()
        issues = config.validate()
        assert any("GITHUB_TOKEN" in issue for issue in issues)

    def test_validate_empty_string_openrouter_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_valid")
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.delenv("MODELS", raising=False)
        config = Config()
        issues = config.validate()
        assert any("OPENROUTER_API_KEY" in issue for issue in issues)


class TestPresets:
    """Tests for model preset configuration."""

    def test_get_preset_fast(self) -> None:
        preset = get_preset("fast")
        assert "models" in preset
        assert "description" in preset
        assert "est_cost_per_100_prs" in preset
        assert preset["models"] == PRESETS["fast"]["models"]

    def test_get_preset_balanced(self) -> None:
        preset = get_preset("balanced")
        assert preset["models"] == PRESETS["balanced"]["models"]
        assert "quality/cost" in preset["description"].lower() or "recommended" in preset["description"].lower()

    def test_get_preset_thorough(self) -> None:
        preset = get_preset("thorough")
        assert preset["models"] == PRESETS["thorough"]["models"]
        assert "highest" in preset["description"].lower() or "quality" in preset["description"].lower()

    def test_get_preset_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset 'nonexistent'"):
            get_preset("nonexistent")

    def test_list_presets_returns_all(self) -> None:
        result = list_presets()
        assert set(result.keys()) == {"fast", "balanced", "thorough"}
        # Ensure it returns a copy
        assert result is not PRESETS

    def test_preset_models_are_valid_format(self) -> None:
        for name, preset in PRESETS.items():
            for model in preset["models"]:
                assert "/" in model, f"Model '{model}' in preset '{name}' missing provider prefix"

    def test_fast_preset_has_three_models(self) -> None:
        assert len(PRESETS["fast"]["models"]) == 3

    def test_balanced_preset_has_three_models(self) -> None:
        assert len(PRESETS["balanced"]["models"]) == 3

    def test_thorough_preset_has_three_models(self) -> None:
        assert len(PRESETS["thorough"]["models"]) == 3
