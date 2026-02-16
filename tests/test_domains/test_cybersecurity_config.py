"""Tests for the cybersecurity domain configuration."""

from __future__ import annotations

from claw_review.domains.cybersecurity import CYBERSECURITY_CONFIG
from claw_review.platform.interfaces import DomainConfig
from claw_review.platform.registry import AdapterRegistry


def test_config_is_domain_config() -> None:
    """CYBERSECURITY_CONFIG should be a DomainConfig instance."""
    assert isinstance(CYBERSECURITY_CONFIG, DomainConfig)


def test_config_domain_name() -> None:
    """Domain should be 'cybersecurity'."""
    assert CYBERSECURITY_CONFIG.domain == "cybersecurity"


def test_config_has_five_scoring_dimensions() -> None:
    """There should be exactly 5 scoring dimensions."""
    dims = CYBERSECURITY_CONFIG.scoring_dimensions
    assert len(dims) == 5
    assert "threat_severity" in dims
    assert "confidence" in dims
    assert "attack_sophistication" in dims
    assert "asset_criticality" in dims
    assert "actionability" in dims


def test_config_recommendation_levels() -> None:
    """Recommendation levels should be BLOCK, INVESTIGATE, MONITOR, DISMISS."""
    expected = ["BLOCK", "INVESTIGATE", "MONITOR", "DISMISS"]
    assert CYBERSECURITY_CONFIG.recommendation_levels == expected


def test_config_thresholds_present() -> None:
    """Default thresholds should contain similarity, disagreement, alignment_reject."""
    thresholds = CYBERSECURITY_CONFIG.default_thresholds
    assert "similarity" in thresholds
    assert "disagreement" in thresholds
    assert "alignment_reject" in thresholds
    assert thresholds["similarity"] == 0.78
    assert thresholds["disagreement"] == 2.5
    assert thresholds["alignment_reject"] == 3.0


def test_clustering_prompt_non_empty() -> None:
    """Clustering prompt should be a non-empty string."""
    assert isinstance(CYBERSECURITY_CONFIG.clustering_prompt, str)
    assert len(CYBERSECURITY_CONFIG.clustering_prompt) > 50


def test_scoring_prompt_non_empty() -> None:
    """Scoring prompt should be a non-empty string."""
    assert isinstance(CYBERSECURITY_CONFIG.scoring_prompt, str)
    assert len(CYBERSECURITY_CONFIG.scoring_prompt) > 50


def test_alignment_prompt_non_empty() -> None:
    """Alignment prompt should be a non-empty string."""
    assert isinstance(CYBERSECURITY_CONFIG.alignment_prompt, str)
    assert len(CYBERSECURITY_CONFIG.alignment_prompt) > 50


def test_registers_with_adapter_registry() -> None:
    """Importing the module should register 'cybersecurity' in AdapterRegistry."""
    adapter_cls, config = AdapterRegistry.get("cybersecurity")
    assert config is CYBERSECURITY_CONFIG
    assert adapter_cls.domain == "cybersecurity"
