"""Tests for the fraud detection domain configuration."""

from __future__ import annotations

from claw_review.domains.fraud_detection import FRAUD_DETECTION_CONFIG
from claw_review.platform.interfaces import DomainConfig
from claw_review.platform.registry import AdapterRegistry


def test_config_is_domain_config() -> None:
    """FRAUD_DETECTION_CONFIG should be a DomainConfig instance."""
    assert isinstance(FRAUD_DETECTION_CONFIG, DomainConfig)


def test_config_domain_name() -> None:
    """Domain should be 'fraud-detection'."""
    assert FRAUD_DETECTION_CONFIG.domain == "fraud-detection"


def test_config_has_five_scoring_dimensions() -> None:
    """There should be exactly 5 scoring dimensions."""
    dims = FRAUD_DETECTION_CONFIG.scoring_dimensions
    assert len(dims) == 5
    assert "anomaly_score" in dims
    assert "pattern_match" in dims
    assert "velocity_risk" in dims
    assert "geographic_risk" in dims
    assert "amount_deviation" in dims


def test_config_recommendation_levels() -> None:
    """Recommendation levels should be APPROVE, FLAG, HOLD, BLOCK."""
    expected = ["APPROVE", "FLAG", "HOLD", "BLOCK"]
    assert FRAUD_DETECTION_CONFIG.recommendation_levels == expected


def test_config_thresholds_present() -> None:
    """Default thresholds should contain similarity, disagreement, alignment_reject."""
    thresholds = FRAUD_DETECTION_CONFIG.default_thresholds
    assert "similarity" in thresholds
    assert "disagreement" in thresholds
    assert "alignment_reject" in thresholds
    assert thresholds["similarity"] == 0.75
    assert thresholds["disagreement"] == 2.0
    assert thresholds["alignment_reject"] == 3.0


def test_clustering_prompt_non_empty() -> None:
    """Clustering prompt should be a non-empty string."""
    assert isinstance(FRAUD_DETECTION_CONFIG.clustering_prompt, str)
    assert len(FRAUD_DETECTION_CONFIG.clustering_prompt) > 50


def test_scoring_prompt_non_empty() -> None:
    """Scoring prompt should be a non-empty string."""
    assert isinstance(FRAUD_DETECTION_CONFIG.scoring_prompt, str)
    assert len(FRAUD_DETECTION_CONFIG.scoring_prompt) > 50


def test_alignment_prompt_non_empty() -> None:
    """Alignment prompt should be a non-empty string."""
    assert isinstance(FRAUD_DETECTION_CONFIG.alignment_prompt, str)
    assert len(FRAUD_DETECTION_CONFIG.alignment_prompt) > 50


def test_registers_with_adapter_registry() -> None:
    """Importing the module should register 'fraud-detection' in AdapterRegistry."""
    adapter_cls, config = AdapterRegistry.get("fraud-detection")
    assert config is FRAUD_DETECTION_CONFIG
    assert adapter_cls.domain == "fraud-detection"
