"""Fraud detection domain configuration — community stub.

Defines the scoring dimensions, recommendation levels, and thresholds
for financial transaction fraud analysis.  Prompts are placeholders;
the full production prompts with fraud-typology scoring guidance and
behavioral-profile alignment are available in the enterprise edition.
"""

from __future__ import annotations

from claw_review.adapters.fraud_detection import FraudDetectionAdapter
from claw_review.platform.interfaces import DomainConfig
from claw_review.platform.registry import AdapterRegistry

FRAUD_DETECTION_CONFIG = DomainConfig(
    domain="fraud-detection",
    scoring_dimensions=[
        "anomaly_score",
        "pattern_match",
        "velocity_risk",
        "geographic_risk",
        "amount_deviation",
    ],
    clustering_prompt=(
        "You are a fraud analyst. Group related transactions that are "
        "likely part of the same fraud scheme.\n\n"
        "Extract as JSON: intent, category (one of: card_not_present, "
        "account_takeover, identity_theft, velocity_abuse, "
        "geographic_anomaly, refund_fraud, synthetic_identity, "
        "merchant_fraud, friendly_fraud, unknown), and affected_area.\n\n"
        "Respond with valid JSON only."
    ),
    scoring_prompt=(
        "You are a fraud analyst. Score this transaction on five "
        "dimensions (1-10): anomaly_score, pattern_match, velocity_risk, "
        "geographic_risk, amount_deviation.\n\n"
        "Respond with valid JSON including a brief summary."
    ),
    alignment_prompt=(
        "You are a fraud analyst. Evaluate whether this transaction "
        "aligns with the customer's behavioral profile and fraud rules.\n\n"
        "Respond with valid JSON: alignment_score (1-10), aligned_aspects, "
        "drift_concerns, recommendation (APPROVE/FLAG/HOLD/BLOCK), "
        "rationale."
    ),
    recommendation_levels=["APPROVE", "FLAG", "HOLD", "BLOCK"],
    default_thresholds={
        "similarity": 0.75,
        "disagreement": 2.0,
        "alignment_reject": 3.0,
    },
)

AdapterRegistry.register(
    "fraud-detection", FraudDetectionAdapter, FRAUD_DETECTION_CONFIG
)
