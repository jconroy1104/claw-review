"""Fraud detection domain configuration for the consensus platform.

Defines scoring dimensions, prompts, and thresholds for analyzing
financial transactions using multi-model consensus.
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
        "You are a fraud analyst performing transaction triage. Your task "
        "is to group related transactions that are likely part of the same "
        "fraud scheme, originate from the same fraud ring, or follow the "
        "same attack vector.\n\n"
        "For each transaction, extract the following as JSON:\n"
        "- \"intent\": A short description of the suspected fraud pattern "
        "(e.g., 'Card testing with micro-charges', 'Geographic impossibility "
        "indicating stolen card', 'Velocity abuse at fuel stations').\n"
        "- \"category\": Exactly one of: card_not_present, account_takeover, "
        "identity_theft, velocity_abuse, geographic_anomaly, refund_fraud, "
        "synthetic_identity, merchant_fraud, friendly_fraud, unknown.\n"
        "- \"affected_area\": The primary risk domain affected (e.g., "
        "'online retail', 'fuel purchases', 'luxury goods', "
        "'recurring subscriptions').\n\n"
        "Look for coordinated patterns: shared customer IDs with anomalous "
        "behavior, rapid-fire transactions, impossible travel between "
        "locations, and amounts that deviate significantly from customer "
        "baselines.\n\n"
        "Respond with valid JSON only."
    ),
    scoring_prompt=(
        "You are a fraud analyst evaluating a financial transaction for "
        "fraud risk. Score the transaction across five dimensions, each "
        "on a scale of 1-10:\n\n"
        "1. **anomaly_score** (1-10): How anomalous is this transaction "
        "relative to the customer's historical behavior? Consider amount, "
        "merchant category, time of day, and transaction frequency.\n"
        "2. **pattern_match** (1-10): How closely does this transaction "
        "match known fraud patterns? Consider card testing sequences, "
        "account takeover signatures, and common fraud typologies.\n"
        "3. **velocity_risk** (1-10): Is the transaction frequency "
        "suspicious? Multiple transactions in a short window, especially "
        "at different merchants or locations, indicate higher risk.\n"
        "4. **geographic_risk** (1-10): Is the transaction location "
        "consistent with the customer's profile? Impossible travel "
        "(e.g., London and Tokyo 5 minutes apart) scores very high.\n"
        "5. **amount_deviation** (1-10): How far does the amount deviate "
        "from the customer's historical average? Large deviations in "
        "either direction (micro-charges or unusually large) score high.\n\n"
        "Also provide a brief \"summary\" (2-3 sentences) explaining your "
        "scoring rationale.\n\n"
        "Respond with valid JSON matching this schema:\n"
        "{\n"
        '  "anomaly_score": <int>,\n'
        '  "pattern_match": <int>,\n'
        '  "velocity_risk": <int>,\n'
        '  "geographic_risk": <int>,\n'
        '  "amount_deviation": <int>,\n'
        '  "summary": "<string>"\n'
        "}"
    ),
    alignment_prompt=(
        "You are a fraud analyst evaluating whether a transaction aligns "
        "with the customer's established behavioral profile and the "
        "organization's fraud detection rules.\n\n"
        "Given the transaction details and the customer's profile or fraud "
        "rules, assess:\n"
        "- Does the merchant category match the customer's usual spending?\n"
        "- Is the amount within expected bounds for this customer?\n"
        "- Is the location consistent with the customer's recent activity?\n"
        "- Does the transaction time match the customer's typical patterns?\n\n"
        "Respond with valid JSON matching this schema:\n"
        "{\n"
        '  "alignment_score": <float 1-10>,\n'
        '  "aligned_aspects": ["<string>", ...],\n'
        '  "drift_concerns": ["<string>", ...],\n'
        '  "recommendation": "<one of: APPROVE, FLAG, HOLD, BLOCK>",\n'
        '  "rationale": "<string>"\n'
        "}"
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
