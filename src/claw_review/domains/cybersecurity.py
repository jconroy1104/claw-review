"""Cybersecurity domain configuration — community stub.

Defines the scoring dimensions, recommendation levels, and thresholds
for cybersecurity SIEM alert analysis.  Prompts are placeholders; the
full production prompts with MITRE ATT&CK mapping and SOC-optimized
scoring guidance are available in the enterprise edition.
"""

from __future__ import annotations

from claw_review.adapters.cybersecurity import CybersecurityAdapter
from claw_review.platform.interfaces import DomainConfig
from claw_review.platform.registry import AdapterRegistry

CYBERSECURITY_CONFIG = DomainConfig(
    domain="cybersecurity",
    scoring_dimensions=[
        "threat_severity",
        "confidence",
        "attack_sophistication",
        "asset_criticality",
        "actionability",
    ],
    clustering_prompt=(
        "You are a cybersecurity threat analyst. Group related SIEM alerts "
        "that are likely part of the same attack campaign.\n\n"
        "Extract as JSON: intent, category (one of: malware, phishing, "
        "brute_force, data_exfil, lateral_movement, privilege_escalation, "
        "reconnaissance, dos, insider_threat, unknown), and affected_area.\n\n"
        "Respond with valid JSON only."
    ),
    scoring_prompt=(
        "You are a cybersecurity analyst. Score this SIEM alert on five "
        "dimensions (1-10): threat_severity, confidence, "
        "attack_sophistication, asset_criticality, actionability.\n\n"
        "Respond with valid JSON including a brief summary."
    ),
    alignment_prompt=(
        "You are a cybersecurity analyst. Evaluate whether this alert "
        "aligns with the organization's threat model and security policies.\n\n"
        "Respond with valid JSON: alignment_score (1-10), aligned_aspects, "
        "drift_concerns, recommendation (BLOCK/INVESTIGATE/MONITOR/DISMISS), "
        "rationale."
    ),
    recommendation_levels=["BLOCK", "INVESTIGATE", "MONITOR", "DISMISS"],
    default_thresholds={
        "similarity": 0.78,
        "disagreement": 2.5,
        "alignment_reject": 3.0,
    },
)

AdapterRegistry.register("cybersecurity", CybersecurityAdapter, CYBERSECURITY_CONFIG)
