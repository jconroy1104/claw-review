"""Cybersecurity domain configuration for the consensus platform.

Defines scoring dimensions, prompts, and thresholds for analyzing
SIEM alerts using multi-model consensus.
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
        "You are a cybersecurity threat analyst performing alert triage on "
        "SIEM alerts. Your task is to group related alerts that are likely "
        "part of the same attack campaign, originate from the same threat "
        "actor, or use the same MITRE ATT&CK technique.\n\n"
        "For each alert, extract the following as JSON:\n"
        "- \"intent\": A short description of what the attacker is trying "
        "to achieve (e.g., 'Establish persistence via SSH brute force', "
        "'Exfiltrate customer database via DNS tunneling').\n"
        "- \"category\": Exactly one of: malware, phishing, brute_force, "
        "data_exfil, lateral_movement, privilege_escalation, reconnaissance, "
        "dos, insider_threat, unknown.\n"
        "- \"affected_area\": The primary system, network segment, or asset "
        "group affected (e.g., 'DMZ web servers', 'finance workstations', "
        "'internal DNS infrastructure').\n\n"
        "Identify correlated alerts that form part of the same incident "
        "chain. Look for shared source IPs, escalating severity patterns, "
        "temporal proximity, and common MITRE ATT&CK kill chain stages.\n\n"
        "Respond with valid JSON only."
    ),
    scoring_prompt=(
        "You are a cybersecurity analyst evaluating a SIEM alert for "
        "operational priority. Score the alert across five dimensions, "
        "each on a scale of 1-10:\n\n"
        "1. **threat_severity** (1-10): How dangerous is this threat if "
        "left unaddressed? Consider potential for data loss, system "
        "compromise, and business impact.\n"
        "2. **confidence** (1-10): How confident are you that this is a "
        "true positive? Consider false positive indicators, corroborating "
        "evidence, and IoC reliability.\n"
        "3. **attack_sophistication** (1-10): How technically sophisticated "
        "is the attack? Script-kiddie tools score low; custom malware, "
        "zero-days, or living-off-the-land techniques score high.\n"
        "4. **asset_criticality** (1-10): How critical is the affected "
        "asset? Development workstations score low; production databases "
        "and domain controllers score high.\n"
        "5. **actionability** (1-10): How actionable is this alert? Can "
        "the SOC team take immediate, concrete steps to contain or "
        "remediate? High if clear containment steps exist.\n\n"
        "Also provide a brief \"summary\" (2-3 sentences) explaining your "
        "scoring rationale.\n\n"
        "Respond with valid JSON matching this schema:\n"
        "{\n"
        '  "threat_severity": <int>,\n'
        '  "confidence": <int>,\n'
        '  "attack_sophistication": <int>,\n'
        '  "asset_criticality": <int>,\n'
        '  "actionability": <int>,\n'
        '  "summary": "<string>"\n'
        "}"
    ),
    alignment_prompt=(
        "You are a cybersecurity analyst evaluating whether a SIEM alert "
        "aligns with known threat patterns described in the organization's "
        "threat model and security policies.\n\n"
        "Given the alert details and the organization's threat model "
        "documents, assess:\n"
        "- Does this alert match a known threat scenario in the threat model?\n"
        "- Is the affected asset covered by existing security policies?\n"
        "- Does the response procedure align with the incident response plan?\n\n"
        "Respond with valid JSON matching this schema:\n"
        "{\n"
        '  "alignment_score": <float 1-10>,\n'
        '  "aligned_aspects": ["<string>", ...],\n'
        '  "drift_concerns": ["<string>", ...],\n'
        '  "recommendation": "<one of: BLOCK, INVESTIGATE, MONITOR, DISMISS>",\n'
        '  "rationale": "<string>"\n'
        "}"
    ),
    recommendation_levels=["BLOCK", "INVESTIGATE", "MONITOR", "DISMISS"],
    default_thresholds={
        "similarity": 0.78,
        "disagreement": 2.5,
        "alignment_reject": 3.0,
    },
)

AdapterRegistry.register("cybersecurity", CybersecurityAdapter, CYBERSECURITY_CONFIG)
