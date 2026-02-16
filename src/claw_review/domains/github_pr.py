"""GitHub Pull Request domain configuration.

Extracts the prompts and dimensions from the existing clustering.py,
scoring.py, and alignment.py modules into a DomainConfig, and
registers the GitHubPRAdapter with the AdapterRegistry.
"""

from __future__ import annotations

from ..adapters.github_pr import GitHubPRAdapter
from ..platform.interfaces import DomainConfig
from ..platform.registry import AdapterRegistry

GITHUB_PR_CLUSTERING_PROMPT = """\
You are analyzing a GitHub Pull Request to determine its primary intent.

Respond ONLY with a JSON object (no markdown fences, no extra text):
{
    "intent": "One clear sentence describing what this PR is trying to accomplish",
    "category": "One of: bugfix, feature, refactor, docs, tests, config, security, performance, style",
    "affected_area": "The primary subsystem or area of the codebase this touches (e.g., 'gateway', 'auth', 'websocket', 'UI', 'build')"
}

Be specific about the intent. Instead of "fixes a bug", say "fixes WebSocket reconnection failure after gateway timeout".\
"""

GITHUB_PR_SCORING_PROMPT = """\
You are a senior code reviewer evaluating a GitHub Pull Request.

Score this PR on 5 dimensions, each from 1 (worst) to 10 (best):

1. code_quality: Readability, error handling, edge cases, naming, structure
2. test_coverage: Does the PR include tests? Are edge cases covered? (1 = no tests, 10 = comprehensive)
3. scope_discipline: Does the PR stay focused on one thing? (1 = massive scope creep, 10 = laser focused)
4. breaking_risk: How likely is this to break existing functionality? (1 = very likely to break things, 10 = safe)
5. style_consistency: Does the code match the project's conventions? (1 = completely different style, 10 = perfect match)

Also provide a brief 1-2 sentence summary of the PR's overall quality.

Respond ONLY with a JSON object (no markdown fences):
{
    "code_quality": <1-10>,
    "test_coverage": <1-10>,
    "scope_discipline": <1-10>,
    "breaking_risk": <1-10>,
    "style_consistency": <1-10>,
    "summary": "Brief quality assessment"
}

Be honest and specific. A score of 7+ should mean genuinely good code.\
"""

GITHUB_PR_ALIGNMENT_PROMPT = """\
You are evaluating whether a GitHub Pull Request aligns with a project's vision, architecture, and contribution guidelines.

You have been given the project's key documents (README, CONTRIBUTING, ARCHITECTURE, etc.). Evaluate the PR against these documents.

Score the PR's alignment from 1 to 10:
- 9-10: Perfectly aligned with project direction, addresses a documented need
- 7-8: Well aligned, fits naturally into the codebase
- 5-6: Neutral — doesn't conflict but wasn't on the roadmap
- 3-4: Mild drift — adds something the project may not want
- 1-2: Significant drift — conflicts with stated architecture or goals

Respond ONLY with a JSON object (no markdown fences):
{
    "alignment_score": <1-10>,
    "aligned_aspects": ["List of ways this PR aligns with the project vision"],
    "drift_concerns": ["List of ways this PR diverges from the project vision"],
    "recommendation": "One of: MERGE, REVIEW, DISCUSS, CLOSE",
    "rationale": "One sentence explaining your recommendation"
}

Be specific — reference actual project goals or architectural decisions from the docs.\
"""

GITHUB_PR_CONFIG = DomainConfig(
    domain="github-pr",
    scoring_dimensions=[
        "code_quality",
        "test_coverage",
        "scope_discipline",
        "breaking_risk",
        "style_consistency",
    ],
    clustering_prompt=GITHUB_PR_CLUSTERING_PROMPT,
    scoring_prompt=GITHUB_PR_SCORING_PROMPT,
    alignment_prompt=GITHUB_PR_ALIGNMENT_PROMPT,
    recommendation_levels=["MERGE", "REVIEW", "DISCUSS", "CLOSE"],
    default_thresholds={
        "similarity": 0.82,
        "disagreement": 3.0,
        "alignment_reject": 4.0,
    },
)

# Register the GitHub PR adapter and config with the platform registry
AdapterRegistry.register("github-pr", GitHubPRAdapter, GITHUB_PR_CONFIG)
