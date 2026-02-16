"""Quality scoring for PRs using multi-model consensus.

Each model independently scores PRs on multiple dimensions.
Scores are fused using weighted averaging with outlier detection.
"""

import json
from dataclasses import dataclass

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from .models import ModelPool
from .github_client import PRData

console = Console()

SCORING_DIMENSIONS = [
    "code_quality",
    "test_coverage",
    "scope_discipline",
    "breaking_risk",
    "style_consistency",
]

SCORING_SYSTEM_PROMPT = """You are a senior code reviewer evaluating a GitHub Pull Request.

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

Be honest and specific. A score of 7+ should mean genuinely good code.
"""


@dataclass
class DimensionScore:
    """Score for a single dimension across models."""

    dimension: str
    scores: dict[str, float]  # provider -> score
    consensus: float
    disagreement: float  # max spread between models
    flagged: bool  # True if models disagree significantly


@dataclass
class QualityScore:
    """Quality assessment for a single PR."""

    pr_number: int
    pr_title: str
    pr_author: str
    pr_url: str
    dimensions: list[DimensionScore]
    overall_score: float  # Weighted average of all dimensions
    summary: str
    provider_summaries: dict[str, str]
    needs_human_review: bool  # True if significant model disagreement
    disagreement_reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "pr_number": self.pr_number,
            "pr_title": self.pr_title,
            "pr_author": self.pr_author,
            "pr_url": self.pr_url,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "needs_human_review": self.needs_human_review,
            "disagreement_reasons": self.disagreement_reasons,
            "dimensions": {
                d.dimension: {
                    "consensus": d.consensus,
                    "scores": d.scores,
                    "disagreement": d.disagreement,
                    "flagged": d.flagged,
                }
                for d in self.dimensions
            },
        }


def _build_scoring_prompt(pr: PRData) -> str:
    """Build user prompt for quality scoring."""
    body_truncated = (pr.body or "No description.")[:2000]
    files_str = "\n".join(f"  - {f}" for f in pr.files_changed[:15])

    return f"""Pull Request #{pr.number}: {pr.title}
Author: {pr.author}
Labels: {', '.join(pr.labels) if pr.labels else 'none'}
Changes: +{pr.additions}/-{pr.deletions} across {len(pr.files_changed)} files

Description:
{body_truncated}

Files changed:
{files_str}

Diff:
{pr.diff_summary[:6000]}
"""


def score_prs(
    prs: list[PRData],
    model_pool: ModelPool,
    disagreement_threshold: float = 3.0,
) -> list[QualityScore]:
    """Score PRs on quality dimensions using multi-model consensus.

    Args:
        prs: List of PR data objects to score
        model_pool: Pool of model providers
        disagreement_threshold: Flag for human review if models
            disagree by more than this on any dimension

    Returns:
        List of QualityScore objects, sorted by overall score (best first)
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Scoring PR quality..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
    ) as progress:
        task = progress.add_task("scoring", total=len(prs))

        for pr in prs:
            user_prompt = _build_scoring_prompt(pr)
            responses = model_pool.query_all(
                system_prompt=SCORING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=800,
            )

            # Parse all model responses
            model_scores: dict[str, dict[str, float]] = {}
            summaries: dict[str, str] = {}

            for resp in responses:
                if resp.model == "error":
                    continue
                try:
                    parsed = resp.parse_json()
                    scores = {}
                    for dim in SCORING_DIMENSIONS:
                        val = parsed.get(dim)
                        if val is not None:
                            scores[dim] = float(
                                max(1, min(10, val))
                            )  # Clamp 1-10
                    if scores:
                        model_scores[resp.provider] = scores
                    summaries[resp.provider] = parsed.get("summary", "")
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

            if not model_scores:
                # No valid responses — skip this PR
                progress.advance(task)
                continue

            # Build consensus scores per dimension
            dimensions = []
            disagreement_reasons = []

            for dim in SCORING_DIMENSIONS:
                dim_scores = {
                    provider: scores[dim]
                    for provider, scores in model_scores.items()
                    if dim in scores
                }

                if dim_scores:
                    values = list(dim_scores.values())
                    consensus = sum(values) / len(values)
                    spread = max(values) - min(values)
                    flagged = spread > disagreement_threshold

                    if flagged:
                        disagreement_reasons.append(
                            f"{dim}: spread of {spread:.1f} "
                            f"({dict((k, f'{v:.0f}') for k, v in dim_scores.items())})"
                        )

                    dimensions.append(
                        DimensionScore(
                            dimension=dim,
                            scores=dim_scores,
                            consensus=round(consensus, 2),
                            disagreement=round(spread, 2),
                            flagged=flagged,
                        )
                    )

            # Overall score: simple average of consensus scores
            if dimensions:
                overall = sum(d.consensus for d in dimensions) / len(dimensions)
            else:
                overall = 0.0

            # Merge summaries
            summary_parts = [s for s in summaries.values() if s]
            merged_summary = summary_parts[0] if summary_parts else "No summary"

            results.append(
                QualityScore(
                    pr_number=pr.number,
                    pr_title=pr.title,
                    pr_author=pr.author,
                    pr_url=pr.url,
                    dimensions=dimensions,
                    overall_score=round(overall, 2),
                    summary=merged_summary,
                    provider_summaries=summaries,
                    needs_human_review=bool(disagreement_reasons),
                    disagreement_reasons=disagreement_reasons,
                )
            )
            progress.advance(task)

    # Sort by overall score, best first
    results.sort(key=lambda r: -r.overall_score)
    return results


def rank_within_clusters(
    clusters: list[dict],
    quality_scores: list[QualityScore],
) -> list[dict]:
    """Attach quality rankings to PRs within each cluster.

    Args:
        clusters: List of cluster dicts from clustering module
        quality_scores: List of quality scores

    Returns:
        Updated clusters with quality_rank and quality_score per PR
    """
    score_map = {qs.pr_number: qs for qs in quality_scores}

    for cluster in clusters:
        for pr in cluster.get("prs", []):
            qs = score_map.get(pr["number"])
            if qs:
                pr["quality_score"] = qs.overall_score
                pr["needs_human_review"] = qs.needs_human_review
                pr["quality_summary"] = qs.summary
            else:
                pr["quality_score"] = 0.0
                pr["needs_human_review"] = True
                pr["quality_summary"] = "Not scored"

        # Rank within cluster by quality score
        sorted_prs = sorted(
            cluster.get("prs", []),
            key=lambda p: -p.get("quality_score", 0),
        )
        for rank, pr in enumerate(sorted_prs, 1):
            pr["quality_rank"] = rank
        cluster["prs"] = sorted_prs

    return clusters
