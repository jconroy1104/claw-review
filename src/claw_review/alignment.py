"""Vision alignment scoring.

Evaluates how well each PR aligns with the project's stated
architecture, roadmap, and contribution guidelines.
"""

import json
from dataclasses import dataclass, asdict

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from .models import ModelPool
from .github_client import PRData

console = Console()

ALIGNMENT_SYSTEM_PROMPT = """You are evaluating whether a GitHub Pull Request aligns with a project's vision, architecture, and contribution guidelines.

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

Be specific — reference actual project goals or architectural decisions from the docs.
"""


@dataclass
class AlignmentScore:
    """Vision alignment assessment for a single PR."""

    pr_number: int
    pr_title: str
    pr_author: str
    pr_url: str
    alignment_score: float  # Consensus score 1-10
    scores_by_provider: dict[str, float]
    aligned_aspects: list[str]
    drift_concerns: list[str]
    recommendation: str  # MERGE, REVIEW, DISCUSS, CLOSE
    rationale: str
    confidence: float  # Agreement between models

    def to_dict(self) -> dict:
        return asdict(self)


def _build_alignment_prompt(
    pr: PRData, vision_docs: dict[str, str]
) -> str:
    """Build user prompt for alignment evaluation."""
    # Compile vision documents
    docs_text = ""
    for path, content in vision_docs.items():
        docs_text += f"\n=== {path} ===\n{content[:5000]}\n"

    body_truncated = (pr.body or "No description.")[:2000]
    files_str = "\n".join(f"  - {f}" for f in pr.files_changed[:15])

    return f"""PROJECT VISION DOCUMENTS:
{docs_text}

---

PULL REQUEST TO EVALUATE:

PR #{pr.number}: {pr.title}
Author: {pr.author}
Labels: {', '.join(pr.labels) if pr.labels else 'none'}
Changes: +{pr.additions}/-{pr.deletions} across {len(pr.files_changed)} files

Description:
{body_truncated}

Files changed:
{files_str}

Diff summary:
{pr.diff_summary[:4000]}
"""


def score_alignment(
    prs: list[PRData],
    vision_docs: dict[str, str],
    model_pool: ModelPool,
    reject_threshold: float = 4.0,
) -> list[AlignmentScore]:
    """Score PR alignment against project vision documents.

    Args:
        prs: List of PR data objects
        vision_docs: Dict of filename -> content for project docs
        model_pool: Pool of model providers
        reject_threshold: Score below which PRs are flagged for closure

    Returns:
        List of AlignmentScore objects, sorted by score (lowest first
        to surface problematic PRs)
    """
    if not vision_docs:
        console.print(
            "[yellow]⚠ No vision documents found. "
            "Skipping alignment scoring."
        )
        return []

    console.print(
        f"[blue]Vision documents loaded: {', '.join(vision_docs.keys())}"
    )

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Scoring vision alignment..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
    ) as progress:
        task = progress.add_task("alignment", total=len(prs))

        for pr in prs:
            user_prompt = _build_alignment_prompt(pr, vision_docs)
            responses = model_pool.query_all(
                system_prompt=ALIGNMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=1000,
            )

            scores_by_provider: dict[str, float] = {}
            all_aligned: list[str] = []
            all_drift: list[str] = []
            recommendations: list[str] = []
            rationales: list[str] = []

            for resp in responses:
                if resp.model == "error":
                    continue
                try:
                    parsed = resp.parse_json()
                    score = float(parsed.get("alignment_score", 5))
                    score = max(1, min(10, score))
                    scores_by_provider[resp.provider] = score

                    aligned = parsed.get("aligned_aspects", [])
                    if isinstance(aligned, list):
                        all_aligned.extend(aligned)

                    drift = parsed.get("drift_concerns", [])
                    if isinstance(drift, list):
                        all_drift.extend(drift)

                    rec = parsed.get("recommendation", "REVIEW")
                    recommendations.append(rec)
                    rationale = parsed.get("rationale", "")
                    if rationale:
                        rationales.append(rationale)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

            if not scores_by_provider:
                progress.advance(task)
                continue

            # Consensus score
            values = list(scores_by_provider.values())
            consensus_score = sum(values) / len(values)
            spread = max(values) - min(values) if len(values) > 1 else 0
            confidence = max(0, 1 - (spread / 10))

            # Consensus recommendation
            from collections import Counter
            rec_counts = Counter(recommendations)
            consensus_rec = rec_counts.most_common(1)[0][0] if rec_counts else "REVIEW"

            # Override recommendation if score is below threshold
            if consensus_score < reject_threshold:
                consensus_rec = "CLOSE"

            # Deduplicate aspects and concerns
            aligned_deduped = list(dict.fromkeys(all_aligned))[:5]
            drift_deduped = list(dict.fromkeys(all_drift))[:5]

            results.append(
                AlignmentScore(
                    pr_number=pr.number,
                    pr_title=pr.title,
                    pr_author=pr.author,
                    pr_url=pr.url,
                    alignment_score=round(consensus_score, 2),
                    scores_by_provider=scores_by_provider,
                    aligned_aspects=aligned_deduped,
                    drift_concerns=drift_deduped,
                    recommendation=consensus_rec,
                    rationale=rationales[0] if rationales else "",
                    confidence=round(confidence, 3),
                )
            )
            progress.advance(task)

    # Sort by alignment score ascending (surface problematic PRs first)
    results.sort(key=lambda r: r.alignment_score)

    low_count = sum(1 for r in results if r.alignment_score < reject_threshold)
    console.print(
        f"[green]✓ Scored {len(results)} PRs for vision alignment. "
        f"[red]{low_count} flagged for potential closure."
    )

    return results
