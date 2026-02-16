"""Intent clustering for PR de-duplication.

Extracts semantic intent from each PR using multiple models,
generates embeddings, and clusters PRs with similar intent
using DBSCAN.
"""

import json
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from .models import ModelPool
from .github_client import PRData

console = Console()

INTENT_SYSTEM_PROMPT = """You are analyzing a GitHub Pull Request to determine its primary intent.

Respond ONLY with a JSON object (no markdown fences, no extra text):
{
    "intent": "One clear sentence describing what this PR is trying to accomplish",
    "category": "One of: bugfix, feature, refactor, docs, tests, config, security, performance, style",
    "affected_area": "The primary subsystem or area of the codebase this touches (e.g., 'gateway', 'auth', 'websocket', 'UI', 'build')"
}

Be specific about the intent. Instead of "fixes a bug", say "fixes WebSocket reconnection failure after gateway timeout".
"""


@dataclass
class IntentResult:
    """Extracted intent for a single PR."""

    pr_number: int
    pr_title: str
    pr_author: str
    pr_url: str
    intent_descriptions: dict[str, str]  # provider -> intent string
    consensus_intent: str
    category: str
    affected_area: str
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)  # Don't serialize embeddings
        return d


@dataclass
class Cluster:
    """A group of PRs with similar intent."""

    cluster_id: str
    intent_summary: str
    category: str
    affected_area: str
    confidence: float
    prs: list[dict]  # List of {number, title, author, url, intent}

    def to_dict(self) -> dict:
        return asdict(self)


def _build_intent_prompt(pr: PRData) -> str:
    """Build the user prompt for intent extraction."""
    files_str = "\n".join(f"  - {f}" for f in pr.files_changed[:20])
    if len(pr.files_changed) > 20:
        files_str += f"\n  ... and {len(pr.files_changed) - 20} more files"

    body_truncated = (pr.body or "No description provided.")[:3000]

    return f"""Pull Request #{pr.number}: {pr.title}
Author: {pr.author}
Labels: {', '.join(pr.labels) if pr.labels else 'none'}
Changes: +{pr.additions}/-{pr.deletions}

Description:
{body_truncated}

Files changed:
{files_str}

Diff summary (truncated):
{pr.diff_summary[:4000]}
"""


def extract_intents(
    prs: list[PRData],
    model_pool: ModelPool,
) -> list[IntentResult]:
    """Extract intent from each PR using all available models.

    Args:
        prs: List of PR data objects
        model_pool: Pool of model providers

    Returns:
        List of IntentResult objects with consensus intents
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Extracting intents..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
    ) as progress:
        task = progress.add_task("intents", total=len(prs))

        for pr in prs:
            user_prompt = _build_intent_prompt(pr)
            responses = model_pool.query_all(
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=500,
            )

            # Parse responses
            intent_descriptions = {}
            categories = []
            areas = []

            for resp in responses:
                if resp.model == "error":
                    continue
                try:
                    parsed = resp.parse_json()
                    intent_descriptions[resp.provider] = parsed.get(
                        "intent", ""
                    )
                    categories.append(parsed.get("category", "unknown"))
                    areas.append(parsed.get("affected_area", "unknown"))
                except (json.JSONDecodeError, ValueError):
                    intent_descriptions[resp.provider] = resp.content[:200]

            # Consensus: pick the most common category/area, merge intents
            consensus_intent = _merge_intents(intent_descriptions)
            category = _majority_vote(categories) if categories else "unknown"
            area = _majority_vote(areas) if areas else "unknown"

            results.append(
                IntentResult(
                    pr_number=pr.number,
                    pr_title=pr.title,
                    pr_author=pr.author,
                    pr_url=pr.url,
                    intent_descriptions=intent_descriptions,
                    consensus_intent=consensus_intent,
                    category=category,
                    affected_area=area,
                )
            )
            progress.advance(task)

    return results


def _merge_intents(descriptions: dict[str, str]) -> str:
    """Merge intent descriptions from multiple models.

    Uses the longest description as the consensus (it's typically
    the most specific). For the PoC, this is simple — production
    would use a more sophisticated fusion approach.
    """
    if not descriptions:
        return "Unable to determine intent"
    # Use the longest non-empty description as it tends to be most specific
    valid = {k: v for k, v in descriptions.items() if v and v.strip()}
    if not valid:
        return "Unable to determine intent"
    return max(valid.values(), key=len)


def _majority_vote(items: list[str]) -> str:
    """Simple majority vote from a list of strings."""
    if not items:
        return "unknown"
    from collections import Counter
    counts = Counter(items)
    return counts.most_common(1)[0][0]


def generate_embeddings(
    intents: list[IntentResult],
    model_pool: ModelPool,
) -> list[IntentResult]:
    """Generate embeddings for each PR's consensus intent.

    Uses OpenAI text-embedding-3-small for cost-effective
    semantic similarity computation.
    """
    console.print("[bold blue]Generating embeddings...")

    texts = [
        f"{r.consensus_intent} | {r.category} | {r.affected_area}"
        for r in intents
    ]

    embeddings = model_pool.get_embeddings(texts)

    for intent_result, embedding in zip(intents, embeddings):
        intent_result.embedding = embedding

    console.print(f"[green]✓ Generated {len(embeddings)} embeddings")
    return intents


def cluster_intents(
    intents: list[IntentResult],
    similarity_threshold: float = 0.82,
) -> list[Cluster]:
    """Cluster PRs by semantic intent similarity using DBSCAN.

    Args:
        intents: List of IntentResult objects with embeddings
        similarity_threshold: Cosine similarity threshold for clustering
            (higher = stricter grouping)

    Returns:
        List of Cluster objects (including singletons)
    """
    console.print("[bold blue]Clustering by intent similarity...")

    # Filter to intents with valid embeddings
    valid = [r for r in intents if r.embedding is not None]
    if not valid:
        return []

    # Build similarity matrix
    embeddings_matrix = np.array([r.embedding for r in valid])
    sim_matrix = cosine_similarity(embeddings_matrix)

    # DBSCAN expects a distance matrix, not similarity
    distance_matrix = 1 - sim_matrix
    eps = 1 - similarity_threshold  # Convert similarity threshold to distance

    clustering = DBSCAN(
        eps=eps,
        min_samples=2,  # Need at least 2 PRs to form a cluster
        metric="precomputed",
    ).fit(distance_matrix)

    labels = clustering.labels_
    unique_labels = set(labels)

    clusters = []

    for label in sorted(unique_labels):
        member_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        members = [valid[i] for i in member_indices]

        if label == -1:
            # Noise points (singletons) — each becomes its own "cluster"
            for member in members:
                clusters.append(
                    Cluster(
                        cluster_id=f"singleton-{member.pr_number}",
                        intent_summary=member.consensus_intent,
                        category=member.category,
                        affected_area=member.affected_area,
                        confidence=0.0,  # No cluster confidence for singletons
                        prs=[
                            {
                                "number": member.pr_number,
                                "title": member.pr_title,
                                "author": member.pr_author,
                                "url": member.pr_url,
                                "intent": member.consensus_intent,
                            }
                        ],
                    )
                )
        else:
            # Real cluster — compute average intra-cluster similarity
            if len(member_indices) > 1:
                sub_sim = sim_matrix[np.ix_(member_indices, member_indices)]
                # Average off-diagonal similarity
                mask = ~np.eye(sub_sim.shape[0], dtype=bool)
                avg_sim = float(sub_sim[mask].mean())
            else:
                avg_sim = 1.0

            # Use the most common category and area
            cats = [m.category for m in members]
            areas = [m.affected_area for m in members]

            clusters.append(
                Cluster(
                    cluster_id=f"cluster-{label}",
                    intent_summary=members[0].consensus_intent,
                    category=_majority_vote(cats),
                    affected_area=_majority_vote(areas),
                    confidence=round(avg_sim, 3),
                    prs=[
                        {
                            "number": m.pr_number,
                            "title": m.pr_title,
                            "author": m.pr_author,
                            "url": m.pr_url,
                            "intent": m.consensus_intent,
                        }
                        for m in members
                    ],
                )
            )

    # Sort: multi-PR clusters first (these are the duplicates), then singletons
    clusters.sort(key=lambda c: (-len(c.prs), c.cluster_id))

    dup_count = sum(1 for c in clusters if len(c.prs) > 1)
    console.print(
        f"[green]✓ Found {dup_count} duplicate clusters "
        f"from {len(valid)} PRs "
        f"({sum(len(c.prs) for c in clusters if len(c.prs) > 1)} PRs grouped)"
    )

    return clusters
