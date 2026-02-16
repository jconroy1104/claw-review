"""Domain-agnostic consensus engine.

Orchestrates the full analysis pipeline (intent extraction, embedding,
clustering, quality scoring, alignment scoring) using the provided
DomainConfig's prompts and dimensions instead of hardcoded ones.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from ..models import ModelPool
from .interfaces import AnalysisResult, DataItem, DomainConfig


class ConsensusEngine:
    """Domain-agnostic consensus analysis engine.

    Uses a ModelPool for API calls and a DomainConfig for prompts,
    dimensions, and thresholds. The actual consensus logic (majority
    vote, averaging, DBSCAN clustering) is implemented inline.

    Args:
        model_pool: Pool of model providers for LLM queries.
        domain_config: Configuration defining prompts and dimensions.
    """

    def __init__(
        self, model_pool: ModelPool, domain_config: DomainConfig
    ) -> None:
        self.model_pool = model_pool
        self.domain_config = domain_config

    async def analyze(
        self,
        items: list[DataItem],
        context_docs: dict[str, str] | None = None,
    ) -> AnalysisResult:
        """Run the full consensus analysis pipeline.

        Steps:
        1. Extract intents from each item using all models.
        2. Generate embeddings for the consensus intents.
        3. Cluster items by semantic similarity (DBSCAN).
        4. Score quality on each item using domain dimensions.
        5. Score alignment against context docs (if provided).

        Args:
            items: List of DataItem objects to analyze.
            context_docs: Optional dict of document name to content
                for alignment scoring. If None, alignment is skipped.

        Returns:
            AnalysisResult with clusters, quality scores, and
            alignment scores.
        """
        if not items:
            return AnalysisResult(
                domain=self.domain_config.domain,
                source="",
                items_analyzed=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Step 1: Extract intents
        intents = await self._extract_intents(items)

        # Step 2: Generate embeddings
        intents_with_embeddings = await self._generate_embeddings(intents)

        # Step 3: Cluster
        similarity_threshold = self.domain_config.default_thresholds.get(
            "similarity", 0.82
        )
        clusters = self._cluster(intents_with_embeddings, similarity_threshold)

        # Step 4: Quality scoring
        quality_scores = await self._score_quality(items)

        # Step 5: Alignment scoring (only if context docs provided)
        alignment_scores: list[dict[str, Any]] = []
        if context_docs:
            alignment_scores = await self._score_alignment(
                items, context_docs
            )

        return AnalysisResult(
            domain=self.domain_config.domain,
            source="",
            items_analyzed=len(items),
            clusters=[c for c in clusters],
            quality_scores=quality_scores,
            alignment_scores=alignment_scores,
            cost={},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    async def _extract_intents(
        self, items: list[DataItem]
    ) -> list[dict[str, Any]]:
        """Extract semantic intent from each item using all models.

        Returns a list of dicts with keys: item, intent_descriptions,
        consensus_intent, category, affected_area.
        """
        results: list[dict[str, Any]] = []

        for item in items:
            user_prompt = (
                f"Item {item.id}: {item.title}\n\n{item.body}"
            )
            responses = await self.model_pool.query_all(
                system_prompt=self.domain_config.clustering_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=500,
            )

            intent_descriptions: dict[str, str] = {}
            categories: list[str] = []
            areas: list[str] = []

            for resp in responses:
                if resp.model == "error":
                    continue
                try:
                    parsed = resp.parse_json()
                    intent_descriptions[resp.provider] = parsed.get(
                        "intent", ""
                    )
                    categories.append(parsed.get("category", "unknown"))
                    areas.append(
                        parsed.get("affected_area", "unknown")
                    )
                except (json.JSONDecodeError, ValueError):
                    intent_descriptions[resp.provider] = resp.content[:200]

            consensus_intent = self._merge_intents(intent_descriptions)
            category = self._majority_vote(categories)
            area = self._majority_vote(areas)

            results.append(
                {
                    "item": item,
                    "intent_descriptions": intent_descriptions,
                    "consensus_intent": consensus_intent,
                    "category": category,
                    "affected_area": area,
                    "embedding": None,
                }
            )

        return results

    async def _generate_embeddings(
        self, intents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate embeddings for each item's consensus intent."""
        texts = [
            f"{r['consensus_intent']} | {r['category']} | {r['affected_area']}"
            for r in intents
        ]

        if not texts:
            return intents

        embeddings = await self.model_pool.get_embeddings(texts)

        for intent_record, embedding in zip(intents, embeddings):
            intent_record["embedding"] = embedding

        return intents

    def _cluster(
        self,
        intents: list[dict[str, Any]],
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        """Cluster items by semantic intent similarity using DBSCAN."""
        valid = [r for r in intents if r["embedding"] is not None]
        if not valid:
            return []

        embeddings_matrix = np.array([r["embedding"] for r in valid])
        sim_matrix = cosine_similarity(embeddings_matrix)
        np.clip(sim_matrix, 0, 1, out=sim_matrix)
        distance_matrix = 1 - sim_matrix
        eps = 1 - similarity_threshold

        clustering = DBSCAN(
            eps=eps,
            min_samples=2,
            metric="precomputed",
        ).fit(distance_matrix)

        labels = clustering.labels_
        unique_labels = set(labels)
        clusters: list[dict[str, Any]] = []

        for label in sorted(unique_labels):
            member_indices = [
                i for i, lbl in enumerate(labels) if lbl == label
            ]
            members = [valid[i] for i in member_indices]

            if label == -1:
                for member in members:
                    item: DataItem = member["item"]
                    clusters.append(
                        {
                            "cluster_id": f"singleton-{item.id}",
                            "intent_summary": member["consensus_intent"],
                            "category": member["category"],
                            "affected_area": member["affected_area"],
                            "confidence": 0.0,
                            "items": [
                                {
                                    "id": item.id,
                                    "title": item.title,
                                    "intent": member["consensus_intent"],
                                }
                            ],
                        }
                    )
            else:
                if len(member_indices) > 1:
                    sub_sim = sim_matrix[
                        np.ix_(member_indices, member_indices)
                    ]
                    mask = ~np.eye(sub_sim.shape[0], dtype=bool)
                    avg_sim = float(sub_sim[mask].mean())
                else:
                    avg_sim = 1.0

                cats = [m["category"] for m in members]
                area_list = [m["affected_area"] for m in members]

                clusters.append(
                    {
                        "cluster_id": f"cluster-{label}",
                        "intent_summary": members[0]["consensus_intent"],
                        "category": self._majority_vote(cats),
                        "affected_area": self._majority_vote(area_list),
                        "confidence": round(avg_sim, 3),
                        "items": [
                            {
                                "id": m["item"].id,
                                "title": m["item"].title,
                                "intent": m["consensus_intent"],
                            }
                            for m in members
                        ],
                    }
                )

        clusters.sort(key=lambda c: (-len(c["items"]), c["cluster_id"]))
        return clusters

    async def _score_quality(
        self, items: list[DataItem]
    ) -> list[dict[str, Any]]:
        """Score each item on quality dimensions using multi-model consensus."""
        results: list[dict[str, Any]] = []
        dimensions = self.domain_config.scoring_dimensions
        disagreement_threshold = self.domain_config.default_thresholds.get(
            "disagreement", 3.0
        )

        for item in items:
            user_prompt = (
                f"Item {item.id}: {item.title}\n\n{item.body}"
            )
            responses = await self.model_pool.query_all(
                system_prompt=self.domain_config.scoring_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=800,
            )

            model_scores: dict[str, dict[str, float]] = {}
            summaries: dict[str, str] = {}

            for resp in responses:
                if resp.model == "error":
                    continue
                try:
                    parsed = resp.parse_json()
                    scores: dict[str, float] = {}
                    for dim in dimensions:
                        val = parsed.get(dim)
                        if val is not None:
                            scores[dim] = float(max(1, min(10, val)))
                    if scores:
                        model_scores[resp.provider] = scores
                    summaries[resp.provider] = parsed.get("summary", "")
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

            if not model_scores:
                continue

            dim_results: list[dict[str, Any]] = []
            disagreement_reasons: list[str] = []

            for dim in dimensions:
                dim_scores = {
                    provider: s[dim]
                    for provider, s in model_scores.items()
                    if dim in s
                }
                if dim_scores:
                    values = list(dim_scores.values())
                    consensus = sum(values) / len(values)
                    spread = max(values) - min(values)
                    flagged = spread > disagreement_threshold

                    if flagged:
                        disagreement_reasons.append(
                            f"{dim}: spread of {spread:.1f}"
                        )

                    dim_results.append(
                        {
                            "dimension": dim,
                            "scores": dim_scores,
                            "consensus": round(consensus, 2),
                            "disagreement": round(spread, 2),
                            "flagged": flagged,
                        }
                    )

            overall = (
                sum(d["consensus"] for d in dim_results) / len(dim_results)
                if dim_results
                else 0.0
            )

            summary_parts = [s for s in summaries.values() if s]
            merged_summary = (
                summary_parts[0] if summary_parts else "No summary"
            )

            results.append(
                {
                    "item_id": item.id,
                    "title": item.title,
                    "overall_score": round(overall, 2),
                    "summary": merged_summary,
                    "dimensions": dim_results,
                    "needs_human_review": bool(disagreement_reasons),
                    "disagreement_reasons": disagreement_reasons,
                }
            )

        results.sort(key=lambda r: -r["overall_score"])
        return results

    async def _score_alignment(
        self,
        items: list[DataItem],
        context_docs: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Score each item's alignment against context documents."""
        results: list[dict[str, Any]] = []
        recommendation_levels = self.domain_config.recommendation_levels
        reject_threshold = self.domain_config.default_thresholds.get(
            "alignment_reject", 4.0
        )

        docs_text = ""
        for path, content in context_docs.items():
            docs_text += f"\n=== {path} ===\n{content[:5000]}\n"

        for item in items:
            user_prompt = (
                f"CONTEXT DOCUMENTS:\n{docs_text}\n\n---\n\n"
                f"ITEM TO EVALUATE:\n"
                f"Item {item.id}: {item.title}\n\n{item.body}"
            )
            responses = await self.model_pool.query_all(
                system_prompt=self.domain_config.alignment_prompt,
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
                continue

            values = list(scores_by_provider.values())
            consensus_score = sum(values) / len(values)
            spread = max(values) - min(values) if len(values) > 1 else 0
            confidence = max(0, 1 - (spread / 10))

            rec_counts = Counter(recommendations)
            consensus_rec = (
                rec_counts.most_common(1)[0][0] if rec_counts else "REVIEW"
            )

            if consensus_score < reject_threshold and recommendation_levels:
                consensus_rec = recommendation_levels[-1]

            aligned_deduped = list(dict.fromkeys(all_aligned))[:5]
            drift_deduped = list(dict.fromkeys(all_drift))[:5]

            results.append(
                {
                    "item_id": item.id,
                    "title": item.title,
                    "alignment_score": round(consensus_score, 2),
                    "scores_by_provider": scores_by_provider,
                    "aligned_aspects": aligned_deduped,
                    "drift_concerns": drift_deduped,
                    "recommendation": consensus_rec,
                    "rationale": rationales[0] if rationales else "",
                    "confidence": round(confidence, 3),
                }
            )

        results.sort(key=lambda r: r["alignment_score"])
        return results

    # ------------------------------------------------------------------
    # Consensus helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_intents(descriptions: dict[str, str]) -> str:
        """Merge intent descriptions by picking the longest one."""
        if not descriptions:
            return "Unable to determine intent"
        valid = {k: v for k, v in descriptions.items() if v and v.strip()}
        if not valid:
            return "Unable to determine intent"
        return max(valid.values(), key=len)

    @staticmethod
    def _majority_vote(items: list[str]) -> str:
        """Simple majority vote from a list of strings."""
        if not items:
            return "unknown"
        counts = Counter(items)
        return counts.most_common(1)[0][0]
