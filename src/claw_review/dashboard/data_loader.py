"""Load and transform claw-review JSON report data for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path


class DataLoader:
    """Load, merge, filter, and sort claw-review analysis reports.

    Supports both the standard JSON report format (with ``repo`` key) and
    the newer AnalysisResult format (with ``domain`` key).
    """

    def __init__(self) -> None:
        """Initialize with empty data."""
        self._data: dict = {
            "repo": "",
            "generated_at": "",
            "providers": [],
            "summary": {
                "total_prs": 0,
                "duplicate_clusters": 0,
                "prs_in_duplicates": 0,
                "unique_prs": 0,
                "flagged_for_drift": 0,
            },
            "clusters": [],
            "quality_scores": [],
            "alignment_scores": [],
            "cost": None,
        }

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_report(self, path: str) -> dict:
        """Load a single JSON report file and validate its structure.

        Args:
            path: Filesystem path to the JSON report.

        Returns:
            The normalised report dict.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        raw = Path(path).read_text()
        data = json.loads(raw)
        self._data = self._normalise(data)
        return self._data

    def load_multiple_reports(self, paths: list[str]) -> dict:
        """Load and merge multiple JSON reports.

        Later reports override earlier ones for the same PR number.

        Args:
            paths: List of filesystem paths to JSON reports.

        Returns:
            Merged and normalised report dict.
        """
        if not paths:
            return self._data

        merged_clusters: list[dict] = []
        merged_quality: list[dict] = []
        merged_alignment: list[dict] = []
        providers: set[str] = set()
        repo = ""
        generated_at = ""
        cost: dict | None = None

        seen_quality: dict[int, int] = {}
        seen_alignment: dict[int, int] = {}
        seen_cluster_prs: set[int] = set()

        for p in paths:
            data = self._normalise(json.loads(Path(p).read_text()))

            if data.get("repo"):
                repo = data["repo"]
            ts = data.get("generated_at", "")
            if ts > generated_at:
                generated_at = ts

            for prov in data.get("providers", []):
                providers.add(prov)

            if data.get("cost") is not None:
                cost = data["cost"]

            for qs in data.get("quality_scores", []):
                pr_num = qs.get("pr_number")
                if pr_num is not None and pr_num in seen_quality:
                    merged_quality[seen_quality[pr_num]] = qs
                else:
                    if pr_num is not None:
                        seen_quality[pr_num] = len(merged_quality)
                    merged_quality.append(qs)

            for als in data.get("alignment_scores", []):
                pr_num = als.get("pr_number")
                if pr_num is not None and pr_num in seen_alignment:
                    merged_alignment[seen_alignment[pr_num]] = als
                else:
                    if pr_num is not None:
                        seen_alignment[pr_num] = len(merged_alignment)
                    merged_alignment.append(als)

            for cluster in data.get("clusters", []):
                new_prs = [
                    pr
                    for pr in cluster.get("prs", [])
                    if pr.get("number") not in seen_cluster_prs
                ]
                if new_prs:
                    merged_cluster = dict(cluster)
                    merged_cluster["prs"] = new_prs
                    merged_clusters.append(merged_cluster)
                    for pr in new_prs:
                        nr = pr.get("number")
                        if nr is not None:
                            seen_cluster_prs.add(nr)

        dup_clusters = [c for c in merged_clusters if len(c.get("prs", [])) > 1]
        total_prs = sum(len(c.get("prs", [])) for c in merged_clusters)
        dup_prs = sum(len(c["prs"]) for c in dup_clusters)

        self._data = {
            "repo": repo,
            "generated_at": generated_at,
            "providers": sorted(providers),
            "summary": {
                "total_prs": total_prs,
                "duplicate_clusters": len(dup_clusters),
                "prs_in_duplicates": dup_prs,
                "unique_prs": total_prs - dup_prs,
                "flagged_for_drift": sum(
                    1
                    for a in merged_alignment
                    if a.get("alignment_score", 10) < 5
                ),
            },
            "clusters": merged_clusters,
            "quality_scores": merged_quality,
            "alignment_scores": merged_alignment,
            "cost": cost,
        }
        return self._data

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return summary stats from the currently loaded data.

        Returns:
            Dict with total_prs, duplicate_clusters, flagged_for_drift,
            avg_quality_score, and total_cost.
        """
        scores = self._data.get("quality_scores", [])
        avg_score = 0.0
        if scores:
            total = sum(s.get("overall_score", 0) for s in scores)
            avg_score = round(total / len(scores), 2)

        cost_data = self._data.get("cost")
        total_cost = cost_data.get("total_cost", 0) if cost_data else 0

        summary = dict(self._data.get("summary", {}))
        summary["avg_quality_score"] = avg_score
        summary["total_cost"] = total_cost
        return summary

    def filter_by_recommendation(self, recommendation: str) -> list[dict]:
        """Filter alignment scores by recommendation level.

        Args:
            recommendation: One of MERGE, REVIEW, DISCUSS, CLOSE (case-insensitive).

        Returns:
            List of matching alignment score dicts.
        """
        rec = recommendation.upper()
        return [
            a
            for a in self._data.get("alignment_scores", [])
            if a.get("recommendation", "").upper() == rec
        ]

    def filter_by_score_range(
        self, min_score: float, max_score: float
    ) -> list[dict]:
        """Filter quality scores within an inclusive range.

        Args:
            min_score: Minimum overall score.
            max_score: Maximum overall score.

        Returns:
            List of matching quality score dicts.
        """
        return [
            q
            for q in self._data.get("quality_scores", [])
            if min_score <= q.get("overall_score", 0) <= max_score
        ]

    def filter_by_category(self, category: str) -> list[dict]:
        """Filter clusters by category (case-insensitive).

        Args:
            category: Cluster category string (e.g. ``bugfix``, ``feature``).

        Returns:
            List of matching cluster dicts.
        """
        cat = category.lower()
        return [
            c
            for c in self._data.get("clusters", [])
            if c.get("category", "").lower() == cat
        ]

    def search(self, query: str) -> list[dict]:
        """Search all items by title (case-insensitive substring match).

        Searches quality scores, alignment scores, and cluster PRs.

        Args:
            query: Substring to match against titles.

        Returns:
            List of dicts with ``type``, ``pr_number``, ``title``, and the
            original item as ``data``.
        """
        q = query.lower()
        results: list[dict] = []

        for qs in self._data.get("quality_scores", []):
            if q in qs.get("title", "").lower():
                results.append(
                    {
                        "type": "quality",
                        "pr_number": qs.get("pr_number"),
                        "title": qs.get("title", ""),
                        "data": qs,
                    }
                )

        for als in self._data.get("alignment_scores", []):
            if q in als.get("title", "").lower():
                results.append(
                    {
                        "type": "alignment",
                        "pr_number": als.get("pr_number"),
                        "title": als.get("title", ""),
                        "data": als,
                    }
                )

        for cluster in self._data.get("clusters", []):
            for pr in cluster.get("prs", []):
                if q in pr.get("title", "").lower():
                    results.append(
                        {
                            "type": "cluster_pr",
                            "pr_number": pr.get("number"),
                            "title": pr.get("title", ""),
                            "data": pr,
                        }
                    )

        return results

    def sort_items(
        self, key: str = "score", ascending: bool = False
    ) -> list[dict]:
        """Sort quality-score items by the given key.

        Args:
            key: Sort key.  ``score`` sorts by overall_score, ``pr_number``
                sorts numerically by PR number, ``title`` sorts alphabetically.
            ascending: If True sort low-to-high, otherwise high-to-low.

        Returns:
            Sorted list of quality score dicts.
        """
        items = list(self._data.get("quality_scores", []))
        key_map: dict[str, str] = {
            "score": "overall_score",
            "pr_number": "pr_number",
            "title": "title",
        }
        sort_field = key_map.get(key, "overall_score")
        items.sort(
            key=lambda x: x.get(sort_field, 0) if sort_field != "title" else x.get(sort_field, ""),
            reverse=not ascending,
        )
        return items

    def get_cost_breakdown(self) -> dict:
        """Return cost breakdown by model if available.

        Returns:
            Cost dict from the report, or an empty dict with zeroed totals.
        """
        cost = self._data.get("cost")
        if cost:
            return dict(cost)
        return {
            "total_cost": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
            "models": {},
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict:
        """Return the full normalised report data."""
        return self._data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(data: dict) -> dict:
        """Normalise a report dict so both old and new formats look the same.

        The old format uses ``repo``; the newer AnalysisResult format may use
        ``domain``.  Missing keys are filled with sensible defaults.
        """
        repo = data.get("repo", data.get("domain", ""))
        generated_at = data.get("generated_at", data.get("timestamp", ""))
        providers = data.get("providers", [])
        clusters = data.get("clusters", [])
        quality_scores = data.get("quality_scores", [])
        alignment_scores = data.get("alignment_scores", [])
        cost = data.get("cost", None)

        summary = data.get("summary", {})
        dup_clusters = [c for c in clusters if len(c.get("prs", [])) > 1]
        total_prs = summary.get(
            "total_prs",
            sum(len(c.get("prs", [])) for c in clusters),
        )
        dup_prs = sum(len(c["prs"]) for c in dup_clusters)

        return {
            "repo": repo,
            "generated_at": generated_at,
            "providers": providers,
            "summary": {
                "total_prs": total_prs,
                "duplicate_clusters": summary.get(
                    "duplicate_clusters", len(dup_clusters)
                ),
                "prs_in_duplicates": summary.get("prs_in_duplicates", dup_prs),
                "unique_prs": summary.get("unique_prs", total_prs - dup_prs),
                "flagged_for_drift": summary.get(
                    "flagged_for_drift",
                    sum(
                        1
                        for a in alignment_scores
                        if a.get("alignment_score", 10) < 5
                    ),
                ),
            },
            "clusters": clusters,
            "quality_scores": quality_scores,
            "alignment_scores": alignment_scores,
            "cost": cost,
        }
