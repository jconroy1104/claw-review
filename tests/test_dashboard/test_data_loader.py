"""Tests for the DataLoader class."""

from __future__ import annotations

import json

import pytest

from claw_review.dashboard.data_loader import DataLoader


SAMPLE_REPORT = {
    "repo": "test/repo",
    "generated_at": "2026-02-16T12:00:00Z",
    "providers": ["model-a", "model-b"],
    "summary": {
        "total_prs": 5,
        "duplicate_clusters": 1,
        "prs_in_duplicates": 2,
        "unique_prs": 3,
        "flagged_for_drift": 1,
    },
    "clusters": [
        {
            "intent_summary": "Fix auth timeout",
            "category": "bugfix",
            "affected_area": "auth",
            "confidence": 0.85,
            "prs": [
                {"number": 1, "title": "Fix auth timeout issue", "intent": "Fix auth timeout", "quality_score": 8.5},
                {"number": 2, "title": "Auth timeout fix v2", "intent": "Fix auth timeout", "quality_score": 7.2},
            ],
        },
        {
            "intent_summary": "Add dark mode",
            "category": "feature",
            "affected_area": "ui",
            "confidence": 0.0,
            "prs": [{"number": 3, "title": "Add dark mode support", "intent": "Add dark mode"}],
        },
    ],
    "quality_scores": [
        {
            "pr_number": 1,
            "title": "Fix auth timeout issue",
            "overall_score": 8.5,
            "summary": "Good fix",
            "dimensions": [
                {"dimension": "code_quality", "scores": {"model-a": 8, "model-b": 9}, "consensus": 8.5, "disagreement": 1.0, "flagged": False},
            ],
            "needs_human_review": False,
            "disagreement_reasons": [],
        },
        {
            "pr_number": 2,
            "title": "Auth timeout fix v2",
            "overall_score": 6.0,
            "summary": "Decent fix",
            "dimensions": [],
            "needs_human_review": True,
            "disagreement_reasons": ["Divergent scores"],
        },
    ],
    "alignment_scores": [
        {
            "pr_number": 1,
            "title": "Fix auth timeout issue",
            "alignment_score": 8.4,
            "scores_by_provider": {"model-a": 8.0, "model-b": 8.8},
            "aligned_aspects": ["Fixes known issue"],
            "drift_concerns": [],
            "recommendation": "MERGE",
            "rationale": "Addresses known auth bug",
            "confidence": 0.92,
        },
        {
            "pr_number": 3,
            "title": "Add dark mode support",
            "alignment_score": 3.0,
            "scores_by_provider": {"model-a": 3.0, "model-b": 3.0},
            "aligned_aspects": [],
            "drift_concerns": ["Not in roadmap"],
            "recommendation": "CLOSE",
            "rationale": "Drifts from project vision",
            "confidence": 0.88,
        },
    ],
}

SAMPLE_REPORT_2 = {
    "repo": "test/repo",
    "generated_at": "2026-02-17T12:00:00Z",
    "providers": ["model-b", "model-c"],
    "summary": {
        "total_prs": 1,
        "duplicate_clusters": 0,
        "prs_in_duplicates": 0,
        "unique_prs": 1,
        "flagged_for_drift": 0,
    },
    "clusters": [
        {
            "intent_summary": "Upgrade deps",
            "category": "chore",
            "affected_area": "build",
            "confidence": 0.5,
            "prs": [{"number": 4, "title": "Upgrade deps", "intent": "Upgrade deps"}],
        },
    ],
    "quality_scores": [
        {
            "pr_number": 4,
            "title": "Upgrade deps",
            "overall_score": 9.0,
            "summary": "Clean upgrade",
            "dimensions": [],
            "needs_human_review": False,
            "disagreement_reasons": [],
        },
    ],
    "alignment_scores": [
        {
            "pr_number": 4,
            "title": "Upgrade deps",
            "alignment_score": 9.0,
            "scores_by_provider": {"model-b": 9.0, "model-c": 9.0},
            "aligned_aspects": ["Routine maintenance"],
            "drift_concerns": [],
            "recommendation": "MERGE",
            "rationale": "Routine dep upgrade",
            "confidence": 0.95,
        },
    ],
}


def _write_report(tmp_path, data, name="report.json"):
    """Write a report dict to a temp file and return the path string."""
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


class TestLoadReport:
    """Tests for DataLoader.load_report."""

    def test_valid_report_loads(self, tmp_path: object) -> None:
        """Loading a valid JSON report populates data correctly."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        result = loader.load_report(path)

        assert result["repo"] == "test/repo"
        assert len(result["clusters"]) == 2
        assert len(result["quality_scores"]) == 2
        assert len(result["alignment_scores"]) == 2

    def test_missing_file_raises(self) -> None:
        """A non-existent path raises FileNotFoundError."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_report("/nonexistent/path/report.json")

    def test_malformed_json_raises(self, tmp_path: object) -> None:
        """Invalid JSON content raises json.JSONDecodeError."""
        p = tmp_path / "bad.json"
        p.write_text("{not valid json!!!")
        loader = DataLoader()
        with pytest.raises(json.JSONDecodeError):
            loader.load_report(str(p))

    def test_domain_key_normalised(self, tmp_path: object) -> None:
        """The newer 'domain' key is normalised to 'repo'."""
        data = {"domain": "cyber/threat", "clusters": [], "quality_scores": [], "alignment_scores": []}
        path = _write_report(tmp_path, data, "domain.json")
        loader = DataLoader()
        result = loader.load_report(path)
        assert result["repo"] == "cyber/threat"

    def test_empty_report(self, tmp_path: object) -> None:
        """An empty report (zero items) loads without error."""
        data = {"repo": "empty/repo", "clusters": [], "quality_scores": [], "alignment_scores": []}
        path = _write_report(tmp_path, data, "empty.json")
        loader = DataLoader()
        result = loader.load_report(path)
        assert result["summary"]["total_prs"] == 0
        assert result["clusters"] == []


class TestLoadMultipleReports:
    """Tests for DataLoader.load_multiple_reports."""

    def test_merge_two_reports(self, tmp_path: object) -> None:
        """Merging two reports combines clusters and scores."""
        p1 = _write_report(tmp_path, SAMPLE_REPORT, "r1.json")
        p2 = _write_report(tmp_path, SAMPLE_REPORT_2, "r2.json")
        loader = DataLoader()
        result = loader.load_multiple_reports([p1, p2])

        assert len(result["clusters"]) == 3
        assert len(result["quality_scores"]) == 3
        assert len(result["alignment_scores"]) == 3
        assert "model-c" in result["providers"]

    def test_empty_list(self) -> None:
        """Passing empty list returns default empty data."""
        loader = DataLoader()
        result = loader.load_multiple_reports([])
        assert result["repo"] == ""
        assert result["summary"]["total_prs"] == 0


class TestGetSummary:
    """Tests for DataLoader.get_summary."""

    def test_correct_counts(self, tmp_path: object) -> None:
        """Summary contains correct counts and averages."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        summary = loader.get_summary()

        # total_prs comes from the report's summary field (5)
        assert summary["total_prs"] == 5
        assert summary["duplicate_clusters"] == 1
        assert summary["flagged_for_drift"] == 1
        # avg of 8.5 and 6.0 = 7.25
        assert summary["avg_quality_score"] == 7.25
        assert summary["total_cost"] == 0  # no cost data in sample


class TestFilterByRecommendation:
    """Tests for DataLoader.filter_by_recommendation."""

    def test_filter_merge(self, tmp_path: object) -> None:
        """Filtering by MERGE returns only MERGE items."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.filter_by_recommendation("MERGE")

        assert len(results) == 1
        assert results[0]["pr_number"] == 1

    def test_filter_close(self, tmp_path: object) -> None:
        """Filtering by CLOSE returns only CLOSE items."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.filter_by_recommendation("close")

        assert len(results) == 1
        assert results[0]["pr_number"] == 3

    def test_filter_nonexistent(self, tmp_path: object) -> None:
        """Filtering by a recommendation with no matches returns empty list."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.filter_by_recommendation("DISCUSS")

        assert results == []


class TestFilterByScoreRange:
    """Tests for DataLoader.filter_by_score_range."""

    def test_within_range(self, tmp_path: object) -> None:
        """Items within the score range are returned."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.filter_by_score_range(7.0, 10.0)

        assert len(results) == 1
        assert results[0]["pr_number"] == 1

    def test_full_range(self, tmp_path: object) -> None:
        """Range 0-10 returns all items."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.filter_by_score_range(0.0, 10.0)

        assert len(results) == 2


class TestSearch:
    """Tests for DataLoader.search."""

    def test_case_insensitive(self, tmp_path: object) -> None:
        """Search is case-insensitive."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.search("AUTH")

        # Should match: quality PR 1, quality PR 2, alignment PR 1, cluster PRs 1 and 2
        titles = [r["title"] for r in results]
        assert any("auth" in t.lower() for t in titles)
        assert len(results) >= 2

    def test_no_match(self, tmp_path: object) -> None:
        """Searching for a string not in any title returns empty list."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.search("xyznotfound")

        assert results == []


class TestSortItems:
    """Tests for DataLoader.sort_items."""

    def test_descending_by_score(self, tmp_path: object) -> None:
        """Default sort is descending by score."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        items = loader.sort_items()

        assert items[0]["overall_score"] >= items[-1]["overall_score"]

    def test_ascending_by_score(self, tmp_path: object) -> None:
        """Ascending sort puts lowest first."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        items = loader.sort_items(ascending=True)

        assert items[0]["overall_score"] <= items[-1]["overall_score"]

    def test_sort_by_pr_number(self, tmp_path: object) -> None:
        """Sorting by pr_number works."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        items = loader.sort_items(key="pr_number", ascending=True)

        assert items[0]["pr_number"] < items[-1]["pr_number"]


class TestGetCostBreakdown:
    """Tests for DataLoader.get_cost_breakdown."""

    def test_no_cost_data(self, tmp_path: object) -> None:
        """When report has no cost data, returns zeroed dict."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        cost = loader.get_cost_breakdown()

        assert cost["total_cost"] == 0
        assert cost["models"] == {}

    def test_with_cost_data(self, tmp_path: object) -> None:
        """When report has cost data, it is returned."""
        data = dict(SAMPLE_REPORT)
        data["cost"] = {
            "total_cost": 0.05,
            "total_input_tokens": 10000,
            "total_output_tokens": 2000,
            "total_requests": 6,
            "models": {"model-a": {"cost": 0.03, "input_tokens": 6000, "output_tokens": 1200, "request_count": 3}},
        }
        path = _write_report(tmp_path, data, "cost.json")
        loader = DataLoader()
        loader.load_report(path)
        cost = loader.get_cost_breakdown()

        assert cost["total_cost"] == 0.05
        assert "model-a" in cost["models"]


class TestFilterByCategory:
    """Tests for DataLoader.filter_by_category."""

    def test_filter_bugfix(self, tmp_path: object) -> None:
        """Filtering by bugfix returns bugfix clusters."""
        path = _write_report(tmp_path, SAMPLE_REPORT)
        loader = DataLoader()
        loader.load_report(path)
        results = loader.filter_by_category("bugfix")

        assert len(results) == 1
        assert results[0]["category"] == "bugfix"
