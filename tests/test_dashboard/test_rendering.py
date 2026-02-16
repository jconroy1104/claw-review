"""Tests for static HTML rendering output."""

from __future__ import annotations

import json
import re

from claw_review.dashboard.app import DashboardApp
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
    ],
}


def _write_report(tmp_path, data, name="report.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def _generate_html(tmp_path, data=None):
    """Generate static HTML and return the content string."""
    report_data = data or SAMPLE_REPORT
    path = _write_report(tmp_path, report_data)
    loader = DataLoader()
    loader.load_report(path)
    app = DashboardApp(loader)
    output = str(tmp_path / "dashboard.html")
    app.generate_static(output)
    with open(output) as f:
        return f.read()


class TestHTMLStructure:
    """Tests for valid HTML structure."""

    def test_has_html_head_body(self, tmp_path) -> None:
        """Generated HTML has <html>, <head>, and <body> tags."""
        html = _generate_html(tmp_path)
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

    def test_has_doctype(self, tmp_path) -> None:
        """Generated HTML starts with DOCTYPE."""
        html = _generate_html(tmp_path)
        assert html.strip().startswith("<!DOCTYPE html>")


class TestHTMLContent:
    """Tests for expected content in the generated HTML."""

    def test_contains_javascript_filtering(self, tmp_path) -> None:
        """Generated HTML includes JavaScript for client-side filtering."""
        html = _generate_html(tmp_path)
        assert "<script>" in html
        assert "applyFilters" in html
        assert "search-input" in html

    def test_contains_dark_theme_css(self, tmp_path) -> None:
        """Generated HTML includes dark theme CSS variables."""
        html = _generate_html(tmp_path)
        assert "#0d1117" in html  # background colour
        assert "#c9d1d9" in html  # text colour
        assert "#161b22" in html  # surface colour
        assert "#30363d" in html  # border colour

    def test_data_json_is_valid(self, tmp_path) -> None:
        """The embedded data_json variable is valid JSON."""
        html = _generate_html(tmp_path)
        # Extract the DATA = {...}; assignment from the script
        match = re.search(r"var DATA = ({.*?});\s*\n", html, re.DOTALL)
        assert match is not None, "Could not find embedded DATA JSON"
        data = json.loads(match.group(1))
        assert data["repo"] == "test/repo"
        assert "clusters" in data

    def test_contains_all_sections(self, tmp_path) -> None:
        """Generated HTML contains summary, clusters, quality, alignment, and cost sections."""
        html = _generate_html(tmp_path)
        assert 'id="panel-overview"' in html
        assert 'id="panel-clusters"' in html
        assert 'id="panel-quality"' in html
        assert 'id="panel-alignment"' in html
        assert 'id="panel-cost"' in html

    def test_contains_tab_buttons(self, tmp_path) -> None:
        """Generated HTML contains tab navigation buttons."""
        html = _generate_html(tmp_path)
        assert 'data-tab="overview"' in html
        assert 'data-tab="clusters"' in html
        assert 'data-tab="quality"' in html
        assert 'data-tab="alignment"' in html
        assert 'data-tab="cost"' in html

    def test_contains_summary_cards(self, tmp_path) -> None:
        """Generated HTML includes summary statistic cards."""
        html = _generate_html(tmp_path)
        assert "Total PRs" in html
        assert "Duplicate Clusters" in html
        assert "Drift Flags" in html
        assert "Avg Quality Score" in html
        assert "Total Cost" in html


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self, tmp_path) -> None:
        """Generating HTML with zero items succeeds."""
        data = {
            "repo": "empty/repo",
            "generated_at": "",
            "providers": [],
            "clusters": [],
            "quality_scores": [],
            "alignment_scores": [],
        }
        html = _generate_html(tmp_path, data)
        assert "<!DOCTYPE html>" in html
        assert "ClawReview Dashboard" in html

    def test_large_dataset(self, tmp_path) -> None:
        """Generating HTML with 100+ items succeeds without error."""
        clusters = []
        quality_scores = []
        alignment_scores = []

        for i in range(120):
            clusters.append({
                "intent_summary": f"Intent {i}",
                "category": "feature" if i % 2 == 0 else "bugfix",
                "affected_area": f"area-{i % 5}",
                "confidence": 0.8,
                "prs": [{"number": i, "title": f"PR {i}", "intent": f"Intent {i}", "quality_score": 5.0 + (i % 5)}],
            })
            quality_scores.append({
                "pr_number": i,
                "title": f"PR {i}",
                "overall_score": 5.0 + (i % 5),
                "summary": f"Summary {i}",
                "dimensions": [],
                "needs_human_review": i % 10 == 0,
                "disagreement_reasons": [],
            })
            alignment_scores.append({
                "pr_number": i,
                "title": f"PR {i}",
                "alignment_score": 6.0 + (i % 4),
                "scores_by_provider": {},
                "aligned_aspects": [],
                "drift_concerns": [],
                "recommendation": ["MERGE", "REVIEW", "DISCUSS", "CLOSE"][i % 4],
                "rationale": f"Rationale {i}",
                "confidence": 0.9,
            })

        data = {
            "repo": "large/repo",
            "generated_at": "2026-02-16T12:00:00Z",
            "providers": ["model-a"],
            "clusters": clusters,
            "quality_scores": quality_scores,
            "alignment_scores": alignment_scores,
        }
        html = _generate_html(tmp_path, data)
        assert "<!DOCTYPE html>" in html
        assert "PR 119" in html  # last item is present
