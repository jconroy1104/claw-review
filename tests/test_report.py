"""Tests for claw_review.report — HTML and JSON report generation."""

import json
from pathlib import Path

from claw_review.report import generate_report, generate_json_report, merge_reports

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_clusters() -> list[dict]:
    return json.loads((FIXTURES / "sample_clusters.json").read_text())


def _load_scores() -> list[dict]:
    return json.loads((FIXTURES / "sample_scores.json").read_text())


def _sample_alignment_scores() -> list[dict]:
    """Build sample alignment scores for testing."""
    return [
        {
            "pr_number": 102,
            "pr_title": "Fix WS reconnect on timeout",
            "pr_author": "bob",
            "pr_url": "https://github.com/test/repo/pull/102",
            "alignment_score": 3.5,
            "recommendation": "CLOSE",
            "rationale": "Does not align with project architecture",
            "drift_concerns": ["Adds REST endpoint to gateway"],
            "aligned_aspects": [],
            "confidence": 0.7,
        },
        {
            "pr_number": 101,
            "pr_title": "Fix WebSocket reconnection",
            "pr_author": "alice",
            "pr_url": "https://github.com/test/repo/pull/101",
            "alignment_score": 8.5,
            "recommendation": "MERGE",
            "rationale": "Addresses a documented WebSocket issue",
            "drift_concerns": [],
            "aligned_aspects": ["Protocol compliance"],
            "confidence": 0.95,
        },
    ]


def _sample_providers() -> list[str]:
    return ["anthropic/claude-sonnet-4", "openai/gpt-4o", "google/gemini-2.0-flash-001"]


# ===================================================================
# generate_report (HTML)
# ===================================================================


class TestGenerateReport:
    def test_produces_valid_html_file(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        result = generate_report(
            repo="owner/repo",
            clusters=_load_clusters(),
            quality_scores=_load_scores(),
            alignment_scores=_sample_alignment_scores(),
            providers=_sample_providers(),
            output_path=str(output),
        )
        assert output.exists()
        html = output.read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert result == str(output)

    def test_html_contains_repo_name(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        generate_report(
            repo="acme/widgets",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=_sample_providers(),
            output_path=str(output),
        )
        html = output.read_text()
        assert "acme/widgets" in html

    def test_html_contains_timestamp(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        generate_report(
            repo="owner/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=_sample_providers(),
            output_path=str(output),
        )
        html = output.read_text()
        assert "UTC" in html

    def test_html_contains_provider_names(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        providers = ["anthropic/claude-sonnet-4", "openai/gpt-4o"]
        generate_report(
            repo="owner/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=providers,
            output_path=str(output),
        )
        html = output.read_text()
        assert "anthropic/claude-sonnet-4" in html
        assert "openai/gpt-4o" in html

    def test_html_has_duplicate_cluster_section(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        clusters = _load_clusters()
        generate_report(
            repo="owner/repo",
            clusters=clusters,
            quality_scores=_load_scores(),
            alignment_scores=[],
            providers=_sample_providers(),
            output_path=str(output),
        )
        html = output.read_text()
        # The template contains "Duplicate Clusters" heading
        assert "Duplicate Clusters" in html
        # Should show the cluster intent summaries
        assert "Fix WebSocket reconnection" in html
        assert "dark mode" in html.lower() or "dark theme" in html.lower()

    def test_html_has_vision_alignment_flagged_section(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        alignment = _sample_alignment_scores()
        generate_report(
            repo="owner/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=alignment,
            providers=_sample_providers(),
            output_path=str(output),
        )
        html = output.read_text()
        assert "Vision Alignment" in html
        # The low-scoring PR (3.5) should appear in flagged section
        assert "CLOSE" in html
        assert "Adds REST endpoint to gateway" in html

    def test_html_has_top_quality_section(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        scores = _load_scores()
        generate_report(
            repo="owner/repo",
            clusters=_load_clusters(),
            quality_scores=scores,
            alignment_scores=[],
            providers=_sample_providers(),
            output_path=str(output),
        )
        html = output.read_text()
        assert "Top Quality" in html
        # Scores >= 7 should appear: 8.2, 7.8, 8.5
        assert "8.2" in html or "8.5" in html

    def test_empty_data_produces_valid_html(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        generate_report(
            repo="owner/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=[],
            output_path=str(output),
        )
        html = output.read_text()
        assert "<!DOCTYPE html>" in html
        # Shows "no duplicates" message
        assert "No duplicate clusters" in html or "all PRs appear to address unique" in html.lower()

    def test_custom_output_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "subdir" / "custom_report.html"
        custom.parent.mkdir(parents=True, exist_ok=True)
        result = generate_report(
            repo="owner/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=[],
            output_path=str(custom),
        )
        assert custom.exists()
        assert result == str(custom)

    def test_pr_count_reflects_all_cluster_prs(self, tmp_path: Path) -> None:
        output = tmp_path / "report.html"
        clusters = _load_clusters()
        generate_report(
            repo="owner/repo",
            clusters=clusters,
            quality_scores=[],
            alignment_scores=[],
            providers=[],
            output_path=str(output),
        )
        html = output.read_text()
        # Total PRs across all clusters: 3 + 2 + 1 + 1 = 7
        assert ">7<" in html


# ===================================================================
# generate_json_report
# ===================================================================


class TestGenerateJsonReport:
    def test_produces_valid_json_file(self, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        result = generate_json_report(
            repo="owner/repo",
            clusters=_load_clusters(),
            quality_scores=_load_scores(),
            alignment_scores=_sample_alignment_scores(),
            providers=_sample_providers(),
            output_path=str(output),
        )
        assert output.exists()
        data = json.loads(output.read_text())
        assert isinstance(data, dict)
        assert result == str(output)

    def test_json_has_expected_keys(self, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        generate_json_report(
            repo="owner/repo",
            clusters=_load_clusters(),
            quality_scores=_load_scores(),
            alignment_scores=_sample_alignment_scores(),
            providers=_sample_providers(),
            output_path=str(output),
        )
        data = json.loads(output.read_text())

        expected_keys = {"repo", "timestamp", "providers", "summary",
                         "clusters", "quality_scores", "alignment_scores"}
        assert expected_keys.issubset(data.keys())

    def test_json_summary_counts_correct(self, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        clusters = _load_clusters()
        alignment = _sample_alignment_scores()

        generate_json_report(
            repo="owner/repo",
            clusters=clusters,
            quality_scores=_load_scores(),
            alignment_scores=alignment,
            providers=_sample_providers(),
            output_path=str(output),
        )
        data = json.loads(output.read_text())
        summary = data["summary"]

        # Total PRs: 3 + 2 + 1 + 1 = 7
        assert summary["total_prs"] == 7
        # Duplicate clusters (>1 PR): cluster-0 (3 PRs), cluster-1 (2 PRs) = 2
        assert summary["duplicate_clusters"] == 2
        # Duplicate PRs: 3 + 2 = 5
        assert summary["duplicate_prs"] == 5
        # Flagged for drift (score < 5): PR 102 with 3.5 = 1
        assert summary["flagged_for_drift"] == 1

    def test_json_empty_data_valid(self, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        generate_json_report(
            repo="empty/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=[],
            output_path=str(output),
        )
        data = json.loads(output.read_text())

        assert data["repo"] == "empty/repo"
        assert data["summary"]["total_prs"] == 0
        assert data["summary"]["duplicate_clusters"] == 0
        assert data["summary"]["duplicate_prs"] == 0
        assert data["summary"]["flagged_for_drift"] == 0

    def test_json_contains_repo_name(self, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        generate_json_report(
            repo="acme/widgets",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=[],
            output_path=str(output),
        )
        data = json.loads(output.read_text())
        assert data["repo"] == "acme/widgets"

    def test_json_contains_timestamp(self, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        generate_json_report(
            repo="owner/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=[],
            output_path=str(output),
        )
        data = json.loads(output.read_text())
        assert "timestamp" in data
        assert len(data["timestamp"]) > 0

    def test_json_contains_providers(self, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        providers = ["model-a", "model-b"]
        generate_json_report(
            repo="owner/repo",
            clusters=[],
            quality_scores=[],
            alignment_scores=[],
            providers=providers,
            output_path=str(output),
        )
        data = json.loads(output.read_text())
        assert data["providers"] == providers


# ===================================================================
# merge_reports
# ===================================================================


def _write_report(path: Path, report: dict) -> str:
    """Helper to write a report dict to a JSON file."""
    path.write_text(json.dumps(report, indent=2))
    return str(path)


class TestMergeReports:
    """Tests for merge_reports."""

    def test_merge_two_reports(self, tmp_path: Path) -> None:
        report1 = {
            "repo": "owner/repo",
            "timestamp": "2025-06-01T00:00:00Z",
            "providers": ["model/a", "model/b"],
            "clusters": [
                {
                    "cluster_id": "c-0",
                    "prs": [{"number": 1, "title": "PR1"}, {"number": 2, "title": "PR2"}],
                }
            ],
            "quality_scores": [{"pr_number": 1, "overall_score": 7.0}],
            "alignment_scores": [{"pr_number": 1, "alignment_score": 8.0}],
        }
        report2 = {
            "repo": "owner/repo",
            "timestamp": "2025-06-02T00:00:00Z",
            "providers": ["model/b", "model/c"],
            "clusters": [
                {
                    "cluster_id": "c-1",
                    "prs": [{"number": 3, "title": "PR3"}],
                }
            ],
            "quality_scores": [{"pr_number": 3, "overall_score": 9.0}],
            "alignment_scores": [{"pr_number": 3, "alignment_score": 6.0}],
        }

        f1 = _write_report(tmp_path / "r1.json", report1)
        f2 = _write_report(tmp_path / "r2.json", report2)

        merged = merge_reports([f1, f2])

        assert merged["repo"] == "owner/repo"
        assert merged["timestamp"] == "2025-06-02T00:00:00Z"
        assert set(merged["providers"]) == {"model/a", "model/b", "model/c"}
        assert len(merged["clusters"]) == 2
        assert len(merged["quality_scores"]) == 2
        assert len(merged["alignment_scores"]) == 2
        # Summary computed correctly
        total_prs = sum(len(c["prs"]) for c in merged["clusters"])
        assert total_prs == 3
        assert merged["summary"]["total_prs"] == 3

    def test_merge_deduplicates_prs(self, tmp_path: Path) -> None:
        report1 = {
            "repo": "owner/repo",
            "timestamp": "2025-06-01T00:00:00Z",
            "providers": ["model/a"],
            "clusters": [
                {"cluster_id": "c-0", "prs": [{"number": 1, "title": "PR1"}]},
            ],
            "quality_scores": [{"pr_number": 1, "overall_score": 5.0}],
            "alignment_scores": [{"pr_number": 1, "alignment_score": 4.0}],
        }
        report2 = {
            "repo": "owner/repo",
            "timestamp": "2025-06-02T00:00:00Z",
            "providers": ["model/a"],
            "clusters": [
                {"cluster_id": "c-0", "prs": [{"number": 1, "title": "PR1-updated"}]},
            ],
            "quality_scores": [{"pr_number": 1, "overall_score": 8.0}],
            "alignment_scores": [{"pr_number": 1, "alignment_score": 9.0}],
        }

        f1 = _write_report(tmp_path / "r1.json", report1)
        f2 = _write_report(tmp_path / "r2.json", report2)

        merged = merge_reports([f1, f2])

        # PR 1 should appear only once in clusters (from first report, since it was seen first)
        all_pr_numbers = []
        for c in merged["clusters"]:
            for pr in c["prs"]:
                all_pr_numbers.append(pr["number"])
        assert all_pr_numbers.count(1) == 1

        # Quality score should be updated to latest (8.0 from report2)
        assert len(merged["quality_scores"]) == 1
        assert merged["quality_scores"][0]["overall_score"] == 8.0

        # Alignment score should be updated to latest (9.0 from report2)
        assert len(merged["alignment_scores"]) == 1
        assert merged["alignment_scores"][0]["alignment_score"] == 9.0

    def test_merge_empty_reports(self, tmp_path: Path) -> None:
        report1 = {
            "repo": "owner/repo",
            "timestamp": "",
            "providers": [],
            "clusters": [],
            "quality_scores": [],
            "alignment_scores": [],
        }
        report2 = {
            "repo": "owner/repo",
            "timestamp": "",
            "providers": [],
            "clusters": [],
            "quality_scores": [],
            "alignment_scores": [],
        }

        f1 = _write_report(tmp_path / "r1.json", report1)
        f2 = _write_report(tmp_path / "r2.json", report2)

        merged = merge_reports([f1, f2])

        assert merged["clusters"] == []
        assert merged["quality_scores"] == []
        assert merged["alignment_scores"] == []
        assert merged["summary"]["total_prs"] == 0
        assert merged["summary"]["duplicate_clusters"] == 0

    def test_merge_single_report(self, tmp_path: Path) -> None:
        report = {
            "repo": "owner/repo",
            "timestamp": "2025-06-01T00:00:00Z",
            "providers": ["model/a"],
            "clusters": [
                {"cluster_id": "c-0", "prs": [{"number": 1}]},
            ],
            "quality_scores": [{"pr_number": 1, "overall_score": 7.0}],
            "alignment_scores": [],
        }

        f1 = _write_report(tmp_path / "r1.json", report)
        merged = merge_reports([f1])

        assert len(merged["clusters"]) == 1
        assert len(merged["quality_scores"]) == 1
        assert merged["summary"]["total_prs"] == 1
