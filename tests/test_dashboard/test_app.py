"""Tests for the DashboardApp class and FastAPI server."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from claw_review.dashboard.app import DashboardApp, generate_static_dashboard
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
    """Write a report dict to a temp file and return the path string."""
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture()
def loaded_loader(tmp_path):
    """Return a DataLoader with SAMPLE_REPORT already loaded."""
    path = _write_report(tmp_path, SAMPLE_REPORT)
    loader = DataLoader()
    loader.load_report(path)
    return loader


@pytest.fixture()
def client(loaded_loader):
    """Return a FastAPI TestClient for the dashboard app."""
    app_instance = DashboardApp(loaded_loader)
    fa = app_instance.create_server_app()
    return TestClient(fa)


class TestCreateServerApp:
    """Tests for DashboardApp.create_server_app."""

    def test_returns_fastapi(self, loaded_loader: DataLoader) -> None:
        """create_server_app returns a FastAPI instance."""
        from fastapi import FastAPI

        app_instance = DashboardApp(loaded_loader)
        fa = app_instance.create_server_app()
        assert isinstance(fa, FastAPI)


class TestAPIRoutes:
    """Tests for the dashboard API routes."""

    def test_index_returns_html(self, client: TestClient) -> None:
        """GET / returns HTML."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "ClawReview Dashboard" in resp.text

    def test_api_data(self, client: TestClient) -> None:
        """GET /api/data returns the full report data."""
        resp = client.get("/api/data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["repo"] == "test/repo"
        assert "clusters" in data
        assert "quality_scores" in data

    def test_api_summary(self, client: TestClient) -> None:
        """GET /api/summary returns summary stats."""
        resp = client.get("/api/summary")
        assert resp.status_code == 200
        summary = resp.json()
        assert "total_prs" in summary
        assert "avg_quality_score" in summary

    def test_api_filter_search(self, client: TestClient) -> None:
        """GET /api/filter?search=auth returns matching results."""
        resp = client.get("/api/filter?search=auth")
        assert resp.status_code == 200
        data = resp.json()
        assert "search_results" in data
        assert len(data["search_results"]) >= 1

    def test_api_filter_recommendation(self, client: TestClient) -> None:
        """GET /api/filter?recommendation=MERGE returns MERGE items."""
        resp = client.get("/api/filter?recommendation=MERGE")
        assert resp.status_code == 200
        data = resp.json()
        assert "alignment" in data
        assert all(a["recommendation"] == "MERGE" for a in data["alignment"])

    def test_api_filter_score(self, client: TestClient) -> None:
        """GET /api/filter with score range filters quality scores."""
        resp = client.get("/api/filter?min_score=8.0&max_score=10.0")
        assert resp.status_code == 200
        data = resp.json()
        assert "quality" in data

    def test_api_filter_no_params(self, client: TestClient) -> None:
        """GET /api/filter with no params returns full data."""
        resp = client.get("/api/filter")
        assert resp.status_code == 200
        data = resp.json()
        assert "repo" in data


class TestGenerateStatic:
    """Tests for DashboardApp.generate_static."""

    def test_produces_html_file(self, loaded_loader: DataLoader, tmp_path) -> None:
        """generate_static produces a valid HTML file."""
        app_instance = DashboardApp(loaded_loader)
        output = str(tmp_path / "dash.html")
        result = app_instance.generate_static(output)

        assert result.endswith("dash.html")
        with open(output) as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "ClawReview Dashboard" in html

    def test_html_contains_embedded_data(self, loaded_loader: DataLoader, tmp_path) -> None:
        """Generated HTML contains embedded JSON data."""
        app_instance = DashboardApp(loaded_loader)
        output = str(tmp_path / "dash.html")
        app_instance.generate_static(output)

        with open(output) as f:
            html = f.read()
        assert "test/repo" in html
        assert "Fix auth timeout issue" in html


class TestGenerateStaticDashboard:
    """Tests for the convenience function."""

    def test_single_report(self, tmp_path) -> None:
        """generate_static_dashboard works with a single report."""
        rp = _write_report(tmp_path, SAMPLE_REPORT)
        output = str(tmp_path / "out.html")
        result = generate_static_dashboard([rp], output=output)

        assert result.endswith("out.html")
        with open(output) as f:
            html = f.read()
        assert "ClawReview Dashboard" in html

    def test_multiple_reports(self, tmp_path) -> None:
        """generate_static_dashboard works with multiple reports."""
        r1 = _write_report(tmp_path, SAMPLE_REPORT, "r1.json")
        r2_data = dict(SAMPLE_REPORT)
        r2_data["providers"] = ["model-c"]
        r2 = _write_report(tmp_path, r2_data, "r2.json")
        output = str(tmp_path / "multi.html")
        result = generate_static_dashboard([r1, r2], output=output)

        assert result.endswith("multi.html")
