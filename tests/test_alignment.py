"""Tests for claw_review.alignment — vision alignment scoring."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from claw_review.github_client import PRData
from claw_review.models import ModelResponse
from claw_review.alignment import (
    AlignmentScore,
    _build_alignment_prompt,
    score_alignment,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pr(**overrides: object) -> PRData:
    """Build a minimal PRData for testing."""
    defaults: dict = {
        "number": 42,
        "title": "Fix memory leak in parser",
        "body": "This PR fixes a memory leak caused by unclosed file handles.",
        "author": "contributor123",
        "created_at": "2025-06-01T10:00:00Z",
        "updated_at": "2025-06-02T12:00:00Z",
        "state": "open",
        "labels": ["bug", "priority:high"],
        "files_changed": ["src/parser.py", "tests/test_parser.py"],
        "additions": 25,
        "deletions": 10,
        "diff_summary": "--- src/parser.py (+25/-10)\n@@ -10,6 +10,8 @@\n+    handle.close()",
        "url": "https://github.com/owner/repo/pull/42",
        "comments_count": 3,
    }
    defaults.update(overrides)
    return PRData(**defaults)


def _alignment_response(
    provider: str,
    alignment_score: float = 7.0,
    aligned_aspects: list[str] | None = None,
    drift_concerns: list[str] | None = None,
    recommendation: str = "MERGE",
    rationale: str = "Well aligned with project goals.",
) -> ModelResponse:
    """Build a ModelResponse with valid alignment JSON."""
    payload = {
        "alignment_score": alignment_score,
        "aligned_aspects": aligned_aspects or ["Fixes a documented issue"],
        "drift_concerns": drift_concerns or [],
        "recommendation": recommendation,
        "rationale": rationale,
    }
    return ModelResponse(
        provider=provider,
        model=provider,
        content=json.dumps(payload),
    )


def _error_response(provider: str) -> ModelResponse:
    return ModelResponse(provider=provider, model="error", content="ERROR: boom")


def _mock_pool(responses: list[ModelResponse]) -> MagicMock:
    pool = MagicMock()
    pool.query_all = AsyncMock(return_value=responses)
    return pool


def _load_vision_docs() -> dict[str, str]:
    return json.loads((FIXTURES / "sample_vision_docs.json").read_text())


# ===================================================================
# _build_alignment_prompt
# ===================================================================


class TestBuildAlignmentPrompt:
    def test_includes_vision_docs(self) -> None:
        pr = _make_pr()
        vision_docs = {"README.md": "# My Project\nGoals: performance"}
        prompt = _build_alignment_prompt(pr, vision_docs)
        assert "README.md" in prompt
        assert "performance" in prompt

    def test_includes_pr_fields(self) -> None:
        pr = _make_pr(
            number=55,
            title="Add REST API",
            author="bob",
            labels=["feature"],
            additions=100,
            deletions=20,
        )
        vision_docs = {"README.md": "# Project"}
        prompt = _build_alignment_prompt(pr, vision_docs)
        assert "#55" in prompt
        assert "Add REST API" in prompt
        assert "bob" in prompt
        assert "feature" in prompt
        assert "+100" in prompt
        assert "-20" in prompt

    def test_includes_diff_summary(self) -> None:
        pr = _make_pr(diff_summary="@@ added new endpoint @@")
        vision_docs = {"README.md": "# Doc"}
        prompt = _build_alignment_prompt(pr, vision_docs)
        assert "added new endpoint" in prompt

    def test_multiple_vision_docs(self) -> None:
        pr = _make_pr()
        vision_docs = {
            "README.md": "Project readme content",
            "CONTRIBUTING.md": "Contribution guidelines here",
        }
        prompt = _build_alignment_prompt(pr, vision_docs)
        assert "README.md" in prompt
        assert "CONTRIBUTING.md" in prompt
        assert "Project readme content" in prompt
        assert "Contribution guidelines here" in prompt


# ===================================================================
# score_alignment
# ===================================================================


class TestScoreAlignment:
    async def test_valid_responses(self) -> None:
        pr = _make_pr(number=10)
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", alignment_score=8.0,
                                aligned_aspects=["Fixes WebSocket bug"],
                                recommendation="MERGE"),
            _alignment_response("model-b", alignment_score=7.0,
                                aligned_aspects=["Protocol compliance"],
                                recommendation="MERGE"),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool)

        assert len(results) == 1
        a = results[0]
        assert a.pr_number == 10
        assert a.alignment_score == 7.5  # average of 8 and 7
        assert "MERGE" in a.recommendation
        assert len(a.aligned_aspects) > 0

    async def test_empty_vision_docs_returns_empty(self) -> None:
        pr = _make_pr()
        pool = _mock_pool([])

        results = await score_alignment([pr], {}, pool)
        assert results == []
        # pool.query_all should not have been called
        pool.query_all.assert_not_called()

    async def test_score_below_reject_threshold_becomes_close(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", alignment_score=2.0,
                                recommendation="DISCUSS"),
            _alignment_response("model-b", alignment_score=3.0,
                                recommendation="REVIEW"),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool,
                                  reject_threshold=4.0)
        assert len(results) == 1
        assert results[0].recommendation == "CLOSE"

    async def test_score_above_threshold_keeps_model_recommendation(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", alignment_score=8.0,
                                recommendation="MERGE"),
            _alignment_response("model-b", alignment_score=7.0,
                                recommendation="MERGE"),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool,
                                  reject_threshold=4.0)
        assert results[0].recommendation == "MERGE"

    async def test_handles_model_errors(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _error_response("model-a"),
            _alignment_response("model-b", alignment_score=6.0,
                                recommendation="REVIEW",
                                rationale="Needs discussion"),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool)
        assert len(results) == 1
        # Only model-b contributes
        assert results[0].scores_by_provider == {"model-b": 6.0}

    async def test_all_models_invalid_json_skips_pr(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        bad1 = ModelResponse(provider="m1", model="m1", content="not json")
        bad2 = ModelResponse(provider="m2", model="m2", content="also bad {{{")
        pool = _mock_pool([bad1, bad2])

        results = await score_alignment([pr], vision_docs, pool)
        assert len(results) == 0

    async def test_deduplicates_aligned_aspects(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", aligned_aspects=["Bug fix", "Performance"]),
            _alignment_response("model-b", aligned_aspects=["Bug fix", "Testing"]),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool)
        # "Bug fix" should appear only once
        assert results[0].aligned_aspects.count("Bug fix") == 1
        assert "Performance" in results[0].aligned_aspects
        assert "Testing" in results[0].aligned_aspects

    async def test_deduplicates_drift_concerns(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", drift_concerns=["Scope creep", "No tests"]),
            _alignment_response("model-b", drift_concerns=["Scope creep", "Style issues"]),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool)
        assert results[0].drift_concerns.count("Scope creep") == 1

    async def test_confidence_high_when_models_agree(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", alignment_score=8.0),
            _alignment_response("model-b", alignment_score=8.0),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool)
        # spread=0, confidence = max(0, 1 - 0/10) = 1.0
        assert results[0].confidence == 1.0

    async def test_confidence_lower_when_models_disagree(self) -> None:
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", alignment_score=2.0),
            _alignment_response("model-b", alignment_score=8.0),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool)
        # spread=6, confidence = max(0, 1 - 6/10) = 0.4
        assert results[0].confidence == 0.4

    async def test_results_sorted_by_alignment_score_ascending(self) -> None:
        """Lowest alignment scores (problematic PRs) should be first."""
        prs = [_make_pr(number=i) for i in range(3)]
        vision_docs = _load_vision_docs()
        responses_list = [
            [_alignment_response("m", alignment_score=8.0)],
            [_alignment_response("m", alignment_score=3.0)],
            [_alignment_response("m", alignment_score=5.0)],
        ]
        pool = MagicMock()
        pool.query_all = AsyncMock(side_effect=responses_list)

        results = await score_alignment(prs, vision_docs, pool)
        scores = [r.alignment_score for r in results]
        assert scores == sorted(scores)

    async def test_single_model_confidence_is_one(self) -> None:
        """Single model response should have confidence=1.0 (spread=0)."""
        pr = _make_pr()
        vision_docs = _load_vision_docs()
        responses = [
            _alignment_response("model-a", alignment_score=5.0),
        ]
        pool = _mock_pool(responses)

        results = await score_alignment([pr], vision_docs, pool)
        assert results[0].confidence == 1.0


# ===================================================================
# AlignmentScore.to_dict
# ===================================================================


class TestAlignmentScoreToDict:
    def test_all_fields_present(self) -> None:
        a = AlignmentScore(
            pr_number=42,
            pr_title="Fix bug",
            pr_author="alice",
            pr_url="https://example.com/pull/42",
            alignment_score=7.5,
            scores_by_provider={"model-a": 8.0, "model-b": 7.0},
            aligned_aspects=["Bug fix", "Performance"],
            drift_concerns=["Scope creep"],
            recommendation="MERGE",
            rationale="Well aligned",
            confidence=0.9,
        )
        d = a.to_dict()

        assert d["pr_number"] == 42
        assert d["pr_title"] == "Fix bug"
        assert d["pr_author"] == "alice"
        assert d["pr_url"] == "https://example.com/pull/42"
        assert d["alignment_score"] == 7.5
        assert d["scores_by_provider"] == {"model-a": 8.0, "model-b": 7.0}
        assert d["aligned_aspects"] == ["Bug fix", "Performance"]
        assert d["drift_concerns"] == ["Scope creep"]
        assert d["recommendation"] == "MERGE"
        assert d["rationale"] == "Well aligned"
        assert d["confidence"] == 0.9
