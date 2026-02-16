"""Tests for claw_review.scoring — quality scoring with multi-model consensus."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from claw_review.github_client import PRData
from claw_review.models import ModelResponse
from claw_review.scoring import (
    DimensionScore,
    QualityScore,
    _build_scoring_prompt,
    score_prs,
    rank_within_clusters,
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


def _score_response(
    provider: str,
    code_quality: float = 7.0,
    test_coverage: float = 6.0,
    scope_discipline: float = 8.0,
    breaking_risk: float = 7.0,
    style_consistency: float = 7.0,
    summary: str = "Looks good overall.",
) -> ModelResponse:
    """Build a ModelResponse with valid scoring JSON."""
    payload = {
        "code_quality": code_quality,
        "test_coverage": test_coverage,
        "scope_discipline": scope_discipline,
        "breaking_risk": breaking_risk,
        "style_consistency": style_consistency,
        "summary": summary,
    }
    return ModelResponse(
        provider=provider,
        model=provider,
        content=json.dumps(payload),
    )


def _error_response(provider: str) -> ModelResponse:
    """Build an error ModelResponse."""
    return ModelResponse(provider=provider, model="error", content="ERROR: boom")


def _mock_pool(responses: list[ModelResponse]) -> MagicMock:
    """Build a mock ModelPool that returns the given responses."""
    pool = MagicMock()
    pool.query_all = AsyncMock(return_value=responses)
    return pool


# ===================================================================
# _build_scoring_prompt
# ===================================================================


class TestBuildScoringPrompt:
    def test_includes_pr_fields(self) -> None:
        pr = _make_pr(
            number=99,
            title="Add caching layer",
            author="alice",
            labels=["enhancement", "perf"],
            additions=200,
            deletions=50,
        )
        prompt = _build_scoring_prompt(pr)
        assert "#99" in prompt
        assert "Add caching layer" in prompt
        assert "alice" in prompt
        assert "enhancement" in prompt
        assert "perf" in prompt
        assert "+200" in prompt
        assert "-50" in prompt

    def test_includes_diff(self) -> None:
        pr = _make_pr(diff_summary="@@ -1,5 +1,10 @@\n+new code here")
        prompt = _build_scoring_prompt(pr)
        assert "+new code here" in prompt

    def test_includes_labels(self) -> None:
        pr = _make_pr(labels=["bug", "urgent"])
        prompt = _build_scoring_prompt(pr)
        assert "bug" in prompt
        assert "urgent" in prompt

    def test_empty_labels(self) -> None:
        pr = _make_pr(labels=[])
        prompt = _build_scoring_prompt(pr)
        assert "none" in prompt

    def test_truncates_body(self) -> None:
        pr = _make_pr(body="x" * 3000)
        prompt = _build_scoring_prompt(pr)
        # Body is truncated to 2000 chars
        assert "x" * 2000 in prompt
        assert "x" * 2001 not in prompt


# ===================================================================
# score_prs
# ===================================================================


class TestScorePrs:
    async def test_single_pr_with_valid_responses(self) -> None:
        pr = _make_pr(number=10)
        responses = [
            _score_response("model-a", code_quality=8, test_coverage=7,
                            scope_discipline=9, breaking_risk=8,
                            style_consistency=7, summary="Good PR"),
            _score_response("model-b", code_quality=7, test_coverage=8,
                            scope_discipline=8, breaking_risk=7,
                            style_consistency=8, summary="Decent PR"),
        ]
        pool = _mock_pool(responses)

        results = await score_prs([pr], pool)

        assert len(results) == 1
        qs = results[0]
        assert qs.pr_number == 10
        assert len(qs.dimensions) == 5
        assert qs.overall_score > 0
        assert qs.summary in ("Good PR", "Decent PR")

    async def test_multiple_prs(self) -> None:
        prs = [_make_pr(number=i) for i in range(3)]
        responses = [
            _score_response("model-a", code_quality=8, test_coverage=7,
                            scope_discipline=9, breaking_risk=8,
                            style_consistency=7),
        ]
        pool = _mock_pool(responses)

        results = await score_prs(prs, pool)
        assert len(results) == 3

    async def test_score_clamping_high(self) -> None:
        """Values above 10 should be clamped to 10."""
        pr = _make_pr()
        payload = {
            "code_quality": 15,
            "test_coverage": 12,
            "scope_discipline": 10,
            "breaking_risk": 8,
            "style_consistency": 7,
            "summary": "Over-scored",
        }
        resp = ModelResponse(
            provider="model-a", model="model-a",
            content=json.dumps(payload),
        )
        pool = _mock_pool([resp])

        results = await score_prs([pr], pool)
        qs = results[0]
        for dim in qs.dimensions:
            for score_val in dim.scores.values():
                assert score_val <= 10.0

    async def test_score_clamping_low(self) -> None:
        """Values below 1 should be clamped to 1."""
        pr = _make_pr()
        payload = {
            "code_quality": -5,
            "test_coverage": 0,
            "scope_discipline": 1,
            "breaking_risk": 8,
            "style_consistency": 7,
            "summary": "Under-scored",
        }
        resp = ModelResponse(
            provider="model-a", model="model-a",
            content=json.dumps(payload),
        )
        pool = _mock_pool([resp])

        results = await score_prs([pr], pool)
        qs = results[0]
        for dim in qs.dimensions:
            for score_val in dim.scores.values():
                assert score_val >= 1.0

    async def test_disagreement_flagging(self) -> None:
        """Spread > 3 on a dimension triggers needs_human_review."""
        pr = _make_pr()
        responses = [
            _score_response("model-a", code_quality=9, test_coverage=9,
                            scope_discipline=9, breaking_risk=9,
                            style_consistency=9),
            _score_response("model-b", code_quality=2, test_coverage=9,
                            scope_discipline=9, breaking_risk=9,
                            style_consistency=9),
        ]
        pool = _mock_pool(responses)

        results = await score_prs([pr], pool, disagreement_threshold=3.0)
        qs = results[0]
        assert qs.needs_human_review is True
        assert len(qs.disagreement_reasons) >= 1
        assert "code_quality" in qs.disagreement_reasons[0]

    async def test_no_disagreement_when_models_agree(self) -> None:
        """Close scores should not flag for human review."""
        pr = _make_pr()
        responses = [
            _score_response("model-a", code_quality=7, test_coverage=7,
                            scope_discipline=7, breaking_risk=7,
                            style_consistency=7),
            _score_response("model-b", code_quality=8, test_coverage=8,
                            scope_discipline=8, breaking_risk=8,
                            style_consistency=8),
        ]
        pool = _mock_pool(responses)

        results = await score_prs([pr], pool, disagreement_threshold=3.0)
        qs = results[0]
        assert qs.needs_human_review is False
        assert qs.disagreement_reasons == []

    async def test_handles_model_errors(self) -> None:
        """Error responses should be skipped, valid ones used."""
        pr = _make_pr()
        responses = [
            _error_response("model-a"),
            _score_response("model-b", code_quality=7, test_coverage=7,
                            scope_discipline=7, breaking_risk=7,
                            style_consistency=7, summary="Only valid model"),
        ]
        pool = _mock_pool(responses)

        results = await score_prs([pr], pool)
        assert len(results) == 1
        qs = results[0]
        assert qs.summary == "Only valid model"
        # Only model-b should be in the scores
        for dim in qs.dimensions:
            assert "model-b" in dim.scores
            assert "model-a" not in dim.scores

    async def test_all_models_invalid_json_skips_pr(self) -> None:
        """If all responses are unparseable, the PR is skipped."""
        pr = _make_pr()
        bad_resp1 = ModelResponse(
            provider="model-a", model="model-a",
            content="not json at all",
        )
        bad_resp2 = ModelResponse(
            provider="model-b", model="model-b",
            content="also garbage {{{",
        )
        pool = _mock_pool([bad_resp1, bad_resp2])

        results = await score_prs([pr], pool)
        assert len(results) == 0

    async def test_results_sorted_by_overall_score_descending(self) -> None:
        """Results should be sorted best first."""
        prs = [_make_pr(number=i) for i in range(3)]
        # Return different scores for each PR
        responses_list = [
            [_score_response("m", code_quality=5, test_coverage=5,
                             scope_discipline=5, breaking_risk=5,
                             style_consistency=5)],
            [_score_response("m", code_quality=9, test_coverage=9,
                             scope_discipline=9, breaking_risk=9,
                             style_consistency=9)],
            [_score_response("m", code_quality=7, test_coverage=7,
                             scope_discipline=7, breaking_risk=7,
                             style_consistency=7)],
        ]
        pool = MagicMock()
        pool.query_all = AsyncMock(side_effect=responses_list)

        results = await score_prs(prs, pool)
        scores = [r.overall_score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_empty_prs_list(self) -> None:
        pool = _mock_pool([])
        results = await score_prs([], pool)
        assert results == []

    async def test_consensus_is_average(self) -> None:
        """Consensus score for each dimension is the arithmetic mean."""
        pr = _make_pr()
        responses = [
            _score_response("model-a", code_quality=6, test_coverage=8,
                            scope_discipline=4, breaking_risk=10,
                            style_consistency=2),
            _score_response("model-b", code_quality=8, test_coverage=6,
                            scope_discipline=6, breaking_risk=8,
                            style_consistency=4),
        ]
        pool = _mock_pool(responses)

        results = await score_prs([pr], pool)
        qs = results[0]
        dim_map = {d.dimension: d for d in qs.dimensions}
        assert dim_map["code_quality"].consensus == 7.0
        assert dim_map["test_coverage"].consensus == 7.0
        assert dim_map["scope_discipline"].consensus == 5.0
        assert dim_map["breaking_risk"].consensus == 9.0
        assert dim_map["style_consistency"].consensus == 3.0


# ===================================================================
# rank_within_clusters
# ===================================================================


class TestRankWithinClusters:
    def test_single_pr_cluster_gets_rank_1(self) -> None:
        clusters = [
            {
                "cluster_id": "singleton-1",
                "prs": [{"number": 1, "title": "PR 1", "author": "a", "url": "u"}],
            }
        ]
        qs = QualityScore(
            pr_number=1, pr_title="PR 1", pr_author="a",
            pr_url="u", dimensions=[], overall_score=8.0,
            summary="Good", provider_summaries={},
            needs_human_review=False, disagreement_reasons=[],
        )
        result = rank_within_clusters(clusters, [qs])
        assert result[0]["prs"][0]["quality_rank"] == 1
        assert result[0]["prs"][0]["quality_score"] == 8.0

    def test_multi_pr_cluster_ranked_by_quality_score(self) -> None:
        clusters = [
            {
                "cluster_id": "cluster-0",
                "prs": [
                    {"number": 1, "title": "PR 1", "author": "a", "url": "u"},
                    {"number": 2, "title": "PR 2", "author": "b", "url": "u2"},
                    {"number": 3, "title": "PR 3", "author": "c", "url": "u3"},
                ],
            }
        ]
        quality_scores = [
            QualityScore(
                pr_number=1, pr_title="PR 1", pr_author="a",
                pr_url="u", dimensions=[], overall_score=6.0,
                summary="OK", provider_summaries={},
                needs_human_review=False, disagreement_reasons=[],
            ),
            QualityScore(
                pr_number=2, pr_title="PR 2", pr_author="b",
                pr_url="u2", dimensions=[], overall_score=9.0,
                summary="Great", provider_summaries={},
                needs_human_review=False, disagreement_reasons=[],
            ),
            QualityScore(
                pr_number=3, pr_title="PR 3", pr_author="c",
                pr_url="u3", dimensions=[], overall_score=7.5,
                summary="Good", provider_summaries={},
                needs_human_review=False, disagreement_reasons=[],
            ),
        ]

        result = rank_within_clusters(clusters, quality_scores)
        prs = result[0]["prs"]
        # Sorted by quality_score descending
        assert prs[0]["number"] == 2
        assert prs[0]["quality_rank"] == 1
        assert prs[1]["number"] == 3
        assert prs[1]["quality_rank"] == 2
        assert prs[2]["number"] == 1
        assert prs[2]["quality_rank"] == 3

    def test_missing_quality_scores_get_zero(self) -> None:
        clusters = [
            {
                "cluster_id": "cluster-0",
                "prs": [
                    {"number": 1, "title": "PR 1", "author": "a", "url": "u"},
                    {"number": 99, "title": "No Score", "author": "z", "url": "u2"},
                ],
            }
        ]
        qs = QualityScore(
            pr_number=1, pr_title="PR 1", pr_author="a",
            pr_url="u", dimensions=[], overall_score=8.0,
            summary="Good", provider_summaries={},
            needs_human_review=False, disagreement_reasons=[],
        )

        result = rank_within_clusters(clusters, [qs])
        prs = result[0]["prs"]
        # PR 99 should get 0.0 score and needs_human_review=True
        unscored = [p for p in prs if p["number"] == 99][0]
        assert unscored["quality_score"] == 0.0
        assert unscored["needs_human_review"] is True
        assert unscored["quality_summary"] == "Not scored"

    def test_empty_clusters(self) -> None:
        result = rank_within_clusters([], [])
        assert result == []


# ===================================================================
# QualityScore.to_dict
# ===================================================================


class TestQualityScoreToDict:
    def test_all_fields_present(self) -> None:
        dim = DimensionScore(
            dimension="code_quality",
            scores={"model-a": 8.0, "model-b": 7.0},
            consensus=7.5,
            disagreement=1.0,
            flagged=False,
        )
        qs = QualityScore(
            pr_number=42,
            pr_title="Fix bug",
            pr_author="alice",
            pr_url="https://example.com/pull/42",
            dimensions=[dim],
            overall_score=7.5,
            summary="Good quality PR",
            provider_summaries={"model-a": "Good", "model-b": "Fine"},
            needs_human_review=False,
            disagreement_reasons=[],
        )
        d = qs.to_dict()

        assert d["pr_number"] == 42
        assert d["pr_title"] == "Fix bug"
        assert d["pr_author"] == "alice"
        assert d["pr_url"] == "https://example.com/pull/42"
        assert d["overall_score"] == 7.5
        assert d["summary"] == "Good quality PR"
        assert d["needs_human_review"] is False
        assert d["disagreement_reasons"] == []

    def test_dimensions_nested_correctly(self) -> None:
        dims = [
            DimensionScore(
                dimension="code_quality",
                scores={"m1": 8.0},
                consensus=8.0,
                disagreement=0.0,
                flagged=False,
            ),
            DimensionScore(
                dimension="test_coverage",
                scores={"m1": 6.0},
                consensus=6.0,
                disagreement=0.0,
                flagged=False,
            ),
        ]
        qs = QualityScore(
            pr_number=1, pr_title="T", pr_author="a", pr_url="u",
            dimensions=dims, overall_score=7.0, summary="S",
            provider_summaries={}, needs_human_review=False,
            disagreement_reasons=[],
        )
        d = qs.to_dict()

        assert "dimensions" in d
        assert "code_quality" in d["dimensions"]
        assert "test_coverage" in d["dimensions"]
        assert d["dimensions"]["code_quality"]["consensus"] == 8.0
        assert d["dimensions"]["code_quality"]["scores"] == {"m1": 8.0}
        assert d["dimensions"]["code_quality"]["disagreement"] == 0.0
        assert d["dimensions"]["code_quality"]["flagged"] is False
