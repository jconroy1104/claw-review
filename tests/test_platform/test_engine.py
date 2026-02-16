"""Tests for the ConsensusEngine."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from claw_review.models import ModelResponse
from claw_review.platform.engine import ConsensusEngine
from claw_review.platform.interfaces import (
    AnalysisResult,
    DataItem,
    DomainConfig,
)


def _make_domain_config(**overrides: Any) -> DomainConfig:
    """Helper to create a DomainConfig with sensible defaults."""
    defaults = {
        "domain": "test",
        "scoring_dimensions": ["quality", "risk"],
        "clustering_prompt": "Cluster this item.",
        "scoring_prompt": "Score this item.",
        "alignment_prompt": "Evaluate alignment.",
        "recommendation_levels": ["ACCEPT", "REVIEW", "REJECT"],
        "default_thresholds": {
            "similarity": 0.82,
            "disagreement": 3.0,
            "alignment_reject": 4.0,
        },
    }
    defaults.update(overrides)
    return DomainConfig(**defaults)


def _make_item(id: str = "1", title: str = "Test", body: str = "Body") -> DataItem:
    return DataItem(id=id, title=title, body=body)


def _make_model_response(
    content_dict: dict,
    provider: str = "test/model-a",
) -> ModelResponse:
    """Create a ModelResponse with JSON content."""
    return ModelResponse(
        provider=provider,
        model=provider,
        content=json.dumps(content_dict),
    )


def _mock_model_pool(
    query_all_responses: list[ModelResponse] | None = None,
    embeddings: list[list[float]] | None = None,
) -> MagicMock:
    """Create a mock ModelPool."""
    pool = MagicMock()
    pool.query_all = AsyncMock(return_value=query_all_responses or [])
    pool.get_embeddings = AsyncMock(return_value=embeddings or [])
    return pool


class TestConsensusEngineInit:
    """Tests for ConsensusEngine initialization."""

    def test_init_stores_config(self) -> None:
        pool = _mock_model_pool()
        config = _make_domain_config()
        engine = ConsensusEngine(pool, config)
        assert engine.model_pool is pool
        assert engine.domain_config is config


class TestConsensusEngineAnalyze:
    """Tests for ConsensusEngine.analyze()."""

    async def test_empty_items_returns_empty_result(self) -> None:
        pool = _mock_model_pool()
        config = _make_domain_config()
        engine = ConsensusEngine(pool, config)

        result = await engine.analyze([])

        assert isinstance(result, AnalysisResult)
        assert result.items_analyzed == 0
        assert result.clusters == []
        assert result.quality_scores == []
        assert result.alignment_scores == []
        assert result.domain == "test"

    async def test_analyze_returns_analysis_result(self) -> None:
        intent_resp = _make_model_response(
            {"intent": "Fix the parser bug", "category": "bugfix", "affected_area": "parser"}
        )
        score_resp = _make_model_response(
            {"quality": 7, "risk": 5, "summary": "Good fix"}
        )

        pool = _mock_model_pool(
            query_all_responses=[intent_resp],
            embeddings=[[0.1, 0.2, 0.3]],
        )
        # Different responses for intent, scoring calls
        pool.query_all = AsyncMock(side_effect=[
            [intent_resp],  # extract_intents
            [score_resp],   # score_quality
        ])
        pool.get_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        config = _make_domain_config()
        engine = ConsensusEngine(pool, config)

        result = await engine.analyze([_make_item()])

        assert isinstance(result, AnalysisResult)
        assert result.items_analyzed == 1
        assert result.domain == "test"
        assert result.timestamp != ""

    async def test_uses_domain_config_prompts(self) -> None:
        """Verify the engine passes domain config prompts to model_pool."""
        intent_resp = _make_model_response(
            {"intent": "Do thing", "category": "feature", "affected_area": "core"}
        )
        score_resp = _make_model_response(
            {"quality": 8, "risk": 3, "summary": "Great"}
        )

        pool = _mock_model_pool()
        pool.query_all = AsyncMock(side_effect=[
            [intent_resp],  # intent
            [score_resp],   # quality
        ])
        pool.get_embeddings = AsyncMock(return_value=[[0.5, 0.5]])

        config = _make_domain_config(
            clustering_prompt="CUSTOM_CLUSTER_PROMPT",
            scoring_prompt="CUSTOM_SCORE_PROMPT",
        )
        engine = ConsensusEngine(pool, config)

        await engine.analyze([_make_item()])

        # First call is intent extraction, should use clustering_prompt
        first_call_args = pool.query_all.call_args_list[0]
        assert first_call_args.kwargs.get("system_prompt") == "CUSTOM_CLUSTER_PROMPT" or \
            first_call_args[1].get("system_prompt") == "CUSTOM_CLUSTER_PROMPT" or \
            (len(first_call_args[0]) > 0 and first_call_args[0][0] == "CUSTOM_CLUSTER_PROMPT")

        # Second call is quality scoring, should use scoring_prompt
        second_call_args = pool.query_all.call_args_list[1]
        assert second_call_args.kwargs.get("system_prompt") == "CUSTOM_SCORE_PROMPT" or \
            second_call_args[1].get("system_prompt") == "CUSTOM_SCORE_PROMPT" or \
            (len(second_call_args[0]) > 0 and second_call_args[0][0] == "CUSTOM_SCORE_PROMPT")

    async def test_alignment_skipped_without_context_docs(self) -> None:
        intent_resp = _make_model_response(
            {"intent": "X", "category": "bugfix", "affected_area": "a"}
        )
        score_resp = _make_model_response(
            {"quality": 6, "risk": 7, "summary": "OK"}
        )

        pool = _mock_model_pool()
        pool.query_all = AsyncMock(side_effect=[
            [intent_resp],
            [score_resp],
        ])
        pool.get_embeddings = AsyncMock(return_value=[[0.1]])

        engine = ConsensusEngine(pool, _make_domain_config())
        result = await engine.analyze([_make_item()], context_docs=None)

        assert result.alignment_scores == []
        # Only 2 calls: intent + quality (no alignment)
        assert pool.query_all.call_count == 2

    async def test_alignment_runs_with_context_docs(self) -> None:
        intent_resp = _make_model_response(
            {"intent": "X", "category": "bugfix", "affected_area": "a"}
        )
        score_resp = _make_model_response(
            {"quality": 6, "risk": 7, "summary": "OK"}
        )
        align_resp = _make_model_response(
            {
                "alignment_score": 8,
                "aligned_aspects": ["Good"],
                "drift_concerns": [],
                "recommendation": "ACCEPT",
                "rationale": "Looks good",
            }
        )

        pool = _mock_model_pool()
        pool.query_all = AsyncMock(side_effect=[
            [intent_resp],
            [score_resp],
            [align_resp],
        ])
        pool.get_embeddings = AsyncMock(return_value=[[0.1]])

        engine = ConsensusEngine(pool, _make_domain_config())
        result = await engine.analyze(
            [_make_item()],
            context_docs={"README.md": "Project docs"},
        )

        assert len(result.alignment_scores) == 1
        assert result.alignment_scores[0]["alignment_score"] == 8.0
        assert pool.query_all.call_count == 3

    async def test_model_error_does_not_crash(self) -> None:
        """An error response from a model should be gracefully skipped."""
        error_resp = ModelResponse(
            provider="test/bad", model="error", content="ERROR: timeout"
        )
        good_resp = _make_model_response(
            {"intent": "Fix", "category": "bugfix", "affected_area": "core"},
            provider="test/good",
        )
        score_resp = _make_model_response(
            {"quality": 5, "risk": 5, "summary": "Meh"},
            provider="test/good",
        )

        pool = _mock_model_pool()
        pool.query_all = AsyncMock(side_effect=[
            [error_resp, good_resp],   # intent
            [error_resp, score_resp],  # quality
        ])
        pool.get_embeddings = AsyncMock(return_value=[[0.2, 0.3]])

        engine = ConsensusEngine(pool, _make_domain_config())
        result = await engine.analyze([_make_item()])

        assert result.items_analyzed == 1
        assert len(result.quality_scores) == 1

    async def test_different_domain_configs_produce_different_results(self) -> None:
        """Two domain configs with different dimensions should produce different scoring keys."""
        resp_a = _make_model_response(
            {"intent": "X", "category": "a", "affected_area": "b"}
        )
        score_a = _make_model_response(
            {"speed": 9, "accuracy": 8, "summary": "Fast"}
        )

        pool = _mock_model_pool()
        pool.query_all = AsyncMock(side_effect=[
            [resp_a],
            [score_a],
        ])
        pool.get_embeddings = AsyncMock(return_value=[[0.1]])

        config = _make_domain_config(
            domain="custom",
            scoring_dimensions=["speed", "accuracy"],
        )
        engine = ConsensusEngine(pool, config)
        result = await engine.analyze([_make_item()])

        assert result.domain == "custom"
        if result.quality_scores:
            dims = [d["dimension"] for d in result.quality_scores[0]["dimensions"]]
            assert "speed" in dims
            assert "accuracy" in dims


class TestConsensusEngineHelpers:
    """Tests for static helper methods."""

    def test_merge_intents_picks_longest(self) -> None:
        descriptions = {
            "a": "Short",
            "b": "This is a much longer description of intent",
        }
        result = ConsensusEngine._merge_intents(descriptions)
        assert result == "This is a much longer description of intent"

    def test_merge_intents_empty(self) -> None:
        assert ConsensusEngine._merge_intents({}) == "Unable to determine intent"

    def test_merge_intents_all_blank(self) -> None:
        assert ConsensusEngine._merge_intents({"a": "", "b": "  "}) == "Unable to determine intent"

    def test_majority_vote(self) -> None:
        assert ConsensusEngine._majority_vote(["a", "b", "a"]) == "a"

    def test_majority_vote_empty(self) -> None:
        assert ConsensusEngine._majority_vote([]) == "unknown"
