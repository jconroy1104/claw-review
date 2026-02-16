"""Tests for claw_review.clustering — intent extraction, embedding, and DBSCAN clustering."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from claw_review.clustering import (
    IntentResult,
    Cluster,
    _build_intent_prompt,
    _majority_vote,
    _merge_intents,
    extract_intents,
    generate_embeddings,
    cluster_intents,
)
from claw_review.github_client import PRData
from claw_review.models import ModelResponse

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pr(**overrides: object) -> PRData:
    """Build a minimal PRData for testing."""
    defaults = dict(
        number=1,
        title="Test PR",
        body="Some description",
        author="tester",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-02T00:00:00Z",
        state="open",
        labels=["bug"],
        files_changed=["src/foo.py", "src/bar.py"],
        additions=10,
        deletions=3,
        diff_summary="diff content here",
        url="https://github.com/test/repo/pull/1",
        comments_count=2,
    )
    defaults.update(overrides)
    return PRData(**defaults)


def _make_intent(
    pr_number: int = 1,
    consensus: str = "Fix a bug",
    category: str = "bugfix",
    area: str = "core",
    embedding: list[float] | None = None,
) -> IntentResult:
    return IntentResult(
        pr_number=pr_number,
        pr_title=f"PR #{pr_number}",
        pr_author="tester",
        pr_url=f"https://github.com/test/repo/pull/{pr_number}",
        intent_descriptions={"test/model": consensus},
        consensus_intent=consensus,
        category=category,
        affected_area=area,
        embedding=embedding,
    )


def _model_resp(provider: str, parsed: dict) -> ModelResponse:
    """Build a successful ModelResponse whose content is valid JSON."""
    return ModelResponse(
        provider=provider,
        model=provider,
        content=json.dumps(parsed),
    )


def _error_resp(provider: str) -> ModelResponse:
    """Build an error ModelResponse."""
    return ModelResponse(provider=provider, model="error", content="ERROR: boom")


# ===================================================================
# _build_intent_prompt
# ===================================================================


class TestBuildIntentPrompt:
    def test_includes_pr_fields(self) -> None:
        pr = _make_pr(
            number=42,
            title="Fix login",
            author="alice",
            labels=["urgent"],
            additions=100,
            deletions=5,
        )
        prompt = _build_intent_prompt(pr)
        assert "#42" in prompt
        assert "Fix login" in prompt
        assert "alice" in prompt
        assert "urgent" in prompt
        assert "+100" in prompt
        assert "-5" in prompt

    def test_truncates_files_over_20(self) -> None:
        pr = _make_pr(files_changed=[f"file{i}.py" for i in range(30)])
        prompt = _build_intent_prompt(pr)
        assert "and 10 more files" in prompt

    def test_handles_empty_body(self) -> None:
        pr = _make_pr(body=None)
        prompt = _build_intent_prompt(pr)
        assert "No description provided." in prompt


# ===================================================================
# _majority_vote
# ===================================================================


class TestMajorityVote:
    def test_clear_winner(self) -> None:
        assert _majority_vote(["bugfix", "bugfix", "feature"]) == "bugfix"

    def test_tie_returns_first_most_common(self) -> None:
        # Counter.most_common picks first-seen on tie
        result = _majority_vote(["feature", "bugfix", "feature", "bugfix"])
        assert result in ("feature", "bugfix")

    def test_single_item(self) -> None:
        assert _majority_vote(["docs"]) == "docs"

    def test_empty_list(self) -> None:
        assert _majority_vote([]) == "unknown"


# ===================================================================
# _merge_intents
# ===================================================================


class TestMergeIntents:
    def test_picks_longest(self) -> None:
        descs = {
            "a": "Short",
            "b": "A much longer description of the intent",
            "c": "Medium length desc",
        }
        assert _merge_intents(descs) == "A much longer description of the intent"

    def test_empty_dict(self) -> None:
        assert _merge_intents({}) == "Unable to determine intent"

    def test_single_entry(self) -> None:
        assert _merge_intents({"a": "Only one"}) == "Only one"

    def test_all_blank_values(self) -> None:
        assert _merge_intents({"a": "", "b": "  "}) == "Unable to determine intent"


# ===================================================================
# extract_intents
# ===================================================================


class TestExtractIntents:
    """Tests for extract_intents() with mocked model pool."""

    def _make_pool(self, responses: list[ModelResponse]) -> MagicMock:
        pool = MagicMock()
        pool.query_all = AsyncMock(return_value=responses)
        return pool

    async def test_single_pr_success(self) -> None:
        pr = _make_pr(number=10, title="Fix bug")
        responses = [
            _model_resp("a", {"intent": "Fix the login bug", "category": "bugfix", "affected_area": "auth"}),
            _model_resp("b", {"intent": "Fix login", "category": "bugfix", "affected_area": "auth"}),
        ]
        pool = self._make_pool(responses)

        results = await extract_intents([pr], pool)

        assert len(results) == 1
        r = results[0]
        assert r.pr_number == 10
        assert r.category == "bugfix"
        assert r.affected_area == "auth"
        assert len(r.intent_descriptions) == 2

    async def test_multiple_prs(self) -> None:
        prs = [_make_pr(number=i) for i in range(3)]
        resp = [_model_resp("a", {"intent": "x", "category": "bugfix", "affected_area": "core"})]
        pool = self._make_pool(resp)

        results = await extract_intents(prs, pool)
        assert len(results) == 3

    async def test_handles_model_failures(self) -> None:
        pr = _make_pr()
        responses = [
            _error_resp("a"),
            _model_resp("b", {"intent": "works fine", "category": "feature", "affected_area": "ui"}),
        ]
        pool = self._make_pool(responses)

        results = await extract_intents([pr], pool)
        assert len(results) == 1
        # Only one successful response should be in descriptions
        assert len(results[0].intent_descriptions) == 1

    async def test_handles_malformed_json(self) -> None:
        pr = _make_pr()
        bad_resp = ModelResponse(provider="bad", model="bad", content="not json at all")
        good_resp = _model_resp("ok", {"intent": "valid", "category": "docs", "affected_area": "docs"})
        pool = self._make_pool([bad_resp, good_resp])

        results = await extract_intents([pr], pool)
        assert len(results) == 1
        # The malformed response should still put the raw text (truncated) in descriptions
        assert "bad" in results[0].intent_descriptions
        assert "ok" in results[0].intent_descriptions


# ===================================================================
# generate_embeddings
# ===================================================================


class TestGenerateEmbeddings:
    async def test_attaches_embeddings(self) -> None:
        intents = [_make_intent(pr_number=i) for i in range(3)]
        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        pool = MagicMock()
        pool.get_embeddings = AsyncMock(return_value=vectors)

        result = await generate_embeddings(intents, pool)

        assert len(result) == 3
        for intent_r, vec in zip(result, vectors):
            assert intent_r.embedding == vec

    async def test_empty_input(self) -> None:
        pool = MagicMock()
        pool.get_embeddings = AsyncMock(return_value=[])

        result = await generate_embeddings([], pool)
        assert result == []


# ===================================================================
# cluster_intents
# ===================================================================


class TestClusterIntents:
    """Tests for DBSCAN-based intent clustering."""

    def _load_fixture_vectors(self) -> list[list[float]]:
        data = json.loads((FIXTURES / "sample_embeddings.json").read_text())
        return data["vectors"]

    def test_empty_input(self) -> None:
        assert cluster_intents([]) == []

    def test_single_pr_becomes_singleton(self) -> None:
        intent = _make_intent(pr_number=1, embedding=[0.1, 0.2, 0.3])
        clusters = cluster_intents([intent])
        assert len(clusters) == 1
        assert clusters[0].cluster_id.startswith("singleton-")
        assert len(clusters[0].prs) == 1

    def test_no_duplicates_all_singletons(self) -> None:
        # Orthogonal vectors -> no cosine similarity
        intents = [
            _make_intent(pr_number=1, embedding=[1.0, 0.0, 0.0]),
            _make_intent(pr_number=2, embedding=[0.0, 1.0, 0.0]),
            _make_intent(pr_number=3, embedding=[0.0, 0.0, 1.0]),
        ]
        clusters = cluster_intents(intents)
        assert len(clusters) == 3
        assert all(c.cluster_id.startswith("singleton-") for c in clusters)

    def test_clear_duplicate_cluster(self) -> None:
        # Nearly identical vectors should cluster together
        base = [0.9, 0.1, 0.0, 0.0, 0.0]
        near = [0.89, 0.11, 0.01, 0.0, 0.0]
        diff = [0.0, 0.0, 0.0, 0.9, 0.1]

        intents = [
            _make_intent(pr_number=1, consensus="Fix WS reconnect", embedding=base),
            _make_intent(pr_number=2, consensus="Fix WS reconnect v2", embedding=near),
            _make_intent(pr_number=3, consensus="Add dark mode", embedding=diff),
        ]
        clusters = cluster_intents(intents, similarity_threshold=0.82)

        dup_clusters = [c for c in clusters if len(c.prs) > 1]
        singletons = [c for c in clusters if len(c.prs) == 1]
        assert len(dup_clusters) == 1
        assert len(dup_clusters[0].prs) == 2
        assert len(singletons) == 1

    def test_mixed_clusters_and_singletons(self) -> None:
        vecs = self._load_fixture_vectors()
        # vecs: 0,1,5 similar; 2,4 similar; 3,6 different
        intents = [_make_intent(pr_number=i + 101, embedding=vecs[i]) for i in range(7)]

        clusters = cluster_intents(intents, similarity_threshold=0.82)
        dup_clusters = [c for c in clusters if len(c.prs) > 1]
        singletons = [c for c in clusters if len(c.prs) == 1]

        assert len(dup_clusters) >= 1
        assert len(singletons) >= 1
        total_prs = sum(len(c.prs) for c in clusters)
        assert total_prs == 7

    def test_threshold_strict(self) -> None:
        # With very strict threshold, moderately similar vectors won't cluster
        v1 = [0.9, 0.1, 0.0, 0.0, 0.0]
        v2 = [0.5, 0.5, 0.5, 0.3, 0.1]
        intents = [
            _make_intent(pr_number=1, embedding=v1),
            _make_intent(pr_number=2, embedding=v2),
        ]
        # threshold=0.98 means eps=0.02 -> only near-identical vectors cluster
        clusters = cluster_intents(intents, similarity_threshold=0.98)
        assert all(c.cluster_id.startswith("singleton-") for c in clusters)

    def test_threshold_loose(self) -> None:
        # With very loose threshold, even somewhat different vectors cluster
        v1 = [0.9, 0.1, 0.0, 0.0, 0.0]
        v2 = [0.7, 0.3, 0.1, 0.0, 0.0]
        intents = [
            _make_intent(pr_number=1, embedding=v1),
            _make_intent(pr_number=2, embedding=v2),
        ]
        clusters = cluster_intents(intents, similarity_threshold=0.5)
        dup_clusters = [c for c in clusters if len(c.prs) > 1]
        assert len(dup_clusters) == 1

    def test_cluster_sorted_dups_first(self) -> None:
        base = [0.9, 0.1, 0.0, 0.0, 0.0]
        near = [0.89, 0.11, 0.01, 0.0, 0.0]
        diff = [0.0, 0.0, 0.0, 0.9, 0.1]
        intents = [
            _make_intent(pr_number=1, embedding=base),
            _make_intent(pr_number=2, embedding=near),
            _make_intent(pr_number=3, embedding=diff),
        ]
        clusters = cluster_intents(intents, similarity_threshold=0.82)
        # Multi-PR clusters should come before singletons
        if len(clusters) > 1:
            assert len(clusters[0].prs) >= len(clusters[-1].prs)

    def test_skips_intents_without_embedding(self) -> None:
        intents = [
            _make_intent(pr_number=1, embedding=[0.9, 0.1]),
            _make_intent(pr_number=2, embedding=None),
        ]
        clusters = cluster_intents(intents)
        total_prs = sum(len(c.prs) for c in clusters)
        assert total_prs == 1  # Only the one with embedding

    def test_cluster_confidence_nonzero_for_real_cluster(self) -> None:
        base = [0.9, 0.1, 0.0, 0.0, 0.0]
        near = [0.89, 0.11, 0.01, 0.0, 0.0]
        intents = [
            _make_intent(pr_number=1, embedding=base),
            _make_intent(pr_number=2, embedding=near),
        ]
        clusters = cluster_intents(intents, similarity_threshold=0.82)
        dup_clusters = [c for c in clusters if len(c.prs) > 1]
        assert len(dup_clusters) == 1
        assert dup_clusters[0].confidence > 0.0


# ===================================================================
# Dataclass serialization
# ===================================================================


class TestIntentResultSerialization:
    def test_to_dict_excludes_embedding(self) -> None:
        intent = _make_intent(embedding=[0.1, 0.2, 0.3])
        d = intent.to_dict()
        assert "embedding" not in d
        assert d["pr_number"] == intent.pr_number
        assert d["consensus_intent"] == intent.consensus_intent

    def test_to_dict_all_fields_present(self) -> None:
        intent = _make_intent()
        d = intent.to_dict()
        expected_keys = {
            "pr_number", "pr_title", "pr_author", "pr_url",
            "intent_descriptions", "consensus_intent", "category",
            "affected_area",
        }
        assert expected_keys.issubset(d.keys())


class TestClusterSerialization:
    def test_to_dict_includes_all_fields(self) -> None:
        cluster = Cluster(
            cluster_id="cluster-0",
            intent_summary="Fix WS reconnect",
            category="bugfix",
            affected_area="websocket",
            confidence=0.95,
            prs=[{"number": 1, "title": "PR 1", "author": "a", "url": "u", "intent": "i"}],
        )
        d = cluster.to_dict()
        assert d["cluster_id"] == "cluster-0"
        assert d["confidence"] == 0.95
        assert len(d["prs"]) == 1
        assert d["prs"][0]["number"] == 1
