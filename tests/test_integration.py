"""Integration tests for claw-review — end-to-end pipeline with all mocks."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from claw_review.cli import cli
from claw_review.github_client import PRData
from claw_review.models import ModelResponse, ModelPool
from claw_review.clustering import (
    extract_intents,
    generate_embeddings,
    cluster_intents,
)
from claw_review.scoring import score_prs, rank_within_clusters
from claw_review.alignment import score_alignment
from claw_review.report import generate_report, generate_json_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_config() -> MagicMock:
    """Build a mock Config that passes validation."""
    config = MagicMock()
    config.github_token = "ghp_test_token"
    config.openrouter_api_key = "sk-or-test"
    config.models = ["model/a", "model/b", "model/c"]
    config.target_repo = "owner/repo"
    config.max_prs = 10
    config.similarity_threshold = 0.82
    config.quality_disagreement_threshold = 3.0
    config.alignment_reject_threshold = 4.0
    config.embedding_model = "openai/text-embedding-3-small"
    config.validate.return_value = []
    config.model_count = 3
    return config


def _make_pr(number: int = 42, title: str = "Fix parser bug", **kw: object) -> PRData:
    """Build a PRData with sensible defaults."""
    defaults: dict = {
        "number": number,
        "title": title,
        "body": "This PR fixes a bug in the parser module.",
        "author": "alice",
        "created_at": "2025-06-01T10:00:00Z",
        "updated_at": "2025-06-02T12:00:00Z",
        "state": "open",
        "labels": ["bug"],
        "files_changed": ["src/parser.py", "tests/test_parser.py"],
        "additions": 30,
        "deletions": 5,
        "diff_summary": "--- src/parser.py\n+fix",
        "url": f"https://github.com/owner/repo/pull/{number}",
        "comments_count": 2,
    }
    defaults.update(kw)
    return PRData(**defaults)


def _mock_model_pool(
    intent_json: dict | None = None,
    score_json: dict | None = None,
    alignment_json: dict | None = None,
) -> MagicMock:
    """Build a mock ModelPool with configurable async JSON responses."""
    pool = MagicMock(spec=ModelPool)
    pool.models = ["model/a", "model/b"]
    pool.model_count = 2
    pool.model_names = ["a", "b"]

    def _make_response(data: dict, provider: str = "model/a") -> ModelResponse:
        return ModelResponse(
            provider=provider,
            model=provider,
            content=json.dumps(data),
        )

    if intent_json is None:
        intent_json = {
            "intent": "Fix parser reconnection bug",
            "category": "bugfix",
            "affected_area": "parser",
        }

    if score_json is None:
        score_json = {
            "code_quality": 8,
            "test_coverage": 7,
            "scope_discipline": 9,
            "breaking_risk": 8,
            "style_consistency": 7,
            "summary": "Good quality fix",
        }

    if alignment_json is None:
        alignment_json = {
            "alignment_score": 8,
            "aligned_aspects": ["Fixes known bug"],
            "drift_concerns": [],
            "recommendation": "MERGE",
            "rationale": "Well-aligned with project goals",
        }

    async def _query_all_side_effect(**kwargs):
        system_prompt = kwargs.get("system_prompt", "")
        if "intent" in system_prompt.lower():
            data = intent_json
        elif "score" in system_prompt.lower():
            data = score_json
        else:
            data = alignment_json
        return [
            _make_response(data, provider="model/a"),
            _make_response(data, provider="model/b"),
        ]

    pool.query_all = AsyncMock(side_effect=_query_all_side_effect)
    pool.get_embeddings = AsyncMock(return_value=[[0.1] * 10, [0.9] * 10])

    return pool


# ---------------------------------------------------------------------------
# Pipeline Stage Integration
# ---------------------------------------------------------------------------


class TestIntentExtractionPipeline:
    """Test intent extraction → embedding → clustering pipeline."""

    @pytest.mark.asyncio
    async def test_extract_and_cluster_two_unique_prs(self) -> None:
        prs = [_make_pr(1, "Fix parser"), _make_pr(2, "Add dark mode")]
        pool = _mock_model_pool()

        # Make query_all return different intents per call
        call_count = {"n": 0}
        intents_data = [
            {"intent": "Fix parser bug", "category": "bugfix", "affected_area": "parser"},
            {"intent": "Add dark mode UI", "category": "feature", "affected_area": "UI"},
        ]

        async def _side_effect(**kwargs):
            idx = min(call_count["n"], len(intents_data) - 1)
            call_count["n"] += 1
            data = intents_data[idx]
            return [
                ModelResponse("model/a", "model/a", json.dumps(data)),
                ModelResponse("model/b", "model/b", json.dumps(data)),
            ]

        pool.query_all = AsyncMock(side_effect=_side_effect)
        # Dissimilar embeddings → should be singletons
        pool.get_embeddings = AsyncMock(return_value=[[1.0] + [0.0] * 9, [0.0] * 9 + [1.0]])

        intents = await extract_intents(prs, pool)
        assert len(intents) == 2

        intents = await generate_embeddings(intents, pool)
        assert all(i.embedding is not None for i in intents)

        clusters = cluster_intents(intents, similarity_threshold=0.82)
        # Dissimilar → all singletons
        assert all(len(c.prs) == 1 for c in clusters)

    @pytest.mark.asyncio
    async def test_extract_and_cluster_duplicate_prs(self) -> None:
        prs = [_make_pr(1, "Fix parser v1"), _make_pr(2, "Fix parser v2")]
        pool = _mock_model_pool()

        same_intent = {"intent": "Fix parser bug", "category": "bugfix", "affected_area": "parser"}
        pool.query_all = AsyncMock(return_value=[
            ModelResponse("model/a", "model/a", json.dumps(same_intent)),
            ModelResponse("model/b", "model/b", json.dumps(same_intent)),
        ])
        # Nearly identical embeddings → should cluster
        pool.get_embeddings = AsyncMock(return_value=[[1.0] * 10, [1.0] * 10])

        intents = await extract_intents(prs, pool)
        intents = await generate_embeddings(intents, pool)
        clusters = cluster_intents(intents, similarity_threshold=0.82)

        dup_clusters = [c for c in clusters if len(c.prs) > 1]
        assert len(dup_clusters) == 1
        assert len(dup_clusters[0].prs) == 2


class TestScoringPipeline:
    """Test quality scoring → ranking pipeline."""

    @pytest.mark.asyncio
    async def test_score_and_rank(self) -> None:
        prs = [_make_pr(1, "Fix A"), _make_pr(2, "Fix B")]
        pool = MagicMock(spec=ModelPool)

        call_count = {"n": 0}
        scores_data = [
            {"code_quality": 9, "test_coverage": 8, "scope_discipline": 9,
             "breaking_risk": 9, "style_consistency": 8, "summary": "Excellent"},
            {"code_quality": 5, "test_coverage": 4, "scope_discipline": 6,
             "breaking_risk": 5, "style_consistency": 5, "summary": "Mediocre"},
        ]

        async def _side_effect(**kwargs):
            idx = min(call_count["n"], len(scores_data) - 1)
            call_count["n"] += 1
            data = scores_data[idx]
            return [
                ModelResponse("model/a", "model/a", json.dumps(data)),
                ModelResponse("model/b", "model/b", json.dumps(data)),
            ]

        pool.query_all = AsyncMock(side_effect=_side_effect)

        quality_scores = await score_prs(prs, pool, disagreement_threshold=3.0)
        assert len(quality_scores) == 2
        # First should be higher score (sorted descending)
        assert quality_scores[0].overall_score >= quality_scores[1].overall_score

        # Rank within a cluster
        cluster_dicts = [
            {
                "cluster_id": "cluster-0",
                "prs": [
                    {"number": 1, "title": "Fix A", "author": "a", "url": "u1"},
                    {"number": 2, "title": "Fix B", "author": "b", "url": "u2"},
                ],
            }
        ]
        ranked = rank_within_clusters(cluster_dicts, quality_scores)
        assert ranked[0]["prs"][0]["quality_rank"] == 1
        assert ranked[0]["prs"][1]["quality_rank"] == 2


class TestAlignmentPipeline:
    """Test vision alignment scoring."""

    @pytest.mark.asyncio
    async def test_score_alignment_with_docs(self) -> None:
        prs = [_make_pr(1)]
        pool = MagicMock(spec=ModelPool)

        alignment_data = {
            "alignment_score": 7.5,
            "aligned_aspects": ["Addresses known issue"],
            "drift_concerns": [],
            "recommendation": "MERGE",
            "rationale": "Aligns with project goals",
        }
        pool.query_all = AsyncMock(return_value=[
            ModelResponse("model/a", "model/a", json.dumps(alignment_data)),
            ModelResponse("model/b", "model/b", json.dumps(alignment_data)),
        ])

        vision_docs = {"README.md": "# MyProject\nFix bugs and ship."}
        results = await score_alignment(prs, vision_docs, pool, reject_threshold=4.0)

        assert len(results) == 1
        assert results[0].alignment_score == 7.5
        assert results[0].recommendation == "MERGE"


class TestReportPipeline:
    """Test report generation from pipeline data."""

    def test_html_report_from_pipeline_data(self, tmp_path: Path) -> None:
        clusters = [
            {
                "cluster_id": "cluster-0",
                "intent_summary": "Fix parser",
                "category": "bugfix",
                "affected_area": "parser",
                "confidence": 0.95,
                "prs": [
                    {"number": 1, "title": "Fix v1", "author": "a", "url": "#",
                     "intent": "Fix parser", "quality_score": 8.5,
                     "quality_rank": 1, "needs_human_review": False,
                     "quality_summary": "Good"},
                    {"number": 2, "title": "Fix v2", "author": "b", "url": "#",
                     "intent": "Fix parser", "quality_score": 6.2,
                     "quality_rank": 2, "needs_human_review": True,
                     "quality_summary": "OK"},
                ],
            },
            {
                "cluster_id": "singleton-3",
                "intent_summary": "Add dark mode",
                "category": "feature",
                "affected_area": "UI",
                "confidence": 0.0,
                "prs": [
                    {"number": 3, "title": "Dark mode", "author": "c", "url": "#",
                     "intent": "Add dark mode"},
                ],
            },
        ]
        quality_scores = [
            {"pr_number": 1, "overall_score": 8.5, "pr_title": "Fix v1",
             "pr_author": "a", "pr_url": "#", "summary": "Good",
             "needs_human_review": False},
            {"pr_number": 2, "overall_score": 6.2, "pr_title": "Fix v2",
             "pr_author": "b", "pr_url": "#", "summary": "OK",
             "needs_human_review": True},
        ]
        alignment_scores = [
            {"pr_number": 1, "alignment_score": 3.5, "pr_title": "Fix v1",
             "pr_author": "a", "pr_url": "#", "recommendation": "CLOSE",
             "drift_concerns": ["Not on roadmap"], "rationale": "Drifts from goals"},
        ]

        html_path = generate_report(
            repo="owner/repo",
            clusters=clusters,
            quality_scores=quality_scores,
            alignment_scores=alignment_scores,
            providers=["model/a", "model/b"],
            output_path=str(tmp_path / "report.html"),
        )

        html = Path(html_path).read_text()
        assert "<!DOCTYPE html>" in html
        assert "owner/repo" in html
        assert "Fix parser" in html
        # "Dark mode" is a singleton — not in duplicate cluster section
        # but the report still renders it as part of total PR count
        assert "3" in html  # total PR count
        assert "CLOSE" in html
        assert "Not on roadmap" in html

    def test_json_report_from_pipeline_data(self, tmp_path: Path) -> None:
        clusters = [{"cluster_id": "s-1", "prs": [{"number": 1}]}]
        quality_scores = [{"pr_number": 1, "overall_score": 7.0}]
        alignment_scores = [{"pr_number": 1, "alignment_score": 8.0}]

        json_path = generate_json_report(
            repo="owner/repo",
            clusters=clusters,
            quality_scores=quality_scores,
            alignment_scores=alignment_scores,
            providers=["model/a"],
            output_path=str(tmp_path / "report.json"),
        )

        data = json.loads(Path(json_path).read_text())
        assert data["repo"] == "owner/repo"
        assert "timestamp" in data
        assert data["summary"]["total_prs"] == 1
        assert len(data["clusters"]) == 1
        assert len(data["quality_scores"]) == 1
        assert len(data["alignment_scores"]) == 1


# ---------------------------------------------------------------------------
# Full End-to-End Pipeline (CLI)
# ---------------------------------------------------------------------------


class TestEndToEndCli:
    """Full pipeline through the CLI with everything mocked."""

    @patch("claw_review.cli.generate_json_report")
    @patch("claw_review.cli.generate_report")
    @patch("claw_review.cli.score_alignment")
    @patch("claw_review.cli.fetch_repo_docs")
    @patch("claw_review.cli.score_prs")
    @patch("claw_review.cli.rank_within_clusters")
    @patch("claw_review.cli.cluster_intents")
    @patch("claw_review.cli.generate_embeddings")
    @patch("claw_review.cli.extract_intents")
    @patch("claw_review.cli.fetch_open_prs")
    @patch("claw_review.cli.ModelPool")
    @patch("claw_review.cli.Config")
    def test_full_pipeline_with_duplicates(
        self,
        mock_config_cls: MagicMock,
        mock_pool_cls: MagicMock,
        mock_fetch_prs: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_cluster: MagicMock,
        mock_rank: MagicMock,
        mock_score_prs: MagicMock,
        mock_fetch_docs: MagicMock,
        mock_score_align: MagicMock,
        mock_gen_report: MagicMock,
        mock_gen_json: MagicMock,
    ) -> None:
        # Config
        config = _valid_config()
        mock_config_cls.return_value = config

        # PRs
        mock_fetch_prs.return_value = [_make_pr(1), _make_pr(2), _make_pr(3)]

        # Intents & clustering
        mock_intent = MagicMock()
        mock_extract.return_value = [mock_intent] * 3
        mock_embed.return_value = [mock_intent] * 3

        dup_cluster = MagicMock()
        dup_cluster.prs = [
            {"number": 1, "title": "PR1", "author": "a", "url": "u1", "intent": "i"},
            {"number": 2, "title": "PR2", "author": "b", "url": "u2", "intent": "i"},
        ]
        dup_cluster.to_dict.return_value = {
            "cluster_id": "cluster-0",
            "prs": dup_cluster.prs,
        }
        singleton = MagicMock()
        singleton.prs = [
            {"number": 3, "title": "PR3", "author": "c", "url": "u3", "intent": "j"},
        ]
        singleton.to_dict.return_value = {
            "cluster_id": "singleton-3",
            "prs": singleton.prs,
        }
        mock_cluster.return_value = [dup_cluster, singleton]

        # Quality scoring
        qs1 = MagicMock()
        qs1.pr_number = 1
        qs1.overall_score = 8.0
        qs1.to_dict.return_value = {"pr_number": 1, "overall_score": 8.0}
        qs2 = MagicMock()
        qs2.pr_number = 2
        qs2.overall_score = 6.0
        qs2.to_dict.return_value = {"pr_number": 2, "overall_score": 6.0}
        mock_score_prs.return_value = [qs1, qs2]
        mock_rank.return_value = [dup_cluster.to_dict(), singleton.to_dict()]

        # Alignment
        mock_fetch_docs.return_value = {"README.md": "# Project"}
        align1 = MagicMock()
        align1.to_dict.return_value = {"pr_number": 1, "alignment_score": 8.0}
        mock_score_align.return_value = [align1]

        # Reports
        mock_gen_json.return_value = "report.json"
        mock_gen_report.return_value = "report.html"

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--repo", "owner/repo"])

        assert result.exit_code == 0

        # Verify full pipeline was called
        mock_fetch_prs.assert_called_once()
        mock_extract.assert_called_once()
        mock_embed.assert_called_once()
        mock_cluster.assert_called_once()
        mock_score_prs.assert_called_once()
        mock_rank.assert_called_once()
        mock_fetch_docs.assert_called_once()
        mock_score_align.assert_called_once()
        mock_gen_json.assert_called_once()
        mock_gen_report.assert_called_once()

    @patch("claw_review.cli.generate_json_report", return_value="report.json")
    @patch("claw_review.cli.generate_report", return_value="report.html")
    @patch("claw_review.cli.cluster_intents")
    @patch("claw_review.cli.generate_embeddings")
    @patch("claw_review.cli.extract_intents")
    @patch("claw_review.cli.fetch_open_prs")
    @patch("claw_review.cli.ModelPool")
    @patch("claw_review.cli.Config")
    def test_pipeline_no_duplicates_skips_scoring(
        self,
        mock_config_cls: MagicMock,
        mock_pool_cls: MagicMock,
        mock_fetch_prs: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_cluster: MagicMock,
        mock_gen_report: MagicMock,
        mock_gen_json: MagicMock,
    ) -> None:
        mock_config_cls.return_value = _valid_config()
        mock_fetch_prs.return_value = [_make_pr(1)]

        mock_intent = MagicMock()
        mock_extract.return_value = [mock_intent]
        mock_embed.return_value = [mock_intent]

        # All singletons → no quality scoring needed
        singleton = MagicMock()
        singleton.prs = [{"number": 1}]
        singleton.to_dict.return_value = {"cluster_id": "s-1", "prs": [{"number": 1}]}
        mock_cluster.return_value = [singleton]

        runner = CliRunner()
        with patch("claw_review.cli.score_prs") as mock_score:
            with patch("claw_review.cli.fetch_repo_docs", return_value={}):
                result = runner.invoke(
                    cli, ["analyze", "--skip-alignment"]
                )

        assert result.exit_code == 0
        mock_score.assert_not_called()

    @patch("claw_review.cli.generate_json_report", return_value="report.json")
    @patch("claw_review.cli.generate_report", return_value="report.html")
    @patch("claw_review.cli.cluster_intents")
    @patch("claw_review.cli.generate_embeddings")
    @patch("claw_review.cli.extract_intents")
    @patch("claw_review.cli.fetch_open_prs")
    @patch("claw_review.cli.ModelPool")
    @patch("claw_review.cli.Config")
    def test_pipeline_no_vision_docs_skips_alignment(
        self,
        mock_config_cls: MagicMock,
        mock_pool_cls: MagicMock,
        mock_fetch_prs: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_cluster: MagicMock,
        mock_gen_report: MagicMock,
        mock_gen_json: MagicMock,
    ) -> None:
        mock_config_cls.return_value = _valid_config()
        mock_fetch_prs.return_value = [_make_pr(1)]

        mock_intent = MagicMock()
        mock_extract.return_value = [mock_intent]
        mock_embed.return_value = [mock_intent]

        singleton = MagicMock()
        singleton.prs = [{"number": 1}]
        singleton.to_dict.return_value = {"cluster_id": "s-1", "prs": [{"number": 1}]}
        mock_cluster.return_value = [singleton]

        runner = CliRunner()
        with patch("claw_review.cli.fetch_repo_docs", return_value={}) as mock_docs:
            with patch("claw_review.cli.score_alignment") as mock_align:
                result = runner.invoke(cli, ["analyze"])

        assert result.exit_code == 0
        mock_docs.assert_called_once()
        mock_align.assert_not_called()


# ---------------------------------------------------------------------------
# Report Regeneration Integration
# ---------------------------------------------------------------------------


class TestReportRegeneration:
    """Test regenerate command with realistic data."""

    def test_regenerate_from_full_report(self, tmp_path: Path) -> None:
        report_data = {
            "repo": "openclaw/openclaw",
            "timestamp": "2025-06-01T00:00:00Z",
            "providers": ["anthropic/claude-sonnet-4", "openai/gpt-4o"],
            "summary": {"total_prs": 3, "duplicate_clusters": 1},
            "clusters": [
                {
                    "cluster_id": "cluster-0",
                    "intent_summary": "Fix WebSocket reconnection",
                    "category": "bugfix",
                    "affected_area": "websocket",
                    "confidence": 0.92,
                    "prs": [
                        {"number": 101, "title": "Fix WS reconnect v1",
                         "author": "alice", "url": "#", "intent": "Fix WS",
                         "quality_score": 8.5, "quality_rank": 1,
                         "needs_human_review": False, "quality_summary": "Great"},
                        {"number": 102, "title": "Fix WS reconnect v2",
                         "author": "bob", "url": "#", "intent": "Fix WS",
                         "quality_score": 6.0, "quality_rank": 2,
                         "needs_human_review": True, "quality_summary": "OK"},
                    ],
                },
            ],
            "quality_scores": [
                {"pr_number": 101, "overall_score": 8.5, "pr_title": "Fix WS v1",
                 "pr_author": "alice", "pr_url": "#", "summary": "Great",
                 "needs_human_review": False},
            ],
            "alignment_scores": [
                {"pr_number": 101, "alignment_score": 3.0, "pr_title": "Fix WS v1",
                 "pr_author": "alice", "pr_url": "#", "recommendation": "CLOSE",
                 "drift_concerns": ["Not on roadmap"],
                 "rationale": "Conflicts with planned rewrite"},
            ],
        }

        json_file = tmp_path / "full-report.json"
        json_file.write_text(json.dumps(report_data))

        runner = CliRunner()
        with patch("claw_review.cli.generate_report") as mock_gen:
            mock_gen.return_value = str(tmp_path / "full-report.html")
            result = runner.invoke(cli, ["regenerate", str(json_file)])

        assert result.exit_code == 0
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["repo"] == "openclaw/openclaw"
        assert len(call_kwargs["clusters"]) == 1
        assert len(call_kwargs["alignment_scores"]) == 1
