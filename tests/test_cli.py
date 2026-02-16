"""Tests for claw_review.cli — Click CLI entry point."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from claw_review.cli import cli, _print_summary
from claw_review.github_client import PRData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pr(**overrides: object) -> PRData:
    """Build a minimal PRData for testing."""
    defaults: dict = {
        "number": 42,
        "title": "Fix memory leak in parser",
        "body": "Fixes a memory leak.",
        "author": "contributor123",
        "created_at": "2025-06-01T10:00:00Z",
        "updated_at": "2025-06-02T12:00:00Z",
        "state": "open",
        "labels": ["bug"],
        "files_changed": ["src/parser.py"],
        "additions": 25,
        "deletions": 10,
        "diff_summary": "--- src/parser.py\n+handle.close()",
        "url": "https://github.com/owner/repo/pull/42",
        "comments_count": 3,
    }
    defaults.update(overrides)
    return PRData(**defaults)


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


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------


class TestCliGroup:
    """Tests for the top-level CLI group."""

    def test_help_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "claw-review" in result.output

    def test_no_command_shows_usage(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [])
        # Click returns exit code 0 for group with no command
        assert "Usage" in result.output or "claw-review" in result.output


# ---------------------------------------------------------------------------
# Check Command
# ---------------------------------------------------------------------------


class TestCheckCommand:
    """Tests for 'claw-review check'."""

    @patch("claw_review.cli.Config")
    def test_check_valid_config(self, mock_config_cls: MagicMock) -> None:
        config = _valid_config()
        mock_config_cls.return_value = config

        runner = CliRunner()
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"open_issues_count": 42}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = runner.invoke(cli, ["check"])

        assert result.exit_code == 0
        assert "Configuration looks good" in result.output

    @patch("claw_review.cli.Config")
    def test_check_invalid_config(self, mock_config_cls: MagicMock) -> None:
        config = _valid_config()
        config.validate.return_value = ["GITHUB_TOKEN is required"]
        mock_config_cls.return_value = config

        runner = CliRunner()
        result = runner.invoke(cli, ["check"])

        assert result.exit_code == 0
        assert "GITHUB_TOKEN is required" in result.output

    @patch("claw_review.cli.Config")
    def test_check_github_api_failure(self, mock_config_cls: MagicMock) -> None:
        config = _valid_config()
        mock_config_cls.return_value = config

        runner = CliRunner()
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            result = runner.invoke(cli, ["check"])

        assert result.exit_code == 0
        assert "GitHub access failed" in result.output

    def test_check_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "Verify configuration" in result.output


# ---------------------------------------------------------------------------
# Analyze Command
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    """Tests for 'claw-review analyze'."""

    def test_analyze_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--repo" in result.output
        assert "--max-prs" in result.output
        assert "--skip-alignment" in result.output
        assert "--skip-quality" in result.output
        assert "--no-cache" in result.output
        assert "--json-only" in result.output

    @patch("claw_review.cli.Config")
    def test_analyze_exits_on_invalid_config(
        self, mock_config_cls: MagicMock
    ) -> None:
        config = _valid_config()
        config.validate.return_value = ["OPENROUTER_API_KEY is required"]
        mock_config_cls.return_value = config

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze"])

        assert result.exit_code == 1
        assert "OPENROUTER_API_KEY is required" in result.output

    @patch("claw_review.cli.generate_json_report")
    @patch("claw_review.cli.generate_report")
    @patch("claw_review.cli.fetch_open_prs")
    @patch("claw_review.cli.ModelPool")
    @patch("claw_review.cli.Config")
    def test_analyze_no_prs_exits_gracefully(
        self,
        mock_config_cls: MagicMock,
        mock_pool_cls: MagicMock,
        mock_fetch: MagicMock,
        mock_gen_report: MagicMock,
        mock_gen_json: MagicMock,
    ) -> None:
        mock_config_cls.return_value = _valid_config()
        mock_fetch.return_value = []

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze"])

        assert result.exit_code == 0
        assert "No open PRs found" in result.output
        mock_gen_report.assert_not_called()
        mock_gen_json.assert_not_called()

    @patch("claw_review.cli.generate_json_report", return_value="report.json")
    @patch("claw_review.cli.generate_report", return_value="report.html")
    @patch("claw_review.cli.cluster_intents")
    @patch("claw_review.cli.generate_embeddings")
    @patch("claw_review.cli.extract_intents")
    @patch("claw_review.cli.fetch_open_prs")
    @patch("claw_review.cli.ModelPool")
    @patch("claw_review.cli.Config")
    def test_analyze_skip_quality_and_alignment(
        self,
        mock_config_cls: MagicMock,
        mock_pool_cls: MagicMock,
        mock_fetch: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_cluster: MagicMock,
        mock_gen_report: MagicMock,
        mock_gen_json: MagicMock,
    ) -> None:
        mock_config_cls.return_value = _valid_config()

        pr = _make_pr()
        mock_fetch.return_value = [pr]

        mock_intent = MagicMock()
        mock_intent.embedding = [0.1] * 10
        mock_extract.return_value = [mock_intent]
        mock_embed.return_value = [mock_intent]

        mock_c = MagicMock()
        mock_c.prs = [{"number": 42, "title": "Fix", "author": "a", "url": "u", "intent": "i"}]
        mock_c.to_dict.return_value = {
            "cluster_id": "singleton-42",
            "prs": mock_c.prs,
            "intent_summary": "Fix",
            "category": "bugfix",
            "affected_area": "parser",
            "confidence": 0.0,
        }
        mock_cluster.return_value = [mock_c]

        runner = CliRunner()
        result = runner.invoke(
            cli, ["analyze", "--skip-quality", "--skip-alignment"]
        )

        assert result.exit_code == 0
        mock_gen_json.assert_called_once()
        mock_gen_report.assert_called_once()

    @patch("claw_review.cli.generate_json_report", return_value="report.json")
    @patch("claw_review.cli.cluster_intents")
    @patch("claw_review.cli.generate_embeddings")
    @patch("claw_review.cli.extract_intents")
    @patch("claw_review.cli.fetch_open_prs")
    @patch("claw_review.cli.ModelPool")
    @patch("claw_review.cli.Config")
    def test_analyze_json_only_skips_html(
        self,
        mock_config_cls: MagicMock,
        mock_pool_cls: MagicMock,
        mock_fetch: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_cluster: MagicMock,
        mock_gen_json: MagicMock,
    ) -> None:
        mock_config_cls.return_value = _valid_config()
        mock_fetch.return_value = [_make_pr()]

        mock_intent = MagicMock()
        mock_extract.return_value = [mock_intent]
        mock_embed.return_value = [mock_intent]

        mock_c = MagicMock()
        mock_c.prs = [{"number": 42}]
        mock_c.to_dict.return_value = {"cluster_id": "s-42", "prs": [{"number": 42}]}
        mock_cluster.return_value = [mock_c]

        runner = CliRunner()
        with patch("claw_review.cli.generate_report") as mock_html:
            result = runner.invoke(
                cli,
                ["analyze", "--skip-quality", "--skip-alignment", "--json-only"],
            )

        assert result.exit_code == 0
        mock_html.assert_not_called()
        mock_gen_json.assert_called_once()

    @patch("claw_review.cli.Config")
    def test_analyze_repo_override(self, mock_config_cls: MagicMock) -> None:
        config = _valid_config()
        mock_config_cls.return_value = config

        runner = CliRunner()
        with patch("claw_review.cli.fetch_open_prs", return_value=[]):
            with patch("claw_review.cli.ModelPool"):
                result = runner.invoke(cli, ["analyze", "--repo", "other/repo"])

        assert result.exit_code == 0
        assert config.target_repo == "other/repo"

    @patch("claw_review.cli.Config")
    def test_analyze_max_prs_override(self, mock_config_cls: MagicMock) -> None:
        config = _valid_config()
        mock_config_cls.return_value = config

        runner = CliRunner()
        with patch("claw_review.cli.fetch_open_prs", return_value=[]):
            with patch("claw_review.cli.ModelPool"):
                result = runner.invoke(cli, ["analyze", "--max-prs", "5"])

        assert result.exit_code == 0
        assert config.max_prs == 5


# ---------------------------------------------------------------------------
# Regenerate Command
# ---------------------------------------------------------------------------


class TestRegenerateCommand:
    """Tests for 'claw-review regenerate'."""

    def test_regenerate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["regenerate", "--help"])
        assert result.exit_code == 0
        assert "Regenerate HTML" in result.output

    @patch("claw_review.cli.generate_report")
    def test_regenerate_from_json(
        self, mock_gen_report: MagicMock, tmp_path: Path
    ) -> None:
        report_data = {
            "repo": "owner/repo",
            "clusters": [],
            "quality_scores": [],
            "alignment_scores": [],
            "providers": ["model/a"],
        }
        json_file = tmp_path / "report.json"
        json_file.write_text(json.dumps(report_data))

        runner = CliRunner()
        result = runner.invoke(cli, ["regenerate", str(json_file)])

        assert result.exit_code == 0
        mock_gen_report.assert_called_once()
        call_kwargs = mock_gen_report.call_args
        assert call_kwargs[1]["repo"] == "owner/repo"

    def test_regenerate_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["regenerate", "/nonexistent/file.json"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Print Summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Tests for the _print_summary helper."""

    def test_summary_with_all_data(self) -> None:
        clusters = [
            {"prs": [{"number": 1}, {"number": 2}]},
            {"prs": [{"number": 3}]},
        ]
        quality = [
            {"overall_score": 8.0, "needs_human_review": False},
            {"overall_score": 6.0, "needs_human_review": True},
        ]
        alignment = [
            {"alignment_score": 3.0},
            {"alignment_score": 7.0},
        ]
        # Should not raise
        _print_summary(clusters, quality, alignment)

    def test_summary_empty_data(self) -> None:
        # Should not raise
        _print_summary([], [], [])

    def test_summary_no_quality_or_alignment(self) -> None:
        clusters = [{"prs": [{"number": 1}]}]
        _print_summary(clusters, [], [])


# ---------------------------------------------------------------------------
# Presets Command
# ---------------------------------------------------------------------------


class TestPresetsCommand:
    """Tests for 'claw-review presets'."""

    def test_presets_shows_all(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["presets"])
        assert result.exit_code == 0
        assert "fast" in result.output
        assert "balanced" in result.output
        assert "thorough" in result.output

    def test_presets_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["presets", "--help"])
        assert result.exit_code == 0
        assert "List available model presets" in result.output


# ---------------------------------------------------------------------------
# Estimate Command
# ---------------------------------------------------------------------------


class TestEstimateCommand:
    """Tests for 'claw-review estimate'."""

    def test_estimate_basic(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "--repo", "owner/repo"])
        assert result.exit_code == 0
        assert "Cost Estimate" in result.output
        assert "owner/repo" in result.output

    def test_estimate_with_preset(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["estimate", "--repo", "owner/repo", "--preset", "fast"]
        )
        assert result.exit_code == 0
        assert "fast" in result.output

    def test_estimate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "--repo" in result.output
        assert "--preset" in result.output
        assert "--max-prs" in result.output


# ---------------------------------------------------------------------------
# Analyze with Preset/Budget Flags
# ---------------------------------------------------------------------------


class TestAnalyzePreset:
    """Tests for analyze command with --preset and --budget flags."""

    def test_analyze_with_preset_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--preset" in result.output

    def test_analyze_with_budget_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--budget" in result.output

    @patch("claw_review.cli.fetch_open_prs", return_value=[])
    @patch("claw_review.cli.ModelPool")
    @patch("claw_review.cli.Config")
    def test_analyze_preset_overrides_models(
        self,
        mock_config_cls: MagicMock,
        mock_pool_cls: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        config = _valid_config()
        mock_config_cls.return_value = config

        runner = CliRunner()
        result = runner.invoke(
            cli, ["analyze", "--preset", "fast"]
        )

        assert result.exit_code == 0
        assert "Preset: fast" in result.output
        # The config models should have been overridden to the fast preset
        assert config.models == [
            "meta-llama/llama-3.1-70b-instruct",
            "mistralai/mistral-large-latest",
            "google/gemini-2.0-flash-001",
        ]


# ---------------------------------------------------------------------------
# Merge Command
# ---------------------------------------------------------------------------


class TestMergeCommand:
    """Tests for 'claw-review merge'."""

    def test_merge_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--help"])
        assert result.exit_code == 0
        assert "Merge multiple JSON reports" in result.output
        assert "--output" in result.output
        assert "--json-only" in result.output

    def test_merge_with_mock_reports(self, tmp_path: Path) -> None:
        report1 = {
            "repo": "owner/repo",
            "timestamp": "2025-06-01T00:00:00Z",
            "providers": ["model/a"],
            "clusters": [
                {"cluster_id": "c-0", "prs": [{"number": 1, "title": "PR1"}]},
            ],
            "quality_scores": [{"pr_number": 1, "overall_score": 7.0}],
            "alignment_scores": [],
        }
        report2 = {
            "repo": "owner/repo",
            "timestamp": "2025-06-02T00:00:00Z",
            "providers": ["model/b"],
            "clusters": [
                {"cluster_id": "c-1", "prs": [{"number": 2, "title": "PR2"}]},
            ],
            "quality_scores": [{"pr_number": 2, "overall_score": 8.0}],
            "alignment_scores": [],
        }

        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        f1.write_text(json.dumps(report1))
        f2.write_text(json.dumps(report2))

        output_base = str(tmp_path / "merged")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["merge", str(f1), str(f2), "--output", output_base, "--json-only"]
        )

        assert result.exit_code == 0
        assert "Merging 2 report" in result.output

        # Verify the merged JSON was written
        merged_json = Path(f"{output_base}.json")
        assert merged_json.exists()
        merged = json.loads(merged_json.read_text())
        assert len(merged["clusters"]) == 2
        assert len(merged["quality_scores"]) == 2

    def test_merge_no_files_shows_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["merge"])
        # Click should show usage error since REPORT_FILES is required
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Status Command
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """Tests for 'claw-review status'."""

    def test_status_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show analysis state" in result.output
        assert "--repo" in result.output

    @patch("claw_review.cli.Config")
    @patch("claw_review.cli.load_state")
    def test_status_shows_state_info(
        self, mock_load_state: MagicMock, mock_config_cls: MagicMock
    ) -> None:
        config = _valid_config()
        mock_config_cls.return_value = config

        from claw_review.state import AnalysisState, AnalyzedPR
        state = AnalysisState(
            repo="owner/repo",
            analyzed_prs={
                1: AnalyzedPR(timestamp="2025-06-01T00:00:00Z"),
                2: AnalyzedPR(timestamp="2025-06-01T00:00:00Z"),
            },
            last_run_timestamp="2025-06-01T00:00:00Z",
            model_config=["model/a", "model/b"],
        )
        mock_load_state.return_value = state

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--repo", "owner/repo"])

        assert result.exit_code == 0
        assert "owner/repo" in result.output
        assert "2" in result.output  # 2 analyzed PRs
        assert "2025-06-01" in result.output

    @patch("claw_review.cli.Config")
    @patch("claw_review.cli.load_state")
    def test_status_fresh_repo(
        self, mock_load_state: MagicMock, mock_config_cls: MagicMock
    ) -> None:
        config = _valid_config()
        mock_config_cls.return_value = config

        from claw_review.state import AnalysisState
        mock_load_state.return_value = AnalysisState(repo="owner/fresh")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--repo", "owner/fresh"])

        assert result.exit_code == 0
        assert "0" in result.output  # 0 analyzed PRs
        assert "Never" in result.output


# ---------------------------------------------------------------------------
# Analyze Batch/Force Flags
# ---------------------------------------------------------------------------


class TestAnalyzeBatchFlags:
    """Tests for analyze --batch-size and --force flags."""

    def test_analyze_batch_size_flag_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--batch-size" in result.output

    def test_analyze_force_flag_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output

    def test_analyze_incremental_flag_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--incremental" in result.output
        assert "--no-incremental" in result.output
