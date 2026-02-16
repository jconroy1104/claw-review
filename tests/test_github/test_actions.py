"""Tests for GitHub Actions workflow generator."""
from __future__ import annotations

from pathlib import Path

from claw_review.github.actions import generate_workflow, save_workflow


class TestGenerateWorkflow:
    """Tests for generate_workflow."""

    def test_default_produces_valid_yaml(self) -> None:
        """Default options should produce a workflow with balanced preset."""
        result = generate_workflow()
        assert "ClawReview PR Triage" in result
        assert "--preset balanced" in result
        assert "actions/checkout@v4" in result
        assert "pip install claw-review" in result

    def test_custom_preset(self) -> None:
        """Custom preset should be included in the run command."""
        result = generate_workflow(preset="thorough")
        assert "--preset thorough" in result

    def test_budget_flag(self) -> None:
        """Budget flag should appear when set."""
        result = generate_workflow(budget=5.0)
        assert "--budget 5.0" in result

    def test_skip_alignment_flag(self) -> None:
        """Skip-alignment flag should appear when enabled."""
        result = generate_workflow(skip_alignment=True)
        assert "--skip-alignment" in result

    def test_all_flags_combined(self) -> None:
        """All optional flags should coexist."""
        result = generate_workflow(preset="fast", budget=2.5, skip_alignment=True)
        assert "--preset fast" in result
        assert "--budget 2.5" in result
        assert "--skip-alignment" in result

    def test_contains_github_token_secret(self) -> None:
        """Workflow should reference GITHUB_TOKEN secret."""
        result = generate_workflow()
        assert "secrets.GITHUB_TOKEN" in result

    def test_contains_openrouter_secret(self) -> None:
        """Workflow should reference OPENROUTER_API_KEY secret."""
        result = generate_workflow()
        assert "secrets.OPENROUTER_API_KEY" in result


class TestSaveWorkflow:
    """Tests for save_workflow."""

    def test_writes_file_to_correct_path(self, tmp_path: Path) -> None:
        """Workflow file should be written to .github/workflows/claw-review.yml."""
        result_path = save_workflow(str(tmp_path))
        expected = tmp_path / ".github" / "workflows" / "claw-review.yml"
        assert expected.exists()
        assert str(expected.resolve()) == result_path

    def test_file_content_matches_generate(self, tmp_path: Path) -> None:
        """Saved file content should match generate_workflow output."""
        save_workflow(str(tmp_path), preset="fast", budget=1.0)
        content = (tmp_path / ".github" / "workflows" / "claw-review.yml").read_text()
        expected = generate_workflow(preset="fast", budget=1.0)
        assert content == expected
