"""Tests for PR comment formatter and poster."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claw_review.github.commenter import COMMENT_MARKER, PRCommenter


@pytest.fixture()
def commenter() -> PRCommenter:
    """Create a PRCommenter with a fake token."""
    return PRCommenter(token="ghp_fake_token")


@pytest.fixture()
def quality_scores() -> dict:
    """Sample quality scores dict."""
    return {
        "dimensions": {
            "Code Quality": 8.0,
            "Test Coverage": 7.0,
            "Documentation": 6.5,
            "Complexity": 9.0,
            "Maintainability": 7.5,
        },
        "overall": 7.6,
        "per_model": {
            "claude": {
                "Code Quality": 8.5,
                "Test Coverage": 7.0,
                "Documentation": 6.0,
                "Complexity": 9.0,
                "Maintainability": 8.0,
            },
            "gpt-4o": {
                "Code Quality": 7.5,
                "Test Coverage": 7.0,
                "Documentation": 7.0,
                "Complexity": 9.0,
                "Maintainability": 7.0,
            },
        },
    }


class TestFormatComment:
    """Tests for PRCommenter.format_comment."""

    def test_renders_quality_scores(self, commenter: PRCommenter, quality_scores: dict) -> None:
        """All dimension scores should appear in the rendered Markdown."""
        result = commenter.format_comment(quality_scores, models=["claude", "gpt-4o"])
        assert "Code Quality" in result
        assert "8.0/10" in result
        assert "7.6/10" in result
        assert "MERGE" in result  # 7.6 >= 7.5

    def test_handles_empty_scores(self, commenter: PRCommenter) -> None:
        """Empty dimensions should produce a comment without table rows."""
        result = commenter.format_comment({"dimensions": {}, "overall": 0.0})
        assert COMMENT_MARKER in result
        assert "0.0/10" in result
        assert "CLOSE" in result  # 0.0 < 3.5

    def test_includes_duplicates(self, commenter: PRCommenter, quality_scores: dict) -> None:
        """Duplicate references should appear in the comment."""
        result = commenter.format_comment(quality_scores, duplicates=["#123", "#456"])
        assert "#123" in result
        assert "#456" in result
        assert "Potential duplicates found" in result

    def test_includes_alignment(self, commenter: PRCommenter, quality_scores: dict) -> None:
        """Alignment data should render when provided."""
        alignment = {"score": 8.5, "summary": "Well aligned with project vision."}
        result = commenter.format_comment(quality_scores, alignment=alignment)
        assert "Vision Alignment" in result
        assert "8.5/10" in result
        assert "Well aligned" in result

    def test_no_alignment_section_when_none(self, commenter: PRCommenter, quality_scores: dict) -> None:
        """No alignment section should appear when alignment is None."""
        result = commenter.format_comment(quality_scores)
        assert "Vision Alignment" not in result

    def test_no_duplicates_section_when_none(self, commenter: PRCommenter, quality_scores: dict) -> None:
        """No duplicates warning should appear when duplicates is None."""
        result = commenter.format_comment(quality_scores)
        assert "Potential duplicates" not in result

    def test_overall_computed_from_dimensions(self, commenter: PRCommenter) -> None:
        """Overall should be computed if not provided."""
        scores = {
            "dimensions": {"A": 6.0, "B": 8.0},
        }
        result = commenter.format_comment(scores)
        assert "7.0/10" in result  # (6+8)/2

    def test_per_model_breakdown_rendered(self, commenter: PRCommenter, quality_scores: dict) -> None:
        """Per-model breakdown should appear in collapsible details."""
        result = commenter.format_comment(quality_scores, models=["claude", "gpt-4o"])
        assert "Per-model breakdown" in result
        assert "claude" in result
        assert "gpt-4o" in result


class TestPostComment:
    """Tests for PRCommenter async methods."""

    @pytest.mark.asyncio
    async def test_post_comment(self, commenter: PRCommenter) -> None:
        """post_comment should POST to the GitHub API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": 1, "body": "test"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("claw_review.github.commenter.httpx.AsyncClient", return_value=mock_client):
            result = await commenter.post_comment("owner/repo", 42, "Hello")
            assert result["id"] == 1
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_or_create_creates_new(self, commenter: PRCommenter) -> None:
        """update_or_create_comment should create when no existing comment."""
        # Mock find_existing_comment to return None
        mock_resp_list = MagicMock()
        mock_resp_list.json.return_value = []
        mock_resp_list.raise_for_status = MagicMock()

        mock_resp_create = MagicMock()
        mock_resp_create.json.return_value = {"id": 99, "body": "new"}
        mock_resp_create.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp_list
        mock_client.post.return_value = mock_resp_create
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("claw_review.github.commenter.httpx.AsyncClient", return_value=mock_client):
            result = await commenter.update_or_create_comment("owner/repo", 1, "body")
            assert result["id"] == 99

    @pytest.mark.asyncio
    async def test_update_or_create_updates_existing(self, commenter: PRCommenter) -> None:
        """update_or_create_comment should PATCH when existing comment found."""
        mock_resp_list = MagicMock()
        mock_resp_list.json.return_value = [
            {"id": 55, "body": f"## 🦞 {COMMENT_MARKER}\nold content"},
        ]
        mock_resp_list.raise_for_status = MagicMock()

        mock_resp_patch = MagicMock()
        mock_resp_patch.json.return_value = {"id": 55, "body": "updated"}
        mock_resp_patch.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp_list
        mock_client.patch.return_value = mock_resp_patch
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("claw_review.github.commenter.httpx.AsyncClient", return_value=mock_client):
            result = await commenter.update_or_create_comment("owner/repo", 1, "new body")
            assert result["id"] == 55
            mock_client.patch.assert_called_once()
