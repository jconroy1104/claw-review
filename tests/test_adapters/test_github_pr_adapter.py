"""Tests for the GitHubPRAdapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from claw_review.adapters.github_pr import GitHubPRAdapter
from claw_review.github_client import PRData
from claw_review.platform.interfaces import DataAdapter, DataItem


def _make_pr(**overrides: Any) -> PRData:
    """Create a PRData with sensible defaults."""
    defaults: dict[str, Any] = {
        "number": 42,
        "title": "Fix memory leak",
        "body": "Fixes unclosed file handles.",
        "author": "alice",
        "created_at": "2025-06-01T00:00:00Z",
        "updated_at": "2025-06-02T00:00:00Z",
        "state": "open",
        "labels": ["bug"],
        "files_changed": ["src/parser.py", "tests/test_parser.py"],
        "additions": 25,
        "deletions": 10,
        "diff_summary": "--- src/parser.py\n+handle.close()",
        "url": "https://github.com/owner/repo/pull/42",
        "comments_count": 3,
    }
    defaults.update(overrides)
    return PRData(**defaults)


class TestGitHubPRAdapterProtocol:
    """Tests that GitHubPRAdapter satisfies the DataAdapter protocol."""

    def test_satisfies_data_adapter(self) -> None:
        adapter = GitHubPRAdapter(token="fake-token")
        assert isinstance(adapter, DataAdapter)

    def test_domain_is_github_pr(self) -> None:
        adapter = GitHubPRAdapter(token="fake")
        assert adapter.domain == "github-pr"


class TestFetchItems:
    """Tests for GitHubPRAdapter.fetch_items()."""

    async def test_returns_data_items(self) -> None:
        prs = [_make_pr(number=1, title="PR one"), _make_pr(number=2, title="PR two")]

        with patch(
            "claw_review.adapters.github_pr.fetch_open_prs",
            return_value=prs,
        ) as mock_fetch:
            adapter = GitHubPRAdapter(token="tok")
            items = await adapter.fetch_items("owner/repo", max_items=10)

        assert len(items) == 2
        assert all(isinstance(i, DataItem) for i in items)
        assert items[0].id == "1"
        assert items[0].title == "PR one"
        assert items[1].id == "2"
        mock_fetch.assert_called_once_with(
            repo="owner/repo", token="tok", max_prs=10, use_cache=True
        )

    async def test_metadata_populated(self) -> None:
        pr = _make_pr(number=99, author="bob")
        with patch(
            "claw_review.adapters.github_pr.fetch_open_prs",
            return_value=[pr],
        ):
            adapter = GitHubPRAdapter(token="tok")
            items = await adapter.fetch_items("owner/repo")

        item = items[0]
        assert item.metadata["author"] == "bob"
        assert "url" in item.metadata
        assert "created_at" in item.metadata


class TestFetchContextDocs:
    """Tests for GitHubPRAdapter.fetch_context_docs()."""

    async def test_returns_docs_dict(self) -> None:
        mock_docs = {"README.md": "# Project", "CONTRIBUTING.md": "Rules"}
        with patch(
            "claw_review.adapters.github_pr.fetch_repo_docs",
            return_value=mock_docs,
        ) as mock_fetch:
            adapter = GitHubPRAdapter(token="tok")
            docs = await adapter.fetch_context_docs("owner/repo")

        assert docs == mock_docs
        mock_fetch.assert_called_once_with(repo="owner/repo", token="tok")


class TestFormatItemForPrompt:
    """Tests for GitHubPRAdapter.format_item_for_prompt()."""

    def test_formats_pr_data(self) -> None:
        pr = _make_pr()
        item = GitHubPRAdapter._pr_to_data_item(pr)
        adapter = GitHubPRAdapter(token="tok")

        formatted = adapter.format_item_for_prompt(item)

        assert "Pull Request #42" in formatted
        assert "Fix memory leak" in formatted
        assert "alice" in formatted
        assert "src/parser.py" in formatted
        assert "+25/-10" in formatted
