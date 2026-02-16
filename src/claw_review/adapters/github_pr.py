"""GitHub Pull Request adapter for the consensus platform.

Wraps the existing github_client.py functions to conform to the
DataAdapter protocol, converting PRData objects to DataItems.
"""

from __future__ import annotations

from typing import Any

from ..github_client import PRData, fetch_open_prs, fetch_repo_docs
from ..platform.interfaces import DataItem


class GitHubPRAdapter:
    """Adapter that fetches GitHub PRs and converts them to DataItems.

    Wraps the existing github_client functions so the consensus
    engine can process PRs without knowing GitHub-specific details.
    """

    domain: str = "github-pr"

    def __init__(self, token: str) -> None:
        """Initialize with a GitHub API token.

        Args:
            token: GitHub personal access token.
        """
        self._token = token

    async def fetch_items(
        self, source: str, max_items: int = 100, **kwargs: Any
    ) -> list[DataItem]:
        """Fetch open PRs from a GitHub repository.

        Args:
            source: Repository in "owner/repo" format.
            max_items: Maximum number of PRs to fetch.
            **kwargs: Passed through to fetch_open_prs (e.g., use_cache).

        Returns:
            List of DataItem objects wrapping PRData.
        """
        use_cache = kwargs.get("use_cache", True)
        prs = fetch_open_prs(
            repo=source,
            token=self._token,
            max_prs=max_items,
            use_cache=use_cache,
        )
        return [self._pr_to_data_item(pr) for pr in prs]

    async def fetch_context_docs(
        self, source: str, **kwargs: Any
    ) -> dict[str, str]:
        """Fetch repository vision documents for alignment scoring.

        Args:
            source: Repository in "owner/repo" format.
            **kwargs: Additional parameters (unused).

        Returns:
            Dict mapping filename to document content.
        """
        return fetch_repo_docs(repo=source, token=self._token)

    def format_item_for_prompt(self, item: DataItem) -> str:
        """Format a DataItem for inclusion in an LLM prompt.

        Reconstructs the prompt format used by the existing
        clustering module's _build_intent_prompt.

        Args:
            item: DataItem wrapping a GitHub PR.

        Returns:
            Formatted string suitable for an LLM prompt.
        """
        raw = item.raw
        files = raw.get("files_changed", [])
        files_str = "\n".join(f"  - {f}" for f in files[:20])
        if len(files) > 20:
            files_str += f"\n  ... and {len(files) - 20} more files"

        body_truncated = (item.body or "No description provided.")[:3000]
        labels = raw.get("labels", [])
        additions = raw.get("additions", 0)
        deletions = raw.get("deletions", 0)
        diff_summary = raw.get("diff_summary", "")

        return (
            f"Pull Request #{item.id}: {item.title}\n"
            f"Author: {item.metadata.get('author', 'unknown')}\n"
            f"Labels: {', '.join(labels) if labels else 'none'}\n"
            f"Changes: +{additions}/-{deletions}\n\n"
            f"Description:\n{body_truncated}\n\n"
            f"Files changed:\n{files_str}\n\n"
            f"Diff summary (truncated):\n{diff_summary[:4000]}\n"
        )

    @staticmethod
    def _pr_to_data_item(pr: PRData) -> DataItem:
        """Convert a PRData object to a DataItem.

        Args:
            pr: The PRData to convert.

        Returns:
            A DataItem wrapping the PR data.
        """
        return DataItem(
            id=str(pr.number),
            title=pr.title,
            body=pr.body or "",
            metadata={
                "author": pr.author,
                "created_at": pr.created_at,
                "updated_at": pr.updated_at,
                "url": pr.url,
                "comments_count": pr.comments_count,
            },
            raw=pr.to_dict(),
        )
