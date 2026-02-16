"""Shared pytest fixtures for claw-review test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from claw_review.github_client import PRData


@pytest.fixture()
def sample_pr_data() -> callable:
    """Factory fixture that returns PRData objects with configurable fields."""

    def _factory(**overrides: Any) -> PRData:
        defaults: dict[str, Any] = {
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

    return _factory


@pytest.fixture()
def mock_github_pr_list_response() -> list[dict[str, Any]]:
    """Mock GitHub API response for listing pull requests."""
    return [
        {
            "number": 1,
            "title": "Add feature A",
            "state": "open",
            "updated_at": "2025-06-01T00:00:00Z",
        },
        {
            "number": 2,
            "title": "Fix bug B",
            "state": "open",
            "updated_at": "2025-06-02T00:00:00Z",
        },
    ]


@pytest.fixture()
def mock_github_pr_detail() -> dict[str, Any]:
    """Mock GitHub API response for a single PR detail."""
    return {
        "number": 1,
        "title": "Add feature A",
        "body": "Implements feature A with tests.",
        "user": {"login": "alice"},
        "created_at": "2025-06-01T00:00:00Z",
        "updated_at": "2025-06-01T12:00:00Z",
        "state": "open",
        "labels": [{"name": "enhancement"}],
        "additions": 50,
        "deletions": 5,
        "html_url": "https://github.com/owner/repo/pull/1",
        "comments": 2,
        "review_comments": 1,
    }


@pytest.fixture()
def mock_github_pr_files() -> list[dict[str, Any]]:
    """Mock GitHub API response for PR files."""
    return [
        {
            "filename": "src/feature_a.py",
            "additions": 40,
            "deletions": 0,
            "patch": "@@ -0,0 +1,40 @@\n+class FeatureA:\n+    pass",
        },
        {
            "filename": "tests/test_feature_a.py",
            "additions": 10,
            "deletions": 5,
            "patch": "@@ -1,5 +1,10 @@\n+import pytest",
        },
    ]


@pytest.fixture()
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory for tests."""
    cache_dir = tmp_path / ".claw-review-cache"
    cache_dir.mkdir()
    return cache_dir
