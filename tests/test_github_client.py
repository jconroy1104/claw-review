"""Tests for claw_review.github_client module."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from claw_review.github_client import (
    GITHUB_API,
    MAX_DIFF_CHARS,
    PRData,
    _cache_key,
    _load_cached,
    _save_cache,
    _truncate_diff,
    fetch_open_prs,
    fetch_repo_docs,
)


class TestPRData:
    """Tests for PRData dataclass."""

    def test_creation_with_all_fields(self, sample_pr_data: callable) -> None:
        pr = sample_pr_data()
        assert pr.number == 42
        assert pr.title == "Fix memory leak in parser"
        assert pr.author == "contributor123"
        assert pr.state == "open"
        assert pr.labels == ["bug", "priority:high"]
        assert pr.files_changed == ["src/parser.py", "tests/test_parser.py"]
        assert pr.additions == 25
        assert pr.deletions == 10
        assert pr.comments_count == 3

    def test_creation_with_overrides(self, sample_pr_data: callable) -> None:
        pr = sample_pr_data(number=99, title="Custom title", author="bob")
        assert pr.number == 99
        assert pr.title == "Custom title"
        assert pr.author == "bob"

    def test_to_dict(self, sample_pr_data: callable) -> None:
        pr = sample_pr_data()
        d = pr.to_dict()
        assert isinstance(d, dict)
        assert d["number"] == 42
        assert d["title"] == "Fix memory leak in parser"
        assert d["labels"] == ["bug", "priority:high"]
        assert d["files_changed"] == ["src/parser.py", "tests/test_parser.py"]

    def test_to_dict_roundtrip(self, sample_pr_data: callable) -> None:
        pr = sample_pr_data()
        d = pr.to_dict()
        pr2 = PRData(**d)
        assert pr == pr2


class TestCacheKey:
    """Tests for _cache_key() function."""

    def test_deterministic(self) -> None:
        key1 = _cache_key("owner/repo", 42)
        key2 = _cache_key("owner/repo", 42)
        assert key1 == key2

    def test_different_repos_different_keys(self) -> None:
        key1 = _cache_key("owner/repo-a", 1)
        key2 = _cache_key("owner/repo-b", 1)
        assert key1 != key2

    def test_different_prs_different_keys(self) -> None:
        key1 = _cache_key("owner/repo", 1)
        key2 = _cache_key("owner/repo", 2)
        assert key1 != key2

    def test_returns_hex_string(self) -> None:
        key = _cache_key("owner/repo", 1)
        assert isinstance(key, str)
        # MD5 hex is 32 chars
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


class TestTruncateDiff:
    """Tests for _truncate_diff() function."""

    def test_under_limit_unchanged(self) -> None:
        diff = "short diff"
        result = _truncate_diff(diff, max_chars=100)
        assert result == diff

    def test_at_limit_unchanged(self) -> None:
        diff = "x" * 100
        result = _truncate_diff(diff, max_chars=100)
        assert result == diff

    def test_over_limit_truncated(self) -> None:
        diff = "a" * 200
        result = _truncate_diff(diff, max_chars=100)
        assert len(result) < 200
        assert "TRUNCATED" in result
        assert "100 chars" in result  # 200 - 100 = 100 truncated chars

    def test_truncated_includes_start_and_end(self) -> None:
        diff = "START" + "x" * 200 + "END"
        result = _truncate_diff(diff, max_chars=100)
        assert result.startswith("START")
        assert result.endswith("END")

    def test_empty_string(self) -> None:
        result = _truncate_diff("")
        assert result == ""

    def test_default_max_chars(self) -> None:
        # Verify the default is MAX_DIFF_CHARS
        diff = "a" * (MAX_DIFF_CHARS + 100)
        result = _truncate_diff(diff)
        assert "TRUNCATED" in result


class TestCacheLoadSave:
    """Tests for _load_cached() and _save_cache()."""

    def test_cache_miss_returns_none(self, tmp_cache_dir: Path) -> None:
        with patch("claw_review.github_client.CACHE_DIR", tmp_cache_dir):
            result = _load_cached("owner/repo", 999)
            assert result is None

    def test_save_then_load_roundtrip(
        self, tmp_cache_dir: Path, sample_pr_data: callable
    ) -> None:
        pr = sample_pr_data(number=7)
        with patch("claw_review.github_client.CACHE_DIR", tmp_cache_dir):
            _save_cache("owner/repo", pr)
            loaded = _load_cached("owner/repo", 7)
            assert loaded is not None
            assert loaded.number == 7
            assert loaded.title == pr.title
            assert loaded.to_dict() == pr.to_dict()

    def test_save_creates_cache_dir(self, tmp_path: Path, sample_pr_data: callable) -> None:
        cache_dir = tmp_path / "new-cache-dir"
        assert not cache_dir.exists()
        pr = sample_pr_data()
        with patch("claw_review.github_client.CACHE_DIR", cache_dir):
            _save_cache("owner/repo", pr)
        assert cache_dir.exists()

    def test_cache_file_is_valid_json(
        self, tmp_cache_dir: Path, sample_pr_data: callable
    ) -> None:
        pr = sample_pr_data(number=10)
        with patch("claw_review.github_client.CACHE_DIR", tmp_cache_dir):
            _save_cache("owner/repo", pr)

        # Find the cache file and verify JSON
        cache_files = list(tmp_cache_dir.glob("*.json"))
        assert len(cache_files) == 1
        data = json.loads(cache_files[0].read_text())
        assert data["number"] == 10


def _make_pr_list_page(numbers: list[int]) -> list[dict[str, Any]]:
    """Helper to create a page of PR list items."""
    return [{"number": n, "title": f"PR #{n}", "state": "open"} for n in numbers]


def _make_pr_detail(number: int) -> dict[str, Any]:
    """Helper to create a PR detail response."""
    return {
        "number": number,
        "title": f"PR #{number}",
        "body": f"Body of PR #{number}",
        "user": {"login": f"author{number}"},
        "created_at": "2025-06-01T00:00:00Z",
        "updated_at": "2025-06-02T00:00:00Z",
        "state": "open",
        "labels": [{"name": "bug"}],
        "additions": 10,
        "deletions": 5,
        "html_url": f"https://github.com/owner/repo/pull/{number}",
        "comments": 1,
        "review_comments": 0,
    }


def _make_pr_files(number: int) -> list[dict[str, Any]]:
    """Helper to create a PR files response."""
    return [
        {
            "filename": f"src/file{number}.py",
            "additions": 10,
            "deletions": 5,
            "patch": "@@ -1,5 +1,10 @@\n+new code",
        }
    ]


class TestFetchOpenPrs:
    """Tests for fetch_open_prs() function."""

    @respx.mock
    def test_single_page(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        # Mock PR list
        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            return_value=httpx.Response(200, json=_make_pr_list_page([1, 2]))
        )
        # Second page empty
        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            side_effect=[
                httpx.Response(200, json=_make_pr_list_page([1, 2])),
                httpx.Response(200, json=[]),
            ]
        )

        # Mock details and files for each PR
        for n in [1, 2]:
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/{n}").mock(
                return_value=httpx.Response(200, json=_make_pr_detail(n))
            )
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/{n}/files").mock(
                return_value=httpx.Response(200, json=_make_pr_files(n))
            )

        prs = fetch_open_prs(repo, token, max_prs=10, use_cache=False)
        assert len(prs) == 2
        assert prs[0].number == 1
        assert prs[1].number == 2
        assert prs[0].author == "author1"

    @respx.mock
    def test_empty_result(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            return_value=httpx.Response(200, json=[])
        )

        prs = fetch_open_prs(repo, token, max_prs=10, use_cache=False)
        assert prs == []

    @respx.mock
    def test_pagination_two_pages(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        # Two pages of results, then empty
        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            side_effect=[
                httpx.Response(200, json=_make_pr_list_page([1, 2])),
                httpx.Response(200, json=_make_pr_list_page([3])),
                httpx.Response(200, json=[]),
            ]
        )

        for n in [1, 2, 3]:
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/{n}").mock(
                return_value=httpx.Response(200, json=_make_pr_detail(n))
            )
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/{n}/files").mock(
                return_value=httpx.Response(200, json=_make_pr_files(n))
            )

        prs = fetch_open_prs(repo, token, max_prs=10, use_cache=False)
        assert len(prs) == 3

    @respx.mock
    def test_max_prs_limits_results(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            side_effect=[
                httpx.Response(200, json=_make_pr_list_page([1, 2, 3, 4, 5])),
                httpx.Response(200, json=[]),
            ]
        )

        for n in [1, 2]:
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/{n}").mock(
                return_value=httpx.Response(200, json=_make_pr_detail(n))
            )
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/{n}/files").mock(
                return_value=httpx.Response(200, json=_make_pr_files(n))
            )

        prs = fetch_open_prs(repo, token, max_prs=2, use_cache=False)
        assert len(prs) == 2

    @respx.mock
    def test_cache_hit_skips_api(self, tmp_cache_dir: Path, sample_pr_data: callable) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        # Pre-populate cache for PR #1
        cached_pr = sample_pr_data(number=1)
        with patch("claw_review.github_client.CACHE_DIR", tmp_cache_dir):
            _save_cache(repo, cached_pr)

            respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
                side_effect=[
                    httpx.Response(200, json=_make_pr_list_page([1])),
                    httpx.Response(200, json=[]),
                ]
            )
            # Do NOT mock detail/files endpoints -- if they get called, respx will error

            prs = fetch_open_prs(repo, token, max_prs=10, use_cache=True)
            assert len(prs) == 1
            assert prs[0].number == 1
            assert prs[0].title == cached_pr.title

    @respx.mock
    def test_use_cache_false_always_hits_api(
        self, tmp_cache_dir: Path, sample_pr_data: callable
    ) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        # Pre-populate cache
        cached_pr = sample_pr_data(number=1, title="CACHED TITLE")
        with patch("claw_review.github_client.CACHE_DIR", tmp_cache_dir):
            _save_cache(repo, cached_pr)

            respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
                side_effect=[
                    httpx.Response(200, json=_make_pr_list_page([1])),
                    httpx.Response(200, json=[]),
                ]
            )
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/1").mock(
                return_value=httpx.Response(200, json=_make_pr_detail(1))
            )
            respx.get(f"{GITHUB_API}/repos/{repo}/pulls/1/files").mock(
                return_value=httpx.Response(200, json=_make_pr_files(1))
            )

            prs = fetch_open_prs(repo, token, max_prs=10, use_cache=False)
            assert len(prs) == 1
            # Should have fetched from API, not cache
            assert prs[0].title == "PR #1"

    @respx.mock
    def test_pr_with_empty_body(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            side_effect=[
                httpx.Response(200, json=_make_pr_list_page([1])),
                httpx.Response(200, json=[]),
            ]
        )
        detail = _make_pr_detail(1)
        detail["body"] = None  # GitHub returns null for empty body
        respx.get(f"{GITHUB_API}/repos/{repo}/pulls/1").mock(
            return_value=httpx.Response(200, json=detail)
        )
        respx.get(f"{GITHUB_API}/repos/{repo}/pulls/1/files").mock(
            return_value=httpx.Response(200, json=_make_pr_files(1))
        )

        prs = fetch_open_prs(repo, token, max_prs=10, use_cache=False)
        assert prs[0].body == ""

    @respx.mock
    def test_http_error_raises(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            return_value=httpx.Response(403, json={"message": "Forbidden"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            fetch_open_prs(repo, token, max_prs=10, use_cache=False)

    @respx.mock
    def test_server_error_raises(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        respx.get(f"{GITHUB_API}/repos/{repo}/pulls").mock(
            return_value=httpx.Response(500, json={"message": "Internal Server Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            fetch_open_prs(repo, token, max_prs=10, use_cache=False)


class TestFetchRepoDocs:
    """Tests for fetch_repo_docs() function."""

    @respx.mock
    def test_found_docs(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"
        readme_content = base64.b64encode(b"# My Project\nReadme content").decode()

        respx.get(f"{GITHUB_API}/repos/{repo}/contents/README.md").mock(
            return_value=httpx.Response(
                200,
                json={"encoding": "base64", "content": readme_content},
            )
        )
        # All other doc paths return 404
        for path in [
            "CONTRIBUTING.md",
            "ARCHITECTURE.md",
            "docs/ARCHITECTURE.md",
            "ROADMAP.md",
            "docs/ROADMAP.md",
        ]:
            respx.get(f"{GITHUB_API}/repos/{repo}/contents/{path}").mock(
                return_value=httpx.Response(404, json={"message": "Not Found"})
            )

        docs = fetch_repo_docs(repo, token)
        assert "README.md" in docs
        assert docs["README.md"] == "# My Project\nReadme content"

    @respx.mock
    def test_missing_docs_404(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        for path in [
            "README.md",
            "CONTRIBUTING.md",
            "ARCHITECTURE.md",
            "docs/ARCHITECTURE.md",
            "ROADMAP.md",
            "docs/ROADMAP.md",
        ]:
            respx.get(f"{GITHUB_API}/repos/{repo}/contents/{path}").mock(
                return_value=httpx.Response(404, json={"message": "Not Found"})
            )

        docs = fetch_repo_docs(repo, token)
        assert docs == {}

    @respx.mock
    def test_mixed_found_and_missing(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        readme_content = base64.b64encode(b"# Readme").decode()
        contributing_content = base64.b64encode(b"# Contributing Guide").decode()

        respx.get(f"{GITHUB_API}/repos/{repo}/contents/README.md").mock(
            return_value=httpx.Response(
                200, json={"encoding": "base64", "content": readme_content}
            )
        )
        respx.get(f"{GITHUB_API}/repos/{repo}/contents/CONTRIBUTING.md").mock(
            return_value=httpx.Response(
                200, json={"encoding": "base64", "content": contributing_content}
            )
        )
        for path in [
            "ARCHITECTURE.md",
            "docs/ARCHITECTURE.md",
            "ROADMAP.md",
            "docs/ROADMAP.md",
        ]:
            respx.get(f"{GITHUB_API}/repos/{repo}/contents/{path}").mock(
                return_value=httpx.Response(404, json={"message": "Not Found"})
            )

        docs = fetch_repo_docs(repo, token)
        assert len(docs) == 2
        assert "README.md" in docs
        assert "CONTRIBUTING.md" in docs

    @respx.mock
    def test_base64_decoding(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"
        original = "Hello, World! Unicode: \u00e9\u00e0\u00fc"
        encoded = base64.b64encode(original.encode("utf-8")).decode()

        respx.get(f"{GITHUB_API}/repos/{repo}/contents/README.md").mock(
            return_value=httpx.Response(
                200, json={"encoding": "base64", "content": encoded}
            )
        )
        for path in [
            "CONTRIBUTING.md",
            "ARCHITECTURE.md",
            "docs/ARCHITECTURE.md",
            "ROADMAP.md",
            "docs/ROADMAP.md",
        ]:
            respx.get(f"{GITHUB_API}/repos/{repo}/contents/{path}").mock(
                return_value=httpx.Response(404, json={"message": "Not Found"})
            )

        docs = fetch_repo_docs(repo, token)
        assert docs["README.md"] == original

    @respx.mock
    def test_http_error_skips_gracefully(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"

        # README raises a connection error, others are 404
        respx.get(f"{GITHUB_API}/repos/{repo}/contents/README.md").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        for path in [
            "CONTRIBUTING.md",
            "ARCHITECTURE.md",
            "docs/ARCHITECTURE.md",
            "ROADMAP.md",
            "docs/ROADMAP.md",
        ]:
            respx.get(f"{GITHUB_API}/repos/{repo}/contents/{path}").mock(
                return_value=httpx.Response(404, json={"message": "Not Found"})
            )

        docs = fetch_repo_docs(repo, token)
        # Should not raise, just skip the failed doc
        assert "README.md" not in docs

    @respx.mock
    def test_long_doc_truncated(self) -> None:
        repo = "owner/repo"
        token = "ghp_test"
        long_content = "x" * 20_000
        encoded = base64.b64encode(long_content.encode()).decode()

        respx.get(f"{GITHUB_API}/repos/{repo}/contents/README.md").mock(
            return_value=httpx.Response(
                200, json={"encoding": "base64", "content": encoded}
            )
        )
        for path in [
            "CONTRIBUTING.md",
            "ARCHITECTURE.md",
            "docs/ARCHITECTURE.md",
            "ROADMAP.md",
            "docs/ROADMAP.md",
        ]:
            respx.get(f"{GITHUB_API}/repos/{repo}/contents/{path}").mock(
                return_value=httpx.Response(404, json={"message": "Not Found"})
            )

        docs = fetch_repo_docs(repo, token)
        assert len(docs["README.md"]) == 15_000
