"""GitHub API client for PR data extraction."""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

GITHUB_API = "https://api.github.com"
CACHE_DIR = Path(".claw-review-cache")
MAX_DIFF_CHARS = 12_000  # Truncate diffs to keep model context manageable


@dataclass
class PRData:
    """Extracted pull request data."""

    number: int
    title: str
    body: str
    author: str
    created_at: str
    updated_at: str
    state: str
    labels: list[str]
    files_changed: list[str]
    additions: int
    deletions: int
    diff_summary: str  # Truncated diff content
    url: str
    comments_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_key(repo: str, pr_number: int) -> str:
    return hashlib.md5(f"{repo}:{pr_number}".encode()).hexdigest()


def _load_cached(repo: str, pr_number: int) -> PRData | None:
    """Load PR data from local cache if available."""
    cache_file = CACHE_DIR / f"{_cache_key(repo, pr_number)}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return PRData(**data)
    return None


def _save_cache(repo: str, pr_data: PRData) -> None:
    """Save PR data to local cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(repo, pr_data.number)}.json"
    cache_file.write_text(json.dumps(pr_data.to_dict(), indent=2))


def _truncate_diff(diff: str, max_chars: int = MAX_DIFF_CHARS) -> str:
    """Truncate diff to fit model context windows."""
    if len(diff) <= max_chars:
        return diff
    half = max_chars // 2
    return (
        diff[:half]
        + f"\n\n... [TRUNCATED {len(diff) - max_chars} chars] ...\n\n"
        + diff[-half:]
    )


def fetch_open_prs(
    repo: str,
    token: str,
    max_prs: int = 100,
    use_cache: bool = True,
) -> list[PRData]:
    """Fetch open PRs from a GitHub repository.

    Args:
        repo: Repository in 'owner/name' format
        token: GitHub personal access token
        max_prs: Maximum number of PRs to fetch
        use_cache: Whether to use local file cache

    Returns:
        List of PRData objects
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    prs: list[PRData] = []

    with httpx.Client(headers=headers, timeout=30.0) as client:
        # Step 1: List open PRs
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Fetching PR list..."),
        ) as progress:
            progress.add_task("fetch", total=None)

            all_pr_items = []
            page = 1
            while len(all_pr_items) < max_prs:
                per_page = min(100, max_prs - len(all_pr_items))
                resp = client.get(
                    f"{GITHUB_API}/repos/{repo}/pulls",
                    params={
                        "state": "open",
                        "sort": "updated",
                        "direction": "desc",
                        "per_page": per_page,
                        "page": page,
                    },
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                all_pr_items.extend(batch)
                page += 1

        pr_items = all_pr_items[:max_prs]

        # Step 2: Fetch details for each PR
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Fetching PR details..."),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("details", total=len(pr_items))

            for item in pr_items:
                pr_number = item["number"]

                # Check cache first
                if use_cache:
                    cached = _load_cached(repo, pr_number)
                    if cached:
                        prs.append(cached)
                        progress.advance(task)
                        continue

                # Fetch PR detail (includes body, labels, etc.)
                detail_resp = client.get(
                    f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
                )
                detail_resp.raise_for_status()
                detail = detail_resp.json()

                # Fetch files changed
                files_resp = client.get(
                    f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
                    params={"per_page": 100},
                )
                files_resp.raise_for_status()
                files = files_resp.json()

                # Build diff summary from file patches
                diff_parts = []
                for f in files[:30]:  # Cap at 30 files
                    patch = f.get("patch", "")
                    if patch:
                        diff_parts.append(
                            f"--- {f['filename']} "
                            f"(+{f.get('additions', 0)}/-{f.get('deletions', 0)})\n"
                            f"{patch}"
                        )

                diff_summary = _truncate_diff("\n\n".join(diff_parts))

                pr_data = PRData(
                    number=pr_number,
                    title=detail.get("title", ""),
                    body=detail.get("body", "") or "",
                    author=detail.get("user", {}).get("login", "unknown"),
                    created_at=detail.get("created_at", ""),
                    updated_at=detail.get("updated_at", ""),
                    state=detail.get("state", "open"),
                    labels=[
                        lbl.get("name", "") for lbl in detail.get("labels", [])
                    ],
                    files_changed=[f["filename"] for f in files],
                    additions=detail.get("additions", 0),
                    deletions=detail.get("deletions", 0),
                    diff_summary=diff_summary,
                    url=detail.get("html_url", ""),
                    comments_count=detail.get("comments", 0)
                    + detail.get("review_comments", 0),
                )

                if use_cache:
                    _save_cache(repo, pr_data)

                prs.append(pr_data)
                progress.advance(task)

    return prs


def fetch_repo_docs(repo: str, token: str) -> dict[str, str]:
    """Fetch key repository documents for vision alignment.

    Returns dict mapping filename to content.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.raw",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    docs = {}
    doc_paths = [
        "README.md",
        "CONTRIBUTING.md",
        "ARCHITECTURE.md",
        "docs/ARCHITECTURE.md",
        "ROADMAP.md",
        "docs/ROADMAP.md",
    ]

    with httpx.Client(headers=headers, timeout=15.0) as client:
        for path in doc_paths:
            try:
                resp = client.get(
                    f"{GITHUB_API}/repos/{repo}/contents/{path}"
                )
                if resp.status_code == 200:
                    content = resp.json()
                    if content.get("encoding") == "base64":
                        import base64
                        decoded = base64.b64decode(
                            content["content"]
                        ).decode("utf-8", errors="replace")
                        # Truncate very long docs
                        docs[path] = decoded[:15_000]
                    elif isinstance(resp.text, str):
                        docs[path] = resp.text[:15_000]
            except (httpx.HTTPError, KeyError):
                continue

    return docs
