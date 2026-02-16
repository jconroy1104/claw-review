"""PR comment formatter and poster for claw-review analysis results."""
from __future__ import annotations

from pathlib import Path

import httpx
import jinja2

GITHUB_API = "https://api.github.com"
COMMENT_MARKER = "ClawReview Analysis"

# Recommendation thresholds
_THRESHOLDS = {
    "MERGE": 7.5,
    "REVIEW": 5.5,
    "DISCUSS": 3.5,
}


def _recommendation(score: float) -> str:
    """Map an overall score to a recommendation label."""
    if score >= _THRESHOLDS["MERGE"]:
        return "MERGE"
    if score >= _THRESHOLDS["REVIEW"]:
        return "REVIEW"
    if score >= _THRESHOLDS["DISCUSS"]:
        return "DISCUSS"
    return "CLOSE"


def _load_template() -> jinja2.Template:
    """Load the PR comment Jinja2 template."""
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template("pr_comment.md.j2")


class PRCommenter:
    """Formats and posts claw-review analysis results as GitHub PR comments.

    Uses httpx.AsyncClient for all GitHub API interactions.
    """

    def __init__(self, token: str) -> None:
        """Initialize with a GitHub token (installation or PAT).

        Args:
            token: A GitHub access token with permission to comment on PRs.
        """
        self.token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

    def format_comment(
        self,
        quality_scores: dict,
        alignment: dict | None = None,
        duplicates: list[str] | None = None,
        models: list[str] | None = None,
    ) -> str:
        """Format analysis results as a Markdown comment.

        Args:
            quality_scores: Dict with 'dimensions' (dict of name->score) and
                optionally 'per_model' (dict of model->dimensions).
            alignment: Optional dict with 'score' and 'summary' keys.
            duplicates: Optional list of duplicate PR references (e.g. '#123').
            models: Optional list of model names used in analysis.

        Returns:
            Formatted Markdown string.
        """
        dimensions = quality_scores.get("dimensions", {})
        per_model = quality_scores.get("per_model", {})
        overall = quality_scores.get("overall", 0.0)

        if dimensions and not overall:
            scores = [v for v in dimensions.values() if isinstance(v, (int, float))]
            overall = sum(scores) / len(scores) if scores else 0.0

        recommendation = _recommendation(overall)

        template = _load_template()
        return template.render(
            dimensions=dimensions,
            overall=round(overall, 1),
            recommendation=recommendation,
            per_model=per_model,
            models=models or [],
            alignment=alignment,
            duplicates=duplicates,
            marker=COMMENT_MARKER,
        )

    async def post_comment(self, repo: str, pr_number: int, body: str) -> dict:
        """Post a new comment on a pull request.

        Args:
            repo: Repository full name (owner/repo).
            pr_number: Pull request number.
            body: Comment body in Markdown.

        Returns:
            GitHub API response as dict.
        """
        async with httpx.AsyncClient(headers=self._headers, timeout=15.0) as client:
            resp = await client.post(
                f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
                json={"body": body},
            )
            resp.raise_for_status()
            return resp.json()

    async def find_existing_comment(self, repo: str, pr_number: int) -> int | None:
        """Search for an existing claw-review comment on a PR.

        Identifies comments by the presence of the COMMENT_MARKER text.

        Args:
            repo: Repository full name.
            pr_number: Pull request number.

        Returns:
            Comment ID if found, None otherwise.
        """
        async with httpx.AsyncClient(headers=self._headers, timeout=15.0) as client:
            resp = await client.get(
                f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
                params={"per_page": 100},
            )
            resp.raise_for_status()
            for comment in resp.json():
                if COMMENT_MARKER in comment.get("body", ""):
                    return comment["id"]
        return None

    async def update_or_create_comment(
        self, repo: str, pr_number: int, body: str
    ) -> dict:
        """Update an existing claw-review comment or create a new one.

        Finds a comment containing the COMMENT_MARKER and patches it,
        or creates a new comment if none exists.

        Args:
            repo: Repository full name.
            pr_number: Pull request number.
            body: Comment body in Markdown.

        Returns:
            GitHub API response as dict.
        """
        existing_id = await self.find_existing_comment(repo, pr_number)
        if existing_id is not None:
            async with httpx.AsyncClient(headers=self._headers, timeout=15.0) as client:
                resp = await client.patch(
                    f"{GITHUB_API}/repos/{repo}/issues/comments/{existing_id}",
                    json={"body": body},
                )
                resp.raise_for_status()
                return resp.json()
        return await self.post_comment(repo, pr_number, body)
