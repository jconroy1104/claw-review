"""Webhook receiver for GitHub App events (FastAPI)."""
from __future__ import annotations

import hashlib
import hmac
from typing import Any

from fastapi import FastAPI, Request, Response

from .app import GitHubApp


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify a GitHub webhook HMAC-SHA256 signature.

    Args:
        payload: Raw request body bytes.
        signature: The X-Hub-Signature-256 header value (sha256=...).
        secret: The webhook secret configured in the GitHub App.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


class WebhookReceiver:
    """Container for webhook event data extracted from a request."""

    def __init__(
        self,
        event: str,
        action: str,
        repo: str,
        pr_number: int,
        installation_id: int,
    ) -> None:
        self.event = event
        self.action = action
        self.repo = repo
        self.pr_number = pr_number
        self.installation_id = installation_id


_HANDLED_ACTIONS = {"opened", "synchronize", "reopened"}


def create_app(
    github_app: GitHubApp,
    allowed_repos: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI application with webhook and health endpoints.

    Args:
        github_app: A configured GitHubApp instance (used for signature verification).
        allowed_repos: Optional list of repo full-names to accept events from.
            If None, all repos are accepted.

    Returns:
        A FastAPI application.
    """
    app = FastAPI(title="claw-review webhook")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/webhook")
    async def webhook(request: Request) -> Response:
        """Receive and validate GitHub webhook events."""
        body = await request.body()

        # Verify signature
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not signature or not verify_signature(body, signature, github_app.webhook_secret):
            return Response(content="Invalid signature", status_code=403)

        # Parse event
        event_type = request.headers.get("X-GitHub-Event", "")
        try:
            payload: dict[str, Any] = await request.json()
        except Exception:
            return Response(content="Invalid JSON", status_code=400)

        action = payload.get("action", "")

        # Only handle pull_request events with specific actions
        if event_type != "pull_request" or action not in _HANDLED_ACTIONS:
            return Response(content="ignored", status_code=200)

        pr = payload.get("pull_request", {})
        repo_data = payload.get("repository", {})
        repo_full_name = repo_data.get("full_name", "")
        installation = payload.get("installation", {})

        # Optional repo filter
        if allowed_repos and repo_full_name not in allowed_repos:
            return Response(content="ignored", status_code=200)

        _receiver = WebhookReceiver(
            event=event_type,
            action=action,
            repo=repo_full_name,
            pr_number=pr.get("number", 0),
            installation_id=installation.get("id", 0),
        )

        return Response(content="accepted", status_code=200)

    return app
