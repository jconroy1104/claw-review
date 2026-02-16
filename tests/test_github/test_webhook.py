"""Tests for webhook receiver (FastAPI endpoint)."""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from claw_review.github.app import GitHubApp
from claw_review.github.webhook import create_app, verify_signature


SECRET = "test-webhook-secret"


def _sign(payload: bytes, secret: str = SECRET) -> str:
    """Compute HMAC-SHA256 signature for a payload."""
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


@pytest.fixture()
def app() -> TestClient:
    """Create a test client for the webhook FastAPI app."""
    github_app = GitHubApp(app_id="1", private_key_path="", webhook_secret=SECRET)
    fastapi_app = create_app(github_app)
    return TestClient(fastapi_app)


class TestVerifySignature:
    """Tests for the standalone verify_signature function."""

    def test_valid_signature_passes(self) -> None:
        payload = b'{"hello": "world"}'
        sig = _sign(payload)
        assert verify_signature(payload, sig, SECRET) is True

    def test_invalid_signature_fails(self) -> None:
        payload = b'{"hello": "world"}'
        assert verify_signature(payload, "sha256=invalid", SECRET) is False

    def test_missing_prefix_fails(self) -> None:
        payload = b'{"hello": "world"}'
        raw_hmac = hmac.new(SECRET.encode(), payload, hashlib.sha256).hexdigest()
        assert verify_signature(payload, raw_hmac, SECRET) is False


class TestWebhookEndpoint:
    """Tests for POST /webhook."""

    def _post_event(
        self,
        client: TestClient,
        event: str,
        payload: dict,
        secret: str = SECRET,
    ):
        body = json.dumps(payload).encode()
        sig = _sign(body, secret)
        return client.post(
            "/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": event,
                "Content-Type": "application/json",
            },
        )

    def test_invalid_signature_returns_403(self, app: TestClient) -> None:
        resp = app.post(
            "/webhook",
            content=b'{"action":"opened"}',
            headers={
                "X-Hub-Signature-256": "sha256=wrong",
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 403

    def test_missing_signature_returns_403(self, app: TestClient) -> None:
        resp = app.post(
            "/webhook",
            content=b'{"action":"opened"}',
            headers={
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 403

    def test_pull_request_opened_returns_accepted(self, app: TestClient) -> None:
        payload = {
            "action": "opened",
            "pull_request": {"number": 42},
            "repository": {"full_name": "owner/repo"},
            "installation": {"id": 1234},
        }
        resp = self._post_event(app, "pull_request", payload)
        assert resp.status_code == 200
        assert resp.text == "accepted"

    def test_pull_request_synchronize_returns_accepted(self, app: TestClient) -> None:
        payload = {
            "action": "synchronize",
            "pull_request": {"number": 10},
            "repository": {"full_name": "owner/repo"},
            "installation": {"id": 1},
        }
        resp = self._post_event(app, "pull_request", payload)
        assert resp.status_code == 200
        assert resp.text == "accepted"

    def test_ignored_event_returns_ignored(self, app: TestClient) -> None:
        payload = {"action": "opened", "issue": {"number": 1}}
        resp = self._post_event(app, "issues", payload)
        assert resp.status_code == 200
        assert resp.text == "ignored"

    def test_pull_request_closed_is_ignored(self, app: TestClient) -> None:
        payload = {
            "action": "closed",
            "pull_request": {"number": 5},
            "repository": {"full_name": "owner/repo"},
            "installation": {"id": 1},
        }
        resp = self._post_event(app, "pull_request", payload)
        assert resp.status_code == 200
        assert resp.text == "ignored"


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, app: TestClient) -> None:
        resp = app.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAllowedRepos:
    """Test the optional repo allow-list."""

    def test_allowed_repo_accepted(self) -> None:
        github_app = GitHubApp(app_id="1", private_key_path="", webhook_secret=SECRET)
        fastapi_app = create_app(github_app, allowed_repos=["owner/repo"])
        client = TestClient(fastapi_app)

        payload = {
            "action": "opened",
            "pull_request": {"number": 1},
            "repository": {"full_name": "owner/repo"},
            "installation": {"id": 1},
        }
        body = json.dumps(payload).encode()
        sig = _sign(body)
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )
        assert resp.text == "accepted"

    def test_blocked_repo_ignored(self) -> None:
        github_app = GitHubApp(app_id="1", private_key_path="", webhook_secret=SECRET)
        fastapi_app = create_app(github_app, allowed_repos=["owner/allowed"])
        client = TestClient(fastapi_app)

        payload = {
            "action": "opened",
            "pull_request": {"number": 1},
            "repository": {"full_name": "owner/blocked"},
            "installation": {"id": 1},
        }
        body = json.dumps(payload).encode()
        sig = _sign(body)
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )
        assert resp.text == "ignored"
