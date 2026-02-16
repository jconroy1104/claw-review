"""Tests for GitHub App JWT auth and installation token management."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from claw_review.github.app import GitHubApp


@pytest.fixture()
def rsa_key_pair(tmp_path: Path) -> tuple[Path, bytes]:
    """Generate an RSA key pair and write the private key to a temp file."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    key_file = tmp_path / "test-app.pem"
    key_file.write_bytes(pem)
    return key_file, pem


@pytest.fixture()
def github_app(rsa_key_pair: tuple[Path, bytes]) -> GitHubApp:
    """Create a GitHubApp instance with a valid test key."""
    key_file, _ = rsa_key_pair
    return GitHubApp(
        app_id="12345",
        private_key_path=str(key_file),
        webhook_secret="test-secret",
    )


class TestGenerateJWT:
    """Tests for GitHubApp.generate_jwt."""

    def test_produces_valid_jwt(self, github_app: GitHubApp, rsa_key_pair: tuple[Path, bytes]) -> None:
        """JWT should decode successfully and contain correct claims."""
        token = github_app.generate_jwt()
        _, pem = rsa_key_pair
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        private_key = load_pem_private_key(pem, password=None)
        public_key = private_key.public_key()
        decoded = pyjwt.decode(token, public_key, algorithms=["RS256"])
        assert decoded["iss"] == "12345"
        assert "exp" in decoded
        assert "iat" in decoded

    def test_jwt_uses_rs256(self, github_app: GitHubApp) -> None:
        """JWT header should specify RS256 algorithm."""
        token = github_app.generate_jwt()
        header = pyjwt.get_unverified_header(token)
        assert header["alg"] == "RS256"

    def test_jwt_expiry_is_ten_minutes(self, github_app: GitHubApp) -> None:
        """JWT exp claim should be ~10 minutes after iat."""
        token = github_app.generate_jwt()
        decoded = pyjwt.decode(token, options={"verify_signature": False})
        assert decoded["exp"] - decoded["iat"] == 11 * 60  # iat is 60s before now

    def test_missing_app_id_raises(self, rsa_key_pair: tuple[Path, bytes]) -> None:
        """Empty app_id should raise ValueError."""
        key_file, _ = rsa_key_pair
        app = GitHubApp(app_id="", private_key_path=str(key_file))
        with pytest.raises(ValueError, match="GITHUB_APP_ID"):
            app.generate_jwt()

    def test_missing_key_file_raises(self, tmp_path: Path) -> None:
        """Non-existent key path should raise FileNotFoundError."""
        app = GitHubApp(
            app_id="12345",
            private_key_path=str(tmp_path / "nonexistent.pem"),
        )
        with pytest.raises(FileNotFoundError, match="private key not found"):
            app.generate_jwt()


class TestInstallationToken:
    """Tests for GitHubApp.get_installation_token."""

    def test_fetches_and_caches_token(self, github_app: GitHubApp) -> None:
        """First call should hit API; second call should return cached token."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"token": "ghs_abc123"}
        mock_resp.raise_for_status = MagicMock()

        with patch("claw_review.github.app.httpx.post", return_value=mock_resp) as mock_post:
            token1 = github_app.get_installation_token(9999)
            assert token1 == "ghs_abc123"
            assert mock_post.call_count == 1

            # Second call should use cache
            token2 = github_app.get_installation_token(9999)
            assert token2 == "ghs_abc123"
            assert mock_post.call_count == 1  # no additional call

    def test_expired_token_triggers_refresh(self, github_app: GitHubApp) -> None:
        """Expired cached token should trigger a new API call."""
        # Pre-populate cache with an expired entry
        github_app._token_cache[9999] = ("old_token", time.time() - 1)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"token": "ghs_refreshed"}
        mock_resp.raise_for_status = MagicMock()

        with patch("claw_review.github.app.httpx.post", return_value=mock_resp):
            token = github_app.get_installation_token(9999)
            assert token == "ghs_refreshed"
