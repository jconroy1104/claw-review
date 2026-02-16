"""GitHub App configuration and JWT authentication."""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import jwt
from cryptography.hazmat.primitives.serialization import load_pem_private_key

GITHUB_API = "https://api.github.com"


class GitHubApp:
    """GitHub App authentication manager.

    Handles JWT generation (RS256) and installation token exchange.
    Config is read from environment variables:
      - GITHUB_APP_ID
      - GITHUB_PRIVATE_KEY_PATH
      - GITHUB_WEBHOOK_SECRET
    """

    def __init__(
        self,
        app_id: str | None = None,
        private_key_path: str | None = None,
        webhook_secret: str | None = None,
    ) -> None:
        self.app_id: str = app_id if app_id is not None else os.getenv("GITHUB_APP_ID", "")
        self.private_key_path: str = private_key_path if private_key_path is not None else os.getenv("GITHUB_PRIVATE_KEY_PATH", "")
        self.webhook_secret: str = webhook_secret if webhook_secret is not None else os.getenv("GITHUB_WEBHOOK_SECRET", "")
        self._private_key: bytes | None = None
        self._token_cache: dict[int, tuple[str, float]] = {}

    def _load_private_key(self) -> bytes:
        """Load RSA private key from the configured file path.

        Returns:
            The PEM-encoded private key bytes.

        Raises:
            FileNotFoundError: If the key file does not exist.
            ValueError: If the key file cannot be parsed as a PEM RSA key.
        """
        if self._private_key is not None:
            return self._private_key

        key_path = Path(self.private_key_path)
        if not key_path.exists():
            raise FileNotFoundError(
                f"GitHub App private key not found at: {self.private_key_path}"
            )

        pem_data = key_path.read_bytes()
        # Validate that it is a valid PEM RSA key
        load_pem_private_key(pem_data, password=None)
        self._private_key = pem_data
        return self._private_key

    def generate_jwt(self) -> str:
        """Create a JWT for GitHub App authentication.

        The JWT uses RS256, expires in 10 minutes, and contains the app_id
        as the issuer claim.

        Returns:
            Encoded JWT string.

        Raises:
            FileNotFoundError: If the private key file is missing.
            ValueError: If app_id is empty.
        """
        if not self.app_id:
            raise ValueError("GITHUB_APP_ID is required to generate a JWT")

        private_key = self._load_private_key()
        now = int(time.time())
        payload = {
            "iat": now - 60,  # issued at (60s in the past for clock drift)
            "exp": now + (10 * 60),  # expires in 10 minutes
            "iss": self.app_id,
        }
        return jwt.encode(payload, private_key, algorithm="RS256")

    def get_installation_token(self, installation_id: int) -> str:
        """Exchange a JWT for an installation access token.

        Tokens are cached for 55 minutes (GitHub issues them for 1 hour).

        Args:
            installation_id: The GitHub App installation ID.

        Returns:
            Installation access token string.
        """
        now = time.time()
        cached = self._token_cache.get(installation_id)
        if cached is not None:
            token, expiry = cached
            if now < expiry:
                return token

        app_jwt = self.generate_jwt()
        resp = httpx.post(
            f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        # Cache for 55 minutes
        self._token_cache[installation_id] = (token, now + 55 * 60)
        return token
