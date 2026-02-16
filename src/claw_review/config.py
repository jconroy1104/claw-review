"""Configuration and environment management."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# Default models for consensus (3-model minimum)
DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
]


@dataclass
class Config:
    """Application configuration loaded from environment."""

    # GitHub
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    target_repo: str = field(
        default_factory=lambda: os.getenv("TARGET_REPO", "openclaw/openclaw")
    )
    max_prs: int = field(
        default_factory=lambda: int(os.getenv("MAX_PRS", "100"))
    )

    # OpenRouter
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )

    # Models
    models: list[str] = field(default_factory=lambda: _parse_models())
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
    )

    # Consensus thresholds
    similarity_threshold: float = 0.82
    quality_disagreement_threshold: float = 3.0
    alignment_reject_threshold: float = 4.0

    @property
    def model_count(self) -> int:
        return len(self.models)

    def validate(self) -> list[str]:
        """Validate configuration, return list of issues."""
        issues = []
        if not self.github_token:
            issues.append("GITHUB_TOKEN is required for API access")
        if not self.openrouter_api_key:
            issues.append("OPENROUTER_API_KEY is required for model access")
        if len(self.models) < 2:
            issues.append(
                f"Need at least 2 models for consensus, "
                f"found {len(self.models)}: {self.models}"
            )
        return issues


def _parse_models() -> list[str]:
    """Parse model list from environment or use defaults."""
    env_models = os.getenv("MODELS", "")
    if env_models.strip():
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return DEFAULT_MODELS.copy()
