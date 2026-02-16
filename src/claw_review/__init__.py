"""claw-review: AI-powered PR triage using multi-model consensus."""

__version__ = "0.1.0"

# Shared types — imported by all modules
from .github_client import PRData
from .models import ModelResponse

__all__ = ["PRData", "ModelResponse", "__version__"]
