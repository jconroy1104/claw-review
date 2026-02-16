"""Multi-model interface via OpenRouter.

OpenRouter provides a single OpenAI-compatible API that routes to any model
(Claude, GPT-4o, Gemini, Llama, Mistral, etc.). This eliminates the need
for separate SDKs per provider.

API docs: https://openrouter.ai/docs
"""

import json
from dataclasses import dataclass

import httpx
from .config import Config

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass
class ModelResponse:
    """Response from a single model."""

    provider: str  # OpenRouter model ID (e.g., "anthropic/claude-sonnet-4")
    model: str     # Same as provider for OpenRouter
    content: str
    usage: dict | None = None  # Token usage stats

    def parse_json(self) -> dict:
        """Extract JSON from response, handling markdown fences."""
        text = self.content.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise


class ModelPool:
    """Pool of models accessed through OpenRouter.

    All models are queried via the same endpoint with the same API key.
    The model ID in each request determines which provider handles it.
    """

    def __init__(self, config: Config):
        self.config = config
        self.models = config.models
        self._headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vectorcertain/claw-review",
            "X-Title": "claw-review",
        }

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def model_names(self) -> list[str]:
        """Return short display names for each model."""
        return [m.split("/")[-1] for m in self.models]

    def query_single(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> ModelResponse:
        """Query a single model via OpenRouter.

        Args:
            model: OpenRouter model ID (e.g., "anthropic/claude-sonnet-4")
            system_prompt: System message content
            user_prompt: User message content
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            ModelResponse with the model's output
        """
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract content from OpenAI-compatible response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage")

        return ModelResponse(
            provider=model,
            model=model,
            content=content,
            usage=usage,
        )

    def query_all(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> list[ModelResponse]:
        """Query ALL configured models and collect responses.

        Each model is queried independently. If one fails, others continue.
        This is the core consensus pattern — independent evaluations
        that are later fused.

        Returns:
            List of ModelResponse objects (may include error responses)
        """
        responses = []
        for model in self.models:
            try:
                resp = self.query_single(
                    model, system_prompt, user_prompt, temperature, max_tokens
                )
                responses.append(resp)
            except Exception as e:
                responses.append(
                    ModelResponse(
                        provider=model,
                        model="error",
                        content=f"ERROR: {e}",
                    )
                )
        return responses

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenRouter.

        Uses the configured embedding model (default: OpenAI text-embedding-3-small).
        Batches in groups of 100 for API compatibility.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        batch_size = 100

        with httpx.Client(timeout=30.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    "model": self.config.embedding_model,
                    "input": batch,
                }
                resp = client.post(
                    f"{OPENROUTER_BASE}/embeddings",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend(
                    [item["embedding"] for item in data["data"]]
                )

        return all_embeddings

    def list_available_models(self) -> list[dict]:
        """List all models available on OpenRouter (for discovery)."""
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{OPENROUTER_BASE}/models",
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
