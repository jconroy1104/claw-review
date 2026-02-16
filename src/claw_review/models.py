"""Multi-model interface via OpenRouter.

OpenRouter provides a single OpenAI-compatible API that routes to any model
(Claude, GPT-4o, Gemini, Llama, Mistral, etc.). This eliminates the need
for separate SDKs per provider.

API docs: https://openrouter.ai/docs
"""

import asyncio
import json
import random
from dataclasses import dataclass

import httpx
from .config import Config

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Status codes that trigger a retry
_RETRYABLE_STATUS_CODES = {429, 500, 503}
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds


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
        self._semaphore = asyncio.Semaphore(10)
        self._client: httpx.AsyncClient | None = None

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def model_names(self) -> list[str]:
        """Return short display names for each model."""
        return [m.split("/")[-1] for m in self.models]

    @property
    def concurrency_limit(self) -> int:
        return self._semaphore._value

    @concurrency_limit.setter
    def concurrency_limit(self, value: int) -> None:
        self._semaphore = asyncio.Semaphore(value)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, pool=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers=self._headers,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def query_single(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> ModelResponse:
        """Query a single model via OpenRouter with retry logic.

        Uses a semaphore to limit concurrency and retries with exponential
        backoff on 429/500/503 status codes.

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

        client = await self._get_client()

        async with self._semaphore:
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.post(
                        f"{OPENROUTER_BASE}/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage")

                    return ModelResponse(
                        provider=model,
                        model=model,
                        content=content,
                        usage=usage,
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in _RETRYABLE_STATUS_CODES:
                        last_exc = e
                        backoff = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
                        await asyncio.sleep(backoff)
                        continue
                    raise
                except httpx.TimeoutException:
                    raise
            # All retries exhausted
            raise last_exc  # type: ignore[misc]

    async def query_all(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> list[ModelResponse]:
        """Query ALL configured models concurrently and collect responses.

        Each model is queried independently via asyncio.gather(). If one fails,
        others continue. This is the core consensus pattern -- independent
        evaluations that are later fused.

        Returns:
            List of ModelResponse objects (may include error responses)
        """
        tasks = [
            self.query_single(model, system_prompt, user_prompt, temperature, max_tokens)
            for model in self.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for model, result in zip(self.models, results):
            if isinstance(result, Exception):
                responses.append(
                    ModelResponse(
                        provider=model,
                        model="error",
                        content=f"ERROR: {result}",
                    )
                )
            else:
                responses.append(result)
        return responses

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenRouter.

        Uses the configured embedding model (default: OpenAI text-embedding-3-small).
        Batches in groups of 100 for API compatibility.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors
        """
        all_embeddings: list[list[float]] = []
        batch_size = 100
        client = await self._get_client()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "model": self.config.embedding_model,
                "input": batch,
            }
            resp = await client.post(
                f"{OPENROUTER_BASE}/embeddings",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(
                [item["embedding"] for item in data["data"]]
            )

        return all_embeddings

    async def list_available_models(self) -> list[dict]:
        """List all models available on OpenRouter (for discovery)."""
        client = await self._get_client()
        resp = await client.get(f"{OPENROUTER_BASE}/models")
        resp.raise_for_status()
        return resp.json().get("data", [])
