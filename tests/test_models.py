"""Tests for claw_review.models — ModelResponse and ModelPool (async)."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claw_review.config import Config
from claw_review.models import (
    ModelResponse,
    ModelPool,
    _MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: object) -> Config:
    """Build a Config with safe test defaults."""
    defaults = {
        "openrouter_api_key": "test-key-123",
        "github_token": "ghp_test",
        "models": [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-001",
        ],
        "embedding_model": "openai/text-embedding-3-small",
    }
    defaults.update(overrides)
    return Config(**defaults)


def _chat_response(content: str, usage: dict | None = None) -> dict:
    """Build a minimal OpenRouter chat-completions response body."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
    }


def _embedding_response(vectors: list[list[float]]) -> dict:
    """Build a minimal OpenRouter embeddings response body."""
    return {
        "data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)]
    }


def _mock_httpx_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


# ===================================================================
# ModelResponse tests
# ===================================================================


class TestModelResponse:
    """Tests for the ModelResponse dataclass and parse_json()."""

    def test_creation_with_all_fields(self) -> None:
        resp = ModelResponse(
            provider="anthropic/claude-sonnet-4",
            model="anthropic/claude-sonnet-4",
            content='{"key": "value"}',
            usage={"prompt_tokens": 5, "completion_tokens": 10},
        )
        assert resp.provider == "anthropic/claude-sonnet-4"
        assert resp.model == "anthropic/claude-sonnet-4"
        assert resp.content == '{"key": "value"}'
        assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 10}

    def test_creation_defaults_usage_to_none(self) -> None:
        resp = ModelResponse(
            provider="openai/gpt-4o",
            model="openai/gpt-4o",
            content="hello",
        )
        assert resp.usage is None

    def test_parse_json_clean(self) -> None:
        resp = ModelResponse(
            provider="test", model="test",
            content='{"intent": "fix bug", "category": "bugfix"}',
        )
        parsed = resp.parse_json()
        assert parsed == {"intent": "fix bug", "category": "bugfix"}

    def test_parse_json_with_json_fences(self) -> None:
        raw = '```json\n{"intent": "add feature"}\n```'
        resp = ModelResponse(provider="test", model="test", content=raw)
        parsed = resp.parse_json()
        assert parsed["intent"] == "add feature"

    def test_parse_json_with_plain_fences(self) -> None:
        raw = '```\n{"intent": "refactor"}\n```'
        resp = ModelResponse(provider="test", model="test", content=raw)
        parsed = resp.parse_json()
        assert parsed["intent"] == "refactor"

    def test_parse_json_embedded_in_text(self) -> None:
        raw = 'Here is my analysis:\n{"intent": "docs update"}\nHope this helps!'
        resp = ModelResponse(provider="test", model="test", content=raw)
        parsed = resp.parse_json()
        assert parsed["intent"] == "docs update"

    def test_parse_json_malformed_raises(self) -> None:
        resp = ModelResponse(
            provider="test", model="test",
            content='{"intent": "missing bracket"',
        )
        with pytest.raises(json.JSONDecodeError):
            resp.parse_json()

    def test_parse_json_no_json_at_all_raises(self) -> None:
        resp = ModelResponse(
            provider="test", model="test",
            content="This is plain text with no JSON.",
        )
        with pytest.raises(json.JSONDecodeError):
            resp.parse_json()

    def test_parse_json_with_whitespace(self) -> None:
        raw = '  \n  {"intent": "trim"}  \n  '
        resp = ModelResponse(provider="test", model="test", content=raw)
        assert resp.parse_json()["intent"] == "trim"


# ===================================================================
# ModelPool tests
# ===================================================================


class TestModelPoolInit:
    """Tests for ModelPool construction and properties."""

    def test_init_with_valid_config(self) -> None:
        cfg = _make_config()
        pool = ModelPool(cfg)
        assert pool.config is cfg
        assert pool.models == cfg.models

    def test_model_count(self) -> None:
        pool = ModelPool(_make_config())
        assert pool.model_count == 3

    def test_model_names_extracts_short(self) -> None:
        pool = ModelPool(_make_config())
        assert pool.model_names == [
            "claude-sonnet-4",
            "gpt-4o",
            "gemini-2.0-flash-001",
        ]

    def test_headers_contain_auth(self) -> None:
        pool = ModelPool(_make_config(openrouter_api_key="sk-abc"))
        assert pool._headers["Authorization"] == "Bearer sk-abc"

    def test_default_concurrency_limit(self) -> None:
        pool = ModelPool(_make_config())
        assert pool.concurrency_limit == 10

    def test_set_concurrency_limit(self) -> None:
        pool = ModelPool(_make_config())
        pool.concurrency_limit = 5
        assert pool.concurrency_limit == 5


class TestModelPoolQuerySingle:
    """Tests for async ModelPool.query_single()."""

    async def test_async_query_single_success(self) -> None:
        pool = ModelPool(_make_config())
        body = _chat_response('{"intent": "test"}')
        mock_resp = _mock_httpx_response(200, body)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        pool._client = mock_client

        result = await pool.query_single(
            model="anthropic/claude-sonnet-4",
            system_prompt="You are a helper.",
            user_prompt="Analyze this.",
        )

        assert isinstance(result, ModelResponse)
        assert result.provider == "anthropic/claude-sonnet-4"
        assert result.content == '{"intent": "test"}'
        assert result.usage is not None

    async def test_payload_structure(self) -> None:
        pool = ModelPool(_make_config())
        body = _chat_response("ok")
        mock_resp = _mock_httpx_response(200, body)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        pool._client = mock_client

        await pool.query_single(
            model="openai/gpt-4o",
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.5,
            max_tokens=1000,
        )

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "openai/gpt-4o"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 1000
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    async def test_async_query_single_timeout(self) -> None:
        pool = ModelPool(_make_config())
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.is_closed = False
        pool._client = mock_client

        with pytest.raises(httpx.TimeoutException):
            await pool.query_single("m", "s", "u")

    async def test_async_query_single_http_error(self) -> None:
        """Non-retryable HTTP error (e.g. 400) propagates immediately."""
        pool = ModelPool(_make_config())
        mock_resp = _mock_httpx_response(400)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        pool._client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await pool.query_single("m", "s", "u")

        # Should NOT retry on 400
        assert mock_client.post.call_count == 1


class TestModelPoolQueryAll:
    """Tests for async ModelPool.query_all() consensus pattern."""

    async def test_all_succeed(self) -> None:
        pool = ModelPool(_make_config())
        resps = [
            ModelResponse(provider=m, model=m, content=f'{{"m": "{m}"}}')
            for m in pool.models
        ]
        pool.query_single = AsyncMock(side_effect=resps)

        results = await pool.query_all("sys", "usr")
        assert len(results) == 3
        assert all(r.model != "error" for r in results)

    async def test_async_query_all_one_fails(self) -> None:
        pool = ModelPool(_make_config())
        ok = ModelResponse(provider="ok", model="ok", content="{}")
        effects: list = [ok, Exception("boom"), ok]
        pool.query_single = AsyncMock(side_effect=effects)

        results = await pool.query_all("sys", "usr")
        assert len(results) == 3
        errors = [r for r in results if r.model == "error"]
        assert len(errors) == 1
        assert "boom" in errors[0].content

    async def test_async_query_all_all_fail(self) -> None:
        pool = ModelPool(_make_config())
        pool.query_single = AsyncMock(side_effect=Exception("fail"))

        results = await pool.query_all("sys", "usr")
        assert len(results) == 3
        assert all(r.model == "error" for r in results)

    async def test_async_query_all_concurrent(self) -> None:
        """Verify that query_all uses asyncio.gather for concurrency."""
        pool = ModelPool(_make_config())
        call_times = []

        async def slow_query(*args, **kwargs):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return ModelResponse(provider="test", model="test", content="{}")

        pool.query_single = AsyncMock(side_effect=slow_query)

        start = time.monotonic()
        await pool.query_all("sys", "usr")
        elapsed = time.monotonic() - start

        # 3 models at 50ms each; if sequential would take ~150ms
        # Concurrent should take ~50ms (+overhead)
        assert elapsed < 0.15, f"Expected concurrent execution, took {elapsed:.3f}s"
        assert len(call_times) == 3


class TestModelPoolRetry:
    """Tests for retry logic with exponential backoff."""

    async def test_retry_on_429(self) -> None:
        pool = ModelPool(_make_config())
        fail_resp = _mock_httpx_response(429)
        ok_body = _chat_response("ok")
        ok_resp = _mock_httpx_response(200, ok_body)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])
        mock_client.is_closed = False
        pool._client = mock_client

        with patch("claw_review.models.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await pool.query_single("m", "s", "u")

        assert result.content == "ok"
        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once()

    async def test_retry_on_500(self) -> None:
        pool = ModelPool(_make_config())
        fail_resp = _mock_httpx_response(500)
        ok_body = _chat_response("recovered")
        ok_resp = _mock_httpx_response(200, ok_body)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])
        mock_client.is_closed = False
        pool._client = mock_client

        with patch("claw_review.models.asyncio.sleep", new_callable=AsyncMock):
            result = await pool.query_single("m", "s", "u")

        assert result.content == "recovered"

    async def test_retry_on_503(self) -> None:
        pool = ModelPool(_make_config())
        fail_resp = _mock_httpx_response(503)
        ok_body = _chat_response("back up")
        ok_resp = _mock_httpx_response(200, ok_body)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])
        mock_client.is_closed = False
        pool._client = mock_client

        with patch("claw_review.models.asyncio.sleep", new_callable=AsyncMock):
            result = await pool.query_single("m", "s", "u")

        assert result.content == "back up"

    async def test_retry_max_retries_exceeded(self) -> None:
        pool = ModelPool(_make_config())
        fail_resp = _mock_httpx_response(429)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=fail_resp)
        mock_client.is_closed = False
        pool._client = mock_client

        with patch("claw_review.models.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await pool.query_single("m", "s", "u")

        assert mock_client.post.call_count == _MAX_RETRIES

    async def test_retry_exponential_backoff(self) -> None:
        pool = ModelPool(_make_config())
        fail_resp = _mock_httpx_response(500)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=fail_resp)
        mock_client.is_closed = False
        pool._client = mock_client

        sleep_durations = []

        async def capture_sleep(duration):
            sleep_durations.append(duration)

        with patch("claw_review.models.asyncio.sleep", side_effect=capture_sleep):
            with patch("claw_review.models.random.uniform", return_value=0.0):
                with pytest.raises(httpx.HTTPStatusError):
                    await pool.query_single("m", "s", "u")

        # Each of the 3 attempts fails and sleeps before the next retry
        assert len(sleep_durations) == _MAX_RETRIES
        assert sleep_durations[0] == 1.0  # 1.0 * 2^0
        assert sleep_durations[1] == 2.0  # 1.0 * 2^1
        assert sleep_durations[2] == 4.0  # 1.0 * 2^2


class TestModelPoolSemaphore:
    """Tests for semaphore-based concurrency control."""

    async def test_semaphore_limits_concurrency(self) -> None:
        pool = ModelPool(_make_config())
        pool.concurrency_limit = 2

        concurrent_count = 0
        max_concurrent = 0

        # Mock the HTTP client so query_single's real code path (with semaphore) runs
        body = _chat_response("ok")
        ok_resp = _mock_httpx_response(200, body)

        async def tracked_post(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return ok_resp

        mock_client = AsyncMock()
        mock_client.post = tracked_post
        mock_client.is_closed = False
        pool._client = mock_client

        # Override models to have more than semaphore limit
        pool.models = ["m1", "m2", "m3", "m4", "m5"]
        await pool.query_all("sys", "usr")

        assert max_concurrent <= 2


class TestModelPoolConnectionPool:
    """Tests for connection pool lifecycle."""

    async def test_connection_pool_reuse(self) -> None:
        pool = ModelPool(_make_config())
        client1 = await pool._get_client()
        client2 = await pool._get_client()
        assert client1 is client2
        await pool.close()

    async def test_async_context_manager_cleanup(self) -> None:
        cfg = _make_config()
        async with ModelPool(cfg) as pool:
            client = await pool._get_client()
            assert client is not None
            assert not client.is_closed
        # After exiting context, client should be closed
        assert pool._client is None

    async def test_close_when_no_client(self) -> None:
        """close() should not raise if no client was created."""
        pool = ModelPool(_make_config())
        await pool.close()  # Should not raise

    async def test_get_client_recreates_after_close(self) -> None:
        pool = ModelPool(_make_config())
        client1 = await pool._get_client()
        await pool.close()
        client2 = await pool._get_client()
        assert client1 is not client2
        await pool.close()


class TestModelPoolEmbeddings:
    """Tests for async ModelPool.get_embeddings()."""

    async def test_async_get_embeddings(self) -> None:
        pool = ModelPool(_make_config())
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp = _mock_httpx_response(200, _embedding_response(vectors))

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        pool._client = mock_client

        result = await pool.get_embeddings(["hello", "world"])

        assert result == vectors
        assert mock_client.post.call_count == 1

    async def test_async_get_embeddings_multi_batch(self) -> None:
        pool = ModelPool(_make_config())
        texts = [f"text-{i}" for i in range(150)]
        batch1 = [[float(i)] for i in range(100)]
        batch2 = [[float(i)] for i in range(100, 150)]

        mock_resp1 = _mock_httpx_response(200, _embedding_response(batch1))
        mock_resp2 = _mock_httpx_response(200, _embedding_response(batch2))

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_resp1, mock_resp2])
        mock_client.is_closed = False
        pool._client = mock_client

        result = await pool.get_embeddings(texts)

        assert len(result) == 150
        assert mock_client.post.call_count == 2


class TestModelPoolListModels:
    """Tests for async ModelPool.list_available_models()."""

    async def test_returns_model_list(self) -> None:
        pool = ModelPool(_make_config())
        data = {
            "data": [{"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet"}]
        }
        mock_resp = _mock_httpx_response(200, data)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        pool._client = mock_client

        models = await pool.list_available_models()

        assert len(models) == 1
        assert models[0]["id"] == "anthropic/claude-sonnet-4"
