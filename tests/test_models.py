"""Tests for claw_review.models — ModelResponse and ModelPool."""

import json
from unittest.mock import patch, MagicMock

import httpx
import pytest

from claw_review.config import Config
from claw_review.models import ModelResponse, ModelPool


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


class TestModelPoolQuerySingle:
    """Tests for ModelPool.query_single()."""

    def test_success_returns_model_response(self) -> None:
        pool = ModelPool(_make_config())
        body = _chat_response('{"intent": "test"}')
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            client_instance = MagicMock()
            client_instance.post.return_value = mock_resp
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = client_instance

            result = pool.query_single(
                model="anthropic/claude-sonnet-4",
                system_prompt="You are a helper.",
                user_prompt="Analyze this.",
            )

        assert isinstance(result, ModelResponse)
        assert result.provider == "anthropic/claude-sonnet-4"
        assert result.content == '{"intent": "test"}'
        assert result.usage is not None

    def test_payload_structure(self) -> None:
        pool = ModelPool(_make_config())
        body = _chat_response("ok")
        mock_resp = MagicMock()
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            client_instance = MagicMock()
            client_instance.post.return_value = mock_resp
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = client_instance

            pool.query_single(
                model="openai/gpt-4o",
                system_prompt="sys",
                user_prompt="usr",
                temperature=0.5,
                max_tokens=1000,
            )

            call_kwargs = client_instance.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["model"] == "openai/gpt-4o"
            assert payload["temperature"] == 0.5
            assert payload["max_tokens"] == 1000
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][1]["role"] == "user"

    def test_http_error_propagates(self) -> None:
        pool = ModelPool(_make_config())
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with patch("httpx.Client") as mock_client_cls:
            client_instance = MagicMock()
            client_instance.post.return_value = mock_resp
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(httpx.HTTPStatusError):
                pool.query_single("m", "s", "u")

    def test_timeout_propagates(self) -> None:
        pool = ModelPool(_make_config())

        with patch("httpx.Client") as mock_client_cls:
            client_instance = MagicMock()
            client_instance.post.side_effect = httpx.TimeoutException("timed out")
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(httpx.TimeoutException):
                pool.query_single("m", "s", "u")


class TestModelPoolQueryAll:
    """Tests for ModelPool.query_all() consensus pattern."""

    def _mock_pool_query_single(
        self, pool: ModelPool, side_effects: list
    ) -> None:
        """Patch query_single with a sequence of return values / exceptions."""
        pool.query_single = MagicMock(side_effect=side_effects)

    def test_all_succeed(self) -> None:
        pool = ModelPool(_make_config())
        resps = [
            ModelResponse(provider=m, model=m, content=f'{{"m": "{m}"}}')
            for m in pool.models
        ]
        pool.query_single = MagicMock(side_effect=resps)

        results = pool.query_all("sys", "usr")
        assert len(results) == 3
        assert all(r.model != "error" for r in results)

    def test_one_fails_others_succeed(self) -> None:
        pool = ModelPool(_make_config())
        ok = ModelResponse(provider="ok", model="ok", content="{}")
        effects: list = [ok, Exception("boom"), ok]
        pool.query_single = MagicMock(side_effect=effects)

        results = pool.query_all("sys", "usr")
        assert len(results) == 3
        errors = [r for r in results if r.model == "error"]
        assert len(errors) == 1
        assert "boom" in errors[0].content

    def test_all_fail(self) -> None:
        pool = ModelPool(_make_config())
        pool.query_single = MagicMock(side_effect=Exception("fail"))

        results = pool.query_all("sys", "usr")
        assert len(results) == 3
        assert all(r.model == "error" for r in results)


class TestModelPoolEmbeddings:
    """Tests for ModelPool.get_embeddings()."""

    def _patch_client(self, responses: list[dict]) -> MagicMock:
        """Build a patched httpx.Client that returns `responses` in order."""
        client_instance = MagicMock()
        mock_resps = []
        for body in responses:
            mr = MagicMock()
            mr.json.return_value = body
            mr.raise_for_status = MagicMock()
            mock_resps.append(mr)
        client_instance.post.side_effect = mock_resps
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        return client_instance

    def test_single_batch(self) -> None:
        pool = ModelPool(_make_config())
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        client_instance = self._patch_client([_embedding_response(vectors)])

        with patch("httpx.Client", return_value=client_instance):
            result = pool.get_embeddings(["hello", "world"])

        assert result == vectors
        assert client_instance.post.call_count == 1

    def test_multi_batch(self) -> None:
        pool = ModelPool(_make_config())
        # 150 texts -> 2 batches (100 + 50)
        texts = [f"text-{i}" for i in range(150)]
        batch1 = [[float(i)] for i in range(100)]
        batch2 = [[float(i)] for i in range(100, 150)]

        client_instance = self._patch_client([
            _embedding_response(batch1),
            _embedding_response(batch2),
        ])

        with patch("httpx.Client", return_value=client_instance):
            result = pool.get_embeddings(texts)

        assert len(result) == 150
        assert client_instance.post.call_count == 2


class TestModelPoolListModels:
    """Tests for ModelPool.list_available_models()."""

    def test_returns_model_list(self) -> None:
        pool = ModelPool(_make_config())
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet"}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            client_instance = MagicMock()
            client_instance.get.return_value = mock_resp
            client_instance.__enter__ = MagicMock(return_value=client_instance)
            client_instance.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = client_instance

            models = pool.list_available_models()

        assert len(models) == 1
        assert models[0]["id"] == "anthropic/claude-sonnet-4"
