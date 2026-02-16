"""Tests for the AdapterRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from claw_review.platform.interfaces import DataItem, DomainConfig
from claw_review.platform.registry import AdapterRegistry


def _make_config(domain: str) -> DomainConfig:
    """Helper to create a minimal DomainConfig for testing."""
    return DomainConfig(
        domain=domain,
        scoring_dimensions=["d1"],
        clustering_prompt="cp",
        scoring_prompt="sp",
        alignment_prompt="ap",
        recommendation_levels=["OK"],
    )


class _FakeAdapter:
    domain: str = "fake"

    async def fetch_items(
        self, source: str, max_items: int = 100, **kwargs: Any
    ) -> list[DataItem]:
        return []

    async def fetch_context_docs(
        self, source: str, **kwargs: Any
    ) -> dict[str, str]:
        return {}

    def format_item_for_prompt(self, item: DataItem) -> str:
        return ""


class TestAdapterRegistry:
    """Tests for AdapterRegistry class methods."""

    def setup_method(self) -> None:
        """Clear the registry before each test."""
        AdapterRegistry._clear()

    def test_register_and_get(self) -> None:
        cfg = _make_config("test-domain")
        AdapterRegistry.register("test-domain", _FakeAdapter, cfg)
        adapter_cls, config = AdapterRegistry.get("test-domain")
        assert adapter_cls is _FakeAdapter
        assert config.domain == "test-domain"

    def test_list_domains(self) -> None:
        AdapterRegistry.register("b-domain", _FakeAdapter, _make_config("b"))
        AdapterRegistry.register("a-domain", _FakeAdapter, _make_config("a"))
        domains = AdapterRegistry.list_domains()
        assert domains == ["a-domain", "b-domain"]

    def test_list_domains_empty(self) -> None:
        assert AdapterRegistry.list_domains() == []

    def test_unknown_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain 'nope'"):
            AdapterRegistry.get("nope")

    def test_duplicate_registration_overwrites(self) -> None:
        cfg1 = _make_config("dup")
        cfg2 = _make_config("dup")
        cfg2.scoring_dimensions = ["new_dim"]

        class AnotherAdapter:
            domain = "dup"

        AdapterRegistry.register("dup", _FakeAdapter, cfg1)
        AdapterRegistry.register("dup", AnotherAdapter, cfg2)

        adapter_cls, config = AdapterRegistry.get("dup")
        assert adapter_cls is AnotherAdapter
        assert config.scoring_dimensions == ["new_dim"]

    def test_unknown_domain_error_lists_available(self) -> None:
        AdapterRegistry.register("alpha", _FakeAdapter, _make_config("alpha"))
        with pytest.raises(ValueError, match="alpha"):
            AdapterRegistry.get("missing")
