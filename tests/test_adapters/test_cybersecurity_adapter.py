"""Tests for the cybersecurity adapter community stub."""

from __future__ import annotations

import pytest

from claw_review.adapters.cybersecurity import CybersecurityAdapter
from claw_review.platform.interfaces import DataAdapter, DataItem


@pytest.fixture()
def adapter() -> CybersecurityAdapter:
    """Return a CybersecurityAdapter instance."""
    return CybersecurityAdapter()


def test_adapter_satisfies_protocol(adapter: CybersecurityAdapter) -> None:
    """CybersecurityAdapter should satisfy the DataAdapter protocol."""
    assert isinstance(adapter, DataAdapter)
    assert adapter.domain == "cybersecurity"


async def test_fetch_items_raises_not_implemented(
    adapter: CybersecurityAdapter,
) -> None:
    """Community stub should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="enterprise feature"):
        await adapter.fetch_items("alerts.json")


async def test_fetch_context_docs_raises_not_implemented(
    adapter: CybersecurityAdapter,
) -> None:
    """Community stub should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="enterprise feature"):
        await adapter.fetch_context_docs("")


def test_format_item_raises_not_implemented(
    adapter: CybersecurityAdapter,
) -> None:
    """Community stub should raise NotImplementedError."""
    item = DataItem(id="test", title="test", body="test")
    with pytest.raises(NotImplementedError, match="enterprise feature"):
        adapter.format_item_for_prompt(item)
