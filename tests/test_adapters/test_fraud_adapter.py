"""Tests for the fraud detection adapter community stub."""

from __future__ import annotations

import pytest

from claw_review.adapters.fraud_detection import FraudDetectionAdapter
from claw_review.platform.interfaces import DataAdapter, DataItem


@pytest.fixture()
def adapter() -> FraudDetectionAdapter:
    """Return a FraudDetectionAdapter instance."""
    return FraudDetectionAdapter()


def test_adapter_satisfies_protocol(adapter: FraudDetectionAdapter) -> None:
    """FraudDetectionAdapter should satisfy the DataAdapter protocol."""
    assert isinstance(adapter, DataAdapter)
    assert adapter.domain == "fraud-detection"


async def test_fetch_items_raises_not_implemented(
    adapter: FraudDetectionAdapter,
) -> None:
    """Community stub should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="enterprise feature"):
        await adapter.fetch_items("transactions.json")


async def test_fetch_context_docs_raises_not_implemented(
    adapter: FraudDetectionAdapter,
) -> None:
    """Community stub should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="enterprise feature"):
        await adapter.fetch_context_docs("")


def test_format_item_raises_not_implemented(
    adapter: FraudDetectionAdapter,
) -> None:
    """Community stub should raise NotImplementedError."""
    item = DataItem(id="test", title="test", body="test")
    with pytest.raises(NotImplementedError, match="enterprise feature"):
        adapter.format_item_for_prompt(item)
