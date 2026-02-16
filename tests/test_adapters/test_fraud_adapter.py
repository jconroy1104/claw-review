"""Tests for the fraud detection adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from claw_review.adapters.fraud_detection import FraudDetectionAdapter
from claw_review.platform.interfaces import DataAdapter, DataItem


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def adapter() -> FraudDetectionAdapter:
    """Return a FraudDetectionAdapter instance."""
    return FraudDetectionAdapter()


@pytest.fixture()
def transactions_path() -> str:
    """Path to the sample transactions fixture."""
    return str(FIXTURES_DIR / "sample_transactions.json")


@pytest.fixture()
def sample_transactions() -> list[dict[str, Any]]:
    """Load sample transactions as raw dicts."""
    path = FIXTURES_DIR / "sample_transactions.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def test_fetch_items_loads_all_transactions(
    adapter: FraudDetectionAdapter, transactions_path: str
) -> None:
    """fetch_items should load all 20 transactions from the fixture file."""
    items = await adapter.fetch_items(transactions_path)
    assert len(items) == 20
    assert all(isinstance(item, DataItem) for item in items)


async def test_fetch_items_respects_max_items(
    adapter: FraudDetectionAdapter, transactions_path: str
) -> None:
    """fetch_items should respect the max_items parameter."""
    items = await adapter.fetch_items(transactions_path, max_items=3)
    assert len(items) == 3


async def test_item_fields_mapped_correctly(
    adapter: FraudDetectionAdapter, transactions_path: str
) -> None:
    """Each item should have correct id, title, body, and metadata fields."""
    items = await adapter.fetch_items(transactions_path, max_items=1)
    item = items[0]
    assert item.id == "TXN-001"
    assert "Starbucks" in item.title
    assert "$42.50" in item.title
    assert item.metadata["amount"] == 42.50
    assert item.metadata["merchant"] == "Starbucks"
    assert item.metadata["location"] == "Seattle, WA"


async def test_format_item_for_prompt_includes_key_fields(
    adapter: FraudDetectionAdapter, transactions_path: str
) -> None:
    """format_item_for_prompt should include amount, merchant, and location."""
    items = await adapter.fetch_items(transactions_path, max_items=1)
    formatted = adapter.format_item_for_prompt(items[0])
    assert "TXN-001" in formatted
    assert "Starbucks" in formatted
    assert "$42.50" in formatted
    assert "Seattle, WA" in formatted
    assert "visa" in formatted.lower()


async def test_load_from_csv(
    adapter: FraudDetectionAdapter, tmp_path: Path
) -> None:
    """fetch_items should load transactions from CSV files."""
    csv_content = (
        "transaction_id,amount,merchant,location,timestamp,card_type,"
        "transaction_type,customer_id,historical_avg\n"
        "CSV-001,99.99,TestMerchant,Test City,2025-06-15T12:00:00Z,"
        "visa,online,CUST-999,50.00\n"
        "CSV-002,25.00,OtherMerchant,Other City,2025-06-15T13:00:00Z,"
        "mastercard,in_store,CUST-998,30.00\n"
    )
    csv_file = tmp_path / "transactions.csv"
    csv_file.write_text(csv_content)

    items = await adapter.fetch_items(str(csv_file))
    assert len(items) == 2
    assert items[0].id == "CSV-001"
    assert items[0].metadata["amount"] == 99.99
    assert items[0].metadata["merchant"] == "TestMerchant"
    assert items[1].id == "CSV-002"


async def test_handle_missing_fields(
    adapter: FraudDetectionAdapter, tmp_path: Path
) -> None:
    """Adapter should handle transactions with missing optional fields."""
    minimal_txn = [{"transaction_id": "MIN-001", "amount": 10.0}]
    txn_file = tmp_path / "minimal.json"
    txn_file.write_text(json.dumps(minimal_txn))

    items = await adapter.fetch_items(str(txn_file))
    assert len(items) == 1
    item = items[0]
    assert item.id == "MIN-001"
    assert item.metadata["amount"] == 10.0
    assert item.metadata["merchant"] == "Unknown"


async def test_empty_transaction_list(
    adapter: FraudDetectionAdapter, tmp_path: Path
) -> None:
    """Adapter should return empty list for empty JSON array."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    items = await adapter.fetch_items(str(empty_file))
    assert items == []


def test_adapter_satisfies_protocol(adapter: FraudDetectionAdapter) -> None:
    """FraudDetectionAdapter should satisfy the DataAdapter protocol."""
    assert isinstance(adapter, DataAdapter)
    assert adapter.domain == "fraud-detection"


async def test_fetch_context_docs_empty_by_default(
    adapter: FraudDetectionAdapter,
) -> None:
    """fetch_context_docs should return empty dict when no paths given."""
    docs = await adapter.fetch_context_docs("")
    assert docs == {}


async def test_fetch_context_docs_reads_profile_files(
    adapter: FraudDetectionAdapter, tmp_path: Path
) -> None:
    """fetch_context_docs should read profile files when paths are provided."""
    profile = tmp_path / "customer_profile.json"
    profile.write_text('{"customer_id": "CUST-100", "avg_spend": 45.00}')

    docs = await adapter.fetch_context_docs(
        "", profile_paths=[str(profile)]
    )
    assert "customer_profile.json" in docs
    assert "CUST-100" in docs["customer_profile.json"]


async def test_csv_custom_column_mapping(
    adapter: FraudDetectionAdapter, tmp_path: Path
) -> None:
    """CSV loading should support custom column mappings."""
    csv_content = (
        "txn_id,total,store,city,time,card,type,cust,avg\n"
        "MAP-001,55.00,CustomStore,CustomCity,2025-01-01T00:00:00Z,"
        "amex,online,C-1,40.00\n"
    )
    csv_file = tmp_path / "custom.csv"
    csv_file.write_text(csv_content)

    mapping = {
        "transaction_id": "txn_id",
        "amount": "total",
        "merchant": "store",
        "location": "city",
        "timestamp": "time",
        "card_type": "card",
        "transaction_type": "type",
        "customer_id": "cust",
        "historical_avg": "avg",
    }

    items = await adapter.fetch_items(
        str(csv_file), column_mapping=mapping
    )
    assert len(items) == 1
    assert items[0].id == "MAP-001"
    assert items[0].metadata["amount"] == 55.00
    assert items[0].metadata["merchant"] == "CustomStore"
