"""Tests for the cybersecurity SIEM alert adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from claw_review.adapters.cybersecurity import CybersecurityAdapter
from claw_review.platform.interfaces import DataAdapter, DataItem


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def adapter() -> CybersecurityAdapter:
    """Return a CybersecurityAdapter instance."""
    return CybersecurityAdapter()


@pytest.fixture()
def alerts_path() -> str:
    """Path to the sample SIEM alerts fixture."""
    return str(FIXTURES_DIR / "sample_siem_alerts.json")


@pytest.fixture()
def sample_alerts() -> list[dict[str, Any]]:
    """Load sample alerts as raw dicts."""
    path = FIXTURES_DIR / "sample_siem_alerts.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def test_fetch_items_loads_all_alerts(
    adapter: CybersecurityAdapter, alerts_path: str
) -> None:
    """fetch_items should load all 20 alerts from the fixture file."""
    items = await adapter.fetch_items(alerts_path)
    assert len(items) == 20
    assert all(isinstance(item, DataItem) for item in items)


async def test_fetch_items_respects_max_items(
    adapter: CybersecurityAdapter, alerts_path: str
) -> None:
    """fetch_items should respect the max_items parameter."""
    items = await adapter.fetch_items(alerts_path, max_items=5)
    assert len(items) == 5


async def test_item_fields_mapped_correctly(
    adapter: CybersecurityAdapter, alerts_path: str
) -> None:
    """Each item should have correct id, title, body, and metadata fields."""
    items = await adapter.fetch_items(alerts_path, max_items=1)
    item = items[0]
    assert item.id == "SIEM-001"
    assert item.title == "Network Reconnaissance Detected"
    assert "Nmap SYN scan" in item.body
    assert item.metadata["severity"] == "medium"
    assert item.metadata["source_ip"] == "10.0.5.23"
    assert item.metadata["dest_ip"] == "10.0.1.50"
    assert item.metadata["alert_type"] == "port_scan"


async def test_format_item_for_prompt_includes_key_fields(
    adapter: CybersecurityAdapter, alerts_path: str
) -> None:
    """format_item_for_prompt should include severity, IPs, and alert type."""
    items = await adapter.fetch_items(alerts_path, max_items=1)
    formatted = adapter.format_item_for_prompt(items[0])
    assert "SIEM-001" in formatted
    assert "medium" in formatted.lower()
    assert "10.0.5.23" in formatted
    assert "10.0.1.50" in formatted
    assert "port_scan" in formatted
    assert "Nmap SYN scan" in formatted


async def test_handle_missing_fields(
    adapter: CybersecurityAdapter, tmp_path: Path
) -> None:
    """Adapter should handle alerts with missing optional fields gracefully."""
    minimal_alert = [{"alert_id": "MIN-001", "alert_type": "unknown_alert"}]
    alert_file = tmp_path / "minimal.json"
    alert_file.write_text(json.dumps(minimal_alert))

    items = await adapter.fetch_items(str(alert_file))
    assert len(items) == 1
    item = items[0]
    assert item.id == "MIN-001"
    assert item.title == "unknown_alert"
    assert item.body == ""
    assert item.metadata["severity"] == "unknown"
    assert item.metadata["source_ip"] == ""


async def test_empty_alert_list(
    adapter: CybersecurityAdapter, tmp_path: Path
) -> None:
    """Adapter should return empty list for empty JSON array."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    items = await adapter.fetch_items(str(empty_file))
    assert items == []


def test_adapter_satisfies_protocol(adapter: CybersecurityAdapter) -> None:
    """CybersecurityAdapter should satisfy the DataAdapter protocol."""
    assert isinstance(adapter, DataAdapter)
    assert adapter.domain == "cybersecurity"


async def test_fetch_context_docs_empty_by_default(
    adapter: CybersecurityAdapter,
) -> None:
    """fetch_context_docs should return empty dict when no paths given."""
    docs = await adapter.fetch_context_docs("")
    assert docs == {}


async def test_fetch_context_docs_reads_policy_files(
    adapter: CybersecurityAdapter, tmp_path: Path
) -> None:
    """fetch_context_docs should read policy files when paths are provided."""
    policy = tmp_path / "threat_model.md"
    policy.write_text("# Threat Model\nAll servers must use MFA.")

    docs = await adapter.fetch_context_docs(
        "", policy_paths=[str(policy)]
    )
    assert "threat_model.md" in docs
    assert "MFA" in docs["threat_model.md"]


async def test_single_alert_dict_wrapped_in_list(
    adapter: CybersecurityAdapter, tmp_path: Path
) -> None:
    """A single alert dict (not in a list) should be treated as a one-item list."""
    single = {"alert_id": "SINGLE-001", "alert_type": "test", "raw_log": "test log"}
    f = tmp_path / "single.json"
    f.write_text(json.dumps(single))

    items = await adapter.fetch_items(str(f))
    assert len(items) == 1
    assert items[0].id == "SINGLE-001"
