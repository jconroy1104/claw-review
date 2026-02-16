"""Tests for platform interface dataclasses and protocols."""

from __future__ import annotations

from typing import Any

from claw_review.platform.interfaces import (
    AnalysisResult,
    CostSummary,
    DataAdapter,
    DataItem,
    DomainConfig,
)


class TestDataItem:
    """Tests for the DataItem dataclass."""

    def test_creation_minimal(self) -> None:
        item = DataItem(id="1", title="Test", body="Body")
        assert item.id == "1"
        assert item.title == "Test"
        assert item.body == "Body"
        assert item.metadata == {}
        assert item.raw == {}

    def test_creation_with_metadata(self) -> None:
        item = DataItem(
            id="42",
            title="Fix bug",
            body="Description",
            metadata={"author": "alice", "priority": "high"},
            raw={"number": 42, "state": "open"},
        )
        assert item.metadata["author"] == "alice"
        assert item.raw["number"] == 42

    def test_to_dict(self) -> None:
        item = DataItem(
            id="1",
            title="Title",
            body="Body",
            metadata={"key": "val"},
            raw={"x": 1},
        )
        d = item.to_dict()
        assert d == {
            "id": "1",
            "title": "Title",
            "body": "Body",
            "metadata": {"key": "val"},
            "raw": {"x": 1},
        }

    def test_to_dict_roundtrip(self) -> None:
        item = DataItem(id="5", title="T", body="B")
        d = item.to_dict()
        restored = DataItem(**d)
        assert restored == item


class TestDomainConfig:
    """Tests for the DomainConfig dataclass."""

    def test_creation(self) -> None:
        cfg = DomainConfig(
            domain="test-domain",
            scoring_dimensions=["dim1", "dim2"],
            clustering_prompt="cluster prompt",
            scoring_prompt="score prompt",
            alignment_prompt="align prompt",
            recommendation_levels=["ACCEPT", "REJECT"],
            default_thresholds={"similarity": 0.8},
        )
        assert cfg.domain == "test-domain"
        assert len(cfg.scoring_dimensions) == 2
        assert cfg.default_thresholds["similarity"] == 0.8

    def test_default_thresholds_empty(self) -> None:
        cfg = DomainConfig(
            domain="d",
            scoring_dimensions=[],
            clustering_prompt="",
            scoring_prompt="",
            alignment_prompt="",
            recommendation_levels=[],
        )
        assert cfg.default_thresholds == {}


class TestAnalysisResult:
    """Tests for the AnalysisResult dataclass."""

    def test_creation_minimal(self) -> None:
        result = AnalysisResult(
            domain="test",
            source="src",
            items_analyzed=5,
        )
        assert result.domain == "test"
        assert result.items_analyzed == 5
        assert result.clusters == []
        assert result.quality_scores == []
        assert result.alignment_scores == []

    def test_to_dict(self) -> None:
        result = AnalysisResult(
            domain="test",
            source="src",
            items_analyzed=2,
            clusters=[{"id": "c1"}],
            timestamp="2026-01-01T00:00:00Z",
        )
        d = result.to_dict()
        assert d["domain"] == "test"
        assert d["clusters"] == [{"id": "c1"}]
        assert d["timestamp"] == "2026-01-01T00:00:00Z"


class TestCostSummary:
    """Tests for the CostSummary dataclass."""

    def test_creation_defaults(self) -> None:
        cost = CostSummary()
        assert cost.total_cost == 0.0
        assert cost.total_input_tokens == 0
        assert cost.total_requests == 0
        assert cost.by_model == {}

    def test_creation_with_values(self) -> None:
        cost = CostSummary(
            total_cost=1.23,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_requests=3,
            by_model={"model-a": {"cost": 0.5}},
        )
        assert cost.total_cost == 1.23
        assert cost.by_model["model-a"]["cost"] == 0.5

    def test_to_dict(self) -> None:
        cost = CostSummary(total_cost=0.5, total_requests=1)
        d = cost.to_dict()
        assert d["total_cost"] == 0.5
        assert d["total_requests"] == 1


class TestDataAdapterProtocol:
    """Tests that the DataAdapter protocol works correctly."""

    def test_mock_adapter_satisfies_protocol(self) -> None:
        class MockAdapter:
            domain: str = "mock"

            async def fetch_items(
                self, source: str, max_items: int = 100, **kwargs: Any
            ) -> list[DataItem]:
                return []

            async def fetch_context_docs(
                self, source: str, **kwargs: Any
            ) -> dict[str, str]:
                return {}

            def format_item_for_prompt(self, item: DataItem) -> str:
                return f"{item.title}: {item.body}"

        adapter = MockAdapter()
        assert isinstance(adapter, DataAdapter)
        assert adapter.domain == "mock"

    def test_incomplete_adapter_does_not_satisfy(self) -> None:
        class Incomplete:
            domain: str = "bad"

        obj = Incomplete()
        assert not isinstance(obj, DataAdapter)
