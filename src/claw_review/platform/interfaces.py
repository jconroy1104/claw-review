"""Core abstractions for the domain-agnostic consensus platform.

Defines the universal data containers and protocols that allow the
consensus engine to work with any domain (GitHub PRs, security
advisories, fraud cases, etc.) without hardcoded assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class DataItem:
    """Universal item container for any domain.

    Wraps domain-specific data (PRs, tickets, advisories) into a
    common shape that the consensus engine can process.
    """

    id: str
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "metadata": self.metadata,
            "raw": self.raw,
        }


@runtime_checkable
class DataAdapter(Protocol):
    """Protocol for domain-specific data adapters.

    Each domain (github-pr, security-advisory, etc.) implements this
    protocol to fetch items and context documents from its source.
    """

    domain: str

    async def fetch_items(
        self, source: str, max_items: int = 100, **kwargs: Any
    ) -> list[DataItem]:
        """Fetch items from the domain source.

        Args:
            source: Domain-specific source identifier (e.g., "owner/repo").
            max_items: Maximum number of items to fetch.
            **kwargs: Additional domain-specific parameters.

        Returns:
            List of DataItem objects.
        """
        ...

    async def fetch_context_docs(
        self, source: str, **kwargs: Any
    ) -> dict[str, str]:
        """Fetch context documents for alignment scoring.

        Args:
            source: Domain-specific source identifier.
            **kwargs: Additional domain-specific parameters.

        Returns:
            Dict mapping document name to content.
        """
        ...

    def format_item_for_prompt(self, item: DataItem) -> str:
        """Format an item for inclusion in an LLM prompt.

        Args:
            item: The DataItem to format.

        Returns:
            A human-readable string representation of the item.
        """
        ...


@dataclass
class DomainConfig:
    """Configuration for a specific domain's analysis pipeline.

    Contains the prompts, dimensions, and thresholds that customize
    how the consensus engine evaluates items in this domain.
    """

    domain: str
    scoring_dimensions: list[str]
    clustering_prompt: str
    scoring_prompt: str
    alignment_prompt: str
    recommendation_levels: list[str]
    default_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class CostSummary:
    """Summary of API costs for an analysis run."""

    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    by_model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "by_model": self.by_model,
        }


@dataclass
class AnalysisResult:
    """Complete result of a consensus analysis run."""

    domain: str
    source: str
    items_analyzed: int
    clusters: list[dict[str, Any]] = field(default_factory=list)
    quality_scores: list[dict[str, Any]] = field(default_factory=list)
    alignment_scores: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "domain": self.domain,
            "source": self.source,
            "items_analyzed": self.items_analyzed,
            "clusters": self.clusters,
            "quality_scores": self.quality_scores,
            "alignment_scores": self.alignment_scores,
            "cost": self.cost,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
