"""Platform abstractions for domain-agnostic consensus analysis."""

from __future__ import annotations

from .interfaces import (
    AnalysisResult,
    CostSummary,
    DataAdapter,
    DataItem,
    DomainConfig,
)
from .engine import ConsensusEngine
from .registry import AdapterRegistry

__all__ = [
    "AnalysisResult",
    "AdapterRegistry",
    "ConsensusEngine",
    "CostSummary",
    "DataAdapter",
    "DataItem",
    "DomainConfig",
]
