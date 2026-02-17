"""Cybersecurity SIEM alert adapter — community stub.

This is a placeholder adapter that demonstrates the interface for building
a cybersecurity domain adapter. The full implementation with production
prompts, SIEM ingestion logic, and threat-scoring heuristics is available
in the enterprise edition.

To build your own cybersecurity adapter, implement the three methods
defined by the :class:`DataAdapter` protocol:

* ``fetch_items`` — Ingest alerts from your SIEM (Splunk, Elastic, etc.)
* ``fetch_context_docs`` — Load threat models or security policies
* ``format_item_for_prompt`` — Format an alert for LLM evaluation

See the GitHub PR adapter (``adapters/github_pr.py``) for a complete
reference implementation.
"""

from __future__ import annotations

from typing import Any

from claw_review.platform.interfaces import DataItem


class CybersecurityAdapter:
    """Stub adapter for cybersecurity SIEM alert analysis.

    This community stub satisfies the :class:`DataAdapter` protocol but
    raises :class:`NotImplementedError` for all operations.  Replace with
    your own implementation or upgrade to the enterprise edition.
    """

    domain: str = "cybersecurity"

    async def fetch_items(
        self, source: str, max_items: int = 100, **kwargs: Any
    ) -> list[DataItem]:
        """Fetch SIEM alerts from a data source.

        Args:
            source: Path to a JSON file containing alerts, or a SIEM endpoint.
            max_items: Maximum number of alerts to return.
            **kwargs: Additional parameters (e.g., time range, severity filter).

        Raises:
            NotImplementedError: This is a community stub.
        """
        raise NotImplementedError(
            "Cybersecurity adapter is an enterprise feature. "
            "See README for details on building your own adapter."
        )

    async def fetch_context_docs(
        self, source: str, **kwargs: Any
    ) -> dict[str, str]:
        """Load threat model or security policy documents for alignment.

        Args:
            source: Path to policy documents directory.
            **kwargs: May contain ``policy_paths`` — a list of file paths.

        Raises:
            NotImplementedError: This is a community stub.
        """
        raise NotImplementedError(
            "Cybersecurity adapter is an enterprise feature. "
            "See README for details on building your own adapter."
        )

    def format_item_for_prompt(self, item: DataItem) -> str:
        """Format a SIEM alert for inclusion in an LLM prompt.

        Args:
            item: The DataItem representing a SIEM alert.

        Raises:
            NotImplementedError: This is a community stub.
        """
        raise NotImplementedError(
            "Cybersecurity adapter is an enterprise feature. "
            "See README for details on building your own adapter."
        )
