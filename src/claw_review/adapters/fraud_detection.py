"""Fraud detection adapter — community stub.

This is a placeholder adapter that demonstrates the interface for building
a fraud detection domain adapter. The full implementation with production
prompts, CSV/JSON ingestion, and transaction-scoring logic is available
in the enterprise edition.

To build your own fraud detection adapter, implement the three methods
defined by the :class:`DataAdapter` protocol:

* ``fetch_items`` — Ingest transactions from JSON, CSV, or a database
* ``fetch_context_docs`` — Load customer profiles or fraud rules
* ``format_item_for_prompt`` — Format a transaction for LLM evaluation

See the GitHub PR adapter (``adapters/github_pr.py``) for a complete
reference implementation.
"""

from __future__ import annotations

from typing import Any

from claw_review.platform.interfaces import DataItem


class FraudDetectionAdapter:
    """Stub adapter for financial transaction fraud analysis.

    This community stub satisfies the :class:`DataAdapter` protocol but
    raises :class:`NotImplementedError` for all operations.  Replace with
    your own implementation or upgrade to the enterprise edition.
    """

    domain: str = "fraud-detection"

    async def fetch_items(
        self, source: str, max_items: int = 100, **kwargs: Any
    ) -> list[DataItem]:
        """Fetch transactions from a data source.

        Args:
            source: Path to a JSON or CSV file containing transactions.
            max_items: Maximum number of transactions to return.
            **kwargs: May contain ``column_mapping`` for CSV files and
                      ``file_format`` to override auto-detection.

        Raises:
            NotImplementedError: This is a community stub.
        """
        raise NotImplementedError(
            "Fraud detection adapter is an enterprise feature. "
            "See README for details on building your own adapter."
        )

    async def fetch_context_docs(
        self, source: str, **kwargs: Any
    ) -> dict[str, str]:
        """Load customer profiles or fraud rules for alignment scoring.

        Args:
            source: Path to a directory or file, or empty string.
            **kwargs: May contain ``profile_paths`` — a list of file paths.

        Raises:
            NotImplementedError: This is a community stub.
        """
        raise NotImplementedError(
            "Fraud detection adapter is an enterprise feature. "
            "See README for details on building your own adapter."
        )

    def format_item_for_prompt(self, item: DataItem) -> str:
        """Format a transaction for inclusion in an LLM prompt.

        Args:
            item: The DataItem representing a transaction.

        Raises:
            NotImplementedError: This is a community stub.
        """
        raise NotImplementedError(
            "Fraud detection adapter is an enterprise feature. "
            "See README for details on building your own adapter."
        )
