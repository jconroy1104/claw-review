"""Fraud detection adapter for the consensus platform.

Reads transaction data from JSON or CSV files and converts each
transaction into a DataItem for multi-model fraud analysis.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from claw_review.platform.interfaces import DataItem


# Default column mapping for CSV ingestion
_DEFAULT_CSV_COLUMNS: dict[str, str] = {
    "transaction_id": "transaction_id",
    "amount": "amount",
    "merchant": "merchant",
    "location": "location",
    "timestamp": "timestamp",
    "card_type": "card_type",
    "transaction_type": "transaction_type",
    "customer_id": "customer_id",
    "historical_avg": "historical_avg",
}


class FraudDetectionAdapter:
    """Adapter that ingests financial transactions for fraud analysis.

    Supports JSON and CSV input formats. Each transaction is converted
    into a :class:`DataItem` suitable for the consensus engine.
    """

    domain: str = "fraud-detection"

    async def fetch_items(
        self, source: str, max_items: int = 100, **kwargs: Any
    ) -> list[DataItem]:
        """Fetch transactions from a JSON or CSV file.

        Args:
            source: Path to a JSON or CSV file containing transactions.
            max_items: Maximum number of transactions to return.
            **kwargs: May contain ``column_mapping`` for CSV files and
                      ``file_format`` ("json" or "csv") to override
                      auto-detection.

        Returns:
            List of DataItem objects, one per transaction.
        """
        file_format: str = kwargs.get("file_format", "")
        column_mapping: dict[str, str] = kwargs.get(
            "column_mapping", _DEFAULT_CSV_COLUMNS
        )

        if not file_format:
            file_format = "csv" if source.endswith(".csv") else "json"

        raw_transactions = self._load_transactions(source, file_format, column_mapping)
        items: list[DataItem] = []
        for txn in raw_transactions[:max_items]:
            items.append(self._transaction_to_item(txn))
        return items

    async def fetch_context_docs(
        self, source: str, **kwargs: Any
    ) -> dict[str, str]:
        """Load customer profiles or fraud rules for alignment scoring.

        Args:
            source: Path to a directory or file, or empty string.
            **kwargs: May contain ``profile_paths`` — a list of file paths.

        Returns:
            Dict mapping document name to content. Empty if none provided.
        """
        docs: dict[str, str] = {}
        profile_paths: list[str] = kwargs.get("profile_paths", [])
        for path_str in profile_paths:
            p = Path(path_str)
            if p.is_file():
                docs[p.name] = p.read_text(encoding="utf-8")
        return docs

    def format_item_for_prompt(self, item: DataItem) -> str:
        """Format a transaction for inclusion in an LLM prompt.

        Args:
            item: The DataItem representing a transaction.

        Returns:
            A human-readable string with transaction details.
        """
        meta = item.metadata
        amount = meta.get("amount", 0)
        hist = meta.get("historical_avg", "N/A")
        lines = [
            f"Transaction ID: {item.id}",
            f"Merchant: {meta.get('merchant', 'N/A')}",
            f"Amount: ${amount:.2f}" if isinstance(amount, (int, float)) else f"Amount: {amount}",
            f"Location: {meta.get('location', 'N/A')}",
            f"Timestamp: {meta.get('timestamp', 'N/A')}",
            f"Card Type: {meta.get('card_type', 'N/A')}",
            f"Transaction Type: {meta.get('transaction_type', 'N/A')}",
            f"Customer ID: {meta.get('customer_id', 'N/A')}",
            f"Historical Average: ${hist:.2f}" if isinstance(hist, (int, float)) else f"Historical Average: {hist}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_transactions(
        source: str,
        file_format: str,
        column_mapping: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Load transactions from a JSON or CSV file.

        Args:
            source: File path.
            file_format: "json" or "csv".
            column_mapping: Mapping from canonical field names to CSV column headers.

        Returns:
            List of raw transaction dictionaries.
        """
        path = Path(source)
        text = path.read_text(encoding="utf-8")

        if file_format == "csv":
            return FraudDetectionAdapter._parse_csv(text, column_mapping)

        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]

    @staticmethod
    def _parse_csv(
        text: str, column_mapping: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Parse CSV text into transaction dicts using column mapping.

        Args:
            text: Raw CSV content.
            column_mapping: Maps canonical names to CSV header names.

        Returns:
            List of transaction dicts with canonical field names.
        """
        reader = csv.DictReader(StringIO(text))
        reverse_map = {v: k for k, v in column_mapping.items()}
        transactions: list[dict[str, Any]] = []
        for row in reader:
            txn: dict[str, Any] = {}
            for csv_col, value in row.items():
                canonical = reverse_map.get(csv_col, csv_col)
                # Attempt numeric conversion for amount and historical_avg
                if canonical in ("amount", "historical_avg"):
                    try:
                        txn[canonical] = float(value)
                    except (ValueError, TypeError):
                        txn[canonical] = value
                else:
                    txn[canonical] = value
            transactions.append(txn)
        return transactions

    @staticmethod
    def _transaction_to_item(txn: dict[str, Any]) -> DataItem:
        """Convert a raw transaction dict into a DataItem.

        Args:
            txn: Raw transaction dictionary.

        Returns:
            A DataItem with transaction fields mapped appropriately.
        """
        amount = txn.get("amount", 0)
        merchant = txn.get("merchant", "Unknown")
        try:
            amount_float = float(amount)
            title = f"{merchant} - ${amount_float:.2f}"
        except (ValueError, TypeError):
            title = f"{merchant} - {amount}"

        body = (
            f"Transaction at {merchant} for ${amount} "
            f"in {txn.get('location', 'unknown location')} "
            f"on {txn.get('timestamp', 'unknown time')}. "
            f"Card: {txn.get('card_type', 'N/A')}, "
            f"Type: {txn.get('transaction_type', 'N/A')}."
        )

        return DataItem(
            id=str(txn.get("transaction_id", "")),
            title=title,
            body=body,
            metadata={
                "amount": amount,
                "merchant": merchant,
                "location": txn.get("location", ""),
                "timestamp": txn.get("timestamp", ""),
                "card_type": txn.get("card_type", ""),
                "transaction_type": txn.get("transaction_type", ""),
                "customer_id": txn.get("customer_id", ""),
                "historical_avg": txn.get("historical_avg", 0),
            },
            raw=txn,
        )
