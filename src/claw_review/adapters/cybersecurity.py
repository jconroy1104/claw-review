"""Cybersecurity SIEM alert adapter for the consensus platform.

Reads SIEM alert data from JSON files or stdin and converts each alert
into a DataItem for multi-model consensus analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from claw_review.platform.interfaces import DataItem


class CybersecurityAdapter:
    """Adapter that ingests SIEM alerts for consensus-based threat analysis.

    Reads JSON alert data from a file path or stdin and converts each
    alert into a :class:`DataItem` suitable for clustering, scoring,
    and alignment analysis.
    """

    domain: str = "cybersecurity"

    async def fetch_items(
        self, source: str, max_items: int = 100, **kwargs: Any
    ) -> list[DataItem]:
        """Fetch SIEM alerts from a JSON file or stdin.

        Args:
            source: Path to a JSON file containing alerts, or "-" for stdin.
            max_items: Maximum number of alerts to return.
            **kwargs: Additional parameters (unused).

        Returns:
            List of DataItem objects, one per alert.
        """
        raw_alerts = self._load_alerts(source)
        items: list[DataItem] = []
        for alert in raw_alerts[:max_items]:
            items.append(self._alert_to_item(alert))
        return items

    async def fetch_context_docs(
        self, source: str, **kwargs: Any
    ) -> dict[str, str]:
        """Load threat model or security policy documents for alignment.

        Args:
            source: Path to a directory or file containing policy docs,
                    or an empty string if none available.
            **kwargs: May contain ``policy_paths`` — a list of file paths.

        Returns:
            Dict mapping document name to content. Empty if none provided.
        """
        docs: dict[str, str] = {}
        policy_paths: list[str] = kwargs.get("policy_paths", [])
        for path_str in policy_paths:
            p = Path(path_str)
            if p.is_file():
                docs[p.name] = p.read_text(encoding="utf-8")
        return docs

    def format_item_for_prompt(self, item: DataItem) -> str:
        """Format a SIEM alert for inclusion in an LLM prompt.

        Args:
            item: The DataItem representing a SIEM alert.

        Returns:
            A human-readable string with alert details.
        """
        meta = item.metadata
        lines = [
            f"Alert ID: {item.id}",
            f"Rule: {item.title}",
            f"Severity: {meta.get('severity', 'unknown')}",
            f"Alert Type: {meta.get('alert_type', 'unknown')}",
            f"Source IP: {meta.get('source_ip', 'N/A')}",
            f"Destination IP: {meta.get('dest_ip', 'N/A')}",
            f"Timestamp: {meta.get('timestamp', 'N/A')}",
            f"Affected Asset: {meta.get('affected_asset', 'N/A')}",
            f"IoC: {meta.get('indicator_of_compromise', 'none')}",
            f"Raw Log: {item.body[:500]}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_alerts(source: str) -> list[dict[str, Any]]:
        """Load alert JSON from file or stdin.

        Args:
            source: File path or "-" for stdin.

        Returns:
            List of raw alert dictionaries.
        """
        if source == "-":
            data = json.load(sys.stdin)
        else:
            path = Path(source)
            data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return [data]

    @staticmethod
    def _alert_to_item(alert: dict[str, Any]) -> DataItem:
        """Convert a raw alert dict into a DataItem.

        Args:
            alert: Raw SIEM alert dictionary.

        Returns:
            A DataItem with alert fields mapped to id, title, body, metadata.
        """
        return DataItem(
            id=str(alert.get("alert_id", "")),
            title=alert.get("rule_name") or alert.get("alert_type", "Unknown Alert"),
            body=alert.get("raw_log", ""),
            metadata={
                "severity": alert.get("severity", "unknown"),
                "source_ip": alert.get("source_ip", ""),
                "dest_ip": alert.get("dest_ip", ""),
                "alert_type": alert.get("alert_type", ""),
                "timestamp": alert.get("timestamp", ""),
                "indicator_of_compromise": alert.get("indicator_of_compromise", ""),
                "affected_asset": alert.get("affected_asset", ""),
            },
            raw=alert,
        )
