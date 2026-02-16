"""Adapter registry for domain discovery and lookup.

Maintains a class-level mapping from domain names to their adapter
classes and domain configs, allowing the CLI and engine to discover
available domains at runtime.
"""

from __future__ import annotations

from .interfaces import DomainConfig


class AdapterRegistry:
    """Class-level registry mapping domain names to adapters and configs."""

    _registry: dict[str, tuple[type, DomainConfig]] = {}

    @classmethod
    def register(
        cls, domain: str, adapter_class: type, config: DomainConfig
    ) -> None:
        """Register an adapter class and its domain config.

        Args:
            domain: Domain identifier (e.g., "github-pr").
            adapter_class: The adapter class implementing DataAdapter.
            config: Domain-specific configuration.
        """
        cls._registry[domain] = (adapter_class, config)

    @classmethod
    def get(cls, domain: str) -> tuple[type, DomainConfig]:
        """Look up an adapter class and config by domain name.

        Args:
            domain: Domain identifier.

        Returns:
            Tuple of (adapter_class, DomainConfig).

        Raises:
            ValueError: If the domain is not registered.
        """
        if domain not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(none)"
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {available}"
            )
        return cls._registry[domain]

    @classmethod
    def list_domains(cls) -> list[str]:
        """Return all registered domain names.

        Returns:
            Sorted list of domain identifier strings.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def _clear(cls) -> None:
        """Clear the registry. Intended for testing only."""
        cls._registry = {}
