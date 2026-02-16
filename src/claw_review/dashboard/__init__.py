"""Interactive web dashboard for exploring claw-review analysis results."""

from __future__ import annotations

from .data_loader import DataLoader
from .app import DashboardApp, generate_static_dashboard

__all__ = ["DashboardApp", "DataLoader", "generate_static_dashboard"]
