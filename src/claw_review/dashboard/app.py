"""Dashboard web application — FastAPI server and static HTML generator."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Template

from .data_loader import DataLoader

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "dashboard.html.j2"


class DashboardApp:
    """Wraps a :class:`DataLoader` to provide a FastAPI server and static export.

    Args:
        data_loader: A :class:`DataLoader` instance with data already loaded.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        """Initialize with a loaded DataLoader."""
        self._loader = data_loader

    # ------------------------------------------------------------------
    # FastAPI server
    # ------------------------------------------------------------------

    def create_server_app(self) -> FastAPI:
        """Create a FastAPI application for serving the dashboard.

        Returns:
            A configured :class:`FastAPI` instance with the following routes:

            * ``GET /`` — HTML dashboard page
            * ``GET /api/data`` — full report data as JSON
            * ``GET /api/summary`` — summary statistics
            * ``GET /api/filter`` — filtered results (query params)
        """
        app = FastAPI(title="ClawReview Dashboard")
        loader = self._loader

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            """Serve the dashboard HTML page."""
            html = self._render_html()
            return HTMLResponse(content=html)

        @app.get("/api/data", response_class=JSONResponse)
        async def api_data() -> JSONResponse:
            """Return the full report data as JSON."""
            return JSONResponse(content=loader.data)

        @app.get("/api/summary", response_class=JSONResponse)
        async def api_summary() -> JSONResponse:
            """Return summary statistics."""
            return JSONResponse(content=loader.get_summary())

        @app.get("/api/filter", response_class=JSONResponse)
        async def api_filter(
            search: str | None = Query(default=None),
            recommendation: str | None = Query(default=None),
            min_score: float | None = Query(default=None),
            max_score: float | None = Query(default=None),
            category: str | None = Query(default=None),
        ) -> JSONResponse:
            """Filter and search the report data.

            Query parameters:
                search: Case-insensitive title substring match.
                recommendation: Filter alignment scores by recommendation.
                min_score / max_score: Filter quality scores by range.
                category: Filter clusters by category.
            """
            result: dict = {}
            if search:
                result["search_results"] = loader.search(search)
            if recommendation:
                result["alignment"] = loader.filter_by_recommendation(recommendation)
            if min_score is not None or max_score is not None:
                lo = min_score if min_score is not None else 0.0
                hi = max_score if max_score is not None else 10.0
                result["quality"] = loader.filter_by_score_range(lo, hi)
            if category:
                result["clusters"] = loader.filter_by_category(category)
            if not result:
                result = loader.data
            return JSONResponse(content=result)

        return app

    # ------------------------------------------------------------------
    # Static HTML export
    # ------------------------------------------------------------------

    def generate_static(self, output_path: str = "dashboard.html") -> str:
        """Generate a self-contained static HTML dashboard file.

        All report data is embedded as JSON inside a ``<script>`` tag so that
        filtering and sorting work client-side with no server required.

        Args:
            output_path: Filesystem path for the output HTML file.

        Returns:
            The absolute path to the generated file.
        """
        html = self._render_html()
        out = Path(output_path)
        out.write_text(html)
        return str(out.resolve())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_html(self) -> str:
        """Render the Jinja2 dashboard template with the loaded data."""
        template_text = _TEMPLATE_PATH.read_text()
        template = Template(template_text)

        data = self._loader.data
        summary = self._loader.get_summary()

        return template.render(
            repo=data.get("repo", ""),
            generated_at=data.get("generated_at", ""),
            summary=data.get("summary", {}),
            avg_quality_score=summary.get("avg_quality_score", 0),
            total_cost=summary.get("total_cost", 0),
            data_json=json.dumps(data, default=str),
        )


def generate_static_dashboard(
    report_paths: list[str], output: str = "dashboard.html"
) -> str:
    """Convenience function: load reports and generate a static HTML dashboard.

    Args:
        report_paths: List of filesystem paths to JSON report files.
        output: Output HTML file path.

    Returns:
        Absolute path to the generated HTML file.
    """
    loader = DataLoader()
    if len(report_paths) == 1:
        loader.load_report(report_paths[0])
    else:
        loader.load_multiple_reports(report_paths)
    app = DashboardApp(loader)
    return app.generate_static(output)
