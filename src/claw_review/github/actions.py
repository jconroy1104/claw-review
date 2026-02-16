"""GitHub Actions workflow generator for claw-review."""
from __future__ import annotations

from pathlib import Path

import jinja2

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _load_template() -> jinja2.Template:
    """Load the Actions workflow Jinja2 template."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template("actions_workflow.yml.j2")


def generate_workflow(
    preset: str = "balanced",
    budget: float | None = None,
    skip_alignment: bool = False,
) -> str:
    """Generate YAML content for a GitHub Actions workflow.

    Args:
        preset: Analysis preset (fast, balanced, thorough).
        budget: Optional maximum budget in dollars.
        skip_alignment: Whether to skip vision alignment analysis.

    Returns:
        The YAML workflow content as a string.
    """
    extra_flags: list[str] = []
    if budget is not None:
        extra_flags.append(f"--budget {budget}")
    if skip_alignment:
        extra_flags.append("--skip-alignment")

    template = _load_template()
    return template.render(
        preset=preset,
        extra_flags=" ".join(extra_flags),
    )


def save_workflow(
    output_dir: str,
    preset: str = "balanced",
    budget: float | None = None,
    skip_alignment: bool = False,
) -> str:
    """Save a GitHub Actions workflow to the filesystem.

    Creates the file at ``<output_dir>/.github/workflows/claw-review.yml``.

    Args:
        output_dir: Root directory of the target repository.
        preset: Analysis preset.
        budget: Optional maximum budget.
        skip_alignment: Whether to skip alignment.

    Returns:
        The absolute path to the written workflow file.
    """
    content = generate_workflow(preset=preset, budget=budget, skip_alignment=skip_alignment)
    workflow_path = Path(output_dir) / ".github" / "workflows" / "claw-review.yml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(content)
    return str(workflow_path.resolve())
