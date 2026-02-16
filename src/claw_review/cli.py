"""CLI entry point for claw-review.

Orchestrates the full analysis pipeline:
1. Fetch PRs from GitHub
2. Extract intents + cluster duplicates
3. Score quality per PR
4. Score vision alignment
5. Generate report
"""

import asyncio
import sys
import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Config
from .github_client import fetch_open_prs, fetch_repo_docs
from .models import ModelPool
from .clustering import extract_intents, generate_embeddings, cluster_intents
from .scoring import score_prs, rank_within_clusters
from .alignment import score_alignment
from .report import generate_report, generate_json_report, merge_reports
from .state import load_state

console = Console()


@click.group()
def cli():
    """🦞 claw-review — AI-powered PR triage using multi-model consensus."""
    pass


@cli.command()
@click.option(
    "--repo",
    default=None,
    help="Repository to analyze (owner/name). Default: from .env",
)
@click.option(
    "--max-prs",
    default=None,
    type=int,
    help="Maximum PRs to analyze. Default: from .env or 100",
)
@click.option(
    "--output",
    default="claw-review-report",
    help="Output filename base (without extension)",
)
@click.option(
    "--skip-alignment",
    is_flag=True,
    help="Skip vision alignment scoring (faster)",
)
@click.option(
    "--skip-quality",
    is_flag=True,
    help="Skip quality scoring (only do clustering)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable PR data caching",
)
@click.option(
    "--json-only",
    is_flag=True,
    help="Only output JSON (no HTML report)",
)
@click.option(
    "--preset",
    type=click.Choice(["fast", "balanced", "thorough"]),
    default=None,
    help="Model preset (overrides MODELS env var). Options: fast, balanced, thorough",
)
@click.option(
    "--budget",
    type=float,
    default=None,
    help="Budget limit in USD. Analysis halts if exceeded.",
)
@click.option(
    "--batch-size",
    type=int,
    default=50,
    help="Number of PRs per batch (default: 50).",
)
@click.option(
    "--incremental/--no-incremental",
    default=True,
    help="Skip already-analyzed PRs (default: enabled).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-analyze all PRs, ignoring saved state.",
)
def analyze(repo, max_prs, output, skip_alignment, skip_quality, no_cache, json_only, preset, budget, batch_size, incremental, force):
    """Run full PR analysis on a GitHub repository."""

    # Banner
    console.print(
        Panel(
            "[bold cyan]🦞 claw-review[/bold cyan]\n"
            "Multi-model consensus PR triage",
            border_style="cyan",
        )
    )

    # Load config
    config = Config()
    if repo:
        config.target_repo = repo
    if max_prs:
        config.max_prs = max_prs

    if preset:
        from .config import get_preset
        preset_config = get_preset(preset)
        config.models = preset_config["models"]
        console.print(f"[blue]Preset: {preset} — {preset_config['description']}")

    # Validate
    issues = config.validate()
    if issues:
        for issue in issues:
            console.print(f"[red]✗ {issue}")
        console.print(
            "\n[yellow]Copy .env.example to .env and fill in your API keys."
        )
        sys.exit(1)

    console.print(f"[blue]Target: {config.target_repo}")
    console.print(f"[blue]Max PRs: {config.max_prs}")
    console.print(f"[blue]Models: {', '.join(config.models)}")
    console.print()

    # Initialize model pool
    model_pool = ModelPool(config)

    # Step 1: Fetch PRs
    console.rule("[bold]Step 1: Fetching PRs")
    prs = fetch_open_prs(
        repo=config.target_repo,
        token=config.github_token,
        max_prs=config.max_prs,
        use_cache=not no_cache,
    )
    console.print(f"[green]✓ Fetched {len(prs)} open PRs\n")

    if not prs:
        console.print("[yellow]No open PRs found. Nothing to analyze.")
        return

    # Run async pipeline
    async def _run_pipeline():
        # Step 2: Extract intents + cluster
        console.rule("[bold]Step 2: Intent Extraction & Clustering")
        intents = await extract_intents(prs, model_pool)
        intents = await generate_embeddings(intents, model_pool)
        clusters = cluster_intents(intents, config.similarity_threshold)
        cluster_dicts = [c.to_dict() for c in clusters]
        console.print()

        # Step 3: Quality scoring
        quality_dicts = []
        if not skip_quality:
            console.rule("[bold]Step 3: Quality Scoring")
            dup_pr_numbers = set()
            for c in clusters:
                if len(c.prs) > 1:
                    for pr_info in c.prs:
                        dup_pr_numbers.add(pr_info["number"])

            if dup_pr_numbers:
                prs_to_score = [p for p in prs if p.number in dup_pr_numbers]
                console.print(
                    f"[blue]Scoring {len(prs_to_score)} PRs in duplicate clusters"
                )
                quality_scores = await score_prs(
                    prs_to_score, model_pool, config.quality_disagreement_threshold
                )
                quality_dicts = [qs.to_dict() for qs in quality_scores]
                cluster_dicts = rank_within_clusters(cluster_dicts, quality_scores)
            else:
                console.print("[yellow]No duplicate clusters — skipping quality scoring")
            console.print()

        # Step 4: Vision alignment
        alignment_dicts = []
        if not skip_alignment:
            console.rule("[bold]Step 4: Vision Alignment")
            vision_docs = fetch_repo_docs(config.target_repo, config.github_token)
            if vision_docs:
                alignment_scores = await score_alignment(
                    prs, vision_docs, model_pool, config.alignment_reject_threshold
                )
                alignment_dicts = [a.to_dict() for a in alignment_scores]
            else:
                console.print("[yellow]No vision documents found — skipping")
            console.print()

        return cluster_dicts, quality_dicts, alignment_dicts

    cluster_dicts, quality_dicts, alignment_dicts = asyncio.run(_run_pipeline())

    # Step 5: Generate reports
    console.rule("[bold]Step 5: Generating Reports")

    json_path = generate_json_report(
        repo=config.target_repo,
        clusters=cluster_dicts,
        quality_scores=quality_dicts,
        alignment_scores=alignment_dicts,
        providers=config.models,
        output_path=f"{output}.json",
    )

    html_path = None
    if not json_only:
        html_path = generate_report(
            repo=config.target_repo,
            clusters=cluster_dicts,
            quality_scores=quality_dicts,
            alignment_scores=alignment_dicts,
            providers=config.models,
            output_path=f"{output}.html",
        )

    # Summary
    console.print()
    _print_summary(cluster_dicts, quality_dicts, alignment_dicts)
    console.print()

    if html_path:
        console.print(
            f"[bold green]📊 Open {html_path} in your browser to view the full report"
        )
    console.print(f"[bold green]📄 JSON data: {json_path}")


@cli.command()
def check():
    """Verify configuration and API access."""
    config = Config()
    issues = config.validate()

    if issues:
        console.print("[bold red]Configuration issues found:\n")
        for issue in issues:
            console.print(f"  [red]✗ {issue}")
        console.print(
            "\n[yellow]Copy .env.example to .env and fill in your API keys."
        )
    else:
        console.print("[bold green]✓ Configuration looks good!")
        console.print(f"  Repository: {config.target_repo}")
        console.print(f"  Models: {', '.join(config.models)}")
        console.print(f"  Embedding: {config.embedding_model}")
        console.print(f"  Max PRs: {config.max_prs}")

        # Test GitHub access
        try:
            import httpx

            resp = httpx.get(
                f"https://api.github.com/repos/{config.target_repo}",
                headers={
                    "Authorization": f"Bearer {config.github_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            repo_data = resp.json()
            console.print(
                f"  [green]✓ GitHub access verified — "
                f"{repo_data.get('open_issues_count', '?')} open issues/PRs"
            )
        except Exception as e:
            console.print(f"  [red]✗ GitHub access failed: {e}")


@cli.command()
@click.argument("report_json")
def regenerate(report_json):
    """Regenerate HTML report from existing JSON data."""
    data = json.loads(Path(report_json).read_text())
    output_base = Path(report_json).stem

    generate_report(
        repo=data["repo"],
        clusters=data.get("clusters", []),
        quality_scores=data.get("quality_scores", []),
        alignment_scores=data.get("alignment_scores", []),
        providers=data.get("providers", []),
        output_path=f"{output_base}.html",
    )


@cli.command()
def presets():
    """List available model presets with cost estimates."""
    from .config import PRESETS
    table = Table(title="Model Presets", border_style="cyan")
    table.add_column("Preset", style="bold")
    table.add_column("Models")
    table.add_column("Est. Cost/100 PRs", justify="right")
    table.add_column("Description")

    for name, info in PRESETS.items():
        model_names = ", ".join(m.split("/")[-1] for m in info["models"])
        table.add_row(name, model_names, info["est_cost_per_100_prs"], info["description"])

    console.print(table)


@cli.command()
@click.option("--repo", required=True, help="Repository (owner/name)")
@click.option("--preset", type=click.Choice(["fast", "balanced", "thorough"]), default="balanced")
@click.option("--max-prs", type=int, default=100, help="Number of PRs to estimate for")
def estimate(repo, preset, max_prs):
    """Estimate analysis cost without making API calls."""
    from .config import get_preset
    from .costs import estimate_cost

    preset_config = get_preset(preset)
    result = estimate_cost(max_prs, preset_config["models"])

    console.print(f"\n[bold]Cost Estimate: {repo}[/bold]")
    console.print(f"Preset: {preset} — {preset_config['description']}")
    console.print(f"PRs: {max_prs}")
    console.print(f"Models: {', '.join(m.split('/')[-1] for m in preset_config['models'])}")
    console.print()

    table = Table(border_style="cyan")
    table.add_column("Model", style="bold")
    table.add_column("Est. Cost (Low)", justify="right")
    table.add_column("Est. Cost (High)", justify="right")

    for model, est in result["models"].items():
        table.add_row(
            model.split("/")[-1],
            f"${est['cost_low']:.4f}",
            f"${est['cost_high']:.4f}",
        )

    table.add_section()
    table.add_row(
        "TOTAL",
        f"${result['total_cost_low']:.4f}",
        f"${result['total_cost_high']:.4f}",
    )
    console.print(table)


@cli.command()
@click.argument("report_files", nargs=-1, required=True)
@click.option("--output", "-o", default="merged-report", help="Output filename base")
@click.option("--json-only", is_flag=True, help="Only output JSON (no HTML report)")
def merge(report_files, output, json_only):
    """Merge multiple JSON reports into a combined report."""
    console.print(f"[blue]Merging {len(report_files)} report(s)...")

    merged = merge_reports(list(report_files))

    # Write JSON
    json_path = f"{output}.json"
    Path(json_path).write_text(json.dumps(merged, indent=2))
    console.print(f"[bold green]✓ Merged JSON written to {json_path}")

    # Write HTML
    if not json_only:
        html_path = generate_report(
            repo=merged["repo"],
            clusters=merged["clusters"],
            quality_scores=merged["quality_scores"],
            alignment_scores=merged["alignment_scores"],
            providers=merged["providers"],
            output_path=f"{output}.html",
        )
        console.print(f"[bold green]📊 Open {html_path} in your browser")

    # Summary
    summary = merged["summary"]
    console.print(f"\n[cyan]Merged report: {summary['total_prs']} PRs, "
                  f"{summary['duplicate_clusters']} duplicate clusters, "
                  f"{summary['flagged_for_drift']} drift flags")


@cli.command()
@click.option("--repo", default=None, help="Repository (owner/name). Default: from .env")
def status(repo):
    """Show analysis state for a repository."""
    config = Config()
    if repo:
        config.target_repo = repo

    target = config.target_repo
    state = load_state(target)

    console.print(Panel(
        f"[bold cyan]Analysis State: {target}",
        border_style="cyan",
    ))

    table = Table(border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Repository", state.repo or target)
    table.add_row("Analyzed PRs", str(len(state.analyzed_prs)))
    table.add_row("Last run", state.last_run_timestamp or "Never")
    table.add_row("Models used", ", ".join(state.model_config) if state.model_config else "N/A")

    from .state import _state_file_path
    state_file = _state_file_path(target)
    table.add_row("State file", str(state_file))
    table.add_row("State file exists", str(state_file.exists()))

    console.print(table)


def _print_summary(clusters, quality_scores, alignment_scores):
    """Print a summary table to the console."""
    table = Table(title="Analysis Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    total_prs = sum(len(c.get("prs", [])) for c in clusters)
    dup_clusters = [c for c in clusters if len(c.get("prs", [])) > 1]
    dup_prs = sum(len(c["prs"]) for c in dup_clusters)

    table.add_row("Total PRs analyzed", str(total_prs))
    table.add_row("Duplicate clusters", str(len(dup_clusters)))
    table.add_row("PRs in duplicate clusters", str(dup_prs))
    table.add_row("Unique PRs (singletons)", str(total_prs - dup_prs))

    if quality_scores:
        avg_quality = sum(q.get("overall_score", 0) for q in quality_scores) / len(
            quality_scores
        )
        needs_review = sum(1 for q in quality_scores if q.get("needs_human_review"))
        table.add_row("Avg quality score", f"{avg_quality:.1f}/10")
        table.add_row("Flagged for human review", str(needs_review))

    if alignment_scores:
        low_align = sum(
            1 for a in alignment_scores if a.get("alignment_score", 10) < 5
        )
        table.add_row("Vision drift flags", str(low_align))

    console.print(table)


if __name__ == "__main__":
    cli()
