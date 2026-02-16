"""HTML report generator for claw-review output."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from jinja2 import Template
from rich.console import Console

console = Console()

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClawReview Report — {{ repo }}</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --yellow: #d29922;
            --red: #f85149;
            --orange: #db6d28;
            --purple: #bc8cff;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        h2 { font-size: 1.4rem; margin: 2rem 0 1rem; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
        h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem; }
        .meta { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 2rem; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.2rem;
            text-align: center;
        }
        .stat-card .number { font-size: 2rem; font-weight: 700; color: var(--accent); }
        .stat-card .label { color: var(--text-muted); font-size: 0.85rem; }
        .cluster {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.2rem;
            margin: 1rem 0;
        }
        .cluster.duplicate { border-left: 4px solid var(--yellow); }
        .cluster.singleton { border-left: 4px solid var(--text-muted); }
        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
        }
        .badge {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-dup { background: rgba(210, 153, 34, 0.2); color: var(--yellow); }
        .badge-category { background: rgba(88, 166, 255, 0.2); color: var(--accent); }
        .badge-merge { background: rgba(63, 185, 80, 0.2); color: var(--green); }
        .badge-review { background: rgba(210, 153, 34, 0.2); color: var(--yellow); }
        .badge-discuss { background: rgba(188, 140, 255, 0.2); color: var(--purple); }
        .badge-close { background: rgba(248, 81, 73, 0.2); color: var(--red); }
        .pr-row {
            display: grid;
            grid-template-columns: 50px 1fr 100px 100px;
            gap: 0.5rem;
            align-items: center;
            padding: 0.6rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }
        .pr-row:last-child { border-bottom: none; }
        .pr-rank {
            font-weight: 700;
            font-size: 1.1rem;
            text-align: center;
        }
        .pr-rank.best { color: var(--green); }
        .pr-link { color: var(--accent); text-decoration: none; }
        .pr-link:hover { text-decoration: underline; }
        .pr-author { color: var(--text-muted); font-size: 0.8rem; }
        .score-bar {
            height: 8px;
            border-radius: 4px;
            background: var(--border);
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .score-fill.high { background: var(--green); }
        .score-fill.med { background: var(--yellow); }
        .score-fill.low { background: var(--red); }
        .alignment-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.8rem 0;
        }
        .alignment-card.flagged { border-left: 4px solid var(--red); }
        .drift-list { margin: 0.5rem 0 0 1.2rem; color: var(--text-muted); font-size: 0.85rem; }
        .drift-list li { margin: 0.2rem 0; }
        .section-desc { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem; }
        .human-review-flag {
            color: var(--orange);
            font-size: 0.8rem;
            font-weight: 600;
        }
        .consensus-detail { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem; }
        footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.8rem; text-align: center; }
    </style>
</head>
<body>
    <h1>🦞 ClawReview Report</h1>
    <div class="meta">
        <strong>{{ repo }}</strong> · {{ pr_count }} open PRs analyzed · {{ timestamp }}
        <br>Models: {{ providers | join(', ') }}
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="number">{{ pr_count }}</div>
            <div class="label">PRs Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="number">{{ dup_cluster_count }}</div>
            <div class="label">Duplicate Clusters</div>
        </div>
        <div class="stat-card">
            <div class="number">{{ dup_pr_count }}</div>
            <div class="label">Duplicate PRs</div>
        </div>
        <div class="stat-card">
            <div class="number">{{ flagged_count }}</div>
            <div class="label">Vision Drift Flags</div>
        </div>
    </div>

    <!-- DUPLICATE CLUSTERS -->
    <h2>🔀 Duplicate Clusters</h2>
    <p class="section-desc">PRs grouped by semantic intent. Within each cluster, PRs are ranked by quality score — #1 is the recommended merge candidate.</p>

    {% for cluster in duplicate_clusters %}
    <div class="cluster duplicate">
        <div class="cluster-header">
            <div>
                <h3>{{ cluster.intent_summary }}</h3>
                <span class="badge badge-dup">{{ cluster.prs | length }} duplicates</span>
                <span class="badge badge-category">{{ cluster.category }}</span>
                <span style="color: var(--text-muted); font-size: 0.8rem; margin-left: 0.5rem;">
                    area: {{ cluster.affected_area }} · confidence: {{ (cluster.confidence * 100) | round(0) }}%
                </span>
            </div>
        </div>
        <div>
            <div class="pr-row" style="font-weight: 600; color: var(--text-muted); font-size: 0.8rem;">
                <div>Rank</div>
                <div>Pull Request</div>
                <div>Quality</div>
                <div>Review</div>
            </div>
            {% for pr in cluster.prs %}
            <div class="pr-row">
                <div class="pr-rank {{ 'best' if pr.quality_rank == 1 else '' }}">
                    {% if pr.quality_rank == 1 %}⭐{% endif %}#{{ pr.quality_rank }}
                </div>
                <div>
                    <a class="pr-link" href="{{ pr.url }}" target="_blank">#{{ pr.number }}</a>
                    {{ pr.title }}
                    <div class="pr-author">@{{ pr.author }}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; margin-bottom: 2px;">{{ pr.quality_score | round(1) }}/10</div>
                    <div class="score-bar">
                        <div class="score-fill {{ 'high' if pr.quality_score >= 7 else 'med' if pr.quality_score >= 5 else 'low' }}"
                             style="width: {{ pr.quality_score * 10 }}%"></div>
                    </div>
                </div>
                <div>
                    {% if pr.needs_human_review %}
                    <span class="human-review-flag">⚠ Human Review</span>
                    {% else %}
                    <span style="color: var(--green); font-size: 0.8rem;">✓ Consensus</span>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endfor %}

    {% if not duplicate_clusters %}
    <p style="color: var(--text-muted); padding: 1rem;">No duplicate clusters found — all PRs appear to address unique intents.</p>
    {% endif %}

    <!-- VISION ALIGNMENT -->
    <h2>🧭 Vision Alignment</h2>
    <p class="section-desc">How well each PR aligns with the project's stated architecture and goals. Low scores indicate potential drift.</p>

    {% for item in alignment_flagged %}
    <div class="alignment-card flagged">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <a class="pr-link" href="{{ item.pr_url }}" target="_blank">#{{ item.pr_number }}</a>
                <strong>{{ item.pr_title }}</strong>
                <span class="pr-author">@{{ item.pr_author }}</span>
            </div>
            <div>
                <span class="badge badge-{{ item.recommendation | lower }}">{{ item.recommendation }}</span>
                <span style="font-size: 0.9rem; margin-left: 0.5rem;">{{ item.alignment_score | round(1) }}/10</span>
            </div>
        </div>
        {% if item.drift_concerns %}
        <ul class="drift-list">
            {% for concern in item.drift_concerns %}
            <li>{{ concern }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        {% if item.rationale %}
        <div class="consensus-detail">{{ item.rationale }}</div>
        {% endif %}
    </div>
    {% endfor %}

    {% if not alignment_flagged %}
    <p style="color: var(--text-muted); padding: 1rem;">All analyzed PRs show acceptable alignment with project vision.</p>
    {% endif %}

    <!-- TOP QUALITY -->
    <h2>⭐ Top Quality PRs</h2>
    <p class="section-desc">Highest-scoring PRs across all quality dimensions.</p>

    {% for qs in top_quality %}
    <div class="alignment-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <a class="pr-link" href="{{ qs.pr_url }}" target="_blank">#{{ qs.pr_number }}</a>
                <strong>{{ qs.pr_title }}</strong>
                <span class="pr-author">@{{ qs.pr_author }}</span>
            </div>
            <div>
                <span style="font-size: 1.1rem; font-weight: 700; color: var(--green);">{{ qs.overall_score | round(1) }}/10</span>
            </div>
        </div>
        <div class="consensus-detail">{{ qs.summary }}</div>
    </div>
    {% endfor %}

    <footer>
        Generated by <strong>claw-review</strong> · Multi-model consensus PR triage
        <br>{{ timestamp }}
    </footer>
</body>
</html>
"""


def generate_report(
    repo: str,
    clusters: list[dict],
    quality_scores: list[dict],
    alignment_scores: list[dict],
    providers: list[str],
    output_path: str = "claw-review-report.html",
) -> str:
    """Generate an HTML report from analysis results.

    Args:
        repo: Repository name
        clusters: List of cluster dicts
        quality_scores: List of quality score dicts
        alignment_scores: List of alignment score dicts
        providers: List of model provider names used
        output_path: Where to write the HTML file

    Returns:
        Path to generated report
    """
    template = Template(REPORT_TEMPLATE)

    # Separate duplicate clusters from singletons
    dup_clusters = [c for c in clusters if len(c.get("prs", [])) > 1]
    dup_pr_count = sum(len(c["prs"]) for c in dup_clusters)

    # Alignment: show flagged PRs (score < 5)
    flagged = [a for a in alignment_scores if a.get("alignment_score", 10) < 5]

    # Top quality PRs (score >= 7, limit 10)
    top_quality = [q for q in quality_scores if q.get("overall_score", 0) >= 7][:10]

    total_prs = sum(len(c.get("prs", [])) for c in clusters)

    html = template.render(
        repo=repo,
        pr_count=total_prs,
        dup_cluster_count=len(dup_clusters),
        dup_pr_count=dup_pr_count,
        flagged_count=len(flagged),
        duplicate_clusters=dup_clusters,
        alignment_flagged=flagged,
        top_quality=top_quality,
        providers=providers,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    output = Path(output_path)
    output.write_text(html)
    console.print(f"[bold green]✓ Report written to {output}")
    return str(output)


def generate_json_report(
    repo: str,
    clusters: list[dict],
    quality_scores: list[dict],
    alignment_scores: list[dict],
    providers: list[str],
    output_path: str = "claw-review-report.json",
) -> str:
    """Generate a JSON report for programmatic consumption."""
    report = {
        "repo": repo,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": providers,
        "summary": {
            "total_prs": sum(len(c.get("prs", [])) for c in clusters),
            "duplicate_clusters": sum(
                1 for c in clusters if len(c.get("prs", [])) > 1
            ),
            "duplicate_prs": sum(
                len(c["prs"])
                for c in clusters
                if len(c.get("prs", [])) > 1
            ),
            "flagged_for_drift": sum(
                1
                for a in alignment_scores
                if a.get("alignment_score", 10) < 5
            ),
        },
        "clusters": clusters,
        "quality_scores": quality_scores,
        "alignment_scores": alignment_scores,
    }

    output = Path(output_path)
    output.write_text(json.dumps(report, indent=2))
    console.print(f"[bold green]✓ JSON report written to {output}")
    return str(output)


def merge_reports(report_files: list[str]) -> dict:
    """Merge multiple JSON reports into a combined report.

    Combines clusters, quality scores, and alignment scores from
    multiple report files. Deduplicates PRs that appear in multiple
    reports (keeps the latest entry based on report file order — later
    files are considered more recent).

    Args:
        report_files: List of paths to JSON report files

    Returns:
        Combined report dict with keys: repo, timestamp, providers,
        clusters, quality_scores, alignment_scores, summary.
    """
    all_clusters: list[dict] = []
    all_quality: list[dict] = []
    all_alignment: list[dict] = []
    all_providers: set[str] = set()
    repo = ""
    latest_timestamp = ""

    # Track seen PR numbers for deduplication
    seen_quality_prs: dict[int, int] = {}  # pr_number -> index in all_quality
    seen_alignment_prs: dict[int, int] = {}  # pr_number -> index in all_alignment
    seen_cluster_prs: set[int] = set()  # pr_numbers already in clusters

    for report_path in report_files:
        data = json.loads(Path(report_path).read_text())

        # Use the repo from the first report, or update if later ones differ
        if data.get("repo"):
            repo = data["repo"]

        timestamp = data.get("timestamp", "")
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp

        # Merge providers (union)
        for provider in data.get("providers", []):
            all_providers.add(provider)

        # Merge quality scores (deduplicate by PR number, keep latest)
        for qs in data.get("quality_scores", []):
            pr_num = qs.get("pr_number")
            if pr_num is not None and pr_num in seen_quality_prs:
                # Replace with the latest entry
                all_quality[seen_quality_prs[pr_num]] = qs
            else:
                if pr_num is not None:
                    seen_quality_prs[pr_num] = len(all_quality)
                all_quality.append(qs)

        # Merge alignment scores (deduplicate by PR number, keep latest)
        for als in data.get("alignment_scores", []):
            pr_num = als.get("pr_number")
            if pr_num is not None and pr_num in seen_alignment_prs:
                all_alignment[seen_alignment_prs[pr_num]] = als
            else:
                if pr_num is not None:
                    seen_alignment_prs[pr_num] = len(all_alignment)
                all_alignment.append(als)

        # Merge clusters — deduplicate PRs within clusters
        for cluster in data.get("clusters", []):
            # Filter out PRs we've already seen in previous clusters
            new_prs = [
                pr for pr in cluster.get("prs", [])
                if pr.get("number") not in seen_cluster_prs
            ]
            if new_prs:
                merged_cluster = dict(cluster)
                merged_cluster["prs"] = new_prs
                all_clusters.append(merged_cluster)
                for pr in new_prs:
                    pr_num = pr.get("number")
                    if pr_num is not None:
                        seen_cluster_prs.add(pr_num)

    # Compute summary
    total_prs = sum(len(c.get("prs", [])) for c in all_clusters)
    dup_clusters = [c for c in all_clusters if len(c.get("prs", [])) > 1]

    return {
        "repo": repo,
        "timestamp": latest_timestamp,
        "providers": sorted(all_providers),
        "summary": {
            "total_prs": total_prs,
            "duplicate_clusters": len(dup_clusters),
            "duplicate_prs": sum(len(c["prs"]) for c in dup_clusters),
            "flagged_for_drift": sum(
                1 for a in all_alignment
                if a.get("alignment_score", 10) < 5
            ),
        },
        "clusters": all_clusters,
        "quality_scores": all_quality,
        "alignment_scores": all_alignment,
    }
