"""Analysis state management for incremental PR analysis.

Stores which PRs have been analyzed, enabling incremental runs
that skip already-processed PRs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .github_client import PRData

DEFAULT_STATE_DIR = ".claw-review-state"


def _repo_hash(repo: str) -> str:
    """Generate a stable hash for a repository name."""
    return hashlib.md5(repo.encode()).hexdigest()


@dataclass
class AnalyzedPR:
    """Record of a single analyzed PR."""

    timestamp: str
    cluster_id: str | None = None
    quality_score: float | None = None
    alignment_score: float | None = None

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "timestamp": self.timestamp,
            "cluster_id": self.cluster_id,
            "quality_score": self.quality_score,
            "alignment_score": self.alignment_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnalyzedPR:
        """Deserialize from dict."""
        return cls(
            timestamp=data.get("timestamp", ""),
            cluster_id=data.get("cluster_id"),
            quality_score=data.get("quality_score"),
            alignment_score=data.get("alignment_score"),
        )


@dataclass
class AnalysisState:
    """Persistent state for incremental analysis.

    Tracks which PRs have already been analyzed so subsequent
    runs can skip them.
    """

    repo: str
    analyzed_prs: dict[int, AnalyzedPR] = field(default_factory=dict)
    last_run_timestamp: str = ""
    model_config: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize state to a JSON-compatible dict."""
        return {
            "repo": self.repo,
            "analyzed_prs": {
                str(pr_num): entry.to_dict()
                for pr_num, entry in self.analyzed_prs.items()
            },
            "last_run_timestamp": self.last_run_timestamp,
            "model_config": self.model_config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnalysisState:
        """Deserialize state from a dict."""
        analyzed_prs: dict[int, AnalyzedPR] = {}
        for pr_num_str, entry_data in data.get("analyzed_prs", {}).items():
            analyzed_prs[int(pr_num_str)] = AnalyzedPR.from_dict(entry_data)

        return cls(
            repo=data.get("repo", ""),
            analyzed_prs=analyzed_prs,
            last_run_timestamp=data.get("last_run_timestamp", ""),
            model_config=data.get("model_config", []),
        )


def _state_file_path(repo: str, state_dir: str = DEFAULT_STATE_DIR) -> Path:
    """Get the state file path for a given repo."""
    return Path(state_dir) / f"{_repo_hash(repo)}.json"


def load_state(repo: str, state_dir: str = DEFAULT_STATE_DIR) -> AnalysisState:
    """Load analysis state from disk, or create a new empty state.

    Args:
        repo: Repository in 'owner/name' format
        state_dir: Directory where state files are stored

    Returns:
        Loaded or freshly created AnalysisState
    """
    state_file = _state_file_path(repo, state_dir)
    if state_file.exists():
        data = json.loads(state_file.read_text())
        return AnalysisState.from_dict(data)
    return AnalysisState(repo=repo)


def save_state(state: AnalysisState, state_dir: str = DEFAULT_STATE_DIR) -> None:
    """Save analysis state to disk using atomic write.

    Writes to a temporary file first, then renames to the final path
    to prevent corruption from interrupted writes.

    Args:
        state: The state to persist
        state_dir: Directory where state files are stored
    """
    state_dir_path = Path(state_dir)
    state_dir_path.mkdir(parents=True, exist_ok=True)

    state_file = _state_file_path(state.repo, state_dir)
    data = json.dumps(state.to_dict(), indent=2)

    # Atomic write: write to temp file, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(state_dir_path), suffix=".tmp", prefix=".state-"
    )
    closed = False
    try:
        os.write(fd, data.encode())
        os.close(fd)
        closed = True
        os.replace(tmp_path, str(state_file))
    except Exception:
        if not closed:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def get_unanalyzed_prs(
    all_prs: list[PRData], state: AnalysisState
) -> list[PRData]:
    """Filter PRs to only those not yet analyzed.

    Args:
        all_prs: All fetched PRs
        state: Current analysis state

    Returns:
        List of PRs not present in the state's analyzed_prs
    """
    return [pr for pr in all_prs if pr.number not in state.analyzed_prs]


def update_state(
    state: AnalysisState,
    clusters: list[dict],
    quality_scores: list[dict],
    alignment_scores: list[dict],
) -> AnalysisState:
    """Merge new analysis results into existing state.

    Updates the state with newly analyzed PRs from clusters,
    quality scores, and alignment scores.

    Args:
        state: Current state to update (modified in place)
        clusters: List of cluster dicts from analysis
        quality_scores: List of quality score dicts
        alignment_scores: List of alignment score dicts

    Returns:
        The updated state (same object, modified in place)
    """
    now = datetime.now(timezone.utc).isoformat()
    state.last_run_timestamp = now

    # Build lookup maps for scores
    quality_map: dict[int, float] = {}
    for qs in quality_scores:
        pr_num = qs.get("pr_number")
        if pr_num is not None:
            quality_map[pr_num] = qs.get("overall_score", 0.0)

    alignment_map: dict[int, float] = {}
    for als in alignment_scores:
        pr_num = als.get("pr_number")
        if pr_num is not None:
            alignment_map[pr_num] = als.get("alignment_score", 0.0)

    # Extract PR numbers from clusters and record them
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id", "")
        for pr_info in cluster.get("prs", []):
            pr_num = pr_info.get("number")
            if pr_num is None:
                continue
            state.analyzed_prs[pr_num] = AnalyzedPR(
                timestamp=now,
                cluster_id=cluster_id,
                quality_score=quality_map.get(pr_num),
                alignment_score=alignment_map.get(pr_num),
            )

    return state
