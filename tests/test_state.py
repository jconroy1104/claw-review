"""Tests for claw_review.state — Analysis state management."""

from __future__ import annotations

import json
from pathlib import Path

from claw_review.github_client import PRData
from claw_review.state import (
    AnalysisState,
    AnalyzedPR,
    _repo_hash,
    _state_file_path,
    get_unanalyzed_prs,
    load_state,
    save_state,
    update_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pr(**overrides: object) -> PRData:
    """Build a minimal PRData for testing."""
    defaults: dict = {
        "number": 42,
        "title": "Fix memory leak in parser",
        "body": "Fixes a memory leak.",
        "author": "contributor123",
        "created_at": "2025-06-01T10:00:00Z",
        "updated_at": "2025-06-02T12:00:00Z",
        "state": "open",
        "labels": ["bug"],
        "files_changed": ["src/parser.py"],
        "additions": 25,
        "deletions": 10,
        "diff_summary": "--- src/parser.py\n+handle.close()",
        "url": "https://github.com/owner/repo/pull/42",
        "comments_count": 3,
    }
    defaults.update(overrides)
    return PRData(**defaults)


def _sample_clusters() -> list[dict]:
    """Build sample cluster dicts."""
    return [
        {
            "cluster_id": "cluster-0",
            "intent_summary": "Fix WebSocket reconnection",
            "category": "bugfix",
            "affected_area": "gateway",
            "confidence": 0.91,
            "prs": [
                {"number": 101, "title": "Fix WS reconnect", "author": "alice", "url": "u1"},
                {"number": 102, "title": "Fix WS timeout", "author": "bob", "url": "u2"},
            ],
        },
        {
            "cluster_id": "singleton-200",
            "intent_summary": "Add dark mode",
            "category": "feature",
            "affected_area": "UI",
            "confidence": 0.0,
            "prs": [
                {"number": 200, "title": "Add dark mode", "author": "charlie", "url": "u3"},
            ],
        },
    ]


def _sample_quality_scores() -> list[dict]:
    """Build sample quality score dicts."""
    return [
        {"pr_number": 101, "overall_score": 8.2},
        {"pr_number": 102, "overall_score": 6.5},
    ]


def _sample_alignment_scores() -> list[dict]:
    """Build sample alignment score dicts."""
    return [
        {"pr_number": 101, "alignment_score": 9.0},
        {"pr_number": 200, "alignment_score": 3.5},
    ]


# ===================================================================
# AnalysisState basics
# ===================================================================


class TestAnalysisState:
    """Tests for the AnalysisState dataclass."""

    def test_new_state_has_empty_analyzed_prs(self) -> None:
        state = AnalysisState(repo="owner/repo")
        assert state.analyzed_prs == {}
        assert state.last_run_timestamp == ""
        assert state.model_config == []

    def test_round_trip_serialization(self) -> None:
        state = AnalysisState(
            repo="owner/repo",
            analyzed_prs={
                42: AnalyzedPR(
                    timestamp="2025-06-01T00:00:00Z",
                    cluster_id="cluster-0",
                    quality_score=8.0,
                    alignment_score=7.5,
                )
            },
            last_run_timestamp="2025-06-01T00:00:00Z",
            model_config=["model/a", "model/b"],
        )
        data = state.to_dict()
        restored = AnalysisState.from_dict(data)

        assert restored.repo == "owner/repo"
        assert 42 in restored.analyzed_prs
        assert restored.analyzed_prs[42].quality_score == 8.0
        assert restored.analyzed_prs[42].alignment_score == 7.5
        assert restored.analyzed_prs[42].cluster_id == "cluster-0"
        assert restored.last_run_timestamp == "2025-06-01T00:00:00Z"
        assert restored.model_config == ["model/a", "model/b"]


# ===================================================================
# load_state
# ===================================================================


class TestLoadState:
    """Tests for load_state."""

    def test_creates_new_state_for_fresh_repo(self, tmp_path: Path) -> None:
        state = load_state("owner/fresh-repo", state_dir=str(tmp_path))
        assert state.repo == "owner/fresh-repo"
        assert state.analyzed_prs == {}

    def test_loads_existing_state(self, tmp_path: Path) -> None:
        # Pre-create a state file
        repo = "owner/existing"
        state_data = {
            "repo": repo,
            "analyzed_prs": {
                "10": {
                    "timestamp": "2025-01-01T00:00:00Z",
                    "cluster_id": "c-0",
                    "quality_score": 7.0,
                    "alignment_score": None,
                }
            },
            "last_run_timestamp": "2025-01-01T00:00:00Z",
            "model_config": ["model/x"],
        }
        state_file = tmp_path / f"{_repo_hash(repo)}.json"
        state_file.write_text(json.dumps(state_data))

        state = load_state(repo, state_dir=str(tmp_path))
        assert state.repo == repo
        assert 10 in state.analyzed_prs
        assert state.analyzed_prs[10].quality_score == 7.0


# ===================================================================
# save_state
# ===================================================================


class TestSaveState:
    """Tests for save_state."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        state = AnalysisState(
            repo="owner/repo",
            analyzed_prs={
                1: AnalyzedPR(timestamp="t1", cluster_id="c-0", quality_score=5.0),
                2: AnalyzedPR(timestamp="t2", cluster_id="c-1"),
            },
            last_run_timestamp="t2",
            model_config=["m1", "m2"],
        )
        save_state(state, state_dir=str(tmp_path))

        loaded = load_state("owner/repo", state_dir=str(tmp_path))
        assert loaded.repo == "owner/repo"
        assert len(loaded.analyzed_prs) == 2
        assert loaded.analyzed_prs[1].quality_score == 5.0
        assert loaded.analyzed_prs[2].cluster_id == "c-1"

    def test_atomic_write_creates_state_dir(self, tmp_path: Path) -> None:
        state_dir = str(tmp_path / "nested" / "state")
        state = AnalysisState(repo="owner/repo")
        save_state(state, state_dir=state_dir)

        assert Path(state_dir).exists()
        loaded = load_state("owner/repo", state_dir=state_dir)
        assert loaded.repo == "owner/repo"

    def test_atomic_write_no_temp_files_left(self, tmp_path: Path) -> None:
        state = AnalysisState(repo="owner/repo")
        save_state(state, state_dir=str(tmp_path))

        # Only the state file should exist, no .tmp files
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".json"

    def test_overwrite_existing_state(self, tmp_path: Path) -> None:
        state1 = AnalysisState(
            repo="owner/repo",
            analyzed_prs={1: AnalyzedPR(timestamp="t1")},
        )
        save_state(state1, state_dir=str(tmp_path))

        state2 = AnalysisState(
            repo="owner/repo",
            analyzed_prs={
                1: AnalyzedPR(timestamp="t1"),
                2: AnalyzedPR(timestamp="t2"),
            },
        )
        save_state(state2, state_dir=str(tmp_path))

        loaded = load_state("owner/repo", state_dir=str(tmp_path))
        assert len(loaded.analyzed_prs) == 2


# ===================================================================
# State file path uses repo hash
# ===================================================================


class TestStateFilePath:
    """Tests for state file path generation."""

    def test_path_uses_repo_hash(self) -> None:
        path = _state_file_path("owner/repo", "/tmp/state")
        expected_hash = _repo_hash("owner/repo")
        assert path.name == f"{expected_hash}.json"
        assert str(path).startswith("/tmp/state/")

    def test_different_repos_different_hashes(self) -> None:
        hash1 = _repo_hash("owner/repo-a")
        hash2 = _repo_hash("owner/repo-b")
        assert hash1 != hash2


# ===================================================================
# get_unanalyzed_prs
# ===================================================================


class TestGetUnanalyzedPrs:
    """Tests for get_unanalyzed_prs."""

    def test_all_new_prs(self) -> None:
        prs = [_make_pr(number=1), _make_pr(number=2), _make_pr(number=3)]
        state = AnalysisState(repo="owner/repo")

        result = get_unanalyzed_prs(prs, state)
        assert len(result) == 3
        assert {pr.number for pr in result} == {1, 2, 3}

    def test_some_already_analyzed(self) -> None:
        prs = [_make_pr(number=1), _make_pr(number=2), _make_pr(number=3)]
        state = AnalysisState(
            repo="owner/repo",
            analyzed_prs={
                1: AnalyzedPR(timestamp="t1"),
                3: AnalyzedPR(timestamp="t2"),
            },
        )

        result = get_unanalyzed_prs(prs, state)
        assert len(result) == 1
        assert result[0].number == 2

    def test_none_new_all_analyzed(self) -> None:
        prs = [_make_pr(number=1), _make_pr(number=2)]
        state = AnalysisState(
            repo="owner/repo",
            analyzed_prs={
                1: AnalyzedPR(timestamp="t1"),
                2: AnalyzedPR(timestamp="t2"),
            },
        )

        result = get_unanalyzed_prs(prs, state)
        assert result == []

    def test_empty_pr_list(self) -> None:
        state = AnalysisState(
            repo="owner/repo",
            analyzed_prs={1: AnalyzedPR(timestamp="t1")},
        )
        result = get_unanalyzed_prs([], state)
        assert result == []


# ===================================================================
# update_state
# ===================================================================


class TestUpdateState:
    """Tests for update_state."""

    def test_merges_new_results(self) -> None:
        state = AnalysisState(repo="owner/repo")
        clusters = _sample_clusters()
        quality = _sample_quality_scores()
        alignment = _sample_alignment_scores()

        updated = update_state(state, clusters, quality, alignment)

        assert updated is state  # Modified in place
        assert len(updated.analyzed_prs) == 3  # PRs 101, 102, 200
        assert 101 in updated.analyzed_prs
        assert 102 in updated.analyzed_prs
        assert 200 in updated.analyzed_prs

        # Quality scores mapped correctly
        assert updated.analyzed_prs[101].quality_score == 8.2
        assert updated.analyzed_prs[102].quality_score == 6.5
        assert updated.analyzed_prs[200].quality_score is None

        # Alignment scores mapped correctly
        assert updated.analyzed_prs[101].alignment_score == 9.0
        assert updated.analyzed_prs[200].alignment_score == 3.5
        assert updated.analyzed_prs[102].alignment_score is None

        # Cluster IDs assigned
        assert updated.analyzed_prs[101].cluster_id == "cluster-0"
        assert updated.analyzed_prs[200].cluster_id == "singleton-200"

    def test_updates_last_run_timestamp(self) -> None:
        state = AnalysisState(repo="owner/repo")
        assert state.last_run_timestamp == ""

        update_state(state, _sample_clusters(), [], [])
        assert state.last_run_timestamp != ""
        assert "T" in state.last_run_timestamp  # ISO format

    def test_preserves_existing_entries(self) -> None:
        state = AnalysisState(
            repo="owner/repo",
            analyzed_prs={
                50: AnalyzedPR(timestamp="old", cluster_id="c-old", quality_score=9.0),
            },
        )

        update_state(state, _sample_clusters(), [], [])

        # Old entry preserved
        assert 50 in state.analyzed_prs
        assert state.analyzed_prs[50].quality_score == 9.0
        # New entries added
        assert 101 in state.analyzed_prs
        assert 102 in state.analyzed_prs
        assert 200 in state.analyzed_prs

    def test_empty_inputs(self) -> None:
        state = AnalysisState(repo="owner/repo")
        update_state(state, [], [], [])
        assert len(state.analyzed_prs) == 0
        assert state.last_run_timestamp != ""
