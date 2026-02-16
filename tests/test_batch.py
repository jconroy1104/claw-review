"""Tests for claw_review.batch — Batch orchestration."""

from __future__ import annotations

from pathlib import Path

from claw_review.batch import BatchProcessor, BatchResult
from claw_review.costs import CostTracker
from claw_review.github_client import PRData
from claw_review.state import AnalysisState, load_state


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


def _make_prs(count: int, start: int = 1) -> list[PRData]:
    """Build a list of PRs with sequential numbers."""
    return [_make_pr(number=i) for i in range(start, start + count)]


# ===================================================================
# BatchResult dataclass
# ===================================================================


class TestBatchResult:
    """Tests for the BatchResult dataclass."""

    def test_default_values(self) -> None:
        result = BatchResult(batch_num=1)
        assert result.batch_num == 1
        assert result.prs_processed == []
        assert result.clusters == []
        assert result.quality_scores == []
        assert result.alignment_scores == []

    def test_populated_result(self) -> None:
        result = BatchResult(
            batch_num=3,
            prs_processed=[10, 20, 30],
            clusters=[{"cluster_id": "c-0", "prs": [{"number": 10}]}],
            quality_scores=[{"pr_number": 10, "overall_score": 8.0}],
            alignment_scores=[{"pr_number": 10, "alignment_score": 7.0}],
        )
        assert result.batch_num == 3
        assert len(result.prs_processed) == 3
        assert len(result.clusters) == 1
        assert len(result.quality_scores) == 1
        assert len(result.alignment_scores) == 1


# ===================================================================
# create_batches
# ===================================================================


class TestCreateBatches:
    """Tests for BatchProcessor.create_batches."""

    def test_splits_correctly(self) -> None:
        prs = _make_prs(10)
        bp = BatchProcessor(batch_size=3)
        batches = bp.create_batches(prs)

        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_fewer_prs_than_batch_size(self) -> None:
        prs = _make_prs(3)
        bp = BatchProcessor(batch_size=50)
        batches = bp.create_batches(prs)

        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_exact_multiple_of_batch_size(self) -> None:
        prs = _make_prs(9)
        bp = BatchProcessor(batch_size=3)
        batches = bp.create_batches(prs)

        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)

    def test_empty_list(self) -> None:
        bp = BatchProcessor(batch_size=50)
        batches = bp.create_batches([])
        assert batches == []

    def test_batch_size_one(self) -> None:
        prs = _make_prs(3)
        bp = BatchProcessor(batch_size=1)
        batches = bp.create_batches(prs)

        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_batch_size_clamps_to_minimum_one(self) -> None:
        bp = BatchProcessor(batch_size=0)
        assert bp.batch_size == 1

        bp2 = BatchProcessor(batch_size=-5)
        assert bp2.batch_size == 1


# ===================================================================
# process_batch
# ===================================================================


class TestProcessBatch:
    """Tests for BatchProcessor.process_batch."""

    def test_returns_batch_result_with_pr_numbers(self) -> None:
        prs = _make_prs(5, start=100)
        bp = BatchProcessor(batch_size=5)
        result = bp.process_batch(prs, batch_num=1, total_batches=1)

        assert isinstance(result, BatchResult)
        assert result.batch_num == 1
        assert result.prs_processed == [100, 101, 102, 103, 104]

    def test_empty_batch(self) -> None:
        bp = BatchProcessor()
        result = bp.process_batch([], batch_num=1, total_batches=1)
        assert result.prs_processed == []


# ===================================================================
# get_progress_message
# ===================================================================


class TestGetProgressMessage:
    """Tests for BatchProcessor.get_progress_message."""

    def test_basic_format(self) -> None:
        bp = BatchProcessor(batch_size=50)
        batch = _make_prs(50, start=101)
        msg = bp.get_progress_message(3, 12, batch)
        assert "Batch 3/12" in msg
        assert "PRs 101-150" in msg

    def test_with_cost_tracker(self) -> None:
        tracker = CostTracker()
        tracker.record_usage("openai/gpt-4o", {"prompt_tokens": 10000, "completion_tokens": 2000})
        bp = BatchProcessor(batch_size=50, cost_tracker=tracker)
        batch = _make_prs(50, start=1)
        msg = bp.get_progress_message(1, 1, batch)
        assert "cost: $" in msg

    def test_single_pr_batch(self) -> None:
        bp = BatchProcessor()
        batch = [_make_pr(number=42)]
        msg = bp.get_progress_message(1, 1, batch)
        assert "PRs 42-42" in msg

    def test_empty_batch(self) -> None:
        bp = BatchProcessor()
        msg = bp.get_progress_message(1, 1, [])
        assert "PRs (none)" in msg


# ===================================================================
# checkpoint
# ===================================================================


class TestCheckpoint:
    """Tests for BatchProcessor.checkpoint."""

    def test_saves_to_state(self, tmp_path: Path) -> None:
        state = AnalysisState(repo="owner/repo")
        bp = BatchProcessor(state=state)

        batch_result = BatchResult(
            batch_num=1,
            prs_processed=[101, 102],
            clusters=[
                {
                    "cluster_id": "cluster-0",
                    "prs": [
                        {"number": 101, "title": "PR 101"},
                        {"number": 102, "title": "PR 102"},
                    ],
                }
            ],
            quality_scores=[{"pr_number": 101, "overall_score": 7.5}],
            alignment_scores=[],
        )

        bp.checkpoint(batch_result, state_dir=str(tmp_path))

        # State should be updated
        assert 101 in state.analyzed_prs
        assert 102 in state.analyzed_prs
        assert state.analyzed_prs[101].quality_score == 7.5

        # State should be persisted to disk
        loaded = load_state("owner/repo", state_dir=str(tmp_path))
        assert 101 in loaded.analyzed_prs
        assert 102 in loaded.analyzed_prs

    def test_checkpoint_without_state_is_noop(self, tmp_path: Path) -> None:
        bp = BatchProcessor(state=None)
        batch_result = BatchResult(batch_num=1, prs_processed=[1])
        # Should not raise
        bp.checkpoint(batch_result, state_dir=str(tmp_path))

    def test_multiple_checkpoints_accumulate(self, tmp_path: Path) -> None:
        state = AnalysisState(repo="owner/repo")
        bp = BatchProcessor(state=state)

        result1 = BatchResult(
            batch_num=1,
            prs_processed=[1],
            clusters=[{"cluster_id": "c-0", "prs": [{"number": 1}]}],
        )
        bp.checkpoint(result1, state_dir=str(tmp_path))

        result2 = BatchResult(
            batch_num=2,
            prs_processed=[2],
            clusters=[{"cluster_id": "c-1", "prs": [{"number": 2}]}],
        )
        bp.checkpoint(result2, state_dir=str(tmp_path))

        assert len(state.analyzed_prs) == 2
        loaded = load_state("owner/repo", state_dir=str(tmp_path))
        assert len(loaded.analyzed_prs) == 2


# ===================================================================
# Custom batch_size
# ===================================================================


class TestCustomBatchSize:
    """Tests for BatchProcessor with custom batch sizes."""

    def test_default_batch_size(self) -> None:
        bp = BatchProcessor()
        assert bp.batch_size == 50

    def test_custom_batch_size(self) -> None:
        bp = BatchProcessor(batch_size=25)
        assert bp.batch_size == 25

    def test_large_batch_size(self) -> None:
        bp = BatchProcessor(batch_size=1000)
        prs = _make_prs(100)
        batches = bp.create_batches(prs)
        assert len(batches) == 1
        assert len(batches[0]) == 100
