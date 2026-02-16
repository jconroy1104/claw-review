"""Batch orchestration for processing large PR sets.

Splits PR lists into manageable batches with checkpointing
after each batch for crash resilience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .costs import CostTracker
from .github_client import PRData
from .state import AnalysisState, update_state, save_state


@dataclass
class BatchResult:
    """Results from processing a single batch of PRs."""

    batch_num: int
    prs_processed: list[int] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)
    quality_scores: list[dict] = field(default_factory=list)
    alignment_scores: list[dict] = field(default_factory=list)


class BatchProcessor:
    """Orchestrates processing of PRs in manageable batches.

    Splits a large PR list into smaller batches, tracks progress,
    and provides checkpointing after each batch for crash resilience.

    The BatchProcessor structures work but does not run the actual
    analysis pipeline — that is handled by the caller (e.g., cli.py).
    """

    def __init__(
        self,
        batch_size: int = 50,
        state: AnalysisState | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            batch_size: Number of PRs per batch
            state: Optional analysis state for checkpointing
            cost_tracker: Optional cost tracker for progress messages
        """
        self.batch_size = max(1, batch_size)
        self.state = state
        self.cost_tracker = cost_tracker

    def create_batches(self, prs: list[PRData]) -> list[list[PRData]]:
        """Split PRs into batches of batch_size.

        Args:
            prs: Full list of PRs to process

        Returns:
            List of PR batches (each batch is a list of PRData)
        """
        if not prs:
            return []
        return [
            prs[i : i + self.batch_size]
            for i in range(0, len(prs), self.batch_size)
        ]

    def process_batch(
        self,
        batch: list[PRData],
        batch_num: int,
        total_batches: int,
    ) -> BatchResult:
        """Create a BatchResult for a batch of PRs.

        This method structures the batch data but does NOT execute
        the actual analysis pipeline. The caller should fill in the
        clusters, quality_scores, and alignment_scores on the returned
        BatchResult after running the pipeline.

        Args:
            batch: List of PRs in this batch
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches

        Returns:
            A BatchResult pre-populated with batch_num and PR numbers
        """
        return BatchResult(
            batch_num=batch_num,
            prs_processed=[pr.number for pr in batch],
        )

    def checkpoint(
        self,
        batch_result: BatchResult,
        state_dir: str = ".claw-review-state",
    ) -> None:
        """Save partial results to state after a batch completes.

        Updates the analysis state with the batch results and writes
        the state to disk for crash resilience.

        Args:
            batch_result: Results from the completed batch
            state_dir: Directory where state files are stored
        """
        if self.state is None:
            return

        update_state(
            self.state,
            batch_result.clusters,
            batch_result.quality_scores,
            batch_result.alignment_scores,
        )
        save_state(self.state, state_dir=state_dir)

    def get_progress_message(
        self,
        batch_num: int,
        total_batches: int,
        batch: list[PRData],
    ) -> str:
        """Format a progress message for the current batch.

        Args:
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches
            batch: PRs in the current batch

        Returns:
            Formatted progress string like
            "Batch 3/12: PRs 101-150 (cost: $0.45)"
        """
        if batch:
            pr_numbers = sorted(pr.number for pr in batch)
            pr_range = f"PRs {pr_numbers[0]}-{pr_numbers[-1]}"
        else:
            pr_range = "PRs (none)"

        msg = f"Batch {batch_num}/{total_batches}: {pr_range}"

        if self.cost_tracker is not None:
            cost = self.cost_tracker.total_cost
            msg += f" (cost: ${cost:.2f})"

        return msg
