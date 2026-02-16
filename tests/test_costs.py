"""Tests for claw_review.costs module."""

from __future__ import annotations

from rich.table import Table

from claw_review.costs import (
    CostTracker,
    ModelUsage,
    DEFAULT_PRICE,
    estimate_cost,
    get_model_price,
)


class TestModelUsage:
    """Tests for ModelUsage dataclass."""

    def test_model_usage_cost_calculation(self) -> None:
        mu = ModelUsage(
            model="openai/gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # gpt-4o: input $2.50/1M, output $10.00/1M
        assert mu.cost == 2.50 + 10.00

    def test_model_usage_known_model(self) -> None:
        mu = ModelUsage(
            model="google/gemini-2.0-flash-001",
            input_tokens=500_000,
            output_tokens=200_000,
        )
        # gemini flash: input $0.10/1M, output $0.40/1M
        expected = (500_000 * 0.10 / 1_000_000) + (200_000 * 0.40 / 1_000_000)
        assert abs(mu.cost - expected) < 1e-9

    def test_model_usage_unknown_model_uses_fallback(self) -> None:
        mu = ModelUsage(
            model="unknown/model-xyz",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # Fallback: input $2.00/1M, output $8.00/1M
        assert mu.cost == 2.00 + 8.00

    def test_model_usage_zero_tokens(self) -> None:
        mu = ModelUsage(model="openai/gpt-4o")
        assert mu.cost == 0.0


class TestCostTracker:
    """Tests for CostTracker dataclass."""

    def test_cost_tracker_record_usage(self) -> None:
        tracker = CostTracker()
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        })
        assert "openai/gpt-4o" in tracker.usage_by_model
        mu = tracker.usage_by_model["openai/gpt-4o"]
        assert mu.input_tokens == 100
        assert mu.output_tokens == 50
        assert mu.request_count == 1

    def test_cost_tracker_total_cost(self) -> None:
        tracker = CostTracker()
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        })
        assert tracker.total_cost == 2.50 + 10.00

    def test_cost_tracker_multiple_models(self) -> None:
        tracker = CostTracker()
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        tracker.record_usage("google/gemini-2.0-flash-001", {
            "prompt_tokens": 2000,
            "completion_tokens": 800,
        })
        assert len(tracker.usage_by_model) == 2
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1300
        assert tracker.total_requests == 2

    def test_cost_tracker_accumulates_same_model(self) -> None:
        tracker = CostTracker()
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        })
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 200,
            "completion_tokens": 100,
        })
        mu = tracker.usage_by_model["openai/gpt-4o"]
        assert mu.input_tokens == 300
        assert mu.output_tokens == 150
        assert mu.request_count == 2

    def test_cost_tracker_budget_not_exceeded(self) -> None:
        tracker = CostTracker(budget_limit=10.0)
        tracker.record_usage("google/gemini-2.0-flash-001", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        assert not tracker.budget_exceeded

    def test_cost_tracker_budget_exceeded(self) -> None:
        tracker = CostTracker(budget_limit=0.001)
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        })
        assert tracker.budget_exceeded

    def test_cost_tracker_budget_remaining(self) -> None:
        tracker = CostTracker(budget_limit=5.0)
        tracker.record_usage("google/gemini-2.0-flash-001", {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        })
        # gemini flash cost: 0.10 + 0.40 = 0.50
        remaining = tracker.budget_remaining
        assert remaining is not None
        assert abs(remaining - 4.50) < 1e-9

    def test_cost_tracker_no_budget(self) -> None:
        tracker = CostTracker()
        assert tracker.budget_limit is None
        assert not tracker.budget_exceeded
        assert tracker.budget_remaining is None

    def test_cost_tracker_to_dict(self) -> None:
        tracker = CostTracker(budget_limit=5.0)
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        d = tracker.to_dict()
        assert "total_cost" in d
        assert "total_input_tokens" in d
        assert "total_output_tokens" in d
        assert "total_requests" in d
        assert "budget_limit" in d
        assert d["budget_limit"] == 5.0
        assert "models" in d
        assert "openai/gpt-4o" in d["models"]
        model_data = d["models"]["openai/gpt-4o"]
        assert model_data["input_tokens"] == 1000
        assert model_data["output_tokens"] == 500
        assert model_data["request_count"] == 1
        assert "cost" in model_data

    def test_cost_tracker_format_report(self) -> None:
        tracker = CostTracker(budget_limit=10.0)
        tracker.record_usage("openai/gpt-4o", {
            "prompt_tokens": 500,
            "completion_tokens": 200,
        })
        result = tracker.format_report()
        assert isinstance(result, Table)

    def test_record_usage_none_usage(self) -> None:
        tracker = CostTracker()
        tracker.record_usage("openai/gpt-4o", None)
        assert len(tracker.usage_by_model) == 0
        assert tracker.total_cost == 0.0


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_estimate_cost_basic(self) -> None:
        result = estimate_cost(10, ["google/gemini-2.0-flash-001"])
        assert result["num_prs"] == 10
        assert result["analysis_types"] == 3
        assert "google/gemini-2.0-flash-001" in result["models"]
        assert result["total_cost_low"] > 0
        assert result["total_cost_high"] > result["total_cost_low"]

    def test_estimate_cost_with_multiple_models(self) -> None:
        models = ["openai/gpt-4o", "google/gemini-2.0-flash-001"]
        result = estimate_cost(50, models)
        assert len(result["models"]) == 2
        assert result["total_cost_high"] > result["total_cost_low"]
        # gpt-4o should be more expensive than gemini flash
        gpt_high = result["models"]["openai/gpt-4o"]["cost_high"]
        gemini_high = result["models"]["google/gemini-2.0-flash-001"]["cost_high"]
        assert gpt_high > gemini_high

    def test_estimate_cost_unknown_model(self) -> None:
        result = estimate_cost(10, ["unknown/model-xyz"])
        assert "unknown/model-xyz" in result["models"]
        assert result["total_cost_low"] > 0


class TestGetModelPrice:
    """Tests for get_model_price function."""

    def test_get_model_price_known(self) -> None:
        price = get_model_price("openai/gpt-4o")
        assert price == (2.50, 10.00)

    def test_get_model_price_unknown(self) -> None:
        price = get_model_price("unknown/model")
        assert price == DEFAULT_PRICE
