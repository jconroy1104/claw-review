"""Cost tracking and estimation for claw-review.

Token prices sourced from OpenRouter pricing (February 2026).
Uses per-1M token pricing for accurate cost calculation.
"""

from dataclasses import dataclass, field
from rich.table import Table
from rich.console import Console

console = Console()

# Prices per 1M tokens (input, output) — sourced from OpenRouter, Feb 2026
TOKEN_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "meta-llama/llama-3.1-70b-instruct": (0.50, 0.50),
    "mistralai/mistral-large-latest": (2.00, 6.00),
}

# Fallback for unknown models (conservative estimate)
DEFAULT_PRICE = (2.00, 8.00)

# Average tokens per PR analysis (estimated from Sprint 1 data)
AVG_INPUT_TOKENS_PER_PR = 2500
AVG_OUTPUT_TOKENS_PER_PR = 500


@dataclass
class ModelUsage:
    """Token usage for a single model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0

    @property
    def cost(self) -> float:
        input_price, output_price = TOKEN_PRICES.get(self.model, DEFAULT_PRICE)
        return (self.input_tokens * input_price / 1_000_000) + (self.output_tokens * output_price / 1_000_000)


@dataclass
class CostTracker:
    """Tracks per-model token usage and cost during analysis."""

    usage_by_model: dict[str, ModelUsage] = field(default_factory=dict)
    budget_limit: float | None = None

    @property
    def total_cost(self) -> float:
        return sum(u.cost for u in self.usage_by_model.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usage_by_model.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usage_by_model.values())

    @property
    def total_requests(self) -> int:
        return sum(u.request_count for u in self.usage_by_model.values())

    @property
    def budget_exceeded(self) -> bool:
        if self.budget_limit is None:
            return False
        return self.total_cost >= self.budget_limit

    @property
    def budget_remaining(self) -> float | None:
        if self.budget_limit is None:
            return None
        return max(0.0, self.budget_limit - self.total_cost)

    def record_usage(self, model: str, usage: dict | None) -> None:
        """Record token usage from an OpenRouter API response.

        Args:
            model: OpenRouter model ID
            usage: Usage dict from API response (has prompt_tokens, completion_tokens)
        """
        if usage is None:
            return
        if model not in self.usage_by_model:
            self.usage_by_model[model] = ModelUsage(model=model)
        mu = self.usage_by_model[model]
        mu.input_tokens += usage.get("prompt_tokens", 0)
        mu.output_tokens += usage.get("completion_tokens", 0)
        mu.request_count += 1

    def format_report(self) -> Table:
        """Generate a Rich table showing cost breakdown."""
        table = Table(title="Cost Summary", border_style="cyan")
        table.add_column("Model", style="bold")
        table.add_column("Requests", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Cost", justify="right", style="green")

        for mu in sorted(self.usage_by_model.values(), key=lambda u: -u.cost):
            table.add_row(
                mu.model.split("/")[-1],
                str(mu.request_count),
                f"{mu.input_tokens:,}",
                f"{mu.output_tokens:,}",
                f"${mu.cost:.4f}",
            )

        table.add_section()
        table.add_row(
            "TOTAL",
            str(self.total_requests),
            f"{self.total_input_tokens:,}",
            f"{self.total_output_tokens:,}",
            f"${self.total_cost:.4f}",
        )

        if self.budget_limit is not None:
            table.add_row(
                "Budget",
                "", "", "",
                f"${self.budget_limit:.2f}",
            )
            table.add_row(
                "Remaining",
                "", "", "",
                f"${self.budget_remaining:.2f}" if self.budget_remaining else "$0.00",
            )

        return table

    def to_dict(self) -> dict:
        """Serialize to dict for inclusion in JSON reports."""
        return {
            "total_cost": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "budget_limit": self.budget_limit,
            "models": {
                model: {
                    "input_tokens": mu.input_tokens,
                    "output_tokens": mu.output_tokens,
                    "request_count": mu.request_count,
                    "cost": round(mu.cost, 6),
                }
                for model, mu in self.usage_by_model.items()
            },
        }


def estimate_cost(
    num_prs: int,
    models: list[str],
    analysis_types: int = 3,  # intent + quality + alignment
) -> dict:
    """Estimate cost for analyzing a given number of PRs.

    Args:
        num_prs: Number of PRs to analyze
        models: List of model IDs to use
        analysis_types: Number of analysis passes (default 3)

    Returns:
        Dict with per-model and total cost estimates
    """
    estimates = {}
    total_low = 0.0
    total_high = 0.0

    for model in models:
        input_price, output_price = TOKEN_PRICES.get(model, DEFAULT_PRICE)

        # Each PR gets analyzed by each model for each analysis type
        total_input = num_prs * analysis_types * AVG_INPUT_TOKENS_PER_PR
        total_output = num_prs * analysis_types * AVG_OUTPUT_TOKENS_PER_PR

        cost = (total_input * input_price / 1_000_000) + (total_output * output_price / 1_000_000)

        # Add 20% variance for high estimate
        estimates[model] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_low": round(cost * 0.8, 4),
            "cost_high": round(cost * 1.2, 4),
        }
        total_low += cost * 0.8
        total_high += cost * 1.2

    return {
        "num_prs": num_prs,
        "models": estimates,
        "total_cost_low": round(total_low, 4),
        "total_cost_high": round(total_high, 4),
        "analysis_types": analysis_types,
    }


def get_model_price(model: str) -> tuple[float, float]:
    """Get the price per 1M tokens (input, output) for a model."""
    return TOKEN_PRICES.get(model, DEFAULT_PRICE)
