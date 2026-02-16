# CLAUDE.md — claw-review Project Context

## Project Overview

**claw-review** is an AI-powered PR triage tool that uses multi-model consensus
to help open-source maintainers manage high-volume pull request inflows.

It addresses three problems:
1. **De-duplication** — Cluster PRs that solve the same problem
2. **Quality ranking** — Identify the best implementation per cluster
3. **Vision alignment** — Flag PRs that drift from the project's stated direction

## Why This Exists

The creator of OpenClaw (145K+ GitHub stars) publicly asked for exactly this tool.
This is a PoC to demonstrate the concept works, built to share with him within 72 hours.

## Architecture

```
GitHub API → PR Data → OpenRouter (Multi-Model) → Analysis → Report
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                  Claude    GPT-4o   Gemini
                  (via OpenRouter.ai)
                    │         │         │
                    └────┬────┘─────────┘
                         ▼
                   Consensus Fusion
                    (vote/average)
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Clusters   Quality    Alignment
           (DBSCAN)   (scores)   (drift)
              │          │          │
              └──────────┼──────────┘
                         ▼
                   HTML + JSON Report
```

## Tech Stack

- **Language:** Python 3.11+
- **HTTP Client:** httpx (for both GitHub API and OpenRouter)
- **LLM Gateway:** OpenRouter.ai (single API key → Claude, GPT-4o, Gemini, Llama, Mistral, etc.)
- **Embeddings:** OpenAI text-embedding-3-small (via OpenRouter)
- **Clustering:** scikit-learn DBSCAN
- **CLI:** Click
- **Console UI:** Rich
- **Templates:** Jinja2
- **Config:** python-dotenv

## Key Design Decisions

### OpenRouter as Unified Gateway
All model calls go through OpenRouter.ai (OpenAI-compatible API). This means:
- One API key, one endpoint, any model
- No separate SDKs for Anthropic/OpenAI/Google
- Easy to add new models (just add the OpenRouter model ID to MODELS env var)
- httpx is the only HTTP dependency
- Models are configurable at runtime without code changes

### Multi-Model Consensus (NOT Single-Model Review)
Every evaluation is performed by ALL available models independently. Results are
fused using simple consensus mechanisms (majority vote, weighted average). This
is the core differentiator — not "GPT reviews your PR" but "3 models independently
evaluate and vote."

### Simple Consensus (Intentional)
The consensus logic uses basic averaging and majority voting. This is intentional
for the PoC. Do NOT implement complex fusion algorithms. Keep it simple:
- Category/area: majority vote
- Scores: arithmetic mean
- Disagreement: max spread > threshold → flag for human review

### Structured JSON Prompts
All model interactions use structured JSON output. System prompts define the exact
JSON schema. Response parsing must handle:
- Clean JSON
- JSON wrapped in markdown code fences (```json ... ```)
- JSON embedded in surrounding text
- Malformed JSON (graceful failure)

### Caching
GitHub API responses are cached locally in `.claw-review-cache/`. This prevents
redundant API calls during development and re-runs. Cache key is MD5 of repo:pr_number.

### Cost Awareness
The tool makes many API calls (3 models × N PRs × 3 analysis types). Design choices
that reduce cost:
- Only quality-score PRs in duplicate clusters (not singletons)
- Truncate diffs to 12K chars
- Use cheap embedding model (text-embedding-3-small)
- Cache everything possible

## File Structure

```
claw-review/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── src/
│   └── claw_review/
│       ├── __init__.py
│       ├── config.py           ← Configuration management
│       ├── github_client.py    ← GitHub API integration
│       ├── models.py           ← Multi-model interface
│       ├── clustering.py       ← Intent extraction + DBSCAN clustering
│       ├── scoring.py          ← Quality scoring (5 dimensions)
│       ├── alignment.py        ← Vision alignment scoring
│       ├── report.py           ← HTML + JSON report generation
│       └── cli.py              ← Click CLI entry point
└── tests/
    ├── __init__.py
    ├── conftest.py             ← Shared fixtures
    ├── test_config.py
    ├── test_github_client.py
    ├── test_models.py
    ├── test_clustering.py
    ├── test_scoring.py
    ├── test_alignment.py
    ├── test_report.py
    ├── test_cli.py
    └── fixtures/
        ├── sample_intents.json
        ├── sample_embeddings.json
        ├── sample_clusters.json
        ├── sample_scores.json
        └── sample_vision_docs.json
```

## Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -x -q

# Type checking
mypy src/claw_review/ --ignore-missing-imports

# Linting
ruff check src/ tests/

# CLI
claw-review check                                  # Verify config
claw-review analyze --repo owner/name --max-prs 20 # Run analysis
claw-review regenerate report.json                  # Regenerate HTML from JSON
```

## Testing Rules

1. **Zero real API calls in tests.** All model and GitHub API calls must be mocked.
2. **Each module's tests pass independently.** No cross-module test dependencies.
3. **Use the shared fixtures** from conftest.py for PRData objects.
4. **Test edge cases:** empty inputs, single items, partial failures, malformed responses.
5. **Minimum coverage:** 80% per module.

## Sprint 1 Agent Team

See AGENTS.md for full agent definitions, file ownership, and quality gates.

| Agent | Owns | Phase |
|-------|------|-------|
| Lead | cli.py, pyproject.toml, integration tests | Coordination |
| @data-layer | config.py, github_client.py, conftest.py | Phase 1 |
| @consensus-engine | models.py, clustering.py | Phase 1 |
| @scoring-reports | scoring.py, alignment.py, report.py | Phase 2 |

## IP Notice

This project demonstrates general multi-model consensus concepts using standard
techniques (majority voting, weighted averaging, DBSCAN clustering). It does NOT
implement any proprietary fusion algorithms. All consensus logic should remain
intentionally simple.


# Sprint 2 Context — Async, Cost Optimization & Scale

## Sprint 1 Results (Live Run)

- **Repo:** openclaw/openclaw (193K stars, 6,000+ open PRs)
- **Sample:** 100 PRs analyzed
- **Findings:** 3 duplicate clusters, 7 duplicate PRs, 0 vision drift flags
- **Cost:** $1.83 total (GPT-4o: $1.71 / Gemini Flash: $0.10 / Claude Sonnet: $0.02)
- **Runtime:** ~20 minutes (sequential API calls)
- **Report:** Published at jconroy1104.github.io/claw-review/claw-review-report.html

## Sprint 2 Goals

1. **Async parallel API calls** — 3-5x faster runtime
2. **Model presets** — 10-50x cost reduction on cheaper models
3. **Incremental analysis** — only process new/changed PRs
4. **Batch processing** — handle 6,000+ PRs with checkpointing

## Async Patterns

### httpx.AsyncClient
```python
# CORRECT: Session-level async client with connection pooling
async with httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10, read=60, pool=5),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    headers=self._headers,
) as client:
    # Client lives for entire analysis run
    ...
```

### Concurrent Model Queries
```python
# CORRECT: All models fire concurrently for each PR
async def query_all(self, system_prompt, user_prompt, ...):
    tasks = [
        self.query_single(model, system_prompt, user_prompt)
        for model in self.models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

### Semaphore for Concurrency Control
```python
# CORRECT: Limit total concurrent requests
semaphore = asyncio.Semaphore(concurrency_limit)

async def _rate_limited_query(self, model, system_prompt, user_prompt):
    async with semaphore:
        return await self.query_single(model, system_prompt, user_prompt)
```

### Batch PR Processing
```python
# CORRECT: Process N PRs concurrently, each with 3 concurrent model calls
async def process_batch(prs, model_pool, batch_size=5):
    for i in range(0, len(prs), batch_size):
        batch = prs[i:i + batch_size]
        results = await asyncio.gather(*[
            process_single_pr(pr, model_pool) for pr in batch
        ])
        yield results  # Checkpoint after each batch
```

## Cost Data (OpenRouter, February 2026)

Per 1M tokens (approximate):

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| anthropic/claude-sonnet-4 | $3.00 | $15.00 | Best reasoning |
| openai/gpt-4o | $2.50 | $10.00 | 93% of Sprint 1 cost |
| openai/gpt-4o-mini | $0.15 | $0.60 | 17x cheaper than GPT-4o |
| google/gemini-2.0-flash-001 | $0.10 | $0.40 | Cheapest name-brand |
| meta-llama/llama-3.1-70b-instruct | $0.50 | $0.50 | Best open-source |
| mistralai/mistral-large-latest | $2.00 | $6.00 | Strong European model |

**Key insight:** Replacing GPT-4o ($1.71) with GPT-4o-mini (~$0.10) in the
"balanced" preset saves 94% of the biggest cost line while maintaining
Claude Sonnet for quality reasoning.

## Testing Async Code

All async tests use pytest-asyncio:

```python
import pytest

@pytest.mark.asyncio
async def test_query_all_concurrent():
    """Verify all models are queried concurrently, not sequentially."""
    ...
```

Mock async httpx responses:
```python
import httpx
from pytest_httpx import HTTPXMock

@pytest.mark.asyncio
async def test_with_mock(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"choices": [...]})
    ...
```

## File Structure After Sprint 2

```
src/claw_review/
├── __init__.py
├── config.py            ← Updated: model presets
├── costs.py             ← NEW: cost tracking and estimation
├── state.py             ← NEW: incremental analysis state
├── batch.py             ← NEW: batch orchestration
├── github_client.py
├── models.py            ← Updated: async httpx
├── clustering.py        ← Updated: async pipeline
├── scoring.py           ← Updated: async pipeline
├── alignment.py         ← Updated: async pipeline
├── report.py            ← Updated: merge capability, cost summary
└── cli.py               ← Updated: new commands and flags
```

## New CLI Commands (Sprint 2)

```bash
# List presets
claw-review presets

# Estimate cost before running
claw-review estimate --repo openclaw/openclaw --preset balanced

# Run with preset (10x cheaper than Sprint 1)
claw-review analyze --repo openclaw/openclaw --preset balanced

# Run with budget limit
claw-review analyze --repo openclaw/openclaw --budget 5.00

# Batch processing for large repos
claw-review analyze --repo openclaw/openclaw --batch-size 50

# Incremental (skip known PRs — default behavior)
claw-review analyze --repo openclaw/openclaw --incremental

# Force re-analysis
claw-review analyze --repo openclaw/openclaw --force

# Merge multiple batch results
claw-review merge batch1.json batch2.json -o full-report.html

# Check analysis state
claw-review status --repo openclaw/openclaw
```

## IP Notice (Unchanged)

This project demonstrates general multi-model consensus concepts using standard
techniques (majority voting, weighted averaging, DBSCAN clustering). It does NOT
implement any proprietary fusion algorithms. All consensus logic should remain
intentionally simple.
