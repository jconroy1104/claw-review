# AGENTS.md — claw-review Sprint 1 Agent Team

## Sprint 1 Mission

Build a working CLI tool that scans a GitHub repository's open PRs and produces
a multi-model consensus report with duplicate clustering, quality ranking, and
vision alignment scoring. Target: functional end-to-end pipeline with tests.

**Sprint Duration:** 48–72 hours (rapid PoC sprint)
**Target Output:** `claw-review analyze --repo openclaw/openclaw` produces HTML + JSON report

---

## Team Structure

| Agent | Role | Phase | Test Target |
|-------|------|-------|-------------|
| @data-layer | GitHub API client, config, caching, data models | Phase 1 (parallel) | 30+ |
| @consensus-engine | Multi-model interface, intent extraction, embeddings, DBSCAN clustering | Phase 1 (parallel) | 35+ |
| @scoring-reports | Quality scoring, vision alignment, HTML/JSON report generation | Phase 2 (sequential) | 30+ |

**Lead** coordinates architecture, defines shared interfaces, runs integration
tests, and owns the CLI entry point.

---

## Dependency Graph

```
  @data-layer          @consensus-engine
   (Phase 1)             (Phase 1)
        \                    /
         \                  /
          \                /
       ──→ BOTH COMPLETE ←──
                 |
                 ▼
         @scoring-reports
            (Phase 2)
```

**Phase 1 (parallel):** @data-layer and @consensus-engine build independently.
They share the data model interfaces defined by Lead but own separate source files.

**Phase 2 (sequential):** @scoring-reports starts ONLY after both Phase 1 agents
complete. It consumes the GitHub client and model pool to build scoring and
report generation.

---

## Shared Interfaces (Defined by Lead)

Before spawning teammates, Lead creates these shared type definitions:

### PRData (dataclass)
```python
@dataclass
class PRData:
    number: int
    title: str
    body: str
    author: str
    created_at: str
    updated_at: str
    state: str
    labels: list[str]
    files_changed: list[str]
    additions: int
    deletions: int
    diff_summary: str
    url: str
    comments_count: int
```

### ModelResponse (dataclass)
```python
@dataclass
class ModelResponse:
    provider: str  # OpenRouter model ID (e.g., "anthropic/claude-sonnet-4")
    model: str     # Same as provider for OpenRouter
    content: str
    usage: dict | None = None
```

### Key Function Signatures
```python
# data-layer
def fetch_open_prs(repo: str, token: str, max_prs: int, use_cache: bool) -> list[PRData]
def fetch_repo_docs(repo: str, token: str) -> dict[str, str]

# consensus-engine
class ModelPool:
    def query_single(model: str, system_prompt: str, user_prompt: str, ...) -> ModelResponse
    def query_all(system_prompt: str, user_prompt: str, ...) -> list[ModelResponse]
    def get_embeddings(texts: list[str]) -> list[list[float]]
    def list_available_models() -> list[dict]

def extract_intents(prs: list[PRData], model_pool: ModelPool) -> list[IntentResult]
def cluster_intents(intents: list[IntentResult], threshold: float) -> list[Cluster]

# scoring-reports (Phase 2)
def score_prs(prs: list[PRData], model_pool: ModelPool, ...) -> list[QualityScore]
def score_alignment(prs: list[PRData], vision_docs: dict, model_pool: ModelPool, ...) -> list[AlignmentScore]
def generate_report(...) -> str
```

---

## @data-layer

**Description:** Implements GitHub API integration, local file caching, configuration
management, and the core data models shared across the project.

### File Ownership

```
src/claw_review/
├── __init__.py                    ← @data-layer
├── config.py                      ← @data-layer
├── github_client.py               ← @data-layer
tests/
├── __init__.py                    ← @data-layer
├── test_config.py                 ← @data-layer
├── test_github_client.py          ← @data-layer
├── conftest.py                    ← @data-layer (shared fixtures)
```

### Responsibilities

1. **config.py** — Config dataclass loading from environment/.env
   - Validate OPENROUTER_API_KEY and GITHUB_TOKEN
   - Parse MODELS env var (comma-separated OpenRouter model IDs)
   - Configurable thresholds (similarity, disagreement, alignment reject)
   - `model_count` property
   - `validate()` method returning list of issues

2. **github_client.py** — GitHub REST API v3 client
   - `fetch_open_prs()` — Paginated PR list + detail + file diffs
   - `fetch_repo_docs()` — Fetch README, CONTRIBUTING, ARCHITECTURE docs
   - Local file cache in `.claw-review-cache/` with hash-based keys
   - Diff truncation to keep model context manageable (12K chars max)
   - Rich progress bars for user feedback
   - Rate limit awareness (respect `X-RateLimit-Remaining` headers)

3. **conftest.py** — Shared pytest fixtures
   - `sample_pr_data()` — Factory for test PRData objects
   - `mock_github_responses()` — httpx mock responses
   - `tmp_cache_dir()` — Temporary cache directory

### Test Requirements (30+ tests)

- Config validation: missing OpenRouter key, missing GitHub token, model count < 2
- Config parsing: MODELS env var parsing, defaults, empty string, whitespace
- GitHub API: pagination, rate limiting, error handling (404, 403, 500)
- Cache: write, read, cache hit, cache miss, cache invalidation
- Diff truncation: under limit, over limit, empty diff
- Repo docs: found docs, missing docs, mixed availability
- PRData serialization: to_dict, round-trip

### Quality Gates

- [ ] All functions have type hints and docstrings
- [ ] httpx used for HTTP (not requests)
- [ ] All API calls have timeout configuration
- [ ] Cache directory is configurable and test-safe
- [ ] No hardcoded API URLs (use constants)
- [ ] Tests use httpx mocking (not real API calls)
- [ ] Rich progress bars for any operation >1 second

---

## @consensus-engine

**Description:** Implements the multi-model interface (Claude, GPT-4o, Gemini),
intent extraction prompting, embedding generation, and DBSCAN-based semantic
clustering for PR de-duplication.

### File Ownership

```
src/claw_review/
├── models.py                      ← @consensus-engine
├── clustering.py                  ← @consensus-engine
tests/
├── test_models.py                 ← @consensus-engine
├── test_clustering.py             ← @consensus-engine
├── fixtures/                      ← @consensus-engine
│   ├── sample_intents.json
│   └── sample_embeddings.json
```

### Responsibilities

1. **models.py** — Unified multi-model query interface via OpenRouter
   - `ModelPool` class using OpenRouter's OpenAI-compatible API
   - Single httpx client, single API key, routes to any model by ID
   - `query_single(model, system_prompt, user_prompt)` → ModelResponse
   - `query_all(system_prompt, user_prompt)` → list[ModelResponse]
   - `get_embeddings(texts)` → list[list[float]] (via OpenRouter)
   - `list_available_models()` for discovery
   - ModelResponse.parse_json() with markdown fence stripping
   - Graceful error handling (one model fails → others continue)
   - Temperature and max_tokens configuration per call

2. **clustering.py** — Intent extraction and semantic clustering
   - Intent extraction system prompt (structured JSON output)
   - `extract_intents(prs, model_pool)` → list[IntentResult]
   - `generate_embeddings(intents, model_pool)` → list[IntentResult] (with embeddings)
   - `cluster_intents(intents, threshold)` → list[Cluster]
   - DBSCAN with cosine distance matrix
   - Majority vote for category and affected area
   - Intent merging across model responses
   - Singleton handling (unclustered PRs)

### Test Requirements (35+ tests)

- ModelPool: initialization with valid/missing OpenRouter key, model counting
- query_single: each model (mocked), error handling, timeout
- query_all: mixed success/failure, all fail gracefully
- parse_json: clean JSON, markdown fences, embedded JSON, malformed
- get_embeddings: batching, empty input, dimension validation
- extract_intents: single PR, multiple PRs, partial model failure
- Intent merging: all agree, all differ, partial agreement
- majority_vote: clear winner, tie, single entry
- cluster_intents: no duplicates, clear duplicates, edge cases
- DBSCAN: threshold sensitivity, singleton handling, large clusters
- Embedding similarity: identical texts, similar texts, unrelated texts

### Quality Gates

- [ ] All model API calls are mocked in tests (zero real API usage)
- [ ] ModelPool handles single-model config gracefully (logs warning, still works)
- [ ] JSON parsing is resilient to markdown fences and extra whitespace
- [ ] DBSCAN parameters are configurable, not hardcoded
- [ ] Clustering works correctly with 0, 1, 2, and 100+ PRs
- [ ] All numpy operations handle edge cases (empty arrays, single element)
- [ ] Test fixtures provide realistic sample data

---

## @scoring-reports

**Description:** Implements multi-model quality scoring, vision alignment
evaluation, and report generation (HTML + JSON). Depends on @data-layer's
GitHub client and @consensus-engine's ModelPool.

### File Ownership

```
src/claw_review/
├── scoring.py                     ← @scoring-reports
├── alignment.py                   ← @scoring-reports
├── report.py                      ← @scoring-reports
├── cli.py                         ← Lead (but @scoring-reports adds subcommands)
tests/
├── test_scoring.py                ← @scoring-reports
├── test_alignment.py              ← @scoring-reports
├── test_report.py                 ← @scoring-reports
├── test_cli.py                    ← Lead (integration)
├── fixtures/
│   ├── sample_clusters.json       ← @scoring-reports
│   ├── sample_scores.json         ← @scoring-reports
│   └── sample_vision_docs.json    ← @scoring-reports
```

### Responsibilities

1. **scoring.py** — Multi-dimensional quality scoring
   - 5 scoring dimensions: code_quality, test_coverage, scope_discipline, breaking_risk, style_consistency
   - System prompt for structured JSON scoring (1-10 per dimension)
   - `score_prs(prs, model_pool, disagreement_threshold)` → list[QualityScore]
   - Consensus fusion: weighted average with spread-based outlier flagging
   - `rank_within_clusters(clusters, quality_scores)` → clusters with rankings
   - Human review flagging when models disagree by >3 points

2. **alignment.py** — Vision document alignment scoring
   - System prompt for alignment evaluation against project docs
   - `score_alignment(prs, vision_docs, model_pool, reject_threshold)` → list[AlignmentScore]
   - Consensus recommendation: MERGE / REVIEW / DISCUSS / CLOSE
   - Automatic CLOSE recommendation below threshold
   - Deduplication of aligned_aspects and drift_concerns

3. **report.py** — HTML and JSON report generation
   - Jinja2-templated HTML report with dark GitHub-style theme
   - Duplicate cluster visualization with quality rankings
   - Vision alignment flags with drift reasons
   - Top quality PRs section
   - Summary statistics cards
   - `generate_report(...)` → HTML file path
   - `generate_json_report(...)` → JSON file path

### Test Requirements (30+ tests)

- Scoring: single PR, multiple PRs, all dimensions present/missing
- Score clamping: values outside 1-10 range
- Disagreement detection: within threshold, beyond threshold
- rank_within_clusters: single PR cluster, multi-PR cluster, tied scores
- Alignment: with vision docs, without vision docs (skip gracefully)
- Recommendation logic: high score → MERGE, low score → CLOSE
- Report HTML: valid HTML output, all sections present, empty data handling
- Report JSON: valid JSON, round-trip serialization
- CLI: --help output, --skip-alignment flag, --max-prs validation

### Quality Gates

- [ ] HTML report renders correctly in Chrome, Firefox, Safari
- [ ] JSON report is valid and parseable
- [ ] All scoring prompts produce valid structured JSON from mocked responses
- [ ] Empty input (0 PRs, 0 clusters) produces valid empty report
- [ ] CLI provides clear error messages for missing config
- [ ] Jinja2 template handles missing/None values without crashing
- [ ] Report file paths are configurable

---

## Lead Responsibilities

The Lead agent owns:

```
src/claw_review/
├── cli.py                         ← Lead
pyproject.toml                     ← Lead
.env.example                       ← Lead
.gitignore                         ← Lead
README.md                          ← Lead
tests/
├── test_cli.py                    ← Lead
├── test_integration.py            ← Lead (end-to-end)
```

### Lead Tasks

1. **Pre-spawn:** Create shared interfaces (PRData, ModelResponse dataclasses), pyproject.toml, .env.example
2. **During Phase 1:** Monitor @data-layer and @consensus-engine for interface compliance
3. **Phase 2 gate:** Verify Phase 1 agents' tests pass before spawning @scoring-reports
4. **Post Phase 2:** Write integration tests, assemble CLI, verify end-to-end pipeline
5. **Final:** Run full test suite, generate sample report, validate README accuracy

### Integration Test Requirements (10+ tests)

- Full pipeline: fetch → extract → cluster → score → report (with all mocks)
- CLI entrypoint: `analyze`, `check`, `regenerate` commands
- Error handling: missing .env, invalid repo, API failures mid-pipeline
- Output validation: HTML file exists and contains expected sections
- JSON output: parseable, contains all expected keys

---

## Conflict Prevention Rules

1. **File lock:** Each agent owns specific files (see ownership above). No agent touches another agent's files.
2. **Interface-first:** Lead defines shared dataclasses and function signatures BEFORE spawning teammates.
3. **Import boundaries:**
   - @data-layer exports: `Config`, `PRData`, `fetch_open_prs`, `fetch_repo_docs`
   - @consensus-engine exports: `ModelPool`, `ModelResponse`, `IntentResult`, `Cluster`, `extract_intents`, `generate_embeddings`, `cluster_intents`
   - @scoring-reports imports from both, exports: `QualityScore`, `AlignmentScore`, `score_prs`, `score_alignment`, `generate_report`
4. **Test isolation:** Each agent's tests must pass independently using mocks. No tests require real API calls.
5. **Shared fixtures:** @data-layer creates `conftest.py` with `sample_pr_data()` factory. Other agents import from conftest.

---

## Global Quality Gates

Before sprint completion:

- [ ] All tests pass: `pytest tests/ -x -q` → 0 failures
- [ ] Type checking: `mypy src/claw_review/ --ignore-missing-imports` → 0 errors
- [ ] Linting: `ruff check src/ tests/` → 0 violations
- [ ] CLI runs: `claw-review check` works with valid .env
- [ ] README is accurate and matches actual CLI interface
- [ ] pyproject.toml installs cleanly: `pip install -e .`
- [ ] All Python files have module-level docstrings
- [ ] No hardcoded API keys anywhere in source


# Sprint 2 — Async Parallelism, Cost Optimization & Scale

## Sprint 2 Mission

Transform claw-review from a sequential PoC into a production-grade tool capable
of analyzing 6,000+ PRs efficiently. Three priorities: (1) async parallel API
calls to cut runtime 3-5x, (2) model presets to reduce costs by 10-50x, and
(3) incremental analysis to avoid re-processing known PRs.

**Sprint Duration:** 48–72 hours
**Predecessor:** Sprint 1 — 197 tests, 99% coverage, 0 failures. Live report
generated against openclaw/openclaw (100 PRs, $1.83 total cost, 865 API calls).

**Key Metrics from Sprint 1 Live Run:**
- 100 PRs analyzed, 865 OpenRouter requests, 2.69M tokens, $1.83 total
- GPT-4o consumed 93.4% of budget ($1.71) despite equal request count
- All 3 models got 287 requests each (perfectly balanced)
- Runtime: ~20 minutes (sequential API calls)

**Sprint 2 Targets:**
- Runtime: <5 minutes for 100 PRs (async parallel)
- Cost: <$0.50 for 100 PRs on "fast" preset
- Scale: Support 6,000+ PRs via batched incremental analysis
- Tests: 60+ new tests, maintain 99%+ coverage, 0 failures

---

## Team Structure

| Agent | Role | Phase | Test Target |
|-------|------|-------|-------------|
| @async-engine | Convert models.py and pipeline to async, connection pooling | Phase 1 (parallel) | 25+ |
| @cost-optimizer | Model presets, cost estimation, token tracking, budget limits | Phase 1 (parallel) | 20+ |
| @scale-layer | Incremental analysis, batch processing, resume capability | Phase 2 (sequential) | 15+ |

---

## Dependency Graph

```
  @async-engine         @cost-optimizer
   (Phase 1)              (Phase 1)
        \                    /
         \                  /
          \                /
       ──→ BOTH COMPLETE ←──
                 |
                 ▼
           @scale-layer
            (Phase 2)
```

**Phase 1 (parallel):** @async-engine rewrites the model interface while
@cost-optimizer adds presets and tracking. They touch different files.

**Phase 2 (sequential):** @scale-layer builds on async+cost features to
enable batched incremental analysis at 6,000+ PR scale.

---

## @async-engine

**Description:** Convert the synchronous httpx model interface to async,
enabling parallel API calls across all models simultaneously. This is the
single biggest performance win — instead of 3 sequential calls per PR,
all 3 fire concurrently.

### File Ownership

```
src/claw_review/
├── models.py                      ← @async-engine (rewrite to async)
├── clustering.py                  ← @async-engine (update to use async models)
├── scoring.py                     ← @async-engine (update to use async models)
├── alignment.py                   ← @async-engine (update to use async models)
tests/
├── test_models.py                 ← @async-engine (update for async)
├── test_clustering.py             ← @async-engine (update for async)
├── test_scoring.py                ← @async-engine (update for async)
├── test_alignment.py              ← @async-engine (update for async)
```

### Responsibilities

1. **models.py — Async rewrite**
   - Convert `ModelPool` to use `httpx.AsyncClient` with connection pooling
   - `async query_single()` — single model, single request
   - `async query_all()` — fires ALL models concurrently via `asyncio.gather()`
   - `async get_embeddings()` — batch embeddings with async client
   - Configurable concurrency limit (default: 10 simultaneous requests)
   - Per-model rate limiting (respect OpenRouter rate limits)
   - Retry with exponential backoff (max 3 retries per request)
   - Connection pool: keep-alive, max_connections=20
   - Timeout configuration: connect=10s, read=60s, pool=5s
   - Session-level client (create once, reuse across all calls)

2. **Pipeline modules — Async updates**
   - Update `extract_intents()` to use `async for` pattern
   - Update `score_prs()` to score all PRs concurrently
   - Update `score_alignment()` to align all PRs concurrently
   - Batch concurrency: process N PRs simultaneously (default: 5)
   - Progress callbacks for Rich progress bars

### Key Design Decisions

- Use `asyncio.Semaphore` for concurrency control, NOT unlimited parallelism
- The semaphore limit should be configurable via CLI `--concurrency N`
- All 3 models for a single PR fire concurrently (inner parallelism)
- Multiple PRs process concurrently up to semaphore limit (outer parallelism)
- Example: --concurrency 5 means 5 PRs × 3 models = 15 simultaneous requests max

### Test Requirements (25+ tests)

- Async query_single: mock responses, timeouts, retries
- Async query_all: concurrent execution verified (not sequential)
- Connection pooling: client reuse, cleanup on exit
- Semaphore: respects concurrency limit
- Rate limiting: backs off when rate limited (429 response)
- Retry logic: exponential backoff, max retries, gives up gracefully
- Pipeline async: extract_intents, score_prs, score_alignment all async
- Batch processing: correct results with concurrent PR processing
- Error isolation: one PR failing doesn't crash the batch
- Progress callbacks: called with correct counts

### Quality Gates

- [ ] All model calls use async httpx (zero synchronous httpx.Client remaining)
- [ ] asyncio.gather used for concurrent model queries
- [ ] Semaphore controls concurrency (no unbounded parallelism)
- [ ] Retry with backoff on 429/500/503 responses
- [ ] Connection pool properly cleaned up (async context manager)
- [ ] All tests use pytest-asyncio
- [ ] Existing test count maintained (no tests deleted, only updated)

---

## @cost-optimizer

**Description:** Add model presets for cost optimization, real-time cost
tracking, budget limits, and a cost estimation command. Based on Sprint 1
data: GPT-4o is 93% of cost, so swapping it for GPT-4o-mini or Llama cuts
total cost by 10-50x.

### File Ownership

```
src/claw_review/
├── config.py                      ← @cost-optimizer (add presets)
├── costs.py                       ← @cost-optimizer (NEW: cost tracking)
├── cli.py                         ← @cost-optimizer (add estimate + preset flags)
tests/
├── test_config.py                 ← @cost-optimizer (update for presets)
├── test_costs.py                  ← @cost-optimizer (NEW)
├── test_cli.py                    ← @cost-optimizer (update for new commands)
```

### Responsibilities

1. **costs.py — NEW: Cost tracking and estimation**
   - Token price table for common OpenRouter models (per 1M input/output tokens)
   - `CostTracker` class: accumulates per-model token usage and cost
   - `estimate_cost(num_prs, models, analysis_types)` → estimated cost
   - `format_cost_report()` → Rich table showing per-model breakdown
   - Real-time cost tracking: update after each API call from usage data
   - Budget limit: `--budget 5.00` stops analysis if budget exceeded
   - Cost summary appended to HTML/JSON report

2. **config.py — Model presets**
   Add preset configurations:
   ```
   PRESETS = {
       "fast": {
           "models": [
               "meta-llama/llama-3.1-70b-instruct",
               "mistralai/mistral-large-latest",
               "google/gemini-2.0-flash-001",
           ],
           "description": "Fastest & cheapest. Good for initial scan.",
           "est_cost_per_100_prs": "$0.15-0.30",
       },
       "balanced": {
           "models": [
               "anthropic/claude-sonnet-4",
               "openai/gpt-4o-mini",
               "google/gemini-2.0-flash-001",
           ],
           "description": "Best quality/cost ratio. Recommended for most use.",
           "est_cost_per_100_prs": "$0.30-0.60",
       },
       "thorough": {
           "models": [
               "anthropic/claude-sonnet-4",
               "openai/gpt-4o",
               "google/gemini-2.0-flash-001",
           ],
           "description": "Highest quality. Sprint 1 default.",
           "est_cost_per_100_prs": "$1.50-2.50",
       },
   }
   ```

3. **cli.py — New flags and commands**
   - `--preset fast|balanced|thorough` (overrides MODELS env var)
   - `--budget 5.00` (halt if estimated cost exceeds budget)
   - `claw-review estimate --repo owner/name --max-prs 100` (dry run cost estimate)
   - `claw-review presets` (list available presets with costs)
   - Cost summary printed at end of analysis run

### Test Requirements (20+ tests)

- Preset loading: each preset returns correct models
- Preset override: --preset overrides MODELS env var
- Cost tracker: accumulates correctly, per-model breakdown
- Cost estimation: correct math for different PR counts and presets
- Budget limit: stops when exceeded, partial results saved
- Token price table: known models have prices, unknown models use fallback
- Format cost report: Rich table renders correctly
- CLI: --preset flag, estimate command, presets command
- Cost in report: HTML and JSON reports include cost summary

### Quality Gates

- [ ] All 3 presets tested with correct model lists
- [ ] Cost tracker updates from real OpenRouter usage response format
- [ ] Budget limit saves partial results (doesn't lose work)
- [ ] Estimate command works without API calls (pure calculation)
- [ ] Token prices are documented with source/date
- [ ] Unknown models use conservative fallback pricing

---

## @scale-layer

**Description:** Enable claw-review to handle 6,000+ PRs through incremental
analysis (skip already-analyzed PRs), batch processing with checkpointing,
and a merge command to combine multiple batch results.

### File Ownership

```
src/claw_review/
├── state.py                       ← @scale-layer (NEW: analysis state management)
├── batch.py                       ← @scale-layer (NEW: batch orchestration)
├── report.py                      ← @scale-layer (update: merge reports)
├── cli.py                         ← @scale-layer (add batch + merge commands)
tests/
├── test_state.py                  ← @scale-layer (NEW)
├── test_batch.py                  ← @scale-layer (NEW)
├── test_report.py                 ← @scale-layer (update: merge tests)
├── test_cli.py                    ← @scale-layer (update: batch + merge)
```

### Responsibilities

1. **state.py — NEW: Analysis state management**
   - `AnalysisState` stored as JSON in `.claw-review-state/{repo_hash}.json`
   - Tracks: analyzed PR numbers, analysis timestamp, model config used
   - `get_unanalyzed_prs(all_prs, state)` → list of new PRs to analyze
   - `update_state(state, new_results)` → updated state
   - State includes per-PR: cluster assignment, quality score, alignment score
   - Force re-analysis with `--force` flag (ignore state)

2. **batch.py — NEW: Batch orchestration**
   - `BatchProcessor` class: splits large PR lists into configurable batches
   - Default batch size: 50 PRs (configurable via `--batch-size`)
   - Checkpoint after each batch (save partial results to state)
   - Resume capability: if interrupted, restart from last checkpoint
   - Progress: "Batch 3/12: analyzing PRs 101-150... (cost so far: $0.45)"
   - Automatic re-clustering after all batches complete (global clusters)

3. **report.py — Merge capability**
   - `merge_reports(report_files)` → combined report
   - Re-cluster across merged results (duplicates may span batches)
   - Re-rank quality scores globally
   - `claw-review merge report1.json report2.json -o combined.html`

4. **cli.py — New commands**
   - `claw-review analyze --batch-size 50` (process in batches)
   - `claw-review analyze --incremental` (skip known PRs, default ON)
   - `claw-review analyze --force` (re-analyze everything)
   - `claw-review merge file1.json file2.json` (combine results)
   - `claw-review status --repo owner/name` (show analysis state)

### Test Requirements (15+ tests)

- State: create, read, update, corrupt file handling
- Unanalyzed PRs: all new, some new, none new, force flag
- Batch processing: splits correctly, respects batch size
- Checkpoint: save after each batch, resume from checkpoint
- Resume: interrupted mid-batch, restart correctly
- Merge: two reports, overlapping PRs, re-clustering
- CLI: batch-size flag, incremental flag, force flag, merge command, status command

### Quality Gates

- [ ] State file is atomic (write to temp, rename)
- [ ] Checkpoint saves after EVERY batch (no lost work on crash)
- [ ] Resume produces identical results to uninterrupted run
- [ ] Merge correctly re-clusters across batch boundaries
- [ ] --incremental is the default (users don't re-analyze accidentally)
- [ ] Status command shows useful summary without running analysis

---

## Lead Responsibilities

### Pre-Sprint

1. Update pyproject.toml: add `pytest-asyncio` dependency
2. Update CLAUDE.md with Sprint 2 context (async patterns, cost data)
3. Define shared interfaces for CostTracker and AnalysisState

### Phase Gate

1. Verify @async-engine's async models pass all tests
2. Verify @cost-optimizer's presets and tracking work
3. Confirm both integrate cleanly before spawning @scale-layer

### Post-Sprint

1. Integration tests: full async pipeline with cost tracking
2. Live validation: re-run against openclaw/openclaw with `--preset balanced`
3. Compare: Sprint 1 (sequential, thorough) vs Sprint 2 (async, balanced)
4. Update README with new commands and presets
5. Update GitHub Pages site

---

## Global Quality Gates

Before sprint completion:

- [ ] All tests pass: `pytest tests/ -x -q` → 0 failures
- [ ] Total test count: 250+ (197 existing + 60 new)
- [ ] Coverage: maintain 99%+
- [ ] Type checking: `mypy src/claw_review/ --ignore-missing-imports` → 0 errors
- [ ] Linting: `ruff check src/ tests/` → 0 violations
- [ ] Async correctness: no blocking calls in async code path
- [ ] Live run: 100 PRs in <5 minutes with --preset balanced at <$0.50

---

## Expected Outcome

After Sprint 2, the full 6,000+ PR OpenClaw analysis becomes viable:

| Preset | Est. Cost (6K PRs) | Est. Runtime | Quality |
|--------|-------------------|-------------|---------|
| fast | ~$9-18 | ~15 min | Good (open-source models) |
| balanced | ~$18-36 | ~20 min | Great (Claude + cheap GPT + Gemini) |
| thorough | ~$90-150 | ~30 min | Best (Sprint 1 models) |

**Recommended strategy for 6K PRs:** Run `fast` first for initial clustering,
then `thorough` only on the duplicate clusters (~200-500 PRs) for quality ranking.
Two-pass approach: ~$12-25 total for full repo analysis.


# Sprint 3 — Platform Architecture, GitHub Integration & Multi-Domain Adapters

## Sprint 3 Mission

Transform claw-review from a GitHub-specific CLI tool into a domain-agnostic
multi-model consensus platform. Four priorities: (1) extract a pluggable core
engine with data adapters and prompt templates, (2) build GitHub App + webhook +
PR comment bot for real-time triage, (3) build an interactive dashboard for
historical analysis and exploration, and (4) create domain adapter templates
for cybersecurity and fraud detection to prove the pattern is portable.

**Sprint Duration:** 72–96 hours
**Predecessor:** Sprint 2 — 298 tests, 99% coverage, 0 failures. Async parallel
pipeline, 3 model presets, cost tracking, batch processing with checkpoints.

**Key Metrics from Sprint 2 Live Run:**
- 100 PRs analyzed in ~7 minutes (3x faster than Sprint 1)
- Balanced preset (Claude Sonnet + GPT-4o-mini + Gemini Flash)
- 3 vision drift flags caught (CLOSE, DISCUSS, REVIEW)
- 1 duplicate cluster, 2 duplicate PRs
- 298 tests, 99% coverage, 7 CLI commands

**Sprint 3 Targets:**
- Platform core: domain-agnostic engine with adapter pattern
- GitHub integration: webhook → analysis → PR comment (end-to-end)
- Dashboard: interactive web UI for exploring results
- Domain adapters: cybersecurity + fraud detection prompt templates
- Tests: 80+ new tests, maintain 99%+ coverage, 0 failures
- Total tests: 375+

---

## Team Structure

| Agent | Role | Phase | Test Target |
|-------|------|-------|-------------|
| @core-engine | Extract domain-agnostic consensus platform with adapter interfaces | Phase 1 | 25+ |
| @domain-adapters | Cybersecurity + fraud detection adapters, prompt templates, scoring dimensions | Phase 1 (parallel) | 20+ |
| @github-integration | GitHub App, webhook receiver, Actions workflow, PR comment bot | Phase 2 (sequential) | 20+ |
| @dashboard | Interactive web dashboard for exploring analysis results | Phase 2 (parallel with github-integration) | 15+ |

---

## Dependency Graph

```
  @core-engine          @domain-adapters
   (Phase 1)              (Phase 1)
        \                    /
         \                  /
          \                /
       ──→ BOTH COMPLETE ←──
              |         |
              ▼         ▼
   @github-integration  @dashboard
      (Phase 2)        (Phase 2)
```

**Phase 1 (parallel):** @core-engine extracts the platform interfaces while
@domain-adapters builds prompt templates for new domains. They coordinate
on the adapter interface but own separate files.

**Phase 2 (parallel):** @github-integration and @dashboard both build on the
core platform but are independent of each other. They can run simultaneously.

---

## Shared Interfaces (Defined by Lead)

Before spawning teammates, Lead creates these platform abstractions:

### DataAdapter (Protocol)
```python
from typing import Protocol, Any

class DataItem:
    """A single item to analyze (PR, alert, transaction, etc.)."""
    id: str
    title: str
    body: str
    metadata: dict[str, Any]
    raw: dict[str, Any]

class DataAdapter(Protocol):
    """Interface for domain-specific data sources."""
    domain: str  # "github-pr", "cybersecurity", "fraud", etc.

    async def fetch_items(
        self, source: str, max_items: int, **kwargs
    ) -> list[DataItem]:
        """Fetch items from the data source."""
        ...

    async def fetch_context_docs(
        self, source: str, **kwargs
    ) -> dict[str, str]:
        """Fetch reference documents for alignment scoring."""
        ...

    def format_item_for_prompt(self, item: DataItem) -> str:
        """Format a single item for inclusion in an LLM prompt."""
        ...
```

### DomainConfig (dataclass)
```python
@dataclass
class DomainConfig:
    """Domain-specific configuration for the consensus engine."""
    domain: str
    scoring_dimensions: list[str]  # e.g., ["code_quality", "test_coverage", ...]
    clustering_prompt: str         # System prompt for intent extraction
    scoring_prompt: str            # System prompt for quality scoring
    alignment_prompt: str          # System prompt for vision/policy alignment
    recommendation_levels: list[str]  # e.g., ["MERGE", "REVIEW", "DISCUSS", "CLOSE"]
    default_thresholds: dict[str, float]
```

### AnalysisResult (dataclass)
```python
@dataclass
class AnalysisResult:
    """Domain-agnostic analysis output."""
    domain: str
    source: str
    items_analyzed: int
    clusters: list[Cluster]
    quality_scores: list[QualityScore]
    alignment_scores: list[AlignmentScore]
    cost: CostSummary
    timestamp: str
    metadata: dict[str, Any]
```

---

## @core-engine

**Description:** Refactor the existing claw-review pipeline into a domain-agnostic
consensus platform. Extract GitHub-specific logic into an adapter, define clean
interfaces for data sources and domain configs, and make the pipeline work with
any domain adapter.

### File Ownership

```
src/claw_review/
├── platform/                      ← @core-engine (NEW directory)
│   ├── __init__.py
│   ├── interfaces.py              ← DataAdapter, DomainConfig, DataItem, AnalysisResult
│   ├── engine.py                  ← ConsensusEngine: domain-agnostic pipeline orchestrator
│   └── registry.py                ← AdapterRegistry: register/discover domain adapters
├── adapters/                      ← @core-engine (NEW directory, GitHub adapter)
│   ├── __init__.py
│   └── github_pr.py               ← GitHubPRAdapter (refactored from github_client.py)
├── domains/                       ← @core-engine (NEW directory)
│   ├── __init__.py
│   └── github_pr.py               ← GITHUB_PR_CONFIG: prompts, dimensions, thresholds
tests/
├── test_platform/                 ← @core-engine (NEW directory)
│   ├── __init__.py
│   ├── test_interfaces.py
│   ├── test_engine.py
│   └── test_registry.py
├── test_adapters/
│   ├── __init__.py
│   └── test_github_pr_adapter.py
```

### Responsibilities

1. **platform/interfaces.py** — Core abstractions
   - `DataItem` dataclass: universal item container
   - `DataAdapter` Protocol: fetch_items, fetch_context_docs, format_item_for_prompt
   - `DomainConfig` dataclass: scoring dimensions, prompts, thresholds, recommendations
   - `AnalysisResult` dataclass: domain-agnostic output container
   - `CostSummary` dataclass: token usage and cost breakdown

2. **platform/engine.py** — ConsensusEngine
   - `ConsensusEngine(model_pool, domain_config)` — generic pipeline
   - `async analyze(items: list[DataItem], context_docs: dict) → AnalysisResult`
   - Internally calls: extract_intents → cluster → score → align
   - Uses domain_config.clustering_prompt, scoring_prompt, alignment_prompt
   - Uses domain_config.scoring_dimensions for quality evaluation
   - Uses domain_config.recommendation_levels for alignment output
   - Pluggable: swap domain_config to change what's analyzed and how

3. **platform/registry.py** — Adapter discovery
   - `AdapterRegistry.register(domain, adapter_class, config)`
   - `AdapterRegistry.get(domain) → (adapter, config)`
   - `AdapterRegistry.list_domains() → list[str]`
   - Auto-discovery: scan adapters/ and domains/ directories
   - CLI: `claw-review domains` lists available domains

4. **adapters/github_pr.py** — Refactored GitHub adapter
   - Implements `DataAdapter` Protocol
   - Wraps existing github_client.py functionality
   - `PRData` maps to `DataItem` via conversion method
   - Preserves all existing caching, pagination, rate limiting

5. **domains/github_pr.py** — GitHub PR domain config
   - Extracts existing prompts from clustering.py, scoring.py, alignment.py
   - Defines 5 scoring dimensions (code_quality, test_coverage, etc.)
   - Defines recommendation levels (MERGE, REVIEW, DISCUSS, CLOSE)
   - Defines default thresholds (similarity: 0.82, disagreement: 3.0, etc.)

### Key Design Decisions

- Existing modules (clustering.py, scoring.py, alignment.py) are NOT deleted
- They become thin wrappers around ConsensusEngine for backward compatibility
- CLI `claw-review analyze` continues to work exactly as before
- New flag: `claw-review analyze --domain github-pr` (default)
- Platform enables: `claw-review analyze --domain cybersecurity --source siem.json`

### Test Requirements (25+ tests)

- DataItem: creation, serialization, metadata handling
- DataAdapter Protocol: mock adapter satisfies interface
- DomainConfig: load, validate, missing fields
- ConsensusEngine: full pipeline with mock adapter and mock models
- ConsensusEngine: different domain configs produce different prompts
- Registry: register, get, list, duplicate domain handling
- GitHub adapter: implements DataAdapter correctly
- Backward compatibility: existing CLI commands still work
- Domain config extraction: GitHub prompts match originals

### Quality Gates

- [ ] All existing tests still pass (zero regression)
- [ ] ConsensusEngine works with any DataAdapter implementation
- [ ] Domain config fully parameterizes all prompts and dimensions
- [ ] Registry pattern allows runtime adapter registration
- [ ] `claw-review analyze` backward compatible (no flags needed)
- [ ] Platform interfaces have complete docstrings with examples

---

## @domain-adapters

**Description:** Create domain adapter templates and configs for cybersecurity
(SIEM alert triage) and fraud detection (transaction analysis). These prove the
platform pattern is portable beyond GitHub PRs and provide the prompt engineering
foundation for future VectorCertain integrations.

### File Ownership

```
src/claw_review/
├── adapters/
│   ├── cybersecurity.py           ← @domain-adapters (NEW)
│   └── fraud_detection.py         ← @domain-adapters (NEW)
├── domains/
│   ├── cybersecurity.py           ← @domain-adapters (NEW)
│   └── fraud_detection.py         ← @domain-adapters (NEW)
tests/
├── test_adapters/
│   ├── test_cybersecurity_adapter.py  ← @domain-adapters (NEW)
│   └── test_fraud_adapter.py          ← @domain-adapters (NEW)
├── test_domains/
│   ├── __init__.py
│   ├── test_cybersecurity_config.py   ← @domain-adapters (NEW)
│   └── test_fraud_config.py           ← @domain-adapters (NEW)
├── fixtures/
│   ├── sample_siem_alerts.json        ← @domain-adapters (NEW)
│   └── sample_transactions.json       ← @domain-adapters (NEW)
```

### Responsibilities

1. **adapters/cybersecurity.py** — SIEM alert adapter
   - Implements `DataAdapter` Protocol
   - `fetch_items()`: reads from JSON file or stdin (real SIEM integration deferred)
   - Input format: standard SIEM alert JSON (severity, source_ip, dest_ip,
     alert_type, timestamp, raw_log, indicator_of_compromise)
   - `format_item_for_prompt()`: structured alert summary for LLM analysis
   - Support for common formats: Splunk JSON, ElasticSearch alerts, generic CEF

2. **domains/cybersecurity.py** — CYBERSECURITY_CONFIG
   - Scoring dimensions: `threat_severity`, `confidence`, `attack_sophistication`,
     `asset_criticality`, `actionability`
   - Clustering prompt: "Group these alerts by attack campaign or common
     threat actor / technique. Identify correlated alerts that are part of
     the same incident."
   - Scoring prompt: "Evaluate this security alert across 5 dimensions..."
   - Alignment prompt: "Does this alert match known threat patterns in the
     organization's threat model? Score alignment with security policy."
   - Recommendation levels: `BLOCK`, `INVESTIGATE`, `MONITOR`, `DISMISS`
   - Default thresholds: similarity 0.78, disagreement 2.5, dismiss below 3.0

3. **adapters/fraud_detection.py** — Transaction adapter
   - Implements `DataAdapter` Protocol
   - `fetch_items()`: reads from JSON/CSV file (real bank API deferred)
   - Input format: transaction JSON (amount, merchant, location, timestamp,
     card_type, transaction_type, customer_id, historical_avg)
   - `format_item_for_prompt()`: structured transaction summary with context
   - Support for: JSON, CSV with configurable column mapping

4. **domains/fraud_detection.py** — FRAUD_DETECTION_CONFIG
   - Scoring dimensions: `anomaly_score`, `pattern_match`, `velocity_risk`,
     `geographic_risk`, `amount_deviation`
   - Clustering prompt: "Group these transactions by suspected fraud pattern.
     Identify coordinated fraud rings or repeated attack vectors."
   - Scoring prompt: "Evaluate this transaction for fraud risk across 5 dimensions..."
   - Alignment prompt: "Does this transaction match the customer's established
     behavioral profile? Score deviation from normal patterns."
   - Recommendation levels: `APPROVE`, `FLAG`, `HOLD`, `BLOCK`
   - Default thresholds: similarity 0.75, disagreement 2.0, block below 3.0

5. **Fixture files** — Realistic sample data
   - `sample_siem_alerts.json`: 20 alerts including correlated attack chain,
     false positives, and genuine threats
   - `sample_transactions.json`: 20 transactions including normal purchases,
     velocity anomalies, geographic impossibility, and micro-fraud patterns

### Test Requirements (20+ tests)

- Cybersecurity adapter: load alerts, format for prompt, handle missing fields
- Cybersecurity config: all prompts present, dimensions validated
- Fraud adapter: load transactions (JSON + CSV), format, column mapping
- Fraud config: all prompts present, dimensions validated
- Both adapters: satisfy DataAdapter Protocol
- Both configs: register correctly with AdapterRegistry
- Fixture data: valid JSON, covers edge cases
- Cross-domain: ConsensusEngine works with both configs (mocked models)
- CLI: `claw-review domains` lists all 3 domains (github-pr, cybersecurity, fraud)
- CLI: `claw-review analyze --domain cybersecurity --source alerts.json` parses correctly

### Quality Gates

- [ ] Both adapters fully implement DataAdapter Protocol
- [ ] Domain configs include realistic, well-crafted prompts
- [ ] Fixture data is realistic and covers genuine analysis scenarios
- [ ] All tests mocked (zero real API calls, zero real SIEM/bank connections)
- [ ] ConsensusEngine produces valid AnalysisResult with both new domains
- [ ] Prompts are structured to produce consistent JSON output from models
- [ ] README section documents each domain with example usage

---

## @github-integration

**Description:** Build the GitHub App infrastructure for real-time PR triage:
webhook receiver to detect new/updated PRs, automated analysis trigger, and
PR comment bot that posts consensus results directly on the PR.

### File Ownership

```
src/claw_review/
├── github/                        ← @github-integration (NEW directory)
│   ├── __init__.py
│   ├── app.py                     ← GitHub App configuration and auth
│   ├── webhook.py                 ← Webhook receiver (FastAPI/Flask endpoint)
│   ├── commenter.py               ← PR comment formatter and poster
│   └── actions.py                 ← GitHub Actions workflow generator
├── templates/
│   ├── pr_comment.md.j2           ← @github-integration (NEW)
│   └── actions_workflow.yml.j2    ← @github-integration (NEW)
tests/
├── test_github/                   ← @github-integration (NEW directory)
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_webhook.py
│   ├── test_commenter.py
│   └── test_actions.py
```

### Responsibilities

1. **github/app.py** — GitHub App configuration
   - App ID, private key, installation token management
   - JWT generation for GitHub App authentication
   - Installation token caching (expires after 1 hour)
   - Permissions: pull_requests (read/write), issues (write for comments)
   - Config via environment: GITHUB_APP_ID, GITHUB_PRIVATE_KEY_PATH

2. **github/webhook.py** — Webhook receiver
   - Lightweight HTTP endpoint (FastAPI or Flask, agent decides)
   - Listens for: `pull_request.opened`, `pull_request.synchronize`,
     `pull_request.reopened` events
   - Webhook signature verification (HMAC-SHA256)
   - Event parsing: extract repo, PR number, action type
   - Queue analysis job (async, non-blocking response to GitHub)
   - Configurable: which repos, which PR events, minimum PR size

3. **github/commenter.py** — PR comment bot
   - Format analysis results as a Markdown PR comment
   - Sections: Quality Score (with bar chart), Duplicate Detection,
     Vision Alignment, Model Consensus, Recommendation
   - Update existing comment (don't spam — find and edit previous comment)
   - Collapsible details section for per-model scores
   - Badge/label: "claw-review: MERGE ✅" or "claw-review: REVIEW ⚠️"
   - Optional: add GitHub labels based on recommendation

4. **github/actions.py** — GitHub Actions workflow generator
   - `claw-review generate-workflow` CLI command
   - Generates `.github/workflows/claw-review.yml`
   - Triggers on: pull_request opened/synchronize
   - Runs claw-review against the single PR
   - Posts comment with results
   - Configurable: preset, budget limit, skip-alignment flag

5. **templates/pr_comment.md.j2** — Comment template
   ```markdown
   ## 🦞 ClawReview Analysis

   | Dimension | Score |
   |-----------|-------|
   | Code Quality | {{ scores.code_quality }}/10 |
   | ... | ... |

   **Overall: {{ overall_score }}/10** · Recommendation: {{ recommendation }}

   <details>
   <summary>Per-model breakdown</summary>
   ...
   </details>

   {% if duplicates %}
   ⚠️ **Potential duplicates found:** {{ duplicate_prs }}
   {% endif %}

   ---
   *Analyzed by [claw-review](https://github.com/jconroy1104/claw-review)
   using multi-model consensus ({{ models }})*
   ```

6. **templates/actions_workflow.yml.j2** — Actions workflow template
   ```yaml
   name: ClawReview PR Triage
   on:
     pull_request:
       types: [opened, synchronize, reopened]
   jobs:
     triage:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
         - run: pip install claw-review
         - run: claw-review analyze-pr ${{ github.event.pull_request.number }}
           env:
             GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
             OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
   ```

### Test Requirements (20+ tests)

- App auth: JWT generation, installation token caching, token expiry
- Webhook: signature verification (valid, invalid, missing)
- Webhook: event parsing (opened, synchronize, reopened, ignored events)
- Webhook: repo filtering (allowed repos, blocked repos)
- Commenter: format quality scores, duplicate detection, alignment flags
- Commenter: update existing comment (find by marker text)
- Commenter: handle empty results, single PR, flagged PR
- Actions: workflow generation, correct triggers, env vars
- Actions: CLI command outputs valid YAML
- Integration: webhook → analysis → comment (mocked GitHub API)

### Quality Gates

- [ ] Webhook signature verification prevents unauthorized triggers
- [ ] Comment bot updates (not duplicates) existing comments
- [ ] Actions workflow is valid YAML that GitHub accepts
- [ ] All GitHub API calls mocked in tests
- [ ] Non-blocking webhook response (analysis runs async)
- [ ] Comment template renders correctly with all edge cases
- [ ] `generate-workflow` command produces copy-paste ready YAML

---

## @dashboard

**Description:** Build an interactive web dashboard for exploring claw-review
analysis results. Supports filtering, sorting, searching across all analyzed
items. Works with any domain's AnalysisResult output.

### File Ownership

```
src/claw_review/
├── dashboard/                     ← @dashboard (NEW directory)
│   ├── __init__.py
│   ├── app.py                     ← Dashboard web app (agent chooses framework)
│   ├── data_loader.py             ← Load AnalysisResult JSON into dashboard
│   └── static/                    ← CSS, JS assets (if static HTML approach)
│       └── ...
├── templates/
│   └── dashboard.html.j2          ← @dashboard (or .jsx if React)
tests/
├── test_dashboard/                ← @dashboard (NEW directory)
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_data_loader.py
│   └── test_rendering.py
```

### Responsibilities

1. **dashboard/data_loader.py** — Load and transform data
   - Load from claw-review JSON report files
   - Support multiple report files (merged view)
   - Transform AnalysisResult into dashboard-friendly format
   - Compute summary statistics: total items, clusters, flags, cost
   - Filter/sort API: by score, by cluster, by recommendation, by date

2. **dashboard/app.py** — Web application
   - Agent decides: static HTML generator OR lightweight server (Flask/FastAPI)
   - `claw-review dashboard --port 8080` for local server mode
   - `claw-review dashboard --static -o dashboard.html` for static export
   - Static export for GitHub Pages deployment (no server required)

3. **Dashboard features (UI)**
   - **Summary cards:** items analyzed, clusters, drift flags, total cost, runtime
   - **Cluster view:** expandable clusters with ranked items, quality bars
   - **Table view:** all items sortable by score, recommendation, author, date
   - **Search:** filter by title, author, label, recommendation
   - **Detail panel:** click any item to see per-model scores, consensus detail
   - **Cost breakdown:** per-model cost chart (mirrors OpenRouter dashboard)
   - **Domain selector:** switch between github-pr, cybersecurity, fraud views
   - **Dark theme:** GitHub-style dark theme (matches Sprint 1/2 reports)
   - **Responsive:** works on desktop and mobile

4. **Static export for GitHub Pages**
   - `claw-review dashboard --static` generates self-contained HTML
   - All data embedded as JSON in a `<script>` tag
   - JavaScript handles filtering/sorting client-side
   - Zero server dependencies for viewing
   - Publishable directly to GitHub Pages

### Test Requirements (15+ tests)

- Data loader: load single report, multiple reports, empty report
- Data loader: filter by recommendation, by score range, by author
- Data loader: sort by quality ascending/descending
- Summary stats: correct counts, averages, totals
- Static export: valid HTML output, embedded data, JavaScript works
- Dashboard CLI: --port flag, --static flag, --output flag
- Multi-domain: loads github-pr and cybersecurity results in same dashboard
- Search: title match, author match, partial match
- Edge cases: zero items, single item, 1000+ items

### Quality Gates

- [ ] Static export is a single self-contained HTML file
- [ ] Dashboard works with any domain's AnalysisResult
- [ ] Client-side filtering is fast (<100ms for 6000 items)
- [ ] Dark theme matches existing report styling
- [ ] GitHub Pages deployable with zero configuration
- [ ] Mobile responsive (readable on phone)
- [ ] Data loader handles malformed/partial JSON gracefully

---

## Lead Responsibilities

### Pre-Sprint

1. Create platform/interfaces.py with shared Protocol and dataclass definitions
2. Update pyproject.toml: add FastAPI/Flask, PyJWT, cryptography dependencies
3. Update CLAUDE.md with Sprint 3 platform architecture context
4. Define DataAdapter Protocol and DomainConfig dataclass before spawning

### Phase Gate (Phase 1 → Phase 2)

1. Verify @core-engine's ConsensusEngine works with mock adapters
2. Verify @domain-adapters' configs register correctly with registry
3. Confirm platform interfaces are stable before Phase 2 agents build on them

### Post-Sprint

1. Integration tests: webhook → engine → commenter (end-to-end mock)
2. Integration tests: dashboard loads real Sprint 2 report data
3. Cross-domain test: same engine, different adapters, valid results
4. Update README: platform architecture diagram, new domains, GitHub App setup
5. Update GitHub Pages with new dashboard
6. Live validation: run dashboard against Sprint 2 report data

---

## Conflict Prevention Rules

1. **File lock:** Each agent owns specific directories (platform/, adapters/,
   domains/, github/, dashboard/). No agent touches another's directory.
2. **Interface-first:** Lead defines DataAdapter, DomainConfig, AnalysisResult
   BEFORE spawning. These interfaces are frozen during the sprint.
3. **Import boundaries:**
   - @core-engine exports: ConsensusEngine, DataAdapter, DomainConfig, DataItem,
     AnalysisResult, AdapterRegistry
   - @domain-adapters exports: CybersecurityAdapter, CYBERSECURITY_CONFIG,
     FraudDetectionAdapter, FRAUD_DETECTION_CONFIG
   - @github-integration exports: GitHubApp, WebhookReceiver, PRCommenter,
     generate_workflow
   - @dashboard exports: DashboardApp, DataLoader, generate_static_dashboard
4. **Backward compatibility:** Existing CLI commands (analyze, check, estimate,
   presets, status, merge, regenerate) MUST continue to work unchanged.
5. **Test isolation:** Each agent's tests pass independently using mocks.

---

## Global Quality Gates

Before sprint completion:

- [ ] All tests pass: `pytest tests/ -x -q` → 0 failures
- [ ] Total test count: 375+ (298 existing + 80 new)
- [ ] Coverage: maintain 99%+
- [ ] Type checking: `mypy src/claw_review/ --ignore-missing-imports` → 0 errors
- [ ] Linting: `ruff check src/ tests/` → 0 violations
- [ ] Backward compatibility: all Sprint 2 CLI commands still work
- [ ] Platform works with 3 domains: github-pr, cybersecurity, fraud-detection
- [ ] Dashboard renders Sprint 2 report data correctly
- [ ] GitHub Actions workflow is valid YAML

---

## Expected Outcome

After Sprint 3, claw-review becomes a multi-domain platform:

```
                    ┌──────────────────────────────────────┐
                    │       claw-review Platform           │
                    │                                      │
                    │   ConsensusEngine (domain-agnostic)  │
                    │   ModelPool (OpenRouter, async)       │
                    │   CostTracker / BatchProcessor       │
                    └──────────┬───────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ GitHub   │    │  Cyber   │    │  Fraud   │
        │ PR       │    │ Security │    │Detection │
        │ Adapter  │    │ Adapter  │    │ Adapter  │
        └────┬─────┘    └──────────┘    └──────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
  Webhook  Actions  Dashboard
  Bot      CI/CD    Explorer
```

| Capability | Status |
|------------|--------|
| CLI analysis (any domain) | ✅ |
| GitHub PR comments | ✅ |
| GitHub Actions workflow | ✅ |
| Interactive dashboard | ✅ |
| Cybersecurity templates | ✅ Ready for real SIEM integration |
| Fraud detection templates | ✅ Ready for real transaction feeds |
| Real-time webhooks | ✅ |
| Static GitHub Pages export | ✅ |

**Next sprint candidates after Sprint 3:**
- Sprint 4: OpenClaw Skill plugin (package as installable OpenClaw skill)
- Sprint 5: Live SIEM integration (Splunk/Elastic adapter with real data)
- Sprint 6: Production hardening (Redis queues, PostgreSQL state, Docker deployment)
