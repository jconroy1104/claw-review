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
