# claw-review Sprint 1 — Agent Team Launch Guide

## Pre-Launch Checklist

- [ ] Clone/create the claw-review repo locally
- [ ] Copy source files from the PoC scaffold into the repo
- [ ] Verify AGENTS.md and CLAUDE.md are in the project root
- [ ] Ensure pyproject.toml and .env.example are in place
- [ ] Open a tmux session: `tmux new -s claw-review`
- [ ] Navigate to the project root: `cd ~/claw-review`
- [ ] Set environment variable: `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`
- [ ] Launch Claude Code: `claude --team`

## Spawn Prompt

Paste this into Claude Code after launching with `--team`:

```
Read CLAUDE.md and AGENTS.md to understand the project context and sprint plan.

We are building claw-review, an AI-powered PR triage CLI that uses multi-model
consensus. Sprint 1 has 3 teammates across 2 phases.

PHASE 1 — Launch these two teammates in parallel:

Teammate "data-layer" should implement:
1. src/claw_review/config.py — Config dataclass with env loading, validation,
   and model_count property. Single OPENROUTER_API_KEY + MODELS env var
   (comma-separated OpenRouter model IDs). Thresholds for similarity (0.82),
   disagreement (3.0), and alignment reject (4.0).
2. src/claw_review/github_client.py — GitHub REST API v3 client using httpx.
   fetch_open_prs() with pagination, detail fetching, file diffs, and local
   file caching in .claw-review-cache/. fetch_repo_docs() for README,
   CONTRIBUTING, ARCHITECTURE docs. Diff truncation at 12K chars. Rich
   progress bars.
3. tests/conftest.py — Shared fixtures: sample_pr_data() factory,
   mock_github_responses(), tmp_cache_dir().
4. tests/test_config.py and tests/test_github_client.py — 30+ tests total,
   all mocked (zero real API calls).

Teammate "consensus-engine" should implement:
1. src/claw_review/models.py — ModelPool class using OpenRouter.ai as a
   unified gateway (OpenAI-compatible API at https://openrouter.ai/api/v1).
   One API key, one httpx client, routes to any model by OpenRouter model ID
   (e.g., "anthropic/claude-sonnet-4", "openai/gpt-4o",
   "google/gemini-2.0-flash-001"). query_single(), query_all(),
   get_embeddings(), list_available_models() methods. ModelResponse dataclass
   with parse_json() that handles markdown fences. Graceful error handling —
   one model failing doesn't block others. NO individual provider SDKs —
   everything goes through OpenRouter via httpx.
2. src/claw_review/clustering.py — Intent extraction using structured JSON
   prompts. extract_intents() queries all models per PR. generate_embeddings()
   via OpenAI text-embedding-3-small. cluster_intents() using scikit-learn
   DBSCAN with cosine distance matrix, configurable similarity threshold.
   Majority voting for category and affected_area. Singleton handling.
3. tests/test_models.py and tests/test_clustering.py — 35+ tests total,
   all model calls mocked. Include fixtures/sample_intents.json and
   fixtures/sample_embeddings.json.

IMPORTANT: Both teammates share PRData and ModelResponse dataclasses. Create
these shared types in __init__.py or a shared types module before teammates
start. Neither teammate touches the other's files.

PHASE 2 — After BOTH Phase 1 teammates complete and their tests pass:

Teammate "scoring-reports" should implement:
1. src/claw_review/scoring.py — 5-dimension quality scoring (code_quality,
   test_coverage, scope_discipline, breaking_risk, style_consistency) on 1-10
   scale. score_prs() uses model_pool.query_all() with structured JSON prompts.
   Consensus via weighted average, disagreement flagging when spread > 3.
   rank_within_clusters() attaches quality rankings to cluster PRs.
2. src/claw_review/alignment.py — Vision alignment scoring against repo docs
   (README, CONTRIBUTING, ARCHITECTURE). score_alignment() with MERGE/REVIEW/
   DISCUSS/CLOSE recommendations. Auto-CLOSE below threshold (4.0). Dedup of
   aligned_aspects and drift_concerns.
3. src/claw_review/report.py — Jinja2-templated HTML report with dark GitHub
   theme. Stats cards, duplicate cluster visualization with quality rankings,
   vision alignment flags, top quality PRs section. Also generate_json_report()
   for machine consumption.
4. tests/test_scoring.py, tests/test_alignment.py, tests/test_report.py —
   30+ tests total, all mocked.

After all teammates complete, I (Lead) will:
- Assemble cli.py (Click CLI with analyze, check, regenerate commands)
- Write integration tests (test_cli.py, test_integration.py)
- Run the full test suite
- Generate a sample report
- Validate the README

Quality gates for ALL teammates:
- All functions have type hints and docstrings
- Zero real API calls in tests
- ruff check passes with no violations
- Each module's tests pass independently
```

## Monitoring Progress

While the sprint is running:

```bash
# Watch test count grow
watch -n 30 'cd ~/claw-review && pytest tests/ -x -q 2>&1 | tail -5'

# Check which files exist
find src/claw_review -name "*.py" | sort

# Check test count per module
pytest tests/ --co -q 2>&1 | grep "test" | wc -l
```

## Phase Gate: Phase 1 → Phase 2

Before allowing @scoring-reports to start, verify:

```bash
# Both Phase 1 agents' tests pass
pytest tests/test_config.py tests/test_github_client.py -x -q
pytest tests/test_models.py tests/test_clustering.py -x -q

# Imports work
python -c "from claw_review.config import Config; print('✓ config')"
python -c "from claw_review.github_client import fetch_open_prs, PRData; print('✓ github')"
python -c "from claw_review.models import ModelPool, ModelResponse; print('✓ models')"
python -c "from claw_review.clustering import extract_intents, cluster_intents; print('✓ clustering')"
```

## Sprint Completion Validation

After all agents complete:

```bash
# Full test suite
pytest tests/ -x -q -v

# Type checking
mypy src/claw_review/ --ignore-missing-imports

# Lint
ruff check src/ tests/

# CLI smoke test (requires .env with real keys)
cp .env.example .env
# Edit .env with your API keys
claw-review check

# Full run (small scope for validation)
claw-review analyze --repo openclaw/openclaw --max-prs 5 --skip-alignment
```

## Expected Sprint Output

| Metric | Target |
|--------|--------|
| Source files | 8 Python modules |
| Test files | 8+ test files |
| Total tests | 95+ (30 + 35 + 30 + integration) |
| Test failures | 0 |
| Type errors | 0 |
| Lint violations | 0 |
| CLI commands | 3 (analyze, check, regenerate) |
| Report formats | 2 (HTML, JSON) |
