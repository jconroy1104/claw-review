# claw-review Sprint 2 — Agent Team Launch Guide

## Pre-Launch Checklist

- [ ] Sprint 1 complete: 197 tests, 99% coverage, 0 failures
- [ ] Live report published: jconroy1104.github.io/claw-review/claw-review-report.html
- [ ] Navigate to project: `cd ~/projects/claw-review`
- [ ] Append Sprint 2 additions to AGENTS.md and CLAUDE.md (see commands below)
- [ ] Open a tmux session: `tmux new -s claw-review-s2`
- [ ] Set environment variable: `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`
- [ ] Launch Claude Code: `claude`

## File Setup Commands

```bash
cd ~/projects/claw-review

# Copy Sprint 2 addition files (adjust download path if needed)
cp "/mnt/c/Users/Joseph/Downloads/AGENTS_Sprint2_Addition.md" .
cp "/mnt/c/Users/Joseph/Downloads/CLAUDE_MD_Sprint2_Addition.md" .

# Append to existing files
echo "" >> AGENTS.md && cat AGENTS_Sprint2_Addition.md >> AGENTS.md
echo "" >> CLAUDE.md && cat CLAUDE_MD_Sprint2_Addition.md >> CLAUDE.md

# Clean up
rm AGENTS_Sprint2_Addition.md CLAUDE_MD_Sprint2_Addition.md

# Commit
git add -A
git commit -m "Add Sprint 2 agent definitions: async, cost optimization, scale"
```

## Spawn Prompt

Paste this into Claude Code after launching:

```
Read CLAUDE.md and AGENTS.md to understand the project context. We completed
Sprint 1 with 197 tests at 99% coverage. Now we're starting Sprint 2 to add
async parallelism, cost optimization, and scale support. The Sprint 2 section
is at the bottom of both files.

PHASE 1 — Launch these two teammates in parallel:

Teammate "async-engine" should:
1. Rewrite src/claw_review/models.py to use httpx.AsyncClient with connection
   pooling (max_connections=20, keep-alive). Convert query_single(), query_all(),
   and get_embeddings() to async. Use asyncio.gather() in query_all() so all
   models fire concurrently. Add asyncio.Semaphore for concurrency control
   (configurable limit, default 10). Add retry with exponential backoff on
   429/500/503 responses (max 3 retries). Session-level client created once
   and reused.
2. Update src/claw_review/clustering.py, src/claw_review/scoring.py, and
   src/claw_review/alignment.py to use async model calls. Process multiple
   PRs concurrently using batch_size parameter (default 5). Add progress
   callback support for Rich progress bars.
3. Update all affected tests (test_models.py, test_clustering.py,
   test_scoring.py, test_alignment.py) to use pytest-asyncio. Add new tests
   for concurrency, retry logic, semaphore behavior, and error isolation.
   25+ new/updated tests. Do NOT delete any existing tests — update them
   to async.

Teammate "cost-optimizer" should:
1. Create src/claw_review/costs.py — CostTracker class that accumulates
   per-model token usage and cost from OpenRouter usage response data.
   Token price table for common models (GPT-4o, GPT-4o-mini, Claude Sonnet,
   Gemini Flash, Llama 3.1 70B, Mistral Large). estimate_cost() for dry-run
   estimation. format_cost_report() for Rich table output. Budget limit that
   halts analysis when exceeded (saving partial results).
2. Update src/claw_review/config.py — Add PRESETS dict with three tiers:
   "fast" (Llama 70B + Mistral Large + Gemini Flash, ~$0.15-0.30/100 PRs),
   "balanced" (Claude Sonnet + GPT-4o-mini + Gemini Flash, ~$0.30-0.60/100 PRs),
   "thorough" (Claude Sonnet + GPT-4o + Gemini Flash, Sprint 1 default).
   --preset flag overrides MODELS env var.
3. Update src/claw_review/cli.py — Add --preset flag to analyze command.
   Add --budget flag. Add "claw-review estimate" command (dry run cost calc).
   Add "claw-review presets" command (list presets with descriptions).
   Print cost summary at end of analysis.
4. Tests: test_costs.py (NEW, 12+ tests), update test_config.py and
   test_cli.py. 20+ new/updated tests total.

IMPORTANT: @async-engine owns models.py, clustering.py, scoring.py,
alignment.py and their tests. @cost-optimizer owns costs.py (new), config.py,
cli.py and their tests. No file overlap.

PHASE 2 — After BOTH Phase 1 teammates complete and their tests pass:

Teammate "scale-layer" should:
1. Create src/claw_review/state.py — AnalysisState class stored as JSON in
   .claw-review-state/{repo_hash}.json. Tracks analyzed PR numbers, timestamps,
   model config. get_unanalyzed_prs() returns only new PRs. Atomic file writes
   (write to temp, rename). --force flag ignores state.
2. Create src/claw_review/batch.py — BatchProcessor that splits PR lists into
   configurable batches (default 50). Checkpoint after each batch (save partial
   to state). Resume from last checkpoint if interrupted. Progress display:
   "Batch 3/12: PRs 101-150 (cost: $0.45)".
3. Update src/claw_review/report.py — Add merge_reports() to combine multiple
   JSON reports. Re-cluster across merged results. Re-rank quality globally.
4. Update src/claw_review/cli.py — Add --batch-size, --incremental (default ON),
   --force flags. Add "claw-review merge" command. Add "claw-review status" command.
5. Tests: test_state.py (NEW), test_batch.py (NEW), update test_report.py and
   test_cli.py. 15+ new/updated tests.

After all teammates complete, I (Lead) will:
- Run the full test suite (target: 250+ tests, 99%+ coverage)
- Integration test: full async pipeline with cost tracking
- Live validation: re-run against openclaw/openclaw with --preset balanced
- Compare runtime and cost vs Sprint 1
- Update README with new commands
- Push to GitHub and update GitHub Pages

Quality gates for ALL teammates:
- All functions have type hints and docstrings
- Zero real API calls in tests
- ruff check passes with no violations
- pytest-asyncio for all async tests
- Each module's tests pass independently
- No reduction in existing test count or coverage
```

## Monitoring Progress

```bash
# Watch test count grow
watch -n 30 'cd ~/projects/claw-review && pytest tests/ -x -q 2>&1 | tail -5'

# Check new files
ls -la src/claw_review/costs.py src/claw_review/state.py src/claw_review/batch.py 2>/dev/null

# Verify async conversion
grep -c "async def" src/claw_review/models.py
```

## Phase Gate: Phase 1 → Phase 2

```bash
# Both Phase 1 agents' tests pass
pytest tests/test_models.py tests/test_clustering.py tests/test_scoring.py tests/test_alignment.py -x -q
pytest tests/test_costs.py tests/test_config.py tests/test_cli.py -x -q

# Async imports work
python3 -c "from claw_review.models import ModelPool; print('✓ async models')"
python3 -c "from claw_review.costs import CostTracker; print('✓ cost tracker')"
python3 -c "from claw_review.config import PRESETS; print('✓ presets')"
```

## Sprint Completion Validation

```bash
# Full test suite
pytest tests/ -x -q -v

# Coverage check
pytest tests/ --cov=claw_review --cov-report=term-missing

# Type checking
mypy src/claw_review/ --ignore-missing-imports

# Lint
ruff check src/ tests/

# CLI smoke tests
claw-review presets
claw-review estimate --repo openclaw/openclaw --preset balanced --max-prs 100
claw-review check

# Live comparison run
time claw-review analyze --repo openclaw/openclaw --max-prs 100 --preset balanced
```

## Expected Sprint Output

| Metric | Sprint 1 | Sprint 2 Target |
|--------|----------|----------------|
| Source files | 9 | 12 (+costs, state, batch) |
| Total tests | 197 | 250+ |
| Coverage | 99% | 99%+ |
| Test failures | 0 | 0 |
| Runtime (100 PRs) | ~20 min | <5 min |
| Cost (100 PRs) | $1.83 | <$0.50 (balanced) |
| Max PRs supported | ~200 | 6,000+ |
| CLI commands | 3 | 7 |
