# 🦞 claw-review

**AI-powered PR triage using multi-model consensus.**

When your repo gets thousands of PRs and you can't tell which ones are duplicates, which is the best implementation, or which ones drift from your architecture — `claw-review` helps.

It queries multiple AI models independently, fuses their judgments, and produces a report showing:

- **Duplicate clusters** — PRs solving the same problem, grouped by semantic intent
- **Quality rankings** — Best implementation per cluster, scored across 5 dimensions
- **Vision alignment** — How well each PR fits your project's stated direction

## How it works

```
                    ┌──────────────────┐
                    │   GitHub API     │
                    │  (open PRs)      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Multi-Model     │
                    │  Consensus       │
                    │                  │
                    │  Claude ──┐      │
                    │  GPT-4o ──┼─ 🗳  │
                    │  Gemini ──┘      │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌─────────────┐ ┌───────────┐
     │  Intent    │  │  Quality    │ │  Vision   │
     │  Clusters  │  │  Scores     │ │  Align    │
     │  (de-dup)  │  │  (ranking)  │ │  (drift)  │
     └────────────┘  └─────────────┘ └───────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌────────────────┐
                    │  HTML + JSON   │
                    │  Report        │
                    └────────────────┘
```

Each model evaluates every PR independently. Their assessments are fused using consensus voting — if 2 out of 3 models agree on a PR's intent, category, or quality, that's the consensus. When models disagree significantly, the PR gets flagged for human review.

This isn't a single-model review bot. It's a **consensus system** — the same pattern used in safety-critical AI for high-stakes decisions.

## Quick start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/claw-review.git
cd claw-review

# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys (need GitHub + at least 2 model providers)

# Verify setup
claw-review check

# Run analysis
claw-review analyze --repo openclaw/openclaw --max-prs 50
```

## Requirements

- Python 3.11+
- GitHub personal access token (with `repo` scope for private repos, or public access)
- [OpenRouter](https://openrouter.ai) API key (single key gives access to Claude, GPT-4o, Gemini, Llama, Mistral, and 200+ other models)

## Usage

### Full analysis

```bash
# Analyze with all features
claw-review analyze --repo openclaw/openclaw

# Faster: skip vision alignment
claw-review analyze --repo openclaw/openclaw --skip-alignment

# Clustering only (no quality scoring)
claw-review analyze --repo openclaw/openclaw --skip-quality --skip-alignment

# Limit PR count (useful for testing)
claw-review analyze --repo openclaw/openclaw --max-prs 20

# JSON output only
claw-review analyze --repo openclaw/openclaw --json-only
```

### Regenerate report

```bash
# Regenerate HTML from existing JSON (no API calls needed)
claw-review regenerate claw-review-report.json
```

### Check configuration

```bash
claw-review check
```

## Output

The tool generates two files:

- **`claw-review-report.html`** — Visual report with duplicate clusters, quality rankings, and vision alignment flags. Open in any browser.
- **`claw-review-report.json`** — Machine-readable data for further processing.

## How consensus works

### Intent clustering (de-duplication)

1. Each model extracts a one-sentence intent from each PR
2. Intents are converted to embeddings (OpenAI text-embedding-3-small)
3. DBSCAN clusters PRs with >82% cosine similarity
4. Category and affected area are determined by majority vote

### Quality scoring

Each model scores PRs on 5 dimensions (1-10):

| Dimension | What it measures |
|-----------|-----------------|
| Code quality | Readability, error handling, edge cases |
| Test coverage | Are tests included? Edge cases covered? |
| Scope discipline | Does the PR stay focused? |
| Breaking risk | Likelihood of breaking existing functionality |
| Style consistency | Match with project conventions |

Consensus: weighted average. If models disagree by >3 points on any dimension, the PR is flagged for human review.

### Vision alignment

1. Loads the repo's README, CONTRIBUTING, and ARCHITECTURE docs
2. Each model scores how well each PR aligns with stated goals (1-10)
3. PRs scoring below 4 get a CLOSE recommendation
4. Specific drift concerns are listed

## Caching

PR data is cached locally in `.claw-review-cache/` to avoid redundant GitHub API calls. Use `--no-cache` to force fresh fetches.

## Cost estimate

For 50 PRs with 3 models:
- ~150 intent extraction calls (~$1-2)
- ~50-100 quality scoring calls (~$2-4) — only duplicates
- ~150 alignment calls (~$2-4)
- ~50 embedding calls (~$0.01)
- **Total: ~$5-10 per run**

Adjust `--max-prs` and use `--skip-alignment` or `--skip-quality` to reduce costs.

## License

MIT
