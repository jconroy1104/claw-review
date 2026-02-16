# 🦞 claw-review

**Domain-agnostic multi-model consensus platform.**

A pluggable platform that queries multiple AI models independently, fuses their judgments using consensus voting, and produces interactive reports. Originally built for GitHub PR triage, now supports any domain through adapters — cybersecurity alert triage, fraud detection, and more.

**Core capabilities:**
- **Duplicate clustering** — Group items solving the same problem using semantic intent + DBSCAN
- **Quality ranking** — Score items across configurable dimensions with multi-model consensus
- **Vision alignment** — Flag items that drift from stated policies or goals
- **Interactive dashboard** — Filter, sort, and explore results in a dark-themed web UI

**Built-in domains:**
- **GitHub PRs** — De-duplicate PRs, rank implementations, flag architectural drift
- **Cybersecurity** — Correlate SIEM alerts by attack campaign, score threat severity
- **Fraud Detection** — Group suspicious transactions, score anomaly patterns

## How it works

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

Each model evaluates every item independently. Their assessments are fused using consensus voting — if 2 out of 3 models agree on intent, category, or quality, that's the consensus. When models disagree significantly, the item gets flagged for human review.

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

### Model presets

Choose a cost/quality tradeoff with `--preset`:

```bash
# List available presets
claw-review presets

# Fast & cheap (~$0.15-0.30 per 100 PRs)
claw-review analyze --repo openclaw/openclaw --preset fast

# Balanced — recommended (~$0.30-0.60 per 100 PRs)
claw-review analyze --repo openclaw/openclaw --preset balanced

# Thorough — highest quality (~$1.50-2.50 per 100 PRs)
claw-review analyze --repo openclaw/openclaw --preset thorough
```

| Preset | Models | Est. Cost/100 PRs |
|--------|--------|-------------------|
| fast | Llama 3.1 70B, Mistral Large, Gemini Flash | $0.15-0.30 |
| balanced | Claude Sonnet, GPT-4o-mini, Gemini Flash | $0.30-0.60 |
| thorough | Claude Sonnet, GPT-4o, Gemini Flash | $1.50-2.50 |

### Cost estimation

```bash
# Estimate cost before running (no API calls)
claw-review estimate --repo openclaw/openclaw --preset balanced --max-prs 100

# Set a budget limit — analysis halts if exceeded
claw-review analyze --repo openclaw/openclaw --budget 5.00
```

### Batch processing & incremental analysis

For large repos (1,000+ PRs), use batching and incremental mode:

```bash
# Process in batches of 50 (default) with checkpointing
claw-review analyze --repo openclaw/openclaw --batch-size 50

# Incremental: only analyze new PRs (enabled by default)
claw-review analyze --repo openclaw/openclaw

# Force re-analysis of all PRs
claw-review analyze --repo openclaw/openclaw --force

# Check what's been analyzed so far
claw-review status --repo openclaw/openclaw

# Merge results from multiple runs
claw-review merge report1.json report2.json -o combined-report
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

### Interactive dashboard

Generate a self-contained HTML dashboard from any report:

```bash
# Generate static dashboard (no server required)
claw-review dashboard --static -o dashboard.html

# Or use the Python API
python -c "
from claw_review.dashboard import generate_static_dashboard
generate_static_dashboard(['claw-review-report.json'], output='dashboard.html')
"
```

The dashboard features:
- Summary cards (PRs analyzed, clusters, drift flags, cost)
- 5 tabs: Overview, Clusters, Quality, Alignment, Cost
- Client-side filtering, sorting, and search
- Dark theme matching GitHub's design
- Zero server dependencies — deploy directly to GitHub Pages

### GitHub App integration

Set up automated PR triage with webhooks:

```bash
# Generate a GitHub Actions workflow for your repo
python -c "
from claw_review.github.actions import save_workflow
save_workflow('.', preset='balanced')
"
# Creates .github/workflows/claw-review.yml
```

The GitHub integration includes:
- **Webhook receiver** — FastAPI endpoint for `pull_request` events with HMAC-SHA256 verification
- **PR comment bot** — Posts consensus results directly on PRs (update-not-spam pattern)
- **GitHub App auth** — JWT/RS256 authentication with installation token caching
- **Actions workflow** — Auto-generated CI/CD for PR-triggered analysis

## Output

The tool generates:

- **`claw-review-report.html`** — Visual report with duplicate clusters, quality rankings, and vision alignment flags. Open in any browser.
- **`claw-review-report.json`** — Machine-readable data for further processing.
- **`dashboard.html`** — Interactive dashboard with filtering, sorting, and exploration (via `--static` flag).

## Platform architecture

claw-review is built on a domain-agnostic consensus engine with pluggable adapters:

| Component | Description |
|-----------|-------------|
| `ConsensusEngine` | Orchestrates intent extraction, clustering, scoring, alignment for any domain |
| `DataAdapter` | Protocol for fetching items from any data source |
| `DomainConfig` | Prompts, dimensions, thresholds, and recommendation levels per domain |
| `AdapterRegistry` | Runtime discovery and registration of domain adapters |

### Built-in domains

| Domain | Adapter | Scoring Dimensions | Recommendations |
|--------|---------|-------------------|-----------------|
| `github-pr` | GitHub API (PRs) | code_quality, test_coverage, scope_discipline, breaking_risk, style_consistency | MERGE, REVIEW, DISCUSS, CLOSE |
| `cybersecurity` | SIEM alerts (JSON) | threat_severity, confidence, attack_sophistication, asset_criticality, actionability | BLOCK, INVESTIGATE, MONITOR, DISMISS |
| `fraud-detection` | Transactions (JSON/CSV) | anomaly_score, pattern_match, velocity_risk, geographic_risk, amount_deviation | APPROVE, FLAG, HOLD, BLOCK |

### How consensus works

**Intent clustering (de-duplication):**
1. Each model extracts a one-sentence intent from each item
2. Intents are converted to embeddings (OpenAI text-embedding-3-small)
3. DBSCAN clusters items with configurable cosine similarity threshold
4. Category and affected area are determined by majority vote

**Quality scoring:**
Each model scores items on domain-specific dimensions (1-10). Consensus is the arithmetic mean. If models disagree by more than the disagreement threshold on any dimension, the item is flagged for human review.

**Vision/policy alignment:**
1. Loads context documents (README, security policy, fraud rules, etc.)
2. Each model scores how well each item aligns with stated goals (1-10)
3. Items scoring below the rejection threshold get flagged
4. Specific drift concerns are listed

## Caching

PR data is cached locally in `.claw-review-cache/` to avoid redundant GitHub API calls. Use `--no-cache` to force fresh fetches.

## Cost

Use `claw-review estimate` to get a cost projection before running. Costs depend on the preset:

| Preset | 100 PRs | 1,000 PRs | 6,000 PRs |
|--------|---------|-----------|-----------|
| fast | $0.15-0.30 | $1.50-3.00 | $9-18 |
| balanced | $0.30-0.60 | $3-6 | $18-36 |
| thorough | $1.50-2.50 | $15-25 | $90-150 |

Use `--skip-alignment`, `--skip-quality`, or `--budget` to control spend.

## All commands

| Command | Description |
|---------|-------------|
| `claw-review analyze` | Run full PR analysis |
| `claw-review check` | Verify configuration and API access |
| `claw-review presets` | List available model presets |
| `claw-review estimate` | Estimate cost without API calls |
| `claw-review status` | Show analysis state for a repo |
| `claw-review merge` | Combine multiple JSON reports |
| `claw-review regenerate` | Regenerate HTML from JSON |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (457 tests, 99% coverage)
pytest tests/ -x -q

# Lint
ruff check src/ tests/

# Type check
mypy src/claw_review/ --ignore-missing-imports
```

## Sprint history

| Sprint | Focus | Tests | Key additions |
|--------|-------|-------|---------------|
| 1 | Core pipeline | 197 | GitHub API, multi-model consensus, DBSCAN clustering, HTML reports |
| 2 | Performance & scale | 298 | Async parallel, model presets, cost tracking, batch processing |
| 3 | Platform & integration | 457 | Domain-agnostic engine, adapters, GitHub App, dashboard |

## License

MIT
