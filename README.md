# WikiLoom

WikiLoom turns raw documents into a persistent, compounding knowledge base. Ingest a PDF, markdown file, or URL; the LLM reads the source and writes structured wiki pages with deterministic linking, structural provenance, and human-edit protection.

## How it works

```
source file
    |
    v
[extract + chunk]  -->  [LLM synthesis]  -->  [deterministic linker]  -->  wiki/
    |                        |                        |
    |                    JSON output             spaCy NER +
    |                    validated               rapidfuzz matching
    |                    per schema               inserts [[wikilinks]]
    v                        v                        v
raw/ copy              pages written            backlinks rebuilt
                       with chunk_ids           stubs auto-created
                       in frontmatter           pending.json for
                                                low-confidence links
```

The LLM handles judgment (reading sources, extracting claims, assessing confidence). Everything after the LLM call is deterministic: linking, backlink graph, index regeneration, git commit. Hand-authored content above the `<!-- wikiloom:auto -->` marker is preserved across re-ingests.

## Installation

```bash
# Clone and install
git clone <repo-url> && cd wikiloom
pip install -e ".[dev]"

# Download the spaCy model (required for the linking engine)
python -m spacy download en_core_web_sm
```

## Quick start

```bash
# 1. Create a project
wikiloom init my-wiki --domain "AI research"
cd my-wiki

# 2. Set your LLM provider key
export ANTHROPIC_API_KEY=sk-...

# 3. Ingest a source
wikiloom ingest path/to/paper.pdf

# 4. See what was created
wikiloom status
ls wiki/concepts/ wiki/entities/ wiki/sources/

# 5. Ask a question
wikiloom query "What are the key contributions of this paper?"

# 6. Save the answer as a synthesis page
wikiloom query "Compare flash attention to standard attention" --save
```

## Commands

### Core

| Command | Description |
|---|---|
| `wikiloom init <name>` | Create a new project with full directory structure |
| `wikiloom ingest <source>` | Ingest a file or URL into the wiki |
| `wikiloom query <question>` | Ask a question grounded in wiki content |

### Maintenance

| Command | Description |
|---|---|
| `wikiloom lint [--fix]` | Run health checks; `--fix` applies auto-repairs |
| `wikiloom reindex` | Regenerate all index files |
| `wikiloom protect [--sync]` | Reconcile human-edit flags with git history |
| `wikiloom rebuild-cache` | Regenerate the SQLite query cache from disk |
| `wikiloom review [--accept-all \| --clear]` | Action low-confidence link candidates |

### Observability

| Command | Description |
|---|---|
| `wikiloom status` | Project overview: pages, sources, tokens, cost |
| `wikiloom log [-n N]` | Recent events from the wiki event log |
| `wikiloom cost` | Token usage and spend breakdown by event type |

### Provenance

| Command | Description |
|---|---|
| `wikiloom source <chunk_id>` | Show the exact source text the LLM saw for a chunk |

Every synthesized page carries `chunk_ids` in its frontmatter. Run `wikiloom source <id>` on any chunk_id to see the raw input text that produced that page — structural provenance without trusting the LLM's self-attribution.

## Project structure

```
my-wiki/
  wikiloom.toml          # Project config (LLM model, budget, thresholds)
  .wikiloom/             # Prompt templates + output format schemas
    prompts/
      ingest.md          # Customizable ingest prompt
      query.md           # Customizable query prompt
    output_formats/
      ingest_response.json
      query_response.json
  wiki/                  # The wiki itself (markdown + frontmatter)
    index.md
    log.md               # Event log (tokens, cost, commits)
    concepts/
    entities/
    sources/
    syntheses/
    decisions/
    archive/
  raw/                   # Copies of ingested source files
    papers/
    articles/
  _registry/             # Derived state (manifest, backlinks, cache)
    manifest.json        # Page registry
    backlinks.json       # Wikilink graph
    pending.json         # Low-confidence link candidates
    sources.json         # Content-addressed source catalog
    wiki.db              # SQLite query cache (git-ignored)
```

## Key concepts

**Human-edit protection.** Every auto-generated page has a `<!-- wikiloom:auto -->` marker. Two layers of protection work together:

- **The `human-edit:` commit prefix.** When you run `wikiloom save`, the resulting commit is classified as human-authored. `lint --fix` skips the page; re-synthesis tools leave it alone. This protection is "soft" — it lasts until the next auto-action (e.g. a re-ingest commit) clears it.
- **The marker is the durable boundary.** Content **above** `<!-- wikiloom:auto -->` survives every operation, including `wikiloom ingest <file> --force` (the only command that wipes the auto region). Content below the marker is LLM-generated and may be replaced on `--force` re-ingest.

**Tip — pinning personal notes:** to make a note that survives every future re-synthesis of a page (including `--force`), put it above the marker. Example:

```markdown
# Transformer

> **My note:** the original paper used post-norm; modern impls use pre-norm.

<!-- wikiloom:auto -->

## Architecture
... (LLM-generated content)
```

Normal `wikiloom ingest <new-source>` updates only **append** to the auto region — your edits below the marker also survive in that case. The marker only matters when you re-synthesize from scratch via `--force`.

**Tiered linking confidence.** The linking engine scores each potential wikilink:
- **High (>= 95):** auto-inserted
- **Medium (>= 85):** auto-inserted, flagged in backlinks
- **Low (>= 70):** deferred to `pending.json` for manual review via `wikiloom review`
- **Below 70:** ignored

**Structural provenance.** Each chunk of a source document is persisted to the SQLite cache with a stable `chunk_id` derived from `sha256(source_hash + chunk_index)`. Pages reference their contributing chunks under each entry in their `sources` frontmatter array — every source dict carries its own `chunk_ids` list, so you can trace every claim back to a specific chunk of a specific document. Use `wikiloom show <page> --field sources` to inspect, or `wikiloom source <chunk_id>` to view the original chunk text.

**Budget enforcement.** Before running LLM synthesis, ingest estimates the token cost and refuses if it would exceed `[llm] monthly_budget_usd` in `wikiloom.toml`. Disable with `[ingest] enable_budget_check = false`.

## Configuration

`wikiloom.toml` lives at the project root:

```toml
[project]
name = "my-wiki"
domain = "AI research"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
max_tokens_per_operation = 8000
monthly_budget_usd = 50.0

[linking]
auto_create_stubs = true
high_confidence_threshold = 95
medium_confidence_threshold = 85
low_confidence_threshold = 70

[ingest]
max_file_size_mb = 50
min_extracted_chars = 16
enable_budget_check = true

[staleness]
default_window_days = 90
```

## Development

### Running tests

```bash
pytest                    # Full suite (live-API tests skipped by default)
pytest -m live            # Run live-API tests (requires ANTHROPIC_API_KEY)
pytest tests/test_llm.py  # Just the LLM client unit tests
```

### Test count

323 tests across 17 test modules. All deterministic; no live API calls in the default suite.

## License

MIT
