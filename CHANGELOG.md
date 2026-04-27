# Changelog

All notable changes to WikiLoom are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-26

Initial public release.

### Added

- **Ingestion pipeline** — `wikiloom ingest <file-or-url>` extracts text from
  PDFs (with a text layer), markdown, plain text, RST, Office docs (`.docx`,
  `.pptx`), code, and config files; URL ingestion via trafilatura for static
  HTML sites. Synthesis is JSON-validated against `ingest_response.json`.
- **Deterministic linker** — spaCy NER + rapidfuzz fuzzy matching with hybrid
  cosine re-ranking. Tiered confidence (high ≥95 / medium ≥85 / low ≥70 /
  ignored) controls auto-insert vs. pending review.
- **Page lifecycle** — `active → dormant → deprecated → purged` with per-type
  configurable dormant windows. `wikiloom merge`, `deprecate`, `purge`, and
  `dormant` cover the full lifecycle.
- **Human-edit protection** — `<!-- wikiloom:auto -->` marker preserves manual
  edits across re-ingest. `wikiloom save` commits human edits with a
  `human-edit:` prefix; writer commands block on uncommitted manual changes.
- **Query** — `wikiloom query "<question>"` returns answers grounded in wiki
  content with sources, confidence, relevance, and follow-up suggestions.
- **Query history** — every successful query appends to
  `_registry/query_history.json` (rolling cache, default 100 entries).
  `wikiloom queries` lists / shows / saves past results without re-running.
- **Auto-commit on every state change** — every command that modifies wiki
  content commits with a classifying prefix (`ingest:`, `lint:`, `merge:`,
  `dormant:`, `deprecate:`, `human-edit:`, `query:`, etc.).
- **Structural provenance** — every chunk persisted to SQLite with a stable
  `chunk_id = sha256(source_hash + chunk_index)`. Pages reference their
  contributing chunks under each entry in their `sources` frontmatter array.
- **Lint** — `wikiloom lint` health checks split into Warnings
  (broken links, duplicates, frontmatter, index drift, contradictions) and
  Tracking (orphans, dormant, stubs, promoted-from-update). `--fix` repairs
  mechanical issues.
- **Multi-provider LLM support** — Anthropic (default), OpenAI, Google, and
  Ollama via litellm. Split-model setup (`ingest_model` cheap, `query_model`
  strong) reduces cost.
- **Multi-provider embeddings** — fastembed (local, default), OpenAI, or
  sentence-transformers. Independent of the LLM provider.
- **Pre-flight budget enforcement** — refuses ingest runs that would exceed
  `[llm] monthly_budget_usd` (default $50/mo).
- **Stochastic-output retries with live progress** — synthesis retries up
  to `[llm] parse_retry_count` times (default 2) when the LLM returns
  unparseable JSON *or* JSON that fails schema validation (e.g. missing
  the `confidence` field on a proposed page — the failure mode hit
  during dogfooding on Haiku). Retries and final failures surface live
  (`↻` and `✗` glyphs) instead of being hidden in the post-run summary.
- **Per-chunk page context (Layer 1)** — semantic retrieval of top-K
  similar pages injected into the synthesis prompt to reduce duplicate page
  creation.
- **Initialization detects missing spaCy model** — `wikiloom init` prompts to
  download `en_core_web_sm` if absent. Honors `--no-interactive` for CI.
- **26 CLI commands** grouped by category: setup, ingest & write, read &
  explore, maintenance, deprecation. Run `wikiloom --help` for the index.

### Notes

- Requires Python 3.10–3.13. spaCy does not yet publish a wheel for 3.14.
- Lint exits 1 only on warnings; tracking-only runs exit 0 (CI-friendly).
- PyPI versions are immutable — bug-fix releases bump to `0.1.1`+.
