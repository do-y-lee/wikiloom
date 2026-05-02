# Changelog

All notable changes to WikiLoom are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] — 2026-05-01

### Fixed

- **Working tree stays clean after every state-changing command**
  — `lint --fix`, `reindex`, `relink`, `related --save/--link`,
  `purge`, and `rebuild-cache` appended a log event to
  `wiki/log.md` *after* their primary commit (so the event could
  carry that commit's hash) but never committed the resulting
  log.md change. The user-visible result was a dirty working
  tree after every wikiloom command — surfacing in `git status`,
  failing dirty-tree preflights on the next command, and forcing
  manual `git add wiki/log.md && git commit` cleanups. Each
  affected command now lands the log.md change in a small
  follow-up commit via `_commit_log_tail` (renamed from
  `_commit_merge_log_tail`, which already used this pattern for
  merge). Ingest already had the equivalent tail commit; the
  stale "acceptable staleness" comment in
  `wikiloom/ingest/processor.py` is updated to match.

## [0.1.5] — 2026-05-01

### Performance

- **Per-chunk page-context retrieval reuses the cached embedding
  matrix** — `retrieve_candidates_for_chunk` opened a fresh SQLite
  connection per chunk and Python-looped `deserialize_embedding`
  + `cosine_similarity` over every page, completely bypassing
  the `SQLiteCache._emb_matrix` cache added in 0.1.2. On a 5K-
  page wiki ingesting 50 chunks that's 250K deserializes and
  250K cosine calls. Now delegates to `SQLiteCache.semantic_search`
  and `run_synthesis` shares one cache instance across every
  worker — one matmul per chunk against the cached `(M, D)`
  matrix regardless of wiki size. `semantic_search` gains an
  `exclude_statuses` kwarg that masks deprecated rows at the
  matmul step, preserving the previous SQL filter's semantics.
- **Module-level `load_embedder` cache** — every prior call
  reloaded the 100–300 MB ONNX/SBert model from disk (~1–3 s
  cold). A single `wikiloom ingest` paid this 3× (synthesis +
  linker + cache sync); a `dormant review` over N pages paid
  it N× because `_sync_cache` fires per candidate. Now cached
  at module level keyed by `(provider, model)`; subsequent
  calls within one process return the same instance. Tests
  get isolation via a new `clear_embedder_cache()` helper plus
  an autouse fixture in `tests/conftest.py`.
- **Event log uses true append mode** — `append_event` read
  `wiki/log.md` in full, concatenated the new entry, and
  rewrote the file on every event. With ingest, lint, merge,
  deprecate, relink, and save all emitting events, that was
  O(N) per call and O(N²) cumulative I/O over a project's
  lifetime. Now opens in append mode and writes only the new
  entry; header-on-first-write is preserved.
- **SQLite cache opens with WAL + perf PRAGMAs** — the cache
  connection ran with default `journal_mode=DELETE` and
  `synchronous=FULL`, fsyncing twice per transaction. Switched
  to `journal_mode=WAL`, `synchronous=NORMAL`,
  `temp_store=MEMORY`, and `mmap_size=256MB`. Roughly 5–10×
  faster writes on rebuild paths and readers no longer block
  on writers. The cache is regenerable via `wikiloom
  rebuild-cache`, so the durability tradeoff is acceptable.
- **Cached alias keys list in `LinkingEngine`** — `_resolve`
  and `_resolve_top_k` rebuilt `list(self.alias_map.keys())`
  on every span. With M aliases and N spans across a link
  pass that's N list reconstructions of size M per pass.
  Built once in `__init__` and refreshed alongside the alias
  map in `refresh()`.
- **Registry hoisted out of `_dormant_review` loop** —
  `Registry(project / "_registry")` was rebuilt on every
  iteration, re-reading and re-parsing `manifest.json` per
  candidate. A 100-candidate review session ran 100 manifest
  loads. Now built once next to `BacklinkRegistry`.
- **Batched cache inserts with `executemany`** — `full_rebuild`
  and `_incremental_sync` inserted pages, FTS rows, aliases,
  and backlinks one `conn.execute()` at a time inside Python
  loops. On a 1K-page rebuild that's roughly 5K Python→C
  round-trips just from boundary crossing. Now builds the
  row lists up front and dispatches each table with one
  `executemany` call.

### Upgrade notes

- WAL mode creates `wiki.db-wal` and `wiki.db-shm` sidecar
  files next to `_registry/wiki.db`. The scaffolded
  `.gitignore` for new projects now uses `_registry/wiki.db*`
  to cover the sidecars. **Existing projects need to widen
  their `.gitignore` line manually** — change
  `_registry/wiki.db` to `_registry/wiki.db*`. Otherwise the
  sidecars will appear as untracked files in `git status`
  after the next ingest.

### Documentation

- New **Performance invariants** section in `CLAUDE.md` —
  each bullet anchors a rule to a real cache or helper
  (`SQLiteCache.semantic_search`, the module-level
  `load_embedder` cache, the long-lived `SQLiteCache._conn`
  with its WAL PRAGMAs, the alias keys cache, the
  `executemany` rule, the explicit "don't construct
  `Registry` / `SQLiteCache` / `load_embedder` /
  `BacklinkRegistry` inside loops" list) so future
  contributors reach for the existing fast path instead of
  reinventing a slow one.

## [0.1.4] — 2026-04-30

### Performance

- **Batched spaCy in `link_all` via `nlp.pipe()`** — the linker's
  multi-page entry point now reads bodies up front and runs spaCy
  in one batched call instead of per-page `nlp(body)`. Same output;
  amortizes pipeline overhead across the batch. Roughly 2–3× faster
  NLP time on ingest-tail link runs. `link_page` accepts an
  optional pre-computed `Doc` so callers that already have one
  (notably `link_all`) avoid the redundant parse; direct callers
  keep working unchanged.
- **Skip stub pages in the linker** — pages with `status: stub`
  carry placeholder bodies (`*Stub — awaiting content.*`) that
  produce no useful link candidates. `link_page` now short-circuits
  before NLP, and `link_all` drops them from the `nlp.pipe` batch
  entirely. Source pages are intentionally still linked because
  their summaries reference real concepts.

### Fixed

- **`wikiloom log` crash on real git output** — the 0.1.3 commit-hash
  backfill parsed `log.md` timestamps as naive datetimes (after
  stripping the trailing `Z`) but git's `%aI` format is
  offset-aware. Subtracting them threw `TypeError: can't subtract
  offset-naive and offset-aware datetimes` on every `wikiloom log`
  invocation that contained a query event. Normalize `Z` to
  `+00:00` so both ends are timezone-aware. Regression test added.

## [0.1.3] — 2026-04-29

### Performance

- **Query path reads from the cache, not the JSON files** — the linked-
  page ranker (`_rank_linked_pages`) used to instantiate
  `BacklinkRegistry` and `Registry` on every query, parsing
  `backlinks.json` and `manifest.json` from disk. Both stores are
  already mirrored into the SQLite cache, so the ranker now joins on
  the cache's `backlinks` and `pages` tables directly. Saves the JSON
  parse cost and removes a per-query handshake from the retrieval
  path.
- **Batched ingest state writes** — `IngestState.mark_chunk_done` /
  `mark_chunk_failed` accept a `flush` keyword. Synthesis's post-
  gather loop now defers persistence with `flush=False` and flushes
  once at the end, replacing N per-chunk JSON writes. Crash safety
  is preserved because the loop runs after parallel synthesis is
  already complete.
- **Linker batches embeddings per page** — `_link_text` collects
  context windows for every candidate span on a page and issues one
  `embed_texts` call instead of N per-span calls in
  `_resolve_with_rerank`. Falls back to the per-span path when the
  batch call raises so behavior degrades gracefully on embedder
  failures.
- **Incremental index rebuilds** — `IndexUpdater.rebuild_for_pages`
  derives affected categories from a touched-page list and only
  rewrites those sub-index files plus the root index, instead of
  walking every category every time. Wired into the ingest hot path
  (post-synthesis indexing) and the post-ingest auto-merge path.
  Other call sites still use `rebuild_all` when the touched set is
  unknown (e.g., post-merge relink, `wikiloom rebuild-cache`).

### Fixed

- **`wikiloom log` now shows commit hashes for query events** — query
  events are written to `log.md` before the commit lands, so the
  hash couldn't be baked into the file at write time. The log
  command now backfills missing hashes post-hoc by matching commit
  subjects (`<event_type>: <description>`) against a recent-commits
  window. Pure display-side change; `log.md` itself is untouched.

## [0.1.2] — 2026-04-29

### Performance

- **Vectorized `cosine_similarity`** — replaces a triple Python loop
  with a numpy implementation. Speeds up every caller (linker rerank,
  page-context retrieval, fallback paths) without changing the
  function signature or the on-disk embedding BLOB format.
- **Cached embedding matrix in `SQLiteCache.semantic_search`** — the
  first query deserializes every page embedding into one contiguous
  numpy matrix; subsequent queries reuse it via a single matmul
  instead of re-reading SQLite, deserializing, and looping in Python.
  Invalidated automatically by `full_rebuild` and incremental sync,
  so callers see no behavior change. Roughly 10–100× speedup on
  semantic fallback for wikis with thousands of pages.
- **Batched duplicate detection** — `find_duplicates` replaces an
  O(n²) Python loop calling `cosine_similarity` and
  `fuzz.token_sort_ratio` per pair with two batched matrix passes
  (`process.cdist` for slug similarity, one matmul for embedding
  similarity). Same input/output contract; same scoring rules.
- **Single SQLite connection per `SQLiteCache`** — the cache now
  holds a long-lived connection guarded by an `RLock` instead of
  opening and closing a fresh connection on every read/write. Saves
  a few ms per call and removes a per-query handshake from the
  query path.

### Notes

- Adds `numpy>=1.24` as an explicit runtime dependency. It was
  already present transitively (via `fastembed` /
  `sentence-transformers` / `spacy`); the move makes the dependency
  visible since `wikiloom` now imports it directly.

## [0.1.1] — 2026-04-28

### Fixed

- **Durable fastembed cache** — `FastEmbedBackend` now stores ONNX
  models under `platformdirs` (`~/Library/Caches/wikiloom/fastembed`
  on macOS, `~/.cache/wikiloom/fastembed` on Linux,
  `%LOCALAPPDATA%\wikiloom\Cache\fastembed` on Windows) instead of
  `tempfile.gettempdir()`. macOS reaped individual files inside the
  temp dir on a schedule, leaving the snapshot directory present but
  the large ONNX file missing — fastembed crashed with a cryptic
  `NO_SUCHFILE` on next load.

### Added

- **`wikiloom init` prefetches the embedding model** — same
  Y/n / `--no-interactive` / non-TTY semantics as the spaCy
  download. Declining points the user at the `[embeddings]` section
  of `wikiloom.toml` so they can disable or swap the provider.
- **`platformdirs` runtime dependency** for the cache routing above.

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
  `dormant` cover the full lifecycle. `deprecate --superseded-by <Y>`
  automatically rewrites every inbound `[[X]]` wikilink on active /
  dormant / stub pages to `[[Y]]`, matching what `merge` does — no
  broken links left behind. Without `--superseded-by`, the preview
  surfaces a warning naming the active pages that will end up with
  broken links so the user can re-run with a replacement.
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
