# Changelog

All notable changes to WikiLoom are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **MCP server (`wikiloom.mcp`)** — agent-callable surface exposing
  WikiLoom's retrieval primitives to MCP clients (Claude Desktop,
  Claude Code, the MCP Inspector, etc.). Seven tools in a 3-layer
  pattern: cheap routers (`search_pages`, `search_chunks` — small
  previews + ids), expensive payloads (`get_pages`, `get_chunks` —
  full bodies), the one-shot orchestrator `get_context` (page
  router → token-budgeted chunks), and two graph hops
  (`get_outbound_links` for outbound edges, `get_backlinks` for
  the inbound pages that cite a given page). Tool docstrings
  teach the router-vs-payload pattern and suggest loop
  strategies for refining results. Built
  on FastMCP from the official `mcp` Python SDK; stdio transport;
  cache + embedder loaded once at startup. Pydantic output models
  live at the boundary (`wikiloom.mcp.models`) so the JSON schema
  the agent sees carries per-field `description=` strings —
  internal types (`Citation`, `ContextResult`, `PageHit`,
  `StoredChunk`) stay frozen dataclasses. Server fails loud at
  startup if no embedder is configured rather than booting with a
  degraded surface that would silently return empty results.

- **`wikiloom mcp` CLI subcommand** — launches the stdio MCP server
  for a project. `--project PATH` defaults to the current directory;
  `--print-config` emits a copy-pasteable Claude Desktop / Claude
  Code config block with absolute paths pre-filled and
  `sys.executable` baked in, removing the two most common wiring
  failure modes (wrong Python interpreter, relative project path).
  Ships `wikiloom/__main__.py` so the printed config can use
  `python -m wikiloom mcp …` instead of relying on the `wikiloom`
  script being on PATH when the MCP client spawns the server.

- **`Citation.token_estimate`** — citations now carry an optional
  approximate token count, threaded through `_hydrate` from
  `chunks.token_estimate`. Lets retrieval callers (notably
  `get_context`) pack chunk lists by budget without re-tokenizing.
  Default `None` for legacy rows; preserves byte-identical behavior
  for existing callers.

- **`ChunkStore.get_chunks(ids)`** — batch fetcher that returns a
  `dict[str, StoredChunk]` keyed by `chunk_id` (missing ids are
  omitted). One SELECT with `WHERE chunk_id IN (...)`, mirroring
  `get_chunks_for_source`'s shape. Backs the MCP `get_chunks` tool
  wrapper.

- **Hybrid context lane (`wikiloom.context.get_context`)** — the
  default agent path for goal-shaped queries. Embeds the goal
  once, routes to the top-N most similar synthesized pages via
  `cache.semantic_search` (cosine over the cached page-embedding
  matrix, deprecated pages excluded), then reranks chunks within
  those pages via `search_chunks` scoped to the routed page_ids.
  Returns a `ContextResult(pages, citations)`: `pages` is a list
  of `PageHit` records (`page_id`, `type`, `title`, `summary`,
  `similarity`) for explainability; `citations` is a list of
  `Citation` records identical to `search_chunks`'s output. No
  LLM call, no second rerank — pure deterministic retrieval. The
  MCP `get_context` tool will wrap this directly.

- **Chunk-direct retrieval lane (`wikiloom.retrieval.search_chunks`)** —
  hybrid BM25 + vector retrieval over verbatim chunks, fused with
  Reciprocal Rank Fusion. Returns immutable `Citation` records
  (`chunk_id`, `page_id`, `source_path`, `parent_heading`, `snippet`,
  `score`) with no LLM call and no rerank. Designed as the cheap-router
  half of the 3-layer agent surface: agents pay for `search_chunks`
  first, then decide whether to fetch verbatim text via
  `ChunkStore.get_chunk` or follow `page_id` into the synthesized wiki
  page. Phase 1.3's MCP `search_chunks` tool will wrap this directly.

- **FTS5 + sqlite-vec virtual tables on `chunks`** — `chunks_fts`
  (external-content FTS5 with porter-ascii tokenizer) plus three
  AFTER INSERT/DELETE/UPDATE triggers keep BM25 in lockstep with
  `chunks` writes. `chunk_vec` (sqlite-vec `vec0`) holds disk-backed
  ANN vectors keyed by `chunks.rowid`, created lazily on first persist
  with an embedder. New `meta` table stamps the embedder fingerprint
  `(provider, model, dim)`; both write and read paths refuse on
  mismatch rather than silently mix vector spaces.

- **Provenance columns on `chunks`** — `source_path`, `parent_heading`,
  `page_id`, and `embedding`. `parent_heading` is captured by the
  chunker via a cheap regex against ATX markdown headings (stamps
  the first heading line found in each chunk's text). `page_id` is
  written back from the page writer after a page commits, so
  retrieval citations carry the chunk → synthesized-page edge
  durably in SQLite. Markdown-only for parent_heading today;
  per-extractor structure capture is a Phase 1.x follow-up. Cross-
  content-type previews are covered by the always-populated
  `Citation.snippet` field (≤200 chars, whitespace-collapsed).

### Changed

- **`get_context` gains an optional token `budget`** — when set,
  after RRF ranking, citations are taken in rank order until the
  running sum of `token_estimate` would exceed the budget. The
  top-ranked chunk always lands even if its cost alone exceeds
  budget (returning empty would be worse for the agent), and NULL
  `token_estimate` rows count as 1 so legacy data still advances
  the loop. Default `budget=None` preserves byte-identical behavior
  for existing `top_pages`/`k` callers. The MCP `get_context` tool
  exposes `budget` as its primary knob (defaults to 2000 tokens)
  and intentionally hides `top_pages`/`k` from agents — drop to
  `search_pages` + `search_chunks` for finer control.

- **`search_chunks` gains optional `page_ids` and `query_vec`
  parameters** — `page_ids` scopes both lanes to chunks belonging
  to the given pages (BM25 lane via FTS5 JOIN+IN; vector lane
  falls back to in-memory numpy cosine over `chunks.embedding`
  since sqlite-vec's `MATCH` can't compose with `WHERE`). The
  scoped path is what `get_context` uses to rerank chunks within
  routed pages. `query_vec` lets callers thread a pre-computed
  query embedding so they don't pay for a second embed inside
  `_vector_lane` — used by `get_context` to amortize one embed
  call across page routing and chunk reranking. Default values
  (`page_ids=None`, `query_vec=None`) preserve the previously
  shipped behavior byte-identical.

- **`ChunkStore` uses the shared `SQLiteCache` connection** instead of
  opening a fresh `sqlite3.connect()` per call. Constructor accepts
  either a `SQLiteCache` (hot path: ingest + retrieval share one
  connection, one extension load) or a `Path` (thin callers: CLI
  status, source lookup, tests). Closes the per-call SQLite open
  flagged by the codebase performance invariants.

- **Chunk persistence embeds at write time** when an embedder is
  available. `persist_chunks` now batches embeddings via the existing
  `_embed_in_batches` helper, writes vectors into `chunk_vec` in the
  same transaction as the `chunks` insert, and stamps the embedder
  fingerprint on first persist. Re-ingest cleans up old `chunk_vec`
  rows by rowid before reinserting; subsequent persists with a
  changed embedder identity raise immediately.

- **`PageWriteResult.chunk_to_page`** — the page writer now records
  the chunk_id → page_id mapping as pages are committed. Source-page
  writes seed every chunk; concept/entity create + update outcomes
  overwrite for chunks that produced specific pages. The processor
  applies the map via `ChunkStore.set_page_ids` so the edge lands
  durably in SQLite.

### Removed

- **`[search] engine` config option** — never wired to anything. The
  scaffolded `wikiloom.toml` shipped a `[search] engine = "grep"`
  block and `Config.load` parsed it into `cfg.search`, but no
  consumer ever read it. Literal-match retrieval is already handled
  by the BM25 (FTS5) lane in `retrieval.py`, fused with the vector
  lane via RRF. `SearchConfig` removed from `wikiloom/config.py`,
  block removed from `wikiloom/scaffold.py` and the README. Existing
  `wikiloom.toml` files with a stray `[search]` section still load —
  unknown sections are ignored.

### Dependencies

- **New `[mcp]` optional extra** — `pip install "wikiloom[mcp]"`
  pulls the official `mcp` Python SDK (which ships FastMCP) and
  its transitive deps (anyio, httpx-sse, sse-starlette, uvicorn,
  pydantic-settings, etc.). Not in core — users who never run the
  agent server don't pay for it. Pydantic v2 arrives transitively
  via the SDK; WikiLoom uses it only at the MCP boundary.

- **`sqlite-vec>=0.1.6`** added to core dependencies. The extension is
  registered on every `SQLiteCache` connection at init; a clear error
  is raised if the platform's Python sqlite3 was built without
  `enable_load_extension` (system Python on macOS and some Linux
  distros). Use a Homebrew/pyenv interpreter or any build with
  `SQLITE_ENABLE_LOAD_EXTENSION` to use chunk retrieval.

## [0.1.9] — 2026-05-04

### Fixed

- **`wikiloom status` now shows accurate deprecated counts after
  a merge or deprecate** — `status` had been pulling
  `total_pages`, `by_type`, `by_status`, `backlinks`, and `aliases`
  from `cache.get_stats()`. The cache is a query-acceleration
  layer for FTS / semantic search, not the source of truth for
  "how many pages are deprecated"; a stale or out-of-sync cache
  could make `status` lie even when the registry knew the right
  answer (concretely: after `wikiloom duplicates --review` merged
  two pairs, `status` reported `0 deprecated` until you ran
  `wikiloom rebuild-cache`). `status` now reads counts from the
  registry + `BacklinkRegistry` directly: `Counter` over
  `registry.pages.values()` for `by_type`/`by_status`, sum of
  `len(entry.aliases)` for the alias count, `len(bl.edges)` for
  backlinks. Other sections (`Storage`, `Last event`, `Usage`)
  already read from their true sources and are unchanged.

- **Incremental cache sync keeps deprecated rows instead of
  dropping them** — `_incremental_sync` lumped two delete
  conditions together (missing manifest entry OR missing file),
  so a deprecate/merge that moved a page to `archive/` dropped
  the cache row entirely. `full_rebuild` meanwhile walks
  `registry.pages.items()` and creates a row for the same case
  with empty body, so the two sync paths produced different
  cache states for the same project. Split the delete condition:
  only drop when the manifest entry is gone (truly retired); when
  the entry exists but the file moved, upsert with empty body
  to mirror `full_rebuild`. Both sync paths now produce the same
  cache state. New regression test
  (`test_incremental_sync_keeps_row_when_file_archived`) locks
  in the contract.

## [0.1.8] — 2026-05-03

### Performance

- **Linker cosine rerank uses one matmul instead of a Python
  loop** — `_resolve_with_rerank` looped per candidate calling
  `cosine_similarity(span_vec, embeddings[page_id])`; the helper
  rewrapped both vectors as numpy arrays and recomputed the span
  norm on every call, and page embeddings were stored as
  `list[float]` so each call also rematerialized a numpy array.
  Page embeddings now load into a single `(M, D)` float32 matrix
  with precomputed L2 norms; rerank slices the candidate row
  indices and computes all cosines in one matmul before
  argmax. Same vectorization that `SQLiteCache.semantic_search`
  already used. Noticeable on link-heavy wikis with many
  candidates per span.

- **`wikiloom lint` walks the wiki once instead of three
  times** — `check_frontmatter`, `check_contradictions`, and
  `check_promoted_from_update` each ran their own
  `_iter_content_pages` loop, reading every page from disk and
  parsing YAML frontmatter independently. `run_all` now
  populates a per-run `self._fm_cache` keyed by absolute path
  and clears it in a `finally`; the three checks read from the
  cache via a `_frontmatter_for` helper that falls back to a
  fresh read when called standalone. 3× → 1× disk reads + YAML
  parses per `wikiloom lint`.

- **`wikiloom lint --fix` resolves protection in one git walk
  instead of per page** — `fix_all` called
  `_is_protected → is_human_edited → iter_commits(paths=rel,
  max_count=1)` once per affected page. On a wiki where lint
  flagged 150 pages that was 150 separate `git log` subprocess
  invocations. `git_ops` already exposed
  `latest_commit_types_bulk` (the same helper
  `HumanEditProtection.scan` uses); `fix_all` now precomputes a
  `self._protected_paths` set from one bulk call over all
  candidate paths (broken-link sources ∪ frontmatter issues).
  Falls back to the per-call query when git is unavailable or
  doesn't expose the bulk method.

- **Slug-collision guard pre-buckets candidates by type** —
  `_find_slug_collision` was called once per `pages_to_create`
  proposal during the post-gather loop and re-filtered the full
  manifest list by type prefix on every call. New
  `_bucket_page_ids` builds a `{type_dir: [page_ids]}` map once
  before the loop; the helper now takes that map and does a
  dict lookup plus a rapidfuzz scan over just the relevant
  bucket. In-loop appends switch to `setdefault(...).append(...)`
  so later chunks in the same ingest still see fresh creates.
  O(M×N) → O(M) once + O(bucket) per proposal.

- **Chunk inserts batch into one `executemany`** —
  `persist_chunks` in `chunk_store.py` looped per chunk calling
  `conn.execute(INSERT)`, so a 50-chunk PDF was 50 separate SQL
  round-trips. Build the row tuple list in the loop and issue
  one `executemany` after; same row order, same single commit.
  Matches the `executemany` pattern already used in
  `SQLiteCache.full_rebuild` and `_incremental_sync`.

- **Embeddings (de)serialize via numpy bulk byte conversion** —
  `serialize_embedding` called `struct.pack(f"{n}f", *vector)`,
  splatting 768 floats per call as positional args through
  Python; `deserialize_embedding` rebuilt the list with
  `struct.unpack` the same way. Replaced with
  `np.asarray(...).tobytes()` and
  `np.frombuffer(..., dtype=np.float32).tolist()` — same bytes
  on disk (float32 little-endian either way), but the conversion
  happens in one C call instead of N. Hot on cache rebuild
  (every page) and link sessions (every page).

- **Frontmatter parse/render uses libyaml when available** —
  PyYAML's default `safe_load`/`dump` fall back to the slow
  pure-Python parser unless `CSafeLoader`/`CSafeDumper` are
  passed explicitly. Frontmatter parsing happens on every page
  read across `lint`, search index rebuild, page-context
  retrieval, merge, save — a steady tax across the CLI. Opt
  into the libyaml-backed C loader/dumper at module load with
  a `SafeLoader`/`SafeDumper` fallback for environments without
  libyaml. 5–10× faster YAML where libyaml is available.

## [0.1.7] — 2026-05-01

> **Note:** 0.1.6 was uploaded to TestPyPI on 2026-05-01 and
> yanked the same day after smoke testing surfaced the cache
> sync bug below. 0.1.7 carries the fix plus everything from
> 0.1.6's changelog. 0.1.6 was never published to PyPI.

### Fixed

- **Cache sync handles duplicate paths in `changed_files`**
  — `_incremental_sync` translated paths to page_ids without
  deduping. `wikiloom duplicates --review` builds its sync
  list by appending `(winner.md, loser.md)` for every merged
  pair; when a page is winner of one pair *and* loser of
  another (or winner across multiple pairs), the same path
  appears twice. Without dedup, the page row landed in
  `page_rows` twice and the executemany INSERT introduced in
  0.1.5 crashed with `UNIQUE constraint failed: pages.page_id`
  at the very end of an otherwise-successful batch review.
  `_incremental_sync` now dedupes `changed_page_ids` while
  preserving first-seen order. Same on-disk result; latent
  prior to 0.1.5's batched INSERT.

## [0.1.6] — 2026-05-01

### Performance

- **`HumanEditProtection.scan` resolves every page's last
  commit-type in one history walk** — `scan()` looped over the
  full manifest calling `GitOps.latest_commit_type(page_path)`,
  which under the hood does
  `iter_commits(paths=rel, max_count=1)` per page. On a 1k-page
  wiki that's 1k separate `git log` invocations through
  GitPython, each walking history independently until it finds
  a commit touching that path. New `latest_commit_types_bulk`
  helper does a single `iter_commits()` walk newest-first,
  records the parsed commit-type prefix for each requested path
  via a set intersection with `commit.stats.files`, and stops
  the moment every path has an answer. Same result, ~N× fewer
  subprocess launches and shared traversal across all paths.
  Speeds up `wikiloom lint` (the only `scan()` caller) on real
  wikis without changing its output.
- **Linker reuses pre-pass body + frontmatter** —
  `LinkingEngine.link_all` already pre-read each page's body to
  feed `nlp.pipe` in a single batched call, then threw the body
  away. `link_page` then re-read the same file and re-parsed
  frontmatter on every page, doing exactly twice the disk I/O
  and frontmatter-parse work the relink needed. `link_page` now
  takes optional `body` and `fm` kwargs that the pre-pass
  threads through alongside the spaCy doc; standalone callers
  still fall back to read+parse. ~Halves file I/O on a relink.
- **Pending-link writes flush once per `link_all`, not per
  page** — `_save_pending` did a full read+JSON-parse+rewrite
  of `_registry/pending.json` for each page that produced
  pending matches. Cumulative cost grew quadratically with the
  number of linked pages (the file gets re-serialized larger
  on every page). `link_all` now activates a `_pending_buffer`
  on the engine before its loop, `link_page` appends to it
  in-memory, and the run flushes once in a `finally` so a
  mid-loop crash still persists what we collected. Standalone
  `link_page` callers are unchanged. Same on-disk shape, O(N²)
  → O(N) write work on a full relink.

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
