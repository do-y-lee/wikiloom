# WikiLoom

LLM-maintained knowledge base with deterministic linking. Data flow:
ingest → chunk → synthesize → link → index → query.

README.md is the source of truth for what the project does and how to use
it. The code is the source of truth for how it works.

## Source layout

- `wikiloom/` — library and CLI (entry point: `wikiloom.cli:main`, exposed as the `wikiloom` command).
- `wikiloom/ingest/` — ingestion pipeline.
- `wikiloom/templates/` — page templates and prompts shipped with the package.
- `tests/` — pytest suite.

On-disk state in a user's project:
- `_registry/wiki.db` — SQLite cache, page registry, backlinks (generated). The cache runs in WAL mode, so `wiki.db-wal` and `wiki.db-shm` sidecars appear next to it; the scaffolded `.gitignore` covers `_registry/wiki.db*`.
- `wiki/` — synthesized markdown pages and generated indexes.

## Commands

Develop in a venv with an editable install (matches the README's "From source" flow).

| Command | Purpose |
|---------|---------|
| `pip install -e ".[dev]"` | Install in editable mode with dev extras |
| `wikiloom <cmd>` | Run the CLI (entry point installed by editable install) |
| `pytest` | Run tests (live tests skipped by default) |
| `pytest tests/test_foo.py::test_bar` | Run a single test |
| `ANTHROPIC_API_KEY=... pytest -m live` | Run live LLM tests |
| `ruff check` | Lint |

## Invariants

- Don't commit `*.local.md` files — they're local-only notes/specs.
- Don't hand-edit anything under `_registry/` or generated index/log files in `wiki/`.
- Don't write wikilinks directly in synthesized pages — the linker owns link insertion.
- Python 3.10 floor — avoid 3.11+ syntax.
- New runtime deps: check the optional-extras split (`audio`, `openai-embeddings`, `local-embeddings`) before adding to core dependencies.
- Version bumps: update `pyproject.toml` and add a CHANGELOG entry.

## Performance invariants

This codebase has a few hot paths and shared caches. Reach for the existing infrastructure before writing your own — reinventing the slow path is the most common regression here.

- **Page-similarity search → `SQLiteCache.semantic_search`.** It builds and caches a process-local `(M, D)` numpy matrix on first call and reuses it via one matmul per query. Don't open a fresh SQLite connection and Python-loop `cosine_similarity` per page — that bypasses the cache. For per-chunk retrieval (e.g., page-context during synthesis), pass the same `SQLiteCache` instance through to every worker.
- **Embedder loading → `load_embedder()` is process-cached.** Keyed by `(provider, model)`. Call it freely and pass the result through to inner stages (synthesis, linker, cache sync) rather than re-loading. Tests get isolation via `clear_embedder_cache()` (autouse fixture in `tests/conftest.py`).
- **Append-only files → `open("a")`, not read-concat-write.** `wiki/log.md` and similar grow-over-time files must open in append mode. Read+rewrite is O(N²) cumulative I/O over the project's lifetime.
- **Hot-loop list reconstructions → cache once, refresh on change.** Rebuilding a list/map of fixed values inside a per-iteration loop is the pattern to watch for. Hoist to instance state and refresh only when the underlying data changes. Example: `LinkingEngine._alias_keys` caches `list(self.alias_map.keys())` so `_resolve` / `_resolve_top_k` don't rebuild it per span.
- **Per-call SQLite opens → reuse `SQLiteCache._conn`.** It's the shared long-lived connection, serialized by `self._lock`, with WAL + perf PRAGMAs already set in `__init__`. New reads/writes should go through `_connect()`, not open their own connections.
- **Bulk SQL inserts → `executemany`, not per-row `execute` in a Python loop.** `cache.full_rebuild` and `_incremental_sync` build row lists in Python and batch each table with one `executemany` call. Same SQL, same data, fewer Python→C round-trips.
- **Never construct expensive objects inside loops.** `Registry(...)`, `SQLiteCache(...)`, `load_embedder(...)`, `BacklinkRegistry(...)` all do real I/O / model loading on every call. Build once above the loop and pass the instance through.
- **Watch "per-X × per-Y" multipliers.** Before adding work inside a worker pool or per-chunk loop, ask: what's the multiplier? Expensive × N chunks × M pages adds up fast. Hoist anything stable to the call site above the loop.

## Environment

- `ANTHROPIC_API_KEY` is required for synthesis and `pytest -m live`.
- `.env` is auto-loaded via `python-dotenv`.
- LLM calls route through `litellm`; model name is configured per-project.
