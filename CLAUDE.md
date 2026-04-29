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
- `_registry/wiki.db` — SQLite cache, page registry, backlinks (generated).
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

## Environment

- `ANTHROPIC_API_KEY` is required for synthesis and `pytest -m live`.
- `.env` is auto-loaded via `python-dotenv`.
- LLM calls route through `litellm`; model name is configured per-project.
