# WikiLoom

See README.md for what the project is and how to use it.

Source layout:
- `wikiloom/` — library and CLI (entry point: `wikiloom.cli:main`, exposed as the `wikiloom` command).
- `wikiloom/ingest/` — ingestion pipeline.
- `wikiloom/templates/` — page templates and prompts shipped with the package.
- `tests/` — pytest suite.

Run tests: `pytest` (live tests that hit a real LLM provider are marked `live` and skipped by default; run with `pytest -m live` and an `ANTHROPIC_API_KEY` set).

The README is the source of truth for what the project does and how to use it. The code is the source of truth for how it works.
