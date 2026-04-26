"""Project configuration loader.

Reads wikiloom.toml and exposes typed dataclass sections.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when wikiloom.toml is malformed.

    The CLI catches this and re-raises as ``click.ClickException`` so
    users see a friendly error instead of a tomllib stacktrace.
    """


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class ProjectConfig:
    name: str = ""
    domain: str = ""
    created: str = ""
    schema_version: int = 1


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    # Default model used by any LLM-backed command that doesn't have
    # its own override. Per-command overrides below stay empty unless
    # the user explicitly wants a split setup (cheap model for ingest,
    # stronger model for query).
    default_model: str = "claude-sonnet-4-6"
    ingest_model: str = ""
    query_model: str = ""
    max_tokens_per_operation: int = 8000
    monthly_budget_usd: float = 50.0

    def for_ingest(self) -> str:
        return self.ingest_model or self.default_model

    def for_query(self) -> str:
        return self.query_model or self.default_model


@dataclass
class LinkingConfig:
    """Hybrid linker knobs.

    The linker matches each span against existing pages in two
    stages: fuzzy pre-filter (cheap shortlist of up to
    ``fuzzy_prefilter_top_k`` candidates at ``fuzzy_prefilter_
    threshold`` or above), then cosine similarity of the span's
    context window against each shortlisted page's body embedding.
    Cosine is the decision metric; fuzzy is plumbing.

    Cosine thresholds partition outcomes:

    - ``>= cosine_high_threshold``   → auto-link.
    - ``>= cosine_medium_threshold`` → auto-link, flagged medium.
    - ``>= cosine_low_threshold``    → defer to ``pending.json``.
    - ``<  cosine_low_threshold``    → drop.

    Defaults calibrated for sentence-transformer / fastembed
    defaults where related content typically lands 0.6–0.85 and
    unrelated lands 0.2–0.4.
    """

    ner_model: str = "en_core_web_sm"
    auto_create_stubs: bool = False
    fuzzy_prefilter_top_k: int = 10
    fuzzy_prefilter_threshold: int = 60
    cosine_high_threshold: float = 0.75
    cosine_medium_threshold: float = 0.60
    cosine_low_threshold: float = 0.50


@dataclass
class DormantConfig:
    """Time-based windows for surfacing pages as dormant candidates.

    A page is *dormant* when its ``frontmatter.modified`` timestamp is
    older than the applicable window. Dormancy is purely informational
    — pages stay visible and interactable across all CLI commands.
    Marking a page dormant is a user action via ``wikiloom dormant``.
    """

    default_window_days: int = 90
    entity_window_days: int = 180
    concept_window_days: int = 120
    synthesis_window_days: int = 60


@dataclass
class SearchConfig:
    engine: str = "grep"


@dataclass
class IngestConfig:
    """Safeguards applied at the ingest boundary.

    ``max_file_size_mb`` fails fast on oversized inputs before the
    extractor or (eventually) the LLM loop touch them. ``0`` disables
    the check.

    ``min_extracted_chars`` is the minimum post-extraction text length
    for a source to be considered useful. Empty / near-empty extraction
    (typically a scanned PDF with no text layer) raises instead of
    silently producing a useless ingest.

    ``enable_budget_check`` gates the pre-flight monthly-budget check
    in the synthesis loop. Defaults to True in production; set to
    False in tests or when running a deliberately uncapped batch.

    ``use_page_context`` enables semantic retrieval of existing pages
    per chunk for prompt injection so the LLM can prefer UPDATE over
    CREATE when a chunk overlaps with existing content. Set to False
    to fall back to the simpler "most-recently-modified" snapshot.
    ``page_context_top_k`` caps how many retrieved pages are injected.

    ``max_workers`` sets the concurrency of the synthesis loop. The
    default of 1 is the safe floor: it sidesteps rate limits on every
    provider's entry tier, so new users don't hit 429s before they
    see the tool work. Bump it up once you know your provider tier
    has headroom. Provider-specific guidance:

    - Anthropic Tier 1 (Haiku): stay at 1 against the 10k OTPM cap.
      Tier 2 comfortably handles 2–4; Tier 3+ handles 4–6.
    - OpenAI Tier 1 (gpt-4o-mini / gpt-4o): headroom for 4+ on
      gpt-4o-mini; gpt-4o is ITPM-bound on large chunks.
    - Gemini free tier: keep at 1 (15 RPM ceiling). Paid tiers
      handle 4–6 easily.
    - Mistral / other low free tiers: stay at 1 and tune up.

    Prompt caching (enabled automatically for Anthropic) reduces ITPM
    pressure since cache reads don't count toward the limit, so
    effective headroom is larger than the raw number suggests.

    ``post_merge`` controls whether a post-ingest auto-merge pass runs
    against pairs touched by the ingest:

    - ``"off"`` (default) — no post-ingest merge pass.
    - ``"preview"`` — list pairs that would auto-merge, do not apply.
    - ``"safe"`` — apply merges flagged ``is_safe_to_auto`` by
      ``suggest_winner`` (plural, prefix, hyphenation, token-drop
      rules). Produces a separate commit after the ingest commit so
      ``git revert`` can undo the merge batch without touching the
      ingest itself.

    ``auto_relink`` runs ``wikiloom relink`` after a successful
    post-ingest merge (only when at least one merge actually applied)
    so pages that gained aliases from merged losers pick up new
    inbound links. Ignored when ``post_merge`` is ``"off"`` or
    ``"preview"``, or when no merges happened.
    """

    max_file_size_mb: int = 50
    min_extracted_chars: int = 16
    enable_budget_check: bool = True
    use_page_context: bool = True
    page_context_top_k: int = 10
    max_workers: int = 1
    post_merge: str = "off"
    auto_relink: bool = True


@dataclass
class EmbeddingsConfig:
    """Which embedding backend to use for semantic search.

    ``provider`` selects the backend: ``fastembed`` (default, local),
    ``openai`` (API), or ``sentence-transformers`` (local, heavy).
    ``model`` overrides the default model for the chosen provider.
    ``enabled`` can disable embedding entirely (FTS5-only search).
    """

    provider: str = "fastembed"
    model: str = ""
    enabled: bool = True


@dataclass
class QueryConfig:
    """Settings for ``wikiloom query`` history retention.

    ``history_enabled`` controls whether each successful query appends
    a full record (question, answer, sources, metrics) to
    ``_registry/query_history.json``. The file is gitignored — it's
    per-machine cache, not project state — but may contain sensitive
    prompts. Set to ``False`` to skip writing entirely.

    ``history_size`` caps how many entries are retained. Older entries
    are trimmed on each append. 100 covers ~3 months of moderate use
    and stays well under 2 MB on disk; bias toward more rather than
    less since querying is paid and remembering is free.
    """

    history_enabled: bool = True
    history_size: int = 100


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    dormant: DormantConfig = field(default_factory=DormantConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def load(cls, project_root: Path) -> Config:
        """Load wikiloom.toml from the given project root.

        Raises ``FileNotFoundError`` if the file is missing and
        ``ConfigError`` if it exists but can't be parsed (typo,
        bad bracket, duplicate key, etc.).
        """
        project_root = Path(project_root)
        toml_path = project_root / "wikiloom.toml"
        if not toml_path.exists():
            raise FileNotFoundError(f"No wikiloom.toml found at {toml_path}")

        try:
            with toml_path.open("rb") as f:
                data: dict[str, Any] = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(
                f"Could not parse {toml_path}: {exc}"
            ) from exc

        cfg = cls(project_root=project_root)
        if "project" in data:
            cfg.project = ProjectConfig(
                **_filter(ProjectConfig, data["project"]))
        if "llm" in data:
            cfg.llm = LLMConfig(**_filter(LLMConfig, data["llm"]))
        if "linking" in data:
            cfg.linking = LinkingConfig(
                **_filter(LinkingConfig, data["linking"]))
        if "dormant" in data:
            cfg.dormant = DormantConfig(
                **_filter(DormantConfig, data["dormant"]))
        if "search" in data:
            cfg.search = SearchConfig(**_filter(SearchConfig, data["search"]))
        if "ingest" in data:
            cfg.ingest = IngestConfig(**_filter(IngestConfig, data["ingest"]))
        if "embeddings" in data:
            cfg.embeddings = EmbeddingsConfig(
                **_filter(EmbeddingsConfig, data["embeddings"]))
        if "query" in data:
            cfg.query = QueryConfig(**_filter(QueryConfig, data["query"]))
        return cfg


def _filter(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys from `data` that aren't fields on `cls`.

    Lets the TOML have extra fields without crashing the loader.
    """
    valid = {f.name for f in cls.__dataclass_fields__.values()
             }  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in valid}
