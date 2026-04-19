"""Project configuration loader.

Reads wikiloom.toml and exposes typed dataclass sections.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    model: str = "claude-sonnet-4-6"
    max_tokens_per_operation: int = 8000
    monthly_budget_usd: float = 50.0


@dataclass
class LinkingConfig:
    ner_model: str = "en_core_web_sm"
    auto_create_stubs: bool = False
    high_confidence_threshold: int = 95
    medium_confidence_threshold: int = 85
    low_confidence_threshold: int = 70


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
    """

    max_file_size_mb: int = 50
    min_extracted_chars: int = 16
    enable_budget_check: bool = True
    use_page_context: bool = True
    page_context_top_k: int = 10


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
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    dormant: DormantConfig = field(default_factory=DormantConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def load(cls, project_root: Path) -> Config:
        """Load wikiloom.toml from the given project root."""
        project_root = Path(project_root)
        toml_path = project_root / "wikiloom.toml"
        if not toml_path.exists():
            raise FileNotFoundError(f"No wikiloom.toml found at {toml_path}")

        with toml_path.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)

        cfg = cls(project_root=project_root)
        if "project" in data:
            cfg.project = ProjectConfig(**_filter(ProjectConfig, data["project"]))
        if "llm" in data:
            cfg.llm = LLMConfig(**_filter(LLMConfig, data["llm"]))
        if "linking" in data:
            cfg.linking = LinkingConfig(**_filter(LinkingConfig, data["linking"]))
        if "dormant" in data:
            cfg.dormant = DormantConfig(**_filter(DormantConfig, data["dormant"]))
        if "search" in data:
            cfg.search = SearchConfig(**_filter(SearchConfig, data["search"]))
        if "ingest" in data:
            cfg.ingest = IngestConfig(**_filter(IngestConfig, data["ingest"]))
        if "embeddings" in data:
            cfg.embeddings = EmbeddingsConfig(**_filter(EmbeddingsConfig, data["embeddings"]))
        return cfg


def _filter(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys from `data` that aren't fields on `cls`.

    Lets the TOML have extra fields without crashing the loader.
    """
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in valid}
