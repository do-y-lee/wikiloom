"""Project configuration loader for WikiLoom.

Reads ``wikiloom.toml`` from a project root and exposes typed sections.
This is a minimal implementation — only the fields needed by current
components are populated. Future components can extend the dataclasses.
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
    model: str = "claude-sonnet-4-20250514"
    max_tokens_per_operation: int = 8000
    monthly_budget_usd: float = 50.0


@dataclass
class LinkingConfig:
    ner_model: str = "en_core_web_sm"
    auto_create_stubs: bool = True
    high_confidence_threshold: int = 95
    medium_confidence_threshold: int = 85
    low_confidence_threshold: int = 70


@dataclass
class StalenessConfig:
    default_window_days: int = 90
    entity_window_days: int = 180
    concept_window_days: int = 120
    synthesis_window_days: int = 60


@dataclass
class SearchConfig:
    engine: str = "grep"


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    staleness: StalenessConfig = field(default_factory=StalenessConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
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
        if "staleness" in data:
            cfg.staleness = StalenessConfig(**_filter(StalenessConfig, data["staleness"]))
        if "search" in data:
            cfg.search = SearchConfig(**_filter(SearchConfig, data["search"]))
        return cfg


def _filter(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys from `data` that aren't fields on `cls`.

    Lets the TOML have extra fields without crashing the loader.
    """
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in valid}
