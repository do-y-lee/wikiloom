"""Project scaffolding for WikiLoom."""

from __future__ import annotations

import json
import shutil
from importlib import resources as importlib_resources
from pathlib import Path

from wikiloom.cache import init_cache
from wikiloom.utils import now_iso

# Wiki subdirectories under wiki/
WIKI_SUBDIRS = [
    "entities",
    "concepts",
    "sources",
    "syntheses",
    "decisions",
    "archive",
]

# Raw source subdirectories under raw/
RAW_SUBDIRS = [
    "papers",
    "articles",
    "images",
    "code",
    "misc",
]

GITIGNORE_CONTENT = """\
# WikiLoom — derived / transient state
_registry/wiki.db
_registry/last_query.json
_registry/ingest_state.json
.wikiloom.lock

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
"""

# Single source of truth for the on-disk schema version. Bump when the
# manifest / frontmatter / sources schema changes in a way that needs a
# migration. Used by both wikiloom.toml and _registry/schema_version.json
# so the two never drift.
SCHEMA_VERSION = 1

# Scaffold defaults. Referenced by both `_generate_config` and the CLI's
# post-init summary so the printed "next steps" never drifts from what
# was actually written to wikiloom.toml.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MONTHLY_BUDGET_USD = 50.0

# Env var each provider reads its API key from. Kept next to
# DEFAULT_PROVIDER so adding a provider in one place updates both the
# scaffold default and the init-time API key hint.
PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _generate_config(name: str, domain: str) -> str:
    """Generate wikiloom.toml content."""
    return f"""\
[project]
name = "{name}"
domain = "{domain}"
created = "{now_iso()}"
schema_version = {SCHEMA_VERSION}

[llm]
provider = "{DEFAULT_PROVIDER}"
model = "{DEFAULT_MODEL}"
max_tokens_per_operation = 8000
monthly_budget_usd = {DEFAULT_MONTHLY_BUDGET_USD}

[linking]
ner_model = "en_core_web_sm"
auto_create_stubs = false
high_confidence_threshold = 95
medium_confidence_threshold = 85
low_confidence_threshold = 70

# Dormant windows. A page becomes a dormant *candidate* when its
# `modified` timestamp is older than the window for its type.
# Marking is a user action via `wikiloom dormant <page>` — wikiloom
# never auto-flips status based on age. Dormant is a hint, not a
# verdict. Per-page override via `dormant_window_days` in frontmatter.
[dormant]
default_window_days = 90
entity_window_days = 180
concept_window_days = 120
synthesis_window_days = 60

[search]
engine = "grep"

[embeddings]
provider = "fastembed"
# provider = "openai"                  # needs OPENAI_API_KEY
# provider = "sentence-transformers"   # heavier install, no API key
# model = ""                           # empty = provider default
enabled = true
"""


def _sub_index_content(category: str) -> str:
    """Generate a sub-index template for a wiki category."""
    title = category.capitalize()
    return f"""\
---
title: "{title} Index"
type: "index"
status: "active"
created: "{now_iso()}"
modified: "{now_iso()}"
summary: "Index of all {category} pages."
aliases: []
sources: []
source_count: 0
confidence: "high"
dormant_window_days: 365
human_edited: false
human_edited_at: null
superseded_by: null
contradictions: []
tags: []
---

# {title}

*No pages yet.*
"""


def _root_index_content(name: str) -> str:
    """Generate the root wiki index."""
    return f"""\
---
title: "{name} Wiki"
type: "index"
status: "active"
created: "{now_iso()}"
modified: "{now_iso()}"
summary: "Root index for the {name} knowledge base."
aliases: []
sources: []
source_count: 0
confidence: "high"
dormant_window_days: 365
human_edited: false
human_edited_at: null
superseded_by: null
contradictions: []
tags: []
---

# {name}

## Sections

- [Entities](entities/index.md)
- [Concepts](concepts/index.md)
- [Sources](sources/index.md)
- [Syntheses](syntheses/index.md)
- [Decisions](decisions/index.md)
- [Archive](archive/index.md)
"""


_INGEST_DOMAIN_PLACEHOLDER = (
    'This wiki documents [GENERAL TOPIC — e.g. "consumer banking products '
    'and processes"]. Prefer pages that someone reading the wiki later '
    'would find useful as standalone reference material.'
)


def _copy_templates(dest: Path, domain: str = "") -> None:
    """Copy template files from the package to .wikiloom/ directory.

    When ``domain`` is non-empty, the ingest prompt's placeholder sentence
    is rewritten to reference the user's domain so the first ingest has
    real context without requiring a manual edit.
    """
    templates_pkg = importlib_resources.files("wikiloom") / "templates"

    # schema.md
    schema_src = templates_pkg / "schema.md"
    (dest / "schema.md").write_text(
        schema_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # prompts/
    prompts_dir = dest / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for prompt_name in ("ingest.md", "query.md", "lint.md"):
        src = templates_pkg / "prompts" / prompt_name
        content = src.read_text(encoding="utf-8")
        if prompt_name == "ingest.md" and domain:
            replacement = (
                f"This wiki documents {domain}. Prefer pages that someone "
                f"reading the wiki later would find useful as standalone "
                f"reference material."
            )
            content = content.replace(_INGEST_DOMAIN_PLACEHOLDER, replacement)
        (prompts_dir / prompt_name).write_text(content, encoding="utf-8")

    # output_formats/
    formats_dir = dest / "output_formats"
    formats_dir.mkdir(exist_ok=True)
    for fmt_name in ("ingest_response.json", "query_response.json"):
        src = templates_pkg / "output_formats" / fmt_name
        (formats_dir / fmt_name).write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )


def _init_git(project_dir: Path) -> None:
    """Initialize a git repository if one doesn't exist, and commit the
    scaffold so the working tree starts clean.

    The commit uses the ``init:`` prefix but ``init`` is deliberately not
    in ``AUTO_COMMIT_TYPES`` — scaffold files are synthetic and not
    treated as either human-edited or LLM-authored; the prefix exists
    only to classify the event for log readers.
    """
    from git import Repo
    from git.exc import InvalidGitRepositoryError

    try:
        repo = Repo(project_dir)
    except InvalidGitRepositoryError:
        repo = Repo.init(project_dir)

    # Skip commit if nothing to commit or HEAD already has it.
    repo.git.add("-A", "--", str(project_dir))
    if repo.head.is_valid():
        if not repo.index.diff(repo.head.commit):
            return
    elif not repo.index.entries:
        return
    repo.index.commit("init: scaffold wikiloom project")


def init_project(
    name: str,
    path: Path | None = None,
    domain: str = "",
) -> Path:
    """Create a new WikiLoom project with full directory structure.

    Args:
        name: Project name (used in config and index).
        path: Parent directory. Defaults to current directory.
        domain: Optional domain description (e.g. "AI safety research").

    Returns:
        Path to the created project directory.
    """
    if path is None:
        path = Path.cwd()

    project_dir = path / name
    project_dir.mkdir(parents=True, exist_ok=True)

    # raw/ subdirectories
    for subdir in RAW_SUBDIRS:
        (project_dir / "raw" / subdir).mkdir(parents=True, exist_ok=True)

    # wiki/ subdirectories with index files
    wiki_dir = project_dir / "wiki"
    wiki_dir.mkdir(exist_ok=True)

    # Root index
    (wiki_dir / "index.md").write_text(
        _root_index_content(name), encoding="utf-8"
    )

    # Event log
    (wiki_dir / "log.md").write_text(
        "# WikiLoom Event Log\n\n", encoding="utf-8"
    )

    # Sub-indexes
    for subdir in WIKI_SUBDIRS:
        sub_path = wiki_dir / subdir
        sub_path.mkdir(exist_ok=True)
        (sub_path / "index.md").write_text(
            _sub_index_content(subdir), encoding="utf-8"
        )

    # _registry/
    registry_dir = project_dir / "_registry"
    registry_dir.mkdir(exist_ok=True)

    # Empty manifest
    manifest = {
        "version": 1,
        "updated_at": now_iso(),
        "pages": {},
    }
    (registry_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Empty backlinks
    (registry_dir / "backlinks.json").write_text(
        json.dumps({"version": 1, "links": {}}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Empty pending
    (registry_dir / "pending.json").write_text(
        json.dumps({"version": 1, "pending": []}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Schema version
    schema_version = {
        "version": SCHEMA_VERSION,
        "created": now_iso(),
        "migrations": [],
    }
    (registry_dir / "schema_version.json").write_text(
        json.dumps(schema_version, indent=2) + "\n", encoding="utf-8"
    )

    # SQLite query cache (schema only; populated on first write via
    # SQLiteCache.sync_from_files or `wikiloom rebuild-cache`).
    init_cache(registry_dir / "wiki.db")

    # .wikiloom/ schema directory
    wikiloom_dir = project_dir / ".wikiloom"
    wikiloom_dir.mkdir(exist_ok=True)
    _copy_templates(wikiloom_dir, domain=domain)

    # wikiloom.toml
    (project_dir / "wikiloom.toml").write_text(
        _generate_config(name, domain), encoding="utf-8"
    )

    # .gitignore
    (project_dir / ".gitignore").write_text(GITIGNORE_CONTENT, encoding="utf-8")

    # Initialize git
    _init_git(project_dir)

    return project_dir
