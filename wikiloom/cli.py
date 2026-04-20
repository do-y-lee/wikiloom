"""CLI entry point for WikiLoom."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="wikiloom")
def main() -> None:
    """WikiLoom — LLM-maintained knowledge bases with deterministic linking."""


@main.command()
@click.argument("name")
@click.option("--path", type=click.Path(path_type=Path), default=None,
              help="Parent directory for the project. Defaults to current directory.")
@click.option("--domain", default="", help="Domain description (e.g. 'AI safety research').")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "google", "ollama"]),
    default="anthropic",
    show_default=True,
    help="LLM provider preset. Sets the default model and API key env var in "
         "the generated wikiloom.toml.",
)
@click.option(
    "--model",
    default=None,
    help="Override the provider's default model (e.g. 'gpt-5-mini', "
         "'gemini/gemini-2.5-flash', 'gemma3').",
)
def init(
    name: str,
    path: Path | None,
    domain: str,
    provider: str,
    model: str | None,
) -> None:
    """Initialize a new WikiLoom project.

    Creates the full directory structure, config files, git repo,
    and empty registry files.
    """
    from wikiloom.scaffold import (
        DEFAULT_MONTHLY_BUDGET_USD,
        PROVIDER_PRESETS,
        init_project,
        resolve_provider_model,
    )

    chosen_provider, chosen_model = resolve_provider_model(provider, model)
    preset = PROVIDER_PRESETS[chosen_provider]

    project_dir = init_project(
        name=name,
        path=path,
        domain=domain,
        provider=chosen_provider,
        model=chosen_model,
    )

    prompt_path = project_dir / ".wikiloom" / "prompts" / "ingest.md"
    config_path = project_dir / "wikiloom.toml"
    domain_line = domain if domain else "(not set — edit the prompt to add one)"

    click.echo(f"✓ Initialized WikiLoom project at {project_dir}")
    click.echo("")
    click.echo(f"  Domain:   {domain_line}")
    click.echo(f"  Provider: {preset['label']}")
    click.echo(f"  Model:    {chosen_model}")
    click.echo(f"  Budget:   ${DEFAULT_MONTHLY_BUDGET_USD:g}/month")
    click.echo("")
    click.echo("Next steps:")
    click.echo("")

    api_key_env = preset["api_key_env"]
    if api_key_env:
        click.echo("  1. Set your API key")
        click.echo(f"     export {api_key_env}=...")
        click.echo(f"     ({preset['api_key_hint']})")
    else:
        click.echo("  1. Start your local LLM runtime")
        click.echo(f"     {preset['api_key_hint']}")
    click.echo("")

    click.echo("  2. (Recommended) Review the synthesis prompt — shapes every page WikiLoom writes")
    click.echo(f"     {prompt_path}")
    click.echo("")

    click.echo("  3. (Optional) Adjust LLM model, budget, or dormant windows")
    click.echo(f"     {config_path}")
    cheap_model = preset["cheap_model"]
    if cheap_model:
        click.echo(f"     Tip: switch to {cheap_model} for cheap iteration,")
        click.echo(f"          back to {chosen_model} once the prompt feels right.")
    click.echo("")

    click.echo("  4. Ingest your first file")
    click.echo(f"     cd {project_dir.name}")
    click.echo("     wikiloom ingest path/to/doc.pdf")
    click.echo("")
    click.echo("Run `wikiloom --help` to see all commands.")


def _find_project_root(start: Path) -> Path | None:
    """Walk upward from `start` looking for a wikiloom.toml."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "wikiloom.toml").exists():
            return candidate
    return None


def _sync_cache(project: Path) -> None:
    """Refresh the SQLite query cache (FTS + embeddings) from on-disk state.

    Every writer command calls this at the end so `wikiloom query` and
    `wikiloom related` see the new state without a manual rebuild-cache.
    """
    from wikiloom.cache import SQLiteCache
    from wikiloom.embeddings import load_embedder

    registry_dir = project / "_registry"
    if not registry_dir.exists():
        return
    SQLiteCache(registry_dir / "wiki.db").sync_from_files(
        project, embedder=load_embedder(project)
    )


def _require_clean_tree(project: Path, command: str) -> None:
    """Block writer commands when wiki/ has uncommitted changes.

    Manual edits sitting in the working tree would get swept into the
    command's auto-commit with the wrong classifying prefix (``lint:``,
    ``ingest:``, etc.), silently marking them as LLM-authored. Raising
    here forces the user to commit their edits with ``wikiloom save``
    first so the classification stays honest.
    """
    from wikiloom.git_ops import GitOps

    try:
        git = GitOps(project)
    except ValueError:
        return  # not a git repo; nothing to guard
    dirty = git.dirty_wiki_paths()
    if not dirty:
        return

    preview = "\n".join(f"    {p}" for p in dirty[:5])
    if len(dirty) > 5:
        preview += f"\n    ... and {len(dirty) - 5} more"
    raise click.ClickException(
        f"Uncommitted changes in wiki/ ({len(dirty)} file(s)):\n"
        f"{preview}\n\n"
        f"These look like manual edits. Commit them first with:\n"
        f"    wikiloom save\n\n"
        f"Then re-run `wikiloom {command}`."
    )


def _warn_if_dirty(project: Path) -> None:
    """Print a passive nudge if wiki/ has uncommitted changes.

    Called at the top of read-only commands so users notice forgotten
    edits without being blocked. Writer commands use the stricter
    ``_require_clean_tree`` instead.
    """
    from wikiloom.git_ops import GitOps

    try:
        dirty = GitOps(project).dirty_wiki_paths()
    except ValueError:
        return
    if dirty:
        click.echo(
            f"⚠ {len(dirty)} uncommitted edit(s) in wiki/ — "
            f"run `wikiloom save` to commit.\n",
            err=True,
        )


def _load_config(project: Path):
    """Load Config, returning None if missing and ClickException if malformed.

    Centralizes the FileNotFoundError + ConfigError handling so individual
    CLI handlers don't need to repeat it. Most callers tolerate a missing
    config (treat as defaults) but a *broken* config should always surface
    a friendly error rather than crashing.
    """
    from wikiloom.config import Config, ConfigError

    try:
        return Config.load(project)
    except FileNotFoundError:
        return None
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _post_flight_budget_warning(project: Path) -> None:
    """Warn if month-to-date LLM spend exceeds the configured budget.

    Read-only check: sums ``cost_usd`` across the current month's
    events and compares against ``[llm] monthly_budget_usd``. Does NOT
    abort — that's the pre-flight check's job. This is the post-run
    "you went over" notice so users see it once the work is already
    done. Silent when within budget or when no config is loaded.
    """
    from wikiloom.events import parse_log

    cfg = _load_config(project)
    if cfg is None:
        return
    budget = cfg.llm.monthly_budget_usd
    if budget <= 0:
        return

    log_path = project / "wiki" / "log.md"
    if not log_path.exists():
        return
    events = parse_log(log_path)
    # Sum cost across the current calendar month.
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    month_prefix = now.strftime("%Y-%m")
    total = sum(
        e.cost_usd for e in events
        if e.timestamp and e.timestamp.startswith(month_prefix)
    )
    if total <= budget:
        return
    click.echo(
        f"\n⚠ Budget warning: month-to-date spend is ${total:.2f}, "
        f"exceeding monthly_budget_usd (${budget:.2f}).\n"
        f"  Subsequent ingests will fail pre-flight until you raise the "
        f"budget in wikiloom.toml or wait for the next month.",
        err=True,
    )


def _auto_commit(project: Path, commit_type: str, description: str) -> None:
    """Stage every dirty file under ``wiki/`` + ``_registry/`` and commit.

    Callers inside writer commands invoke this at the end, after the
    cache sync, to persist their changes with a classifying prefix so
    the human-edit classifier can distinguish them from manual edits.
    No-ops silently when the repo is missing or nothing is staged.
    """
    from wikiloom.git_ops import GitOps

    try:
        git = GitOps(project)
    except ValueError:
        return
    # Stage modified + untracked files under the wiki-managed dirs.
    # `git add` on a directory path picks up both; deleted files are
    # captured via the -A flag so renames/removals also land.
    for scope in ("wiki", "_registry"):
        if (project / scope).exists():
            git.repo.git.add("-A", "--", scope)
    git.commit([], f"{commit_type}: {description}")


@main.command()
@click.argument("source")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory to find wikiloom.toml.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run the full pipeline even if the source is already in the catalog.",
)
@click.option(
    "--no-page-context",
    is_flag=True,
    default=False,
    help="Disable per-chunk semantic retrieval of existing pages for this run.",
)
def ingest(
    source: str, project: Path | None, force: bool, no_page_context: bool
) -> None:
    """Ingest a source file or URL into the wiki.

    Extracts content, copies local files to raw/, rebuilds backlinks and
    indexes, and commits the result. Re-ingesting an identical local
    file is a cheap no-op (catalog dedup) unless ``--force`` is passed.
    """
    from wikiloom.config import ConfigError
    from wikiloom.ingest.errors import IngestError
    from wikiloom.ingest.processor import ingest as run_ingest

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found). "
                "Run inside a project directory or pass --project."
            )

    _require_clean_tree(project, "ingest")
    # CLI flag is a one-way opt-out. None leaves the config value in
    # effect; False forces the behavior off for this run only.
    use_page_context_override = False if no_page_context else None
    try:
        result = run_ingest(
            source,
            project_root=project,
            force=force,
            use_page_context=use_page_context_override,
        )
    except IngestError as exc:
        raise click.ClickException(str(exc)) from exc
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    # Summary
    created = len(result.pages_created)
    updated = len(result.pages_updated)
    if created or updated:
        click.echo(
            f"Done: {created} page(s) created, {updated} updated"
            f" ({result.total_tokens_in + result.total_tokens_out:,} tokens, "
            f"${result.total_cost_usd:.2f})"
        )
    else:
        click.echo("Done: no pages synthesized.")
    for note in result.notes:
        click.echo(f"Note: {note}")
    _post_flight_budget_warning(project)


@main.command()
@click.option(
    "--fix",
    is_flag=True,
    default=False,
    help="Apply auto-fixes for broken links and missing frontmatter.",
)
@click.option(
    "--check-only",
    is_flag=True,
    default=False,
    help="Report issues and exit non-zero if any are found (CI-friendly).",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def lint(fix: bool, check_only: bool, project: Path | None) -> None:
    """Run health checks over a WikiLoom project.

    Default behavior prints a report and exits 1 if issues are found.
    ``--fix`` applies mechanical repairs (respecting human-edit
    protection). ``--check-only`` is the default behavior with an
    explicit name.
    """
    from wikiloom.lint import WikiLinter
    from wikiloom.locking import FileLock

    if fix and check_only:
        raise click.UsageError("--fix and --check-only are mutually exclusive.")

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    cfg = _load_config(project)
    dormant_cfg = cfg.dormant if cfg is not None else None

    linter = WikiLinter(project, dormant=dormant_cfg)

    if fix:
        _require_clean_tree(project, "lint --fix")
        with FileLock(project):
            report = linter.run_all()
            fixes = linter.fix_all(report)
            _sync_cache(project)
            if fixes.total_fixed:
                parts: list[str] = []
                if fixes.broken_links_fixed:
                    parts.append(f"{fixes.broken_links_fixed} broken link(s)")
                if fixes.frontmatter_repaired:
                    parts.append(f"{fixes.frontmatter_repaired} frontmatter")
                detail = f" [{', '.join(parts)}]" if parts else ""
                _auto_commit(
                    project,
                    "lint",
                    f"repaired {fixes.total_fixed} page(s){detail}",
                )
        _print_report(report)
        click.echo("")
        click.echo(
            f"Fixed: {fixes.total_fixed} "
            f"(broken links: {fixes.broken_links_fixed}, "
            f"frontmatter: {fixes.frontmatter_repaired})"
        )
        if fixes.skipped_human_edited:
            click.echo(f"Skipped {fixes.skipped_human_edited} human-edited page(s).")
        return

    report = linter.run_all()
    _print_report(report)
    if not report.is_healthy:
        raise click.exceptions.Exit(code=1)


@main.command()
@click.option(
    "--sync",
    is_flag=True,
    default=False,
    help="Apply git truth to manifest + frontmatter for drifted pages.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def protect(sync: bool, project: Path | None) -> None:
    """Reconcile human-edit flags with git history.

    Default behavior scans for pages whose manifest flag disagrees
    with git and prints a report. ``--sync`` applies the fix: updates
    the manifest + frontmatter and emits a HUMAN_EDIT event.
    """
    from wikiloom.locking import FileLock
    from wikiloom.protection import HumanEditProtection

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    pp = HumanEditProtection(project)
    if sync:
        _require_clean_tree(project, "protect --sync")
        with FileLock(project):
            drifted = pp.sync()
            if drifted:
                _sync_cache(project)
                _auto_commit(
                    project,
                    "protect",
                    f"reclassified {len(drifted)} page(s)",
                )
    else:
        drifted = pp.scan()

    if not drifted:
        click.echo("Human-edit flags are in sync with git.")
        return

    verb = "Reclassified" if sync else "Drift detected on"
    click.echo(f"{verb} {len(drifted)} page(s):")
    for page in drifted:
        arrow = "→" if page.git_says else "←"
        click.echo(
            f"  {page.page_id} {arrow} human_edited={page.git_says} "
            f"(last commit: {page.last_commit_type})"
        )
    if not sync:
        raise click.exceptions.Exit(code=1)


@main.command()
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def reindex(project: Path | None) -> None:
    """Regenerate the root index and every non-archive sub-index.

    Reads live state from the manifest + on-disk frontmatter, preserves
    each index file's existing YAML header, and produces deterministic
    output so unchanged rebuilds don't create cosmetic git diffs.
    """
    from wikiloom.locking import FileLock
    from wikiloom.search import IndexUpdater

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    _require_clean_tree(project, "reindex")
    with FileLock(project):
        written = IndexUpdater(project / "wiki").rebuild_all()
        if written:
            _auto_commit(
                project,
                "reindex",
                f"rebuilt {len(written)} index file(s)",
            )
    click.echo(f"Rebuilt {len(written)} index file(s).")


@main.command("relink")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def relink(project: Path | None) -> None:
    """Re-run the linker across all wiki pages.

    Pages created early in an ingest may have missed links to pages
    that didn't exist yet. This command re-links every page against
    the full current manifest, catching connections the first pass
    missed.
    """
    from wikiloom.backlinks import BacklinkRegistry
    from wikiloom.linker import LinkingEngine
    from wikiloom.locking import FileLock
    from wikiloom.registry import Registry
    from wikiloom.search import IndexUpdater

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    cfg = _load_config(project)
    linking_cfg = cfg.linking if cfg is not None else None

    wiki_dir = project / "wiki"
    all_pages = sorted(
        p for p in wiki_dir.rglob("*.md")
        if p.name != "index.md" and p.name != "log.md"
        and "archive" not in p.parts
    )

    if not all_pages:
        click.echo("No pages to link.")
        return

    _require_clean_tree(project, "relink")
    click.echo(f"Re-linking {len(all_pages)} page(s)...")

    with FileLock(project):
        registry = Registry(project / "_registry")
        linker = LinkingEngine(registry, config=linking_cfg)
        linked = linker.link_all(all_pages)

        # Rebuild backlinks after re-linking
        backlinks = BacklinkRegistry(project / "_registry")
        backlinks.rebuild()
        backlinks.save()

        # Rebuild indexes
        IndexUpdater(wiki_dir, registry=registry).rebuild_all()

        _sync_cache(project)
        if linked:
            _auto_commit(
                project,
                "relink",
                f"updated wikilinks across {len(linked)} page(s)",
            )

    click.echo(f"Linked {len(linked)} page(s) with new wikilinks.")
    click.echo(f"Backlinks rebuilt.")


@main.command("query")
@click.argument("question", required=False, default=None)
@click.option(
    "--detail",
    is_flag=True,
    default=False,
    help="Show sources, confidence, cost, and follow-ups alongside the answer.",
)
@click.option(
    "--last-detail",
    is_flag=True,
    default=False,
    help="Show sources, confidence, and cost from the most recent query.",
)
@click.option(
    "--save-last",
    is_flag=True,
    default=False,
    help="Save the most recent query answer as a wiki page in syntheses/.",
)
@click.option(
    "--max-pages",
    type=int,
    default=5,
    help="Maximum number of wiki pages to inject as LLM context.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def query(
    question: str | None,
    detail: bool,
    last_detail: bool,
    save_last: bool,
    max_pages: int,
    project: Path | None,
) -> None:
    """Ask a question and get an answer grounded in the wiki's content.

    Default output shows just the answer. Use ``--detail`` to include
    sources, confidence, cost, and suggested follow-ups. Use ``--last-detail``
    to view detail from the most recent query. Use ``--save-last``
    to save the most recent answer as a synthesis page.
    """
    import json as json_mod

    from wikiloom.llm import LLMClient
    from wikiloom.query import run_query
    from wikiloom.utils import now_iso

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    if not save_last:
        _warn_if_dirty(project)

    last_query_path = project / "_registry" / "last_query.json"

    # --save-last: save the cached last query as a synthesis page
    if save_last:
        if not last_query_path.exists():
            raise click.ClickException(
                "No previous query result found. Run a query first."
            )
        _require_clean_tree(project, "query --save-last")
        data = json_mod.loads(last_query_path.read_text(encoding="utf-8"))
        # _save_query_as_page performs the cache sync and auto-commit.
        _save_query_as_page(data, project)
        return

    # --last-detail: show detail from the cached result, no LLM call
    if last_detail:
        if not last_query_path.exists():
            raise click.ClickException("No previous query result found.")
        data = json_mod.loads(last_query_path.read_text(encoding="utf-8"))
        click.echo(data.get("answer", ""))
        click.echo("")
        _print_query_detail(data, project)
        return

    if not question:
        raise click.UsageError("Missing argument 'QUESTION'. Use --last-detail or --save-last for the previous result.")

    cfg = _load_config(project)
    if cfg is None:
        raise click.ClickException(
            "Could not load wikiloom.toml. Run inside a project directory."
        )

    llm_client = LLMClient(cfg)

    # Load embedder for semantic fallback if enabled
    embedder = None
    if cfg.embeddings.enabled:
        try:
            from wikiloom.embeddings import get_embedder
            embedder = get_embedder(cfg.embeddings)
        except (ImportError, ValueError):
            pass  # embedding provider not installed; FTS5-only

    import threading
    import sys

    stop_spinner = threading.Event()

    def _spinner() -> None:
        frames = ["Searching wiki...", "Reading pages...", "Thinking..."]
        i = 0
        while not stop_spinner.is_set():
            msg = frames[min(i, len(frames) - 1)]
            sys.stderr.write(f"\r{msg:<30}")
            sys.stderr.flush()
            i += 1
            stop_spinner.wait(timeout=2.0)
        sys.stderr.write(f"\r{'':<30}\r")
        sys.stderr.flush()

    spinner_thread = threading.Thread(target=_spinner, daemon=True)
    spinner_thread.start()

    try:
        answer = run_query(
            question=question,
            project_root=project,
            llm_client=llm_client,
            max_context_pages=max_pages,
            embedder=embedder,
        )
    except Exception as exc:
        stop_spinner.set()
        spinner_thread.join()
        raise click.ClickException(str(exc)) from exc

    stop_spinner.set()
    spinner_thread.join()

    # Save result for --last
    result_data = {
        "question": question,
        "answer": answer.answer,
        "sources_consulted": [
            {"page_path": s.page_path, "relevance": s.relevance}
            for s in answer.sources_consulted
        ],
        "confidence": answer.confidence,
        "suggest_synthesis": answer.suggest_synthesis,
        "suggested_followups": answer.suggested_followups,
        "tokens_in": answer.metrics.tokens_in,
        "tokens_out": answer.metrics.tokens_out,
        "cost_usd": answer.metrics.cost_usd,
        "timestamp": now_iso(),
    }
    last_query_path.parent.mkdir(parents=True, exist_ok=True)
    last_query_path.write_text(
        json_mod.dumps(result_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Print the answer
    click.echo(answer.answer)

    # --detail: show metadata inline
    if detail:
        click.echo("")
        _print_query_detail(result_data, project)

    if not detail:
        hints = []
        hints.append("--detail for sources and metadata")
        hints.append("`wikiloom query --last-detail` to review later")
        if answer.suggest_synthesis:
            hints.append("`wikiloom query --save-last` to save as a wiki page")
        click.echo(f"\nRun with {', '.join(hints)}.")


def _save_query_as_page(data: dict, project: Path) -> None:
    """Save a cached query result as a synthesis page."""
    from wikiloom.frontmatter import Frontmatter, write_page
    from wikiloom.registry import PageEntry, Registry
    from wikiloom.utils import now_iso, slugify

    question = data.get("question", "query-answer")
    answer_text = data.get("answer", "")
    confidence = data.get("confidence", "medium")
    sources = data.get("sources_consulted", [])

    slug = slugify(question)[:60] or "query-answer"
    page_id = f"syntheses/{slug}"
    page_path = project / "wiki" / "syntheses" / f"{slug}.md"

    fm = Frontmatter(
        title=question,
        type="synthesis",
        status="active",
        created=now_iso(),
        modified=now_iso(),
        summary=answer_text[:160].replace("\n", " "),
        sources=sources,
        source_count=len(sources),
        confidence=confidence,
    )
    write_page(page_path, fm, answer_text)

    registry = Registry(project / "_registry")
    entry = PageEntry(
        title=question,
        type="synthesis",
        summary=answer_text[:160].replace("\n", " "),
        confidence=confidence,
    )
    registry.register_page(page_id, entry)
    registry.save()
    _sync_cache(project)
    title_snippet = question[:60]
    _auto_commit(project, "query", f'saved synthesis "{title_snippet}"')

    click.echo(f"Saved to {page_path.relative_to(project)}")


def _print_query_detail(data: dict, project: Path) -> None:
    """Print the detail view for a query result."""
    from wikiloom.frontmatter import read_page
    from wikiloom.registry import Registry

    sources = data.get("sources_consulted", [])
    if sources:
        registry = Registry(project / "_registry")
        click.echo("Sources:")
        for src in sources:
            page_path = src.get("page_path", "")
            relevance = src.get("relevance", "")

            # Resolve friendly name: page title + original source file
            title = page_path
            source_file = ""
            modified = ""
            page_status = ""
            entry = registry.get_page(page_path) if page_path else None
            if entry:
                title = entry.title
                modified = (entry.modified or "")[:10]  # YYYY-MM-DD
                if entry.status and entry.status != "active":
                    page_status = entry.status
            page_file = project / "wiki" / f"{page_path}.md"
            if page_file.exists():
                fm, _ = read_page(page_file)
                if fm and fm.sources:
                    for s in fm.sources:
                        if isinstance(s, dict) and s.get("name"):
                            source_file = s["name"]
                            break

            line = f"  [{relevance}] {title}"
            if source_file:
                line += f" (from {source_file})"
            if modified:
                line += f" — modified {modified}"
            if page_status:
                line += f" [{page_status}]"
            click.echo(line)
            click.echo(f"            → {page_path}.md")

    confidence = data.get("confidence", "")
    tokens_in = data.get("tokens_in", 0)
    tokens_out = data.get("tokens_out", 0)
    cost = data.get("cost_usd", 0.0)
    click.echo(f"\nConfidence: {confidence}")
    click.echo(f"Tokens: {tokens_in + tokens_out} (${cost:.4f})")

    followups = data.get("suggested_followups", [])
    if followups:
        click.echo("\nSuggested follow-ups:")
        for f in followups:
            click.echo(f"  - {f}")


@main.command("status")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def status(project: Path | None) -> None:
    """Show a project summary: page counts, last ingest, monthly cost."""
    from wikiloom.cache import SQLiteCache
    from wikiloom.chunk_store import ChunkStore
    from wikiloom.events import parse_log
    from wikiloom.source_catalog import SourceCatalog

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    _warn_if_dirty(project)

    registry_dir = project / "_registry"
    cache = SQLiteCache(registry_dir / "wiki.db")
    cache.full_rebuild(project)
    stats = cache.get_stats()

    click.echo(f"WikiLoom project: {project.name}")
    click.echo(f"  Pages: {stats['total_pages']}")
    if stats["by_type"]:
        for t, count in sorted(stats["by_type"].items()):
            click.echo(f"    {t}: {count}")
    if stats["by_status"]:
        active_n = stats["by_status"].get("active", 0)
        dormant_n = stats["by_status"].get("dormant", 0)
        deprecated_n = stats["by_status"].get("deprecated", 0)
        click.echo(
            f"  Status: active {active_n}, dormant {dormant_n}, "
            f"deprecated {deprecated_n}"
        )
    click.echo(f"  Human-edited: {stats['human_edited']}")
    click.echo(f"  Backlinks: {stats['backlinks']}")
    click.echo(f"  Aliases: {stats['aliases']}")

    # Orphan count
    from wikiloom.backlinks import BacklinkRegistry
    from wikiloom.registry import Registry

    registry_obj = Registry(registry_dir)
    bl = BacklinkRegistry(registry_dir)
    linked_pages: set[str] = set()
    for edge in bl.edges:
        linked_pages.add(edge.source)
        linked_pages.add(edge.target)
    orphan_count = sum(
        1 for pid, entry in registry_obj.pages.items()
        if entry.status != "deprecated"
        and entry.type != "index"
        and pid not in linked_pages
    )
    click.echo(f"  Orphans: {orphan_count}")

    chunk_store = ChunkStore(registry_dir / "wiki.db")
    click.echo(f"  Chunks stored: {chunk_store.count()}")

    if registry_dir.exists():
        catalog = SourceCatalog(registry_dir)
        source_count = len(catalog._entries)  # noqa: SLF001
        click.echo(f"  Sources ingested: {source_count}")

    from wikiloom.ingest.state import IngestState

    incomplete = IngestState.load(registry_dir)
    if incomplete is not None:
        pending = incomplete.pending_indices()
        total_chunks = len(incomplete.chunks)
        done = total_chunks - len(pending)
        click.echo("")
        click.echo(
            f"  WARNING: incomplete ingest for {incomplete.source_name}"
        )
        click.echo(
            f"           ({done}/{total_chunks} chunks synthesized "
            f"— re-run with --force)"
        )

    events = parse_log(project / "wiki" / "log.md")
    if events:
        last = events[0]
        click.echo(
            f"  Last event: {last['event_type']} | "
            f"{last['description']} ({last['timestamp']})"
        )
        total_tokens = sum(int(e.get("tokens_used", 0)) for e in events)
        total_cost = sum(float(e.get("cost_usd", 0.0)) for e in events)
        click.echo(f"  Total tokens: {total_tokens:,}")
        click.echo(f"  Total cost: ${total_cost:.2f}")


@main.command("log")
@click.option("--limit", "-n", type=int, default=10, help="Number of recent events to show.")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def log_cmd(limit: int, project: Path | None) -> None:
    """Show recent events from the wiki event log."""
    from wikiloom.events import parse_log

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    _warn_if_dirty(project)

    events = parse_log(project / "wiki" / "log.md")
    if not events:
        click.echo("No events recorded yet.")
        return

    shown = events[:limit]
    for event in shown:
        ts = event.get("timestamp", "?")
        etype = event.get("event_type", "?")
        desc = event.get("description", "")
        tokens = event.get("tokens_used", 0)
        cost = event.get("cost_usd", 0.0)
        commit = event.get("commit", "")

        line = f"[{ts}] {etype} | {desc}"
        extras = []
        if tokens:
            extras.append(f"{int(tokens):,}t")
        if cost:
            extras.append(f"${float(cost):.2f}")
        if commit:
            extras.append(str(commit)[:8])
        if extras:
            line += f"  ({', '.join(extras)})"
        click.echo(line)

    if len(events) > limit:
        click.echo(f"\n... {len(events) - limit} more event(s). Use -n to see more.")


@main.command("cost")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def cost(project: Path | None) -> None:
    """Show token usage and spend breakdown."""
    from wikiloom.events import parse_log

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    _warn_if_dirty(project)

    events = parse_log(project / "wiki" / "log.md")
    if not events:
        click.echo("No events with cost data yet.")
        return

    by_type: dict[str, dict[str, float]] = {}
    for event in events:
        etype = str(event.get("event_type", "other"))
        tokens = int(event.get("tokens_used", 0))
        cost_usd = float(event.get("cost_usd", 0.0))
        bucket = by_type.setdefault(etype, {"tokens": 0, "cost": 0.0, "count": 0})
        bucket["tokens"] += tokens
        bucket["cost"] += cost_usd
        bucket["count"] += 1

    total_tokens = 0
    total_cost = 0.0
    total_events = 0

    click.echo("Event type       Count    Tokens       Cost")
    click.echo("---------------- -------- ------------ --------")
    for etype in sorted(by_type):
        b = by_type[etype]
        t = int(b["tokens"])
        c = b["cost"]
        n = int(b["count"])
        total_tokens += t
        total_cost += c
        total_events += n
        click.echo(f"{etype:<16} {n:>8} {t:>12,} ${c:>7.2f}")
    click.echo("---------------- -------- ------------ --------")
    click.echo(f"{'Total':<16} {total_events:>8} {total_tokens:>12,} ${total_cost:>7.2f}")

    cfg = _load_config(project)
    if cfg is not None:
        budget = cfg.llm.monthly_budget_usd
        pct = (total_cost / budget * 100) if budget > 0 else 0
        click.echo(f"\nMonthly budget: ${budget:.2f} ({pct:.1f}% used)")


@main.command("show")
@click.argument("page_id")
@click.option(
    "--field",
    "field",
    type=str,
    default=None,
    help="Print just one frontmatter field (e.g. sources, aliases, modified).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit JSON instead of pretty-printed YAML.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def show(
    page_id: str,
    field: str | None,
    as_json: bool,
    project: Path | None,
) -> None:
    """Show a page's frontmatter metadata.

    Default mode pretty-prints the full frontmatter. Use --field to
    extract a single field; chunk_ids is computed by flattening every
    source's chunk_ids list.
    """
    import json as json_mod

    from wikiloom.frontmatter import read_page

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    _warn_if_dirty(project)

    page_id = page_id.replace(".md", "").strip("/")
    if page_id.startswith("wiki/"):
        page_id = page_id[len("wiki/"):]
    page_path = project / "wiki" / f"{page_id}.md"
    if not page_path.exists():
        raise click.ClickException(f"Page not found: {page_id}")

    fm, _ = read_page(page_path)
    if fm is None:
        raise click.ClickException(f"No frontmatter in {page_id}")

    data = fm.to_dict()
    # Synthetic field: flat chunk_ids across all sources.
    data["chunk_ids"] = fm.all_chunk_ids()

    if field is not None:
        if field not in data:
            available = ", ".join(sorted(data.keys()))
            raise click.ClickException(
                f"Unknown field {field!r}. Available: {available}"
            )
        value = data[field]
        if as_json:
            click.echo(json_mod.dumps(value, indent=2, ensure_ascii=False))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    click.echo(
                        json_mod.dumps(item, ensure_ascii=False)
                    )
                else:
                    click.echo(str(item))
        elif value is None:
            click.echo("(none)")
        else:
            click.echo(str(value))
        return

    if as_json:
        click.echo(json_mod.dumps(data, indent=2, ensure_ascii=False))
        return

    # Pretty default — key: value, lists/dicts compact.
    click.echo(f"page: {page_id}\n")
    for key, value in data.items():
        if value in ([], {}, None):
            click.echo(f"  {key}: -")
            continue
        if isinstance(value, list):
            click.echo(f"  {key} ({len(value)}):")
            for item in value:
                if isinstance(item, dict):
                    name = (
                        item.get("name")
                        or item.get("page_id")
                        or item.get("hash")
                        or json_mod.dumps(item, ensure_ascii=False)
                    )
                    extra = ""
                    if "chunk_ids" in item:
                        extra = f" (chunks: {len(item.get('chunk_ids') or [])})"
                    click.echo(f"    - {name}{extra}")
                else:
                    click.echo(f"    - {item}")
        elif isinstance(value, dict):
            click.echo(f"  {key}: {json_mod.dumps(value, ensure_ascii=False)}")
        else:
            click.echo(f"  {key}: {value}")


@main.command("links")
@click.argument("page_id")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def links(page_id: str, project: Path | None) -> None:
    """Show all pages linked to and from a given page."""
    from wikiloom.backlinks import BacklinkRegistry
    from wikiloom.registry import Registry

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    _warn_if_dirty(project)

    page_id = page_id.replace(".md", "").strip("/")
    if page_id.startswith("wiki/"):
        page_id = page_id[len("wiki/"):]

    registry = Registry(project / "_registry")
    page = registry.get_page(page_id)
    if page is None:
        raise click.ClickException(f"Page not found: {page_id}")

    bl = BacklinkRegistry(project / "_registry")

    outbound = []
    inbound = []
    for edge in bl.edges:
        if edge.source == page_id:
            outbound.append(edge)
        if edge.target == page_id:
            inbound.append(edge)

    click.echo(f"Links for: {page.title} ({page_id})\n")

    if outbound:
        click.echo(f"Outbound ({len(outbound)}):")
        for edge in outbound:
            target = registry.get_page(edge.target)
            title = target.title if target else edge.target
            click.echo(f"  → {title}")
            click.echo(f"    {edge.target}.md")
    else:
        click.echo("Outbound: none")

    click.echo("")

    if inbound:
        click.echo(f"Inbound ({len(inbound)}):")
        for edge in inbound:
            source = registry.get_page(edge.source)
            title = source.title if source else edge.source
            click.echo(f"  ← {title}")
            click.echo(f"    {edge.source}.md")
    else:
        click.echo("Inbound: none")

    total = len(outbound) + len(inbound)
    click.echo(f"\nTotal: {total} link(s)")


@main.command("related")
@click.argument("page_id")
@click.option("-n", "--limit", type=int, default=5, help="Number of related pages (max 10).")
@click.option("--save", is_flag=True, default=False, help="Write related pages into the page's frontmatter.")
@click.option("--link", is_flag=True, default=False, help="Also append wikilinks in a Related Pages section in the page body.")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def related(page_id: str, limit: int, save: bool, link: bool, project: Path | None) -> None:
    """Find pages semantically similar to a given page.

    Uses embedding cosine similarity to find related pages that may
    not have explicit wikilinks between them.
    """
    from wikiloom.cache import SQLiteCache
    from wikiloom.embeddings import deserialize_embedding
    from wikiloom.frontmatter import read_page, write_page

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    if not (save or link):
        _warn_if_dirty(project)

    limit = min(limit, 10)

    # Strip .md suffix if user included it
    page_id = page_id.replace(".md", "").strip("/")
    if page_id.startswith("wiki/"):
        page_id = page_id[len("wiki/"):]

    cache = SQLiteCache(project / "_registry" / "wiki.db")
    page = cache.get_page(page_id)
    if page is None:
        raise click.ClickException(f"Page not found: {page_id}")

    # Get this page's embedding
    import sqlite3
    conn = sqlite3.connect(str(project / "_registry" / "wiki.db"))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT embedding FROM pages WHERE page_id = ?", (page_id,)
    ).fetchone()
    conn.close()

    if row is None or row["embedding"] is None:
        raise click.ClickException(
            f"No embedding for {page_id}. Run: wikiloom rebuild-cache"
        )

    page_vec = deserialize_embedding(row["embedding"])

    # Exclude pages already linked to/from the target
    from wikiloom.backlinks import BacklinkRegistry

    bl = BacklinkRegistry(project / "_registry")
    linked_ids: set[str] = set()
    for edge in bl.edges:
        if edge.source == page_id:
            linked_ids.add(edge.target)
        if edge.target == page_id:
            linked_ids.add(edge.source)

    results = cache.semantic_search(page_vec, limit=limit + len(linked_ids) + 1)

    # Filter out the page itself, already-linked pages, and apply threshold
    threshold = 0.60
    related_pages = []
    for r in results:
        if r["page_id"] == page_id:
            continue
        if r["page_id"] in linked_ids:
            continue
        sim = r.get("similarity", 0.0)
        if sim < threshold:
            continue
        related_pages.append((r["page_id"], r["title"], sim))
        if len(related_pages) >= limit:
            break

    if not related_pages:
        click.echo(f"No related pages found for {page_id}.")
        return

    click.echo(f"Related to: {page['title']} ({page_id})\n")
    for pid, title, sim in related_pages:
        click.echo(f"  {sim:.0%}  {title}")
        click.echo(f"       → {pid}.md")

    if save or link:
        _require_clean_tree(project, "related")
        page_path = project / "wiki" / f"{page_id}.md"
        if not page_path.exists():
            raise click.ClickException(f"Page file not found: {page_path}")

        fm, body = read_page(page_path)
        if fm is None:
            raise click.ClickException(f"No frontmatter in {page_id}")

        if save:
            existing = set(fm.related_pages or [])
            for pid, _, _ in related_pages:
                if pid not in existing:
                    fm.related_pages.append(pid)
                    existing.add(pid)

        if link:
            # Append a Related Pages section with wikilinks, deduplicating
            related_section_header = "## Related Pages"
            existing_links = set()
            if related_section_header in body:
                for line in body.splitlines():
                    if line.strip().startswith("- [["):
                        # Extract page_id from [[page_id|title]]
                        inner = line.strip()[4:]  # after "- [["
                        pid_end = inner.find("|")
                        if pid_end == -1:
                            pid_end = inner.find("]]")
                        if pid_end > 0:
                            existing_links.add(inner[:pid_end])

            new_links = []
            for pid, title, _ in related_pages:
                if pid not in existing_links:
                    new_links.append(f"- [[{pid}|{title}]]")

            if new_links:
                if related_section_header not in body:
                    body = body.rstrip() + f"\n\n{related_section_header}\n\n"
                else:
                    body = body.rstrip() + "\n"
                body += "\n".join(new_links) + "\n"

        write_page(page_path, fm, body)
        _sync_cache(project)
        _auto_commit(
            project, "related", f"updated {page_id} with related pages"
        )

        actions = []
        if save:
            actions.append(f"{len(related_pages)} related page(s) to frontmatter")
        if link:
            actions.append(f"{len(new_links)} wikilink(s) to page body")
        click.echo(f"\nSaved: {', '.join(actions)}.")

    if not save and not link and related_pages:
        click.echo(
            "\nRun with --save to record these in frontmatter, --link to append"
            "\nwikilinks in the page body, or both flags together to do both."
        )


@main.command("orphans")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def orphans(project: Path | None) -> None:
    """List pages with no inbound or outbound wikilinks."""
    from wikiloom.backlinks import BacklinkRegistry
    from wikiloom.registry import Registry

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    _warn_if_dirty(project)

    registry = Registry(project / "_registry")
    bl = BacklinkRegistry(project / "_registry")

    # Build set of all pages that have any link (inbound or outbound)
    linked_pages: set[str] = set()
    for edge in bl.edges:
        linked_pages.add(edge.source)
        linked_pages.add(edge.target)

    # Find pages in manifest that aren't in the linked set. Dormant
    # pages count as orphans the same as active ones — being old
    # doesn't change whether anything links to them.
    orphan_list = []
    for page_id, entry in registry.pages.items():
        if entry.status == "deprecated":
            continue
        if entry.type == "index":
            continue
        if page_id not in linked_pages:
            orphan_list.append((page_id, entry.title, entry.type))

    if not orphan_list:
        click.echo("No orphan pages found.")
        return

    click.echo(f"Orphan pages ({len(orphan_list)}):\n")
    for pid, title, ptype in sorted(orphan_list):
        click.echo(f"  [{ptype}] {title}")
        click.echo(f"          {pid}.md")

    example_pid = sorted(orphan_list)[0][0]
    click.echo(
        f"\nTo find connections for a page, pass its path (without .md). Example:"
        f"\n  wikiloom related {example_pid}"
        f"\n\nOr run `wikiloom relink` to re-run the linker on all pages."
    )


@main.command("dormant")
@click.argument("page", required=False)
@click.option(
    "--list-marked",
    is_flag=True,
    default=False,
    help="List currently-marked dormant pages (instead of candidates).",
)
@click.option(
    "--windows",
    is_flag=True,
    default=False,
    help="Show the dormant window configuration.",
)
@click.option(
    "--unmark",
    is_flag=True,
    default=False,
    help="Unmark a page (flip dormant → active). Requires PAGE.",
)
@click.option(
    "--review",
    is_flag=True,
    default=False,
    help="Walk through dormant candidates interactively.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def dormant(
    page: str | None,
    list_marked: bool,
    windows: bool,
    unmark: bool,
    review: bool,
    project: Path | None,
) -> None:
    """Manage dormant pages: pages older than their window.

    Dormancy is a time-driven label, not a verdict on usefulness.
    A page is a candidate when its modified date exceeds the window
    configured in [dormant] for its type. Marking is a user decision.

    Modes:

    \b
      wikiloom dormant                 list candidates (active past window)
      wikiloom dormant --list-marked   list pages currently marked dormant
      wikiloom dormant --windows       show window config
      wikiloom dormant <page>          mark a page as dormant
      wikiloom dormant <page> --unmark flip a dormant page back to active
      wikiloom dormant --review        walk through candidates interactively
    """
    mode_count = sum([bool(page), list_marked, windows, review])
    if mode_count > 1:
        raise click.UsageError(
            "Choose one mode: PAGE, --list-marked, --windows, or --review."
        )
    if unmark and not page:
        raise click.UsageError("--unmark requires a PAGE argument.")

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    if windows:
        _dormant_show_windows(project)
        return

    if list_marked:
        _warn_if_dirty(project)
        _dormant_list_marked(project)
        return

    if review:
        _require_clean_tree(project, "dormant --review")
        _dormant_review(project)
        return

    if page:
        _require_clean_tree(project, "dormant")
        page_id = page.replace(".md", "").strip("/")
        if page_id.startswith("wiki/"):
            page_id = page_id[len("wiki/"):]
        _dormant_set_status(project, page_id, mark=not unmark)
        return

    # Default: list candidates
    _warn_if_dirty(project)
    _dormant_list_candidates(project)


def _dormant_show_windows(project: Path) -> None:
    from wikiloom.config import DormantConfig

    loaded = _load_config(project)
    cfg = loaded.dormant if loaded is not None else DormantConfig()
    click.echo("Dormant windows (change in wikiloom.toml [dormant]):")
    click.echo(f"  entity:    {cfg.entity_window_days} days")
    click.echo(f"  concept:   {cfg.concept_window_days} days")
    click.echo(f"  synthesis: {cfg.synthesis_window_days} days")
    click.echo(f"  default:   {cfg.default_window_days} days")
    click.echo(
        "\nPer-page overrides via `dormant_window_days` in frontmatter."
    )


def _dormant_list_candidates(project: Path) -> None:
    from wikiloom.lint import WikiLinter

    loaded = _load_config(project)
    cfg = loaded.dormant if loaded is not None else None
    linter = WikiLinter(project, dormant=cfg)
    candidates = linter.check_dormant()

    if not candidates:
        click.echo("No dormant candidates — all active pages are within their windows.")
        return

    click.echo(
        f"Dormant candidates ({len(candidates)}, active pages past window):\n"
    )
    for c in sorted(candidates, key=lambda x: -x.age_days):
        click.echo(
            f"  {c.page_id}  ({c.age_days}d old, window {c.window_days}d)"
        )
    click.echo(
        "\nMark a candidate with `wikiloom dormant <page>`, "
        "or walk through interactively with `wikiloom dormant --review`."
    )


def _dormant_list_marked(project: Path) -> None:
    from wikiloom.registry import Registry

    registry = Registry(project / "_registry")
    marked = [
        (pid, entry)
        for pid, entry in registry.pages.items()
        if entry.status == "dormant"
    ]
    if not marked:
        click.echo("No pages currently marked dormant.")
        return

    click.echo(f"Marked dormant ({len(marked)}):\n")
    for pid, entry in sorted(marked):
        modified = (entry.modified or "")[:10]
        click.echo(f"  {pid}  ({entry.title}) — last modified {modified}")
    click.echo(
        "\nUnmark with `wikiloom dormant <page> --unmark`."
    )


def _dormant_set_status(project: Path, page_id: str, mark: bool) -> None:
    """Flip a single page between active and dormant."""
    from wikiloom.frontmatter import parse_frontmatter, render_frontmatter
    from wikiloom.locking import FileLock
    from wikiloom.registry import Registry

    target_status = "dormant" if mark else "active"
    verb = "mark" if mark else "unmark"

    page_path = project / "wiki" / f"{page_id}.md"
    if not page_path.exists():
        raise click.ClickException(f"Page not found: {page_id}")

    with FileLock(project):
        registry = Registry(project / "_registry")
        entry = registry.get_page(page_id)
        if entry is None:
            raise click.ClickException(f"Page not in manifest: {page_id}")
        if mark and entry.status == "dormant":
            click.echo(f"{page_id} is already dormant.")
            return
        if not mark and entry.status != "dormant":
            click.echo(f"{page_id} is not dormant (status: {entry.status}).")
            return

        text = page_path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if fm is None:
            raise click.ClickException(f"No frontmatter in {page_id}")
        fm.status = target_status
        page_path.write_text(
            render_frontmatter(fm) + "\n" + body, encoding="utf-8"
        )
        entry.status = target_status
        registry.save()

        _sync_cache(project)
        _auto_commit(project, "dormant", f"{verb} {page_id}")

    click.echo(f"{verb.capitalize()}ed {page_id}.")


def _dormant_review(project: Path) -> None:
    """Interactive triage of dormant candidates."""
    from wikiloom.config import Config
    from wikiloom.frontmatter import parse_frontmatter, render_frontmatter
    from wikiloom.lint import WikiLinter
    from wikiloom.locking import FileLock
    from wikiloom.registry import Registry

    loaded = _load_config(project)
    cfg = loaded.dormant if loaded is not None else None
    candidates = WikiLinter(project, dormant=cfg).check_dormant()

    if not candidates:
        click.echo("No dormant candidates to review.")
        return

    total = len(candidates)
    marked = 0
    skipped = 0
    click.echo(f"Reviewing {total} candidate(s).")
    click.echo("For each: [m]ark dormant / [n]ext (skip) / [q]uit\n")

    for i, candidate in enumerate(sorted(candidates, key=lambda x: -x.age_days), start=1):
        registry = Registry(project / "_registry")
        entry = registry.get_page(candidate.page_id)
        if entry is None:
            continue
        click.echo(
            f"--- {i}/{total}: {candidate.page_id} ({entry.type})"
        )
        click.echo(f"  title: {entry.title}")
        click.echo(
            f"  age:   {candidate.age_days}d (window {candidate.window_days}d)"
        )
        if entry.summary:
            click.echo(f"  summary: {entry.summary[:100]}")

        choice = click.prompt(
            "Action",
            type=click.Choice(["m", "n", "q"], case_sensitive=False),
            default="n",
            show_choices=True,
            show_default=True,
        ).lower()

        if choice == "q":
            click.echo("Quit.")
            break
        if choice == "n":
            skipped += 1
            continue

        page_path = project / "wiki" / f"{candidate.page_id}.md"
        if not page_path.exists():
            click.echo(f"  ✗ skipped: file missing")
            skipped += 1
            continue
        with FileLock(project):
            text = page_path.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(text)
            if fm is None:
                click.echo(f"  ✗ skipped: no frontmatter")
                skipped += 1
                continue
            fm.status = "dormant"
            page_path.write_text(
                render_frontmatter(fm) + "\n" + body, encoding="utf-8"
            )
            entry.status = "dormant"
            registry.save()
            _sync_cache(project)
            _auto_commit(project, "dormant", f"mark {candidate.page_id}")
        click.echo(f"  ✓ marked dormant\n")
        marked += 1

    click.echo(f"\nDone. Marked: {marked}, skipped: {skipped}.")


@main.command("duplicates")
@click.option(
    "--slug-threshold",
    type=float,
    default=80.0,
    help="Slug fuzzy-match threshold 0-100 (default: 80).",
)
@click.option(
    "--embedding-threshold",
    type=float,
    default=0.85,
    help="Embedding cosine threshold 0-1 (default: 0.85).",
)
@click.option(
    "--cross-type",
    is_flag=True,
    default=False,
    help="Also compare pages across different types (default: same-type only).",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Max number of pairs to print.",
)
@click.option(
    "--review",
    is_flag=True,
    default=False,
    help="Walk through every pair interactively, choosing merge/skip/swap/quit.",
)
@click.option(
    "--auto-merge",
    is_flag=True,
    default=False,
    help="Auto-merge only safe pairs (singular/plural, prefix variants) above thresholds.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="With --auto-merge: print the plan without executing.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def duplicates(
    slug_threshold: float,
    embedding_threshold: float,
    cross_type: bool,
    limit: int,
    review: bool,
    auto_merge: bool,
    dry_run: bool,
    project: Path | None,
) -> None:
    """Find pages that may be duplicates of each other.

    Default mode lists suspect pairs sorted by similarity. Use
    --review for an interactive merge workflow, or --auto-merge to
    batch-resolve only the safe singular/plural and prefix variants.
    """
    from wikiloom.duplicates import find_duplicates, suggest_winner

    if review and auto_merge:
        raise click.UsageError("--review and --auto-merge are mutually exclusive.")

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    if review or auto_merge:
        _require_clean_tree(project, "duplicates")
    else:
        _warn_if_dirty(project)

    pairs = find_duplicates(
        project,
        slug_threshold=slug_threshold,
        embedding_threshold=embedding_threshold,
        same_type_only=not cross_type,
    )

    if not pairs:
        click.echo("No suspected duplicates found.")
        return

    if review:
        _run_review_mode(project, pairs)
        return

    if auto_merge:
        _run_auto_merge_mode(project, pairs, dry_run=dry_run)
        return

    shown = pairs[:limit]
    click.echo(f"Suspected duplicate pairs ({len(pairs)}):\n")
    for pair in shown:
        if pair.embedding_score >= 0:
            emb = f"emb {pair.embedding_score:.2f}"
        else:
            emb = "emb n/a"
        click.echo(f"  slug {pair.slug_score:.0f}% | {emb}")
        click.echo(f"    {pair.page_a}  ({pair.title_a})")
        click.echo(f"    {pair.page_b}  ({pair.title_b})")
        suggestion = suggest_winner(pair)
        click.echo(
            f"    → wikiloom merge {suggestion.winner_page_id} "
            f"{suggestion.loser_page_id}  ({suggestion.reason})\n"
        )

    if len(pairs) > limit:
        click.echo(
            f"... and {len(pairs) - limit} more. Pass --limit N to see more."
        )
    click.echo(
        "\nTip: use `wikiloom duplicates --review` to walk through them "
        "interactively, or `--auto-merge` for safe singular/plural variants."
    )


def _run_review_mode(project: Path, pairs: list) -> None:
    """Interactive walkthrough: prompt y/s/n/q for each pair."""
    from wikiloom.duplicates import suggest_winner
    from wikiloom.locking import FileLock
    from wikiloom.merge import merge_pages

    merged = 0
    skipped = 0
    total = len(pairs)
    click.echo(f"Reviewing {total} suspected duplicate pair(s).")
    click.echo("For each pair: [y]es merge / [s]wap winner-loser / [n]o skip / [q]uit\n")

    for i, pair in enumerate(pairs, start=1):
        suggestion = suggest_winner(pair)
        emb = f"{pair.embedding_score:.2f}" if pair.embedding_score >= 0 else "n/a"
        click.echo(f"--- Pair {i}/{total} (slug {pair.slug_score:.0f}%, emb {emb})")
        click.echo(f"  WINNER (kept):    {suggestion.winner_page_id}  ({pair.title_a if suggestion.winner_page_id == pair.page_a else pair.title_b})")
        click.echo(f"  LOSER (archived): {suggestion.loser_page_id}  ({pair.title_b if suggestion.winner_page_id == pair.page_a else pair.title_a})")
        click.echo(f"  reason: {suggestion.reason}")

        choice = click.prompt(
            "Action",
            type=click.Choice(["y", "s", "n", "q"], case_sensitive=False),
            default="n",
            show_choices=True,
            show_default=True,
        ).lower()

        if choice == "q":
            click.echo("Quit.")
            break
        if choice == "n":
            skipped += 1
            continue

        winner = suggestion.winner_page_id
        loser = suggestion.loser_page_id
        if choice == "s":
            winner, loser = loser, winner

        try:
            with FileLock(project):
                merge_pages(project, winner, loser)
                _sync_cache(project)
                _auto_commit(project, "merge", f"{loser} into {winner}")
            click.echo(f"  ✓ merged {loser} → {winner}\n")
            merged += 1
        except ValueError as exc:
            click.echo(f"  ✗ skipped: {exc}\n")
            skipped += 1

    click.echo(f"\nDone. Merged: {merged}, skipped: {skipped}.")


def _run_auto_merge_mode(project: Path, pairs: list, dry_run: bool) -> None:
    """Batch-merge only pairs flagged is_safe_to_auto by suggest_winner."""
    from wikiloom.duplicates import suggest_winner
    from wikiloom.locking import FileLock
    from wikiloom.merge import merge_pages

    safe_plan: list = []
    unsafe: list = []
    for pair in pairs:
        suggestion = suggest_winner(pair)
        if suggestion.is_safe_to_auto:
            safe_plan.append((pair, suggestion))
        else:
            unsafe.append((pair, suggestion))

    if not safe_plan:
        click.echo("No pairs match the safe auto-merge criteria.")
        if unsafe:
            click.echo(
                f"\n{len(unsafe)} pair(s) need manual review. "
                f"Run `wikiloom duplicates --review` to walk through them."
            )
        return

    click.echo(f"Plan ({len(safe_plan)} safe merge(s)):\n")
    for pair, suggestion in safe_plan:
        emb = f"{pair.embedding_score:.2f}" if pair.embedding_score >= 0 else "n/a"
        click.echo(
            f"  {suggestion.loser_page_id}  →  {suggestion.winner_page_id}"
            f"  (slug {pair.slug_score:.0f}%, emb {emb}, {suggestion.reason})"
        )

    if unsafe:
        click.echo(
            f"\nSkipping {len(unsafe)} ambiguous pair(s) — "
            f"use `wikiloom duplicates --review` for those."
        )

    if dry_run:
        click.echo("\nDry run. Nothing executed.")
        return

    if not click.confirm(f"\nProceed with {len(safe_plan)} merge(s)?"):
        click.echo("Aborted.")
        return

    merged = 0
    skipped = 0
    for pair, suggestion in safe_plan:
        try:
            with FileLock(project):
                merge_pages(
                    project,
                    suggestion.winner_page_id,
                    suggestion.loser_page_id,
                )
                _sync_cache(project)
                _auto_commit(
                    project,
                    "merge",
                    f"{suggestion.loser_page_id} into {suggestion.winner_page_id}",
                )
            merged += 1
        except ValueError as exc:
            click.echo(
                f"  ✗ {suggestion.loser_page_id}: {exc}"
            )
            skipped += 1

    click.echo(f"\nDone. Merged: {merged}, skipped: {skipped}.")


@main.command("merge")
@click.argument("winner")
@click.argument("loser")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def merge(winner: str, loser: str, yes: bool, project: Path | None) -> None:
    """Merge LOSER page into WINNER page.

    Combines bodies (loser appended under a "Merged content" section
    for human reconciliation), unions aliases/sources/chunk_ids,
    rewrites inbound [[loser]] wikilinks to [[winner]], deprecates the
    loser to wiki/archive/, and commits with a merge: prefix.
    """
    from wikiloom.locking import FileLock
    from wikiloom.merge import merge_pages

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    # Strip .md / wiki/ prefix as the other commands do
    winner = winner.replace(".md", "").strip("/")
    if winner.startswith("wiki/"):
        winner = winner[len("wiki/"):]
    loser = loser.replace(".md", "").strip("/")
    if loser.startswith("wiki/"):
        loser = loser[len("wiki/"):]

    if winner == loser:
        raise click.UsageError("WINNER and LOSER must be different pages.")

    _require_clean_tree(project, "merge")

    if not yes:
        click.echo(f"Merge: {loser}  →  {winner}")
        click.echo("This will:")
        click.echo(f"  - append {loser}'s body into {winner}")
        click.echo(f"  - rewrite all [[{loser}]] wikilinks to [[{winner}]]")
        click.echo(f"  - move {loser}.md into wiki/archive/")
        click.echo(f"  - record {loser} as superseded_by {winner}")
        if not click.confirm("\nProceed?"):
            click.echo("Aborted.")
            return

    try:
        with FileLock(project):
            result = merge_pages(project, winner, loser)
            _sync_cache(project)
            _auto_commit(
                project,
                "merge",
                f"{loser} into {winner}",
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Merged {loser} into {winner}.")
    if result.rewrote_links_in:
        click.echo(
            f"Rewrote wikilinks in {len(result.rewrote_links_in)} page(s)."
        )
    if result.archive_path is not None:
        click.echo(
            f"Archived {loser} → {result.archive_path.relative_to(project)}"
        )


@main.command("deprecate")
@click.argument("page_id")
@click.option(
    "--superseded-by",
    "superseded_by",
    type=str,
    default=None,
    help="page_id of the replacement page (optional).",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def deprecate(
    page_id: str,
    superseded_by: str | None,
    yes: bool,
    project: Path | None,
) -> None:
    """Soft-remove an active page: move to wiki/archive/ and set status=deprecated.

    The page file moves out of its category directory but stays on
    disk in wiki/archive/. The manifest entry stays, with status flipped
    to deprecated. Auto-tools and CLI commands stop surfacing the page,
    but readers can still find it in archive.

    To undo: `git revert` the deprecate commit.
    To remove permanently: `wikiloom purge <page_id>` after deprecation.
    """
    from wikiloom.locking import FileLock
    from wikiloom.registry import Registry

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    page_id = page_id.replace(".md", "").strip("/")
    if page_id.startswith("wiki/"):
        page_id = page_id[len("wiki/"):]
    if superseded_by:
        superseded_by = superseded_by.replace(".md", "").strip("/")
        if superseded_by.startswith("wiki/"):
            superseded_by = superseded_by[len("wiki/"):]

    _require_clean_tree(project, "deprecate")

    registry = Registry(project / "_registry", wiki_dir=project / "wiki")
    entry = registry.get_page(page_id)
    if entry is None:
        raise click.ClickException(f"Page not found in manifest: {page_id}")
    if entry.status == "deprecated":
        raise click.ClickException(f"{page_id} is already deprecated.")
    if superseded_by and registry.get_page(superseded_by) is None:
        raise click.ClickException(
            f"--superseded-by target not found: {superseded_by}"
        )

    if not yes:
        click.echo(f"Deprecate: {page_id}  ({entry.title})")
        if superseded_by:
            click.echo(f"  superseded_by: {superseded_by}")
        click.echo(f"  Will move wiki/{page_id}.md → wiki/archive/")
        if not click.confirm("\nProceed?"):
            click.echo("Aborted.")
            return

    with FileLock(project):
        archive_path = registry.deprecate_page(
            page_id,
            superseded_by=superseded_by,
            move_to_archive=True,
            emit_event=True,
        )
        registry.save()
        _sync_cache(project)
        suffix = f" (superseded by {superseded_by})" if superseded_by else ""
        _auto_commit(project, "deprecate", f"{page_id}{suffix}")

    click.echo(f"Deprecated {page_id}.")
    if archive_path is not None:
        click.echo(f"Archived to {archive_path.relative_to(project)}")
    click.echo(
        "To undo: run `git revert HEAD` to restore the previous state."
    )


@main.command("purge")
@click.argument("page_id")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the typed-confirmation prompt. Use only in scripts.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def purge(page_id: str, yes: bool, project: Path | None) -> None:
    """Permanently remove an already-deprecated page.

    Deletes the file from wiki/archive/ and removes the manifest entry.
    This is destructive — the page cannot be recovered through wikiloom
    after this command (only via git history).

    Refuses to run on active pages — deprecate them first with
    `wikiloom deprecate <page_id>`.
    """
    from wikiloom.locking import FileLock
    from wikiloom.registry import Registry

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    page_id = page_id.replace(".md", "").strip("/")
    if page_id.startswith("wiki/"):
        page_id = page_id[len("wiki/"):]

    _require_clean_tree(project, "purge")

    registry = Registry(project / "_registry", wiki_dir=project / "wiki")
    entry = registry.get_page(page_id)
    if entry is None:
        raise click.ClickException(f"Page not found in manifest: {page_id}")
    if entry.status != "deprecated":
        raise click.ClickException(
            f"{page_id} is not deprecated (status: {entry.status}). "
            f"Run `wikiloom deprecate {page_id}` first to soft-remove it."
        )

    archive_name = page_id.replace("/", "__") + ".md"
    archive_file = project / "wiki" / "archive" / archive_name

    if not yes:
        click.echo(f"⚠ PURGE: {page_id}  ({entry.title})")
        click.echo("  This will permanently:")
        if archive_file.exists():
            click.echo(f"  - delete {archive_file.relative_to(project)}")
        else:
            click.echo("  - delete the manifest entry (archive file already missing)")
        click.echo(f"  - remove the manifest entry for {page_id}")
        click.echo("  This cannot be undone via wikiloom (only via git revert).")
        typed = click.prompt(f"\nType the page_id to confirm", default="", show_default=False)
        if typed.strip() != page_id:
            click.echo("Aborted: confirmation did not match.")
            return

    with FileLock(project):
        if archive_file.exists():
            archive_file.unlink()
        # Remove manifest entry directly — there's no Registry method for this
        # because nothing else needs it. Purge is the only consumer.
        if page_id in registry._pages:  # noqa: SLF001
            del registry._pages[page_id]  # noqa: SLF001
            registry.save()
        _sync_cache(project)
        _auto_commit(project, "deprecate", f"purge {page_id}")

    click.echo(f"Purged {page_id}.")


@main.command("review")
@click.option(
    "--accept-all",
    is_flag=True,
    default=False,
    help="Accept every pending link without prompting.",
)
@click.option(
    "--clear",
    is_flag=True,
    default=False,
    help="Discard all pending links without inserting any.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def review(accept_all: bool, clear: bool, project: Path | None) -> None:
    """Review and action low-confidence link candidates.

    The linking engine defers links below the medium-confidence
    threshold to ``_registry/pending.json`` instead of auto-inserting
    them. This command lets you batch-accept or batch-clear those
    candidates. ``--accept-all`` inserts every pending link into its
    source page; ``--clear`` discards them. Without flags, prints the
    list for manual inspection.
    """
    import json

    from wikiloom.frontmatter import parse_frontmatter, render_frontmatter
    from wikiloom.locking import FileLock

    if accept_all and clear:
        raise click.UsageError("--accept-all and --clear are mutually exclusive.")

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    pending_path = project / "_registry" / "pending.json"
    if not pending_path.exists():
        click.echo("No pending links.")
        return

    data = json.loads(pending_path.read_text(encoding="utf-8"))
    items = data.get("pending", []) if isinstance(data, dict) else data
    if not items:
        click.echo("No pending links.")
        return

    if clear:
        _require_clean_tree(project, "review --clear")
        with FileLock(project):
            data["pending"] = []
            pending_path.write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
            _auto_commit(
                project, "review", f"cleared {len(items)} pending link(s)"
            )
        click.echo(f"Cleared {len(items)} pending link(s).")
        return

    if not accept_all:
        click.echo(f"Pending links ({len(items)}):")
        for item in items:
            click.echo(
                f"  {item.get('source_page', '?')} → "
                f"[[{item.get('candidate_page_id', '?')}]] "
                f"(matched: {item.get('matched_text', '?')!r}, "
                f"score: {item.get('score', '?')})"
            )
        click.echo("")
        click.echo(
            "Run with --accept-all to insert all, or --clear to discard."
        )
        return

    # --accept-all: insert each pending link into its source page.
    _require_clean_tree(project, "review --accept-all")
    wiki_dir = project / "wiki"
    inserted = 0
    with FileLock(project):
        for item in items:
            source_page = item.get("source_page", "")
            target = item.get("candidate_page_id", "")
            matched_text = item.get("matched_text", "")
            if not source_page or not target or not matched_text:
                continue

            page_path = wiki_dir / f"{source_page}.md"
            if not page_path.exists():
                continue

            text = page_path.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(text)
            wikilink = f"[[{target}|{matched_text}]]"
            new_body = body.replace(matched_text, wikilink, 1)
            if new_body != body:
                if fm is not None:
                    page_path.write_text(
                        render_frontmatter(fm) + "\n" + new_body,
                        encoding="utf-8",
                    )
                else:
                    page_path.write_text(new_body, encoding="utf-8")
                inserted += 1

        data["pending"] = []
        pending_path.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        if inserted:
            _sync_cache(project)
            _auto_commit(
                project, "review", f"accepted {inserted} pending link(s)"
            )

    click.echo(f"Inserted {inserted} link(s), cleared pending list.")


@main.command("source")
@click.argument("chunk_id")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def source(chunk_id: str, project: Path | None) -> None:
    """Show the raw source text for a chunk referenced by a wiki page.

    Pages synthesized via ``wikiloom ingest`` carry ``chunk_ids`` in
    their frontmatter. Pass one of those ids here to see the exact
    text the LLM saw when it produced that page — structural
    provenance click-through without trusting the LLM's self-
    attribution.
    """
    from wikiloom.chunk_store import ChunkStore
    from wikiloom.source_catalog import SourceCatalog

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    chunk_store = ChunkStore(project / "_registry" / "wiki.db")
    chunk = chunk_store.get_chunk(chunk_id)
    if chunk is None:
        raise click.ClickException(
            f"No chunk found with id {chunk_id!r}. It may have been "
            f"removed by a re-ingest with different content."
        )

    catalog = SourceCatalog(project / "_registry")
    source_entry = catalog.get(chunk.source_hash)
    source_name = source_entry.name if source_entry else "<unknown source>"
    raw_path = (
        source_entry.raw_path if source_entry and source_entry.raw_path else "—"
    )

    click.echo(f"chunk_id:     {chunk.chunk_id}")
    click.echo(f"source:       {source_name}")
    click.echo(f"raw_path:     {raw_path}")
    click.echo(f"chunk:        {chunk.chunk_index + 1} of {chunk.chunk_total}")
    click.echo(f"content_type: {chunk.content_type}")
    click.echo(f"tokens:       {chunk.token_estimate}")
    click.echo(f"created_at:   {chunk.created_at}")
    click.echo("---")
    click.echo(chunk.text)


@main.command("save")
@click.option(
    "-m",
    "--message",
    type=str,
    default=None,
    help="Optional commit message. Defaults to 'human-edit: N page(s) [protected]'.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be committed without creating a commit.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def save(message: str | None, dry_run: bool, project: Path | None) -> None:
    """Commit manual edits under wiki/ with a human-edit: prefix.

    Use after editing wiki pages in your editor. The resulting commit
    is classified as human-authored so auto-tools (lint --fix, re-ingest)
    leave the affected pages alone.
    """
    from wikiloom.git_ops import GitOps
    from wikiloom.locking import FileLock

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    try:
        git = GitOps(project)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    dirty = git.dirty_wiki_paths()
    if not dirty:
        click.echo("Nothing to save — working tree is clean.")
        return

    if dry_run:
        click.echo(f"Would commit {len(dirty)} file(s):")
        for p in dirty:
            click.echo(f"  {p}")
        default_msg = message or f"human-edit: {len(dirty)} page(s) [protected]"
        click.echo(f"\nMessage: {default_msg}")
        return

    commit_msg = message or f"human-edit: {len(dirty)} page(s) [protected]"
    if not commit_msg.startswith("human-edit:"):
        commit_msg = f"human-edit: {commit_msg}"

    with FileLock(project):
        # Auto-bump frontmatter.modified on each saved page so manual
        # edits don't silently roll into dormant. Also flip dormant →
        # active since the user just touched the page.
        freshened = _bump_modified_and_freshen(project, dirty)
        git.repo.git.add("-A", "--", "wiki")
        git.commit([], commit_msg)
        _sync_cache(project)

    click.echo(f"Saved {len(dirty)} file(s).")
    if freshened:
        click.echo(f"Freshened {freshened} dormant page(s) back to active.")


def _bump_modified_and_freshen(project: Path, paths: list[Path]) -> int:
    """For each wiki page in ``paths``, bump frontmatter.modified to now
    and flip dormant → active. Returns the count of pages freshened.

    Skips files without frontmatter, missing files, and non-page files
    (everything outside ``wiki/`` or named ``log.md`` / ``index.md``).
    """
    from wikiloom.frontmatter import parse_frontmatter, render_frontmatter
    from wikiloom.utils import now_iso

    freshened = 0
    timestamp = now_iso()
    for rel in paths:
        if rel.parts[:1] != ("wiki",) or rel.name in ("log.md", "index.md"):
            continue
        full = project / rel
        if not full.exists() or full.suffix != ".md":
            continue
        text = full.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if fm is None:
            continue
        original_status = fm.status
        fm.modified = timestamp
        if fm.status == "dormant":
            fm.status = "active"
            freshened += 1
        new_text = render_frontmatter(fm) + "\n" + body
        if new_text != text or original_status != fm.status:
            full.write_text(new_text, encoding="utf-8")
    return freshened


@main.command("rebuild-cache")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def rebuild_cache(project: Path | None) -> None:
    """Regenerate the SQLite query cache from manifest + backlinks.

    The cache at ``_registry/wiki.db`` is a git-ignored derived index.
    Run this if it's missing, corrupt, or suspected to be out of sync.
    """
    from wikiloom.cache import SQLiteCache
    from wikiloom.embeddings import load_embedder
    from wikiloom.locking import FileLock

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    embedder = load_embedder(project)
    if embedder is not None:
        click.echo("Computing embeddings...")

    with FileLock(project):
        cache = SQLiteCache(project / "_registry" / "wiki.db")
        count = cache.full_rebuild(project, embedder=embedder)
    stats = cache.get_stats()
    click.echo(
        f"Cache rebuilt: {count} page(s), "
        f"{stats['aliases']} alias(es), "
        f"{stats['backlinks']} backlink(s)."
    )


def _print_report(report) -> None:
    """Render a ``LintReport`` to stdout."""
    if report.is_healthy:
        click.echo("Wiki is healthy.")
        return

    click.echo(f"Issues found: {report.total_issues}")
    if report.broken_links:
        click.echo(f"  Broken links ({len(report.broken_links)}):")
        for b in report.broken_links[:10]:
            click.echo(f"    {b.source} → {b.target} ({b.reason})")
    if report.orphans:
        click.echo(f"  Orphans ({len(report.orphans)}): {', '.join(report.orphans[:10])}")
    if report.dormant:
        click.echo(
            f"  Dormant candidates ({len(report.dormant)}, informational — "
            f"use `wikiloom dormant <page>` to mark):"
        )
        for d in report.dormant[:10]:
            click.echo(f"    {d.page_id} ({d.age_days}d > {d.window_days}d)")
    if report.duplicates:
        click.echo(f"  Duplicates ({len(report.duplicates)}):")
        for d in report.duplicates[:10]:
            click.echo(f"    {' ~ '.join(d.pages)} ({d.reason}, {d.score}%)")
    if report.frontmatter_issues:
        click.echo(
            f"  Frontmatter issues ({len(report.frontmatter_issues)}): "
            f"{', '.join(report.frontmatter_issues[:10])}"
        )
    if report.index_drift:
        click.echo(
            f"  Index drift ({len(report.index_drift)}): "
            f"{', '.join(report.index_drift)}"
        )
    if report.contradictions:
        click.echo(f"  Contradictions ({len(report.contradictions)}):")
        for c in report.contradictions[:10]:
            click.echo(f"    {c.page_id}: {c.existing[:40]} vs {c.new[:40]}")
    if report.stubs:
        click.echo(f"  Stubs ({len(report.stubs)}): {', '.join(report.stubs[:10])}")


if __name__ == "__main__":
    main()
