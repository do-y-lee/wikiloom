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
def init(name: str, path: Path | None, domain: str) -> None:
    """Initialize a new WikiLoom project.

    Creates the full directory structure, config files, git repo,
    and empty registry files.
    """
    from wikiloom.scaffold import init_project

    project_dir = init_project(name=name, path=path, domain=domain)
    click.echo(f"Initialized WikiLoom project at {project_dir}")


def _find_project_root(start: Path) -> Path | None:
    """Walk upward from `start` looking for a wikiloom.toml."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "wikiloom.toml").exists():
            return candidate
    return None


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
def ingest(source: str, project: Path | None, force: bool) -> None:
    """Ingest a source file or URL into the wiki.

    Extracts content, copies local files to raw/, rebuilds backlinks and
    indexes, and commits the result. Re-ingesting an identical local
    file is a cheap no-op (catalog dedup) unless ``--force`` is passed.
    """
    from wikiloom.ingest.errors import IngestError
    from wikiloom.ingest.processor import ingest as run_ingest

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found). "
                "Run inside a project directory or pass --project."
            )

    try:
        result = run_ingest(source, project_root=project, force=force)
    except IngestError as exc:
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


@main.command()
@click.option(
    "--fix",
    is_flag=True,
    default=False,
    help="Apply auto-fixes for broken links, missing frontmatter, and stale pages.",
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
    from wikiloom.config import Config
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

    try:
        cfg = Config.load(project)
        staleness = cfg.staleness
    except FileNotFoundError:
        staleness = None

    linter = WikiLinter(project, staleness=staleness)

    if fix:
        with FileLock(project):
            report = linter.run_all()
            fixes = linter.fix_all(report)
        _print_report(report)
        click.echo("")
        click.echo(
            f"Fixed: {fixes.total_fixed} "
            f"(broken links: {fixes.broken_links_fixed}, "
            f"stale: {fixes.stale_marked}, "
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
        with FileLock(project):
            drifted = pp.sync()
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

    with FileLock(project):
        written = IndexUpdater(project / "wiki").rebuild_all()
    click.echo(f"Rebuilt {len(written)} index file(s).")


@main.command("query")
@click.argument("question", required=False, default=None)
@click.option(
    "--detail",
    is_flag=True,
    default=False,
    help="Show sources, confidence, cost, and follow-ups alongside the answer.",
)
@click.option(
    "--last",
    is_flag=True,
    default=False,
    help="Show detail from the most recent query without making an LLM call.",
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help="File the answer as a synthesis page in wiki/syntheses/.",
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
    last: bool,
    save: bool,
    max_pages: int,
    project: Path | None,
) -> None:
    """Ask a question and get an answer grounded in the wiki's content.

    Default output shows just the answer. Use ``--detail`` to include
    sources, confidence, cost, and suggested follow-ups. Use ``--last``
    to view detail from the most recent query without another LLM call.
    """
    import json as json_mod

    from wikiloom.config import Config
    from wikiloom.frontmatter import Frontmatter, write_page
    from wikiloom.llm import LLMClient
    from wikiloom.query import run_query
    from wikiloom.registry import PageEntry, Registry
    from wikiloom.utils import now_iso, slugify

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    last_query_path = project / "_registry" / "last_query.json"

    # --last: show detail from the cached result, no LLM call
    if last:
        if not last_query_path.exists():
            raise click.ClickException("No previous query result found.")
        data = json_mod.loads(last_query_path.read_text(encoding="utf-8"))
        click.echo(data.get("answer", ""))
        click.echo("")
        _print_query_detail(data, project)
        return

    if not question:
        raise click.UsageError("Missing argument 'QUESTION'. Use --last to view the previous result.")

    try:
        cfg = Config.load(project)
    except FileNotFoundError:
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

    try:
        answer = run_query(
            question=question,
            project_root=project,
            llm_client=llm_client,
            max_context_pages=max_pages,
            embedder=embedder,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

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

    if not detail and answer.suggest_synthesis and not save:
        click.echo(
            "\nThis answer could be a good synthesis page. "
            "Re-run with --save to file it."
        )

    if not detail and not save:
        click.echo(
            "\nRun with --detail for sources and metadata, "
            "or `wikiloom query --last` to see the most recent query with metadata and sources."
        )

    # --save: write as a synthesis page
    if save:
        slug = slugify(question)[:60] or "query-answer"
        page_id = f"syntheses/{slug}"
        page_path = project / "wiki" / "syntheses" / f"{slug}.md"

        sources_list = [
            {"page_path": s.page_path, "relevance": s.relevance}
            for s in answer.sources_consulted
        ]
        fm = Frontmatter(
            title=question,
            type="synthesis",
            status="active",
            created=now_iso(),
            modified=now_iso(),
            summary=answer.answer[:160].replace("\n", " "),
            sources=sources_list,
            source_count=len(sources_list),
            confidence=answer.confidence,
        )
        write_page(page_path, fm, answer.answer)

        registry = Registry(project / "_registry")
        entry = PageEntry(
            title=question,
            type="synthesis",
            summary=answer.answer[:160].replace("\n", " "),
            confidence=answer.confidence,
        )
        registry.register_page(page_id, entry)
        registry.save()

        click.echo(f"\nSaved to {page_path.relative_to(project)}")


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
            entry = registry.get_page(page_path) if page_path else None
            if entry:
                title = entry.title
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
        for s, count in sorted(stats["by_status"].items()):
            if s != "active":
                click.echo(f"    ({s}): {count}")
    click.echo(f"  Human-edited: {stats['human_edited']}")
    click.echo(f"  Backlinks: {stats['backlinks']}")
    click.echo(f"  Aliases: {stats['aliases']}")

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

    try:
        from wikiloom.config import Config
        cfg = Config.load(project)
        budget = cfg.llm.monthly_budget_usd
        pct = (total_cost / budget * 100) if budget > 0 else 0
        click.echo(f"\nMonthly budget: ${budget:.2f} ({pct:.1f}% used)")
    except FileNotFoundError:
        pass


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
        with FileLock(project):
            data["pending"] = []
            pending_path.write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
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
    from wikiloom.locking import FileLock

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    embedder = None
    try:
        from wikiloom.config import Config
        from wikiloom.embeddings import get_embedder

        cfg = Config.load(project)
        if cfg.embeddings.enabled:
            embedder = get_embedder(cfg.embeddings)
            click.echo("Computing embeddings...")
    except (FileNotFoundError, ImportError, ValueError):
        pass

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
    if report.stale:
        click.echo(f"  Stale ({len(report.stale)}):")
        for s in report.stale[:10]:
            click.echo(f"    {s.page_id} ({s.age_days}d > {s.window_days}d)")
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
