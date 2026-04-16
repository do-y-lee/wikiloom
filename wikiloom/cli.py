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

    click.echo(f"Extracted: {result.content.content_type} "
               f"({result.content.token_estimate} tokens estimated)")
    if result.raw_path:
        click.echo(f"Copied to: {result.raw_path.relative_to(project)}")
    click.echo(f"Chunks: {len(result.chunks)} "
               f"(needs_chunking={result.budget.needs_chunking})")
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
@click.argument("question")
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
def query(question: str, save: bool, max_pages: int, project: Path | None) -> None:
    """Ask a question and get an answer grounded in the wiki's content.

    Retrieves relevant pages via full-text search, injects them as LLM
    context, and returns a structured answer with source citations.
    Pass ``--save`` to file the answer as a synthesis page.
    """
    from wikiloom.config import Config
    from wikiloom.frontmatter import Frontmatter, write_page
    from wikiloom.ingest.errors import IngestError
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

    try:
        cfg = Config.load(project)
    except FileNotFoundError:
        raise click.ClickException(
            "Could not load wikiloom.toml. Run inside a project directory."
        )

    llm_client = LLMClient(cfg)

    try:
        answer = run_query(
            question=question,
            project_root=project,
            llm_client=llm_client,
            max_context_pages=max_pages,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    # Print the answer
    click.echo(answer.answer)
    click.echo("")

    # Sources
    if answer.sources_consulted:
        click.echo("Sources:")
        for src in answer.sources_consulted:
            click.echo(f"  [{src.relevance}] {src.page_path}")

    click.echo(f"\nConfidence: {answer.confidence}")
    click.echo(
        f"Tokens: {answer.metrics.tokens_in + answer.metrics.tokens_out} "
        f"(${answer.metrics.cost_usd:.4f})"
    )

    if answer.suggested_followups:
        click.echo("\nSuggested follow-ups:")
        for followup in answer.suggested_followups:
            click.echo(f"  - {followup}")

    if answer.suggest_synthesis and not save:
        click.echo(
            "\nThis answer could be a good synthesis page. "
            "Re-run with --save to file it."
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

    with FileLock(project):
        cache = SQLiteCache(project / "_registry" / "wiki.db")
        count = cache.full_rebuild(project)
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
