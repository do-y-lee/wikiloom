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
    from wikiloom.ingest.processor import ingest as run_ingest

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found). "
                "Run inside a project directory or pass --project."
            )

    result = run_ingest(source, project_root=project, force=force)

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
