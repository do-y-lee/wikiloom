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
def ingest(source: str, project: Path | None) -> None:
    """Ingest a source file or URL into the wiki.

    Runs extraction, copies the source to raw/, plans the token budget,
    and chunks oversized files. The LLM synthesis / linking / commit
    steps will be wired up as their components land.
    """
    from wikiloom.ingest.processor import ingest as run_ingest

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found). "
                "Run inside a project directory or pass --project."
            )

    result = run_ingest(source, project_root=project)

    click.echo(f"Extracted: {result.content.content_type} "
               f"({result.content.token_estimate} tokens estimated)")
    if result.raw_path:
        click.echo(f"Copied to: {result.raw_path.relative_to(project)}")
    click.echo(f"Chunks: {len(result.chunks)} "
               f"(needs_chunking={result.budget.needs_chunking})")
    for note in result.notes:
        click.echo(f"Note: {note}")


if __name__ == "__main__":
    main()
