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


if __name__ == "__main__":
    main()
