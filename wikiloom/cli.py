"""CLI entry point for WikiLoom."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import click
from dotenv import find_dotenv, load_dotenv

from wikiloom.cli_output import (
    check as _check,
    cross as _cross,
    dim as _dim,
    done_summary,
    skip_mark as _skip_mark,
)


# Ordered list of (category, commands) tuples. The main --help uses
# this to render commands in logical groups instead of one flat
# alphabetical list. Any command not listed here falls into a trailing
# "Other" section so the help output never silently drops a command.
_COMMAND_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Setup", ["init"]),
    ("Ingest & write", ["ingest", "save", "merge", "relink"]),
    (
        "Read & explore",
        [
            "status",
            "log",
            "edits",
            "cost",
            "links",
            "show",
            "orphans",
            "related",
            "query",
            "source",
        ],
    ),
    (
        "Maintenance",
        ["duplicates", "dormant", "lint", "protect", "reindex", "rebuild-cache"],
    ),
    ("Deprecation", ["deprecate", "purge"]),
]


def _bold(text: str) -> str:
    """Wrap ``text`` in ANSI bold — shared styling primitive for help output."""
    return click.style(text, bold=True)


def _cyan(text: str) -> str:
    """Wrap ``text`` in cyan — used for command names and option flags in help.

    Same hue the rest of the CLI uses for page_ids and other user-
    pointable identifiers, so ``--help`` teaches the same visual
    rule: cyan = something you type.
    """
    return click.style(text, fg="cyan")


class _StyledCommand(click.Command):
    """Command with bold section headers and surrounding blank lines.

    Click's default ``--help`` is cramped: no leading/trailing
    whitespace, plain section labels. This subclass adds consistent
    air around the whole block, emphasizes section names, bolds the
    "Usage:" prefix, and tints option flags cyan so they stand out
    from their descriptions. Applied to every command in the group
    via the ``_CategorizedGroup.command_class`` hook.
    """

    def format_usage(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # Click's default pushes ``[OPTIONS]`` to the front — standard
        # CLI genre convention. We reorder so positional args come
        # first and ``[OPTIONS]`` trails, matching the shape every
        # command's Examples block teaches.
        pieces = _usage_pieces_positional_first(self, ctx)
        formatter.write_usage(
            ctx.command_path, " ".join(pieces), prefix=_bold("Usage:") + " "
        )

    def format_options(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        opts: list[tuple[str, str]] = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                flags, description = rv
                opts.append((_cyan(flags), description))
        if opts:
            with formatter.section(_bold("Options")):
                formatter.write_dl(opts)

    def get_help(self, ctx: click.Context) -> str:
        return "\n" + super().get_help(ctx) + "\n"


def _usage_pieces_positional_first(
    cmd: click.Command, ctx: click.Context
) -> list[str]:
    """Reorder Click's usage pieces so positionals come before ``[OPTIONS]``.

    Click's default puts ``[OPTIONS]`` at index 0, producing
    ``[OPTIONS] SOURCE``. The rule here is plain: keep every other
    piece in its original order and move ``[OPTIONS]`` to the end.
    That gives ``SOURCE [OPTIONS]`` for leaf commands with a
    positional, ``COMMAND [ARGS]... [OPTIONS]`` for the top-level
    group, and ``[OPTIONS]`` alone for commands without positionals.
    """
    raw = list(cmd.collect_usage_pieces(ctx))
    options_metavar = cmd.options_metavar or "[OPTIONS]"
    if options_metavar not in raw:
        return raw
    raw.remove(options_metavar)
    raw.append(options_metavar)
    return raw


class _CategorizedGroup(click.Group):
    """Click Group that renders commands in named categories.

    The default Click formatter emits one flat alphabetical list,
    which doesn't scale past ~10 commands. This override groups
    commands by purpose so a new user can scan the help and know
    where to look. Also emits bold section headers and a blank-line
    frame around the whole help block so every ``--help`` invocation
    has the same visual shape as the rest of the CLI.
    """

    command_class = _StyledCommand

    def format_usage(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # Two-line Usage so the distinction between top-level flags
        # (wikiloom --version / --help) and subcommand invocations
        # (wikiloom ingest SOURCE --force) is visible up front. Leaf
        # commands still use the single-line reorder via
        # ``_StyledCommand.format_usage``.
        prog = ctx.command_path
        formatter.write(f"{_bold('Usage:')} {prog} [OPTIONS]\n")
        formatter.write(f"   or: {prog} COMMAND [ARGS]... [OPTIONS]\n")

    def format_options(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # Render the group's own options with a bold header, then
        # delegate to format_commands (Click's default glues the two
        # together inside format_options — we have to replicate that
        # or the command list disappears).
        opts: list[tuple[str, str]] = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                flags, description = rv
                opts.append((_cyan(flags), description))
        if opts:
            with formatter.section(_bold("Options")):
                formatter.write_dl(opts)
        self.format_commands(ctx, formatter)

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        commands: dict[str, click.Command] = {
            name: self.get_command(ctx, name) for name in self.list_commands(ctx)
        }
        listed: set[str] = set()

        for category, names in _COMMAND_CATEGORIES:
            rows: list[tuple[str, str]] = []
            for name in names:
                cmd = commands.get(name)
                if cmd is None or cmd.hidden:
                    continue
                rows.append((_cyan(name), cmd.get_short_help_str(limit=60)))
                listed.add(name)
            if rows:
                with formatter.section(_bold(category)):
                    formatter.write_dl(rows)

        leftover = [
            (_cyan(name), cmd.get_short_help_str(limit=60))
            for name, cmd in commands.items()
            if name not in listed and not cmd.hidden
        ]
        if leftover:
            with formatter.section(_bold("Other")):
                formatter.write_dl(leftover)

    def get_help(self, ctx: click.Context) -> str:
        return "\n" + super().get_help(ctx) + "\n"


@click.group(cls=_CategorizedGroup)
@click.version_option(version="0.1.0", prog_name="wikiloom")
def main() -> None:
    """WikiLoom — LLM-maintained knowledge bases with deterministic linking."""
    # Load `.env` from the project root (walks up from cwd until it
    # finds one). Existing shell exports win — `load_dotenv` does not
    # override, so users who set keys in `~/.zshrc` keep that behavior.
    load_dotenv(find_dotenv(usecwd=True))


def _maybe_create_env_file(
    project_dir: Path,
    provider: str,
    api_key_env: str | None,
    no_interactive: bool,
) -> str:
    """Optionally prompt to create `.env` and capture the API key.

    Returns one of:
      - ``"saved"``  — .env was created with the key filled in.
      - ``"empty"``  — .env was created but the key field is empty
                       (user hit Enter at the key prompt).
      - ``"skipped"`` — no .env was created; user should cp from
                        .env.example themselves.

    Ollama (no api_key_env) and non-interactive contexts always
    return ``"skipped"``.
    """
    from wikiloom.scaffold import _generate_env_example

    if not api_key_env:
        return "skipped"
    if no_interactive or not sys.stdin.isatty():
        return "skipped"

    click.echo("")
    if not click.confirm(
        f"Create .env now and paste your {api_key_env}?", default=True
    ):
        return "skipped"

    key_value = click.prompt(
        f"Paste your {api_key_env} (input hidden, press Enter to skip)",
        default="",
        show_default=False,
        hide_input=True,
    ).strip()

    env_content = _generate_env_example(provider)
    if key_value:
        env_content = env_content.replace(
            f"{api_key_env}=", f"{api_key_env}={key_value}", 1
        )
    (project_dir / ".env").write_text(env_content, encoding="utf-8")

    if key_value:
        click.echo(f"✓ Saved {api_key_env} to {project_dir}/.env")
        return "saved"
    click.echo(f"✓ Created {project_dir}/.env (key field left empty)")
    return "empty"


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
@click.option(
    "--no-interactive",
    is_flag=True,
    default=False,
    help="Skip the interactive .env / API-key prompt. Useful for scripted "
         "or CI-driven init. The prompt is also auto-skipped when stdin "
         "isn't a terminal.",
)
def init(
    name: str,
    path: Path | None,
    domain: str,
    provider: str,
    model: str | None,
    no_interactive: bool,
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

    click.echo("")
    click.echo(
        f"{_check()} Initialized WikiLoom project at "
        f"{click.style(str(project_dir), fg='cyan')}"
    )
    click.echo("")
    click.echo(click.style("Project", bold=True))
    click.echo(f"  {_dim('Domain:')}   {domain_line}")
    click.echo(f"  {_dim('Provider:')} {preset['label']}")
    click.echo(f"  {_dim('Model:')}    {click.style(chosen_model, fg='cyan')}")
    click.echo(f"  {_dim('Budget:')}   ${DEFAULT_MONTHLY_BUDGET_USD:g}/month")
    click.echo("")

    api_key_env = preset["api_key_env"]
    env_status = _maybe_create_env_file(
        project_dir=project_dir,
        provider=chosen_provider,
        api_key_env=api_key_env,
        no_interactive=no_interactive,
    )

    click.echo(click.style("Next steps", bold=True))
    click.echo("")

    if api_key_env:
        click.echo("  1. API key")
        if env_status == "saved":
            click.echo(
                f"     {_check()} {api_key_env} saved to {project_dir}/.env"
            )
        elif env_status == "empty":
            click.echo(
                f"     Edit {project_dir}/.env and set {api_key_env}=...")
            click.echo(_dim(f"     ({preset['api_key_hint']})"))
        else:
            click.echo(f"     cd {project_dir.name} && cp .env.example .env")
            click.echo(f"     Edit .env and set {api_key_env}=...")
            click.echo(_dim(f"     ({preset['api_key_hint']})"))
    else:
        click.echo("  1. Start your local LLM runtime")
        click.echo(_dim(f"     {preset['api_key_hint']}"))
    click.echo("")

    click.echo(
        "  2. (Recommended) Review the synthesis prompt — shapes every page WikiLoom writes")
    click.echo(f"     {click.style(str(prompt_path), fg='cyan')}")
    click.echo("")

    click.echo("  3. (Optional) Adjust LLM model, budget, or dormant windows")
    click.echo(f"     {click.style(str(config_path), fg='cyan')}")
    cheap_model = preset["cheap_model"]
    if cheap_model:
        click.echo(
            _dim(
                f"     Tip: switch to {cheap_model} for cheap iteration, "
                f"back to {chosen_model} once the prompt feels right."
            )
        )
    click.echo("")

    click.echo("  4. Ingest your first file")
    click.echo(f"     cd {project_dir.name}")
    click.echo(
        f"     {click.style('wikiloom ingest path/to/doc.pdf', fg='cyan')}"
    )
    click.echo("")
    click.echo(_dim("Run `wikiloom --help` to see all commands."))
    click.echo("")


def _find_project_root(start: Path) -> Path | None:
    """Walk upward from `start` looking for a wikiloom.toml."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "wikiloom.toml").exists():
            return candidate
    return None


def _sync_cache(
    project: Path, changed_files: list[Path] | None = None
) -> None:
    """Refresh the SQLite query cache (FTS + embeddings) from on-disk state.

    Every writer command calls this at the end so `wikiloom query` and
    `wikiloom related` see the new state without a manual rebuild-cache.

    ``changed_files``, when known, restricts the re-embed to just
    those page files — turning single-page writer commands from a
    multi-second full-wiki rebuild into a sub-100ms upsert. Leave
    as ``None`` when the change set is unknown or genuinely wiki-wide
    (relink, merges that rewrite inbound links everywhere) — the
    fallback is a full rebuild.
    """
    from wikiloom.cache import SQLiteCache
    from wikiloom.embeddings import load_embedder

    registry_dir = project / "_registry"
    if not registry_dir.exists():
        return
    SQLiteCache(registry_dir / "wiki.db").sync_from_files(
        project,
        changed_files=changed_files,
        embedder=load_embedder(project),
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
    """Print a passive nudge if any human-tracked file is uncommitted.

    Covers ``wiki/`` pages, ``wikiloom.toml``, and
    ``.wikiloom/prompts/*.md`` — everything ``wikiloom save`` handles.
    Called at the top of read-only and writer commands so users notice
    forgotten edits without being blocked. Writer commands that would
    collide with an auto-commit additionally use ``_require_clean_tree``
    (scoped to ``wiki/``) for hard protection.
    """
    from wikiloom.git_ops import GitOps

    try:
        dirty = GitOps(project).dirty_human_paths()
    except ValueError:
        return
    if dirty:
        n = len(dirty)
        plural = "" if n == 1 else "s"
        click.echo(
            f"⚠ {n} uncommitted edit{plural} — run `wikiloom save` to commit.\n",
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
        float(e.get("cost_usd", 0.0) or 0.0)
        for e in events
        if str(e.get("timestamp", "")).startswith(month_prefix)
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


def _format_event_timestamp(iso_str: str) -> str:
    """Render ISO timestamps from the event log as ``YYYY-MM-DD HH:MM``.

    log.md stores ISO-UTC strings like ``2026-04-22T22:07:16Z`` via
    ``now_iso()``. Display them in the same format ``wikiloom edits``
    uses so timestamps are consistent across read-side commands.
    Falls back to the raw string if parsing fails.
    """
    try:
        from datetime import datetime

        s = iso_str.rstrip("Z")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str


def _page_not_found_message(page_id: str) -> str:
    """Shared error text for commands that take a page_id argument.

    Adds a consistent recovery hint so users aren't left guessing
    how to find the right page_id. Used by related, show, links,
    dormant, deprecate, purge, and similar commands.
    """
    return (
        f"Page not found: {page_id}\n"
        f"Tip: run `wikiloom search <keyword>` to find the right "
        f"page_id, or `wikiloom orphans` to list candidates."
    )


def _auto_commit(project: Path, commit_type: str, description: str) -> str | None:
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
        return None
    # Stage modified + untracked files under the wiki-managed dirs.
    # `git add` on a directory path picks up both; deleted files are
    # captured via the -A flag so renames/removals also land.
    for scope in ("wiki", "_registry"):
        if (project / scope).exists():
            git.repo.git.add("-A", "--", scope)
    return git.commit([], f"{commit_type}: {description}") or None


def _commit_merge_log_tail(project: Path, subject: str) -> None:
    """Commit wiki/log.md after a merge flow.

    MERGE events are appended *after* the primary merge commit lands so
    the event can record that commit's hash (matching the ingest-tail
    pattern in processor.py step 16). This follow-up picks up the
    resulting log.md change. No-ops when the repo or log are missing or
    when log.md is unchanged.
    """
    from wikiloom.git_ops import GitOps

    log_path = project / "wiki" / "log.md"
    if not log_path.exists():
        return
    try:
        git = GitOps(project)
    except ValueError:
        return
    git.commit([log_path], subject)


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
    file is a cheap no-op (catalog dedup) unless --force is passed.

    \b
    Examples:
      \x1b[36mwikiloom ingest ~/docs/paper.pdf\x1b[0m
      \x1b[36mwikiloom ingest https://en.wikipedia.org/wiki/Chase_Bank\x1b[0m
      \x1b[36mwikiloom ingest ~/docs/paper.pdf --force\x1b[0m
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

    # Pre-flight: for local sources, fail fast with a friendly message
    # when the path is wrong. URLs are resolved at fetch time by the
    # extractor; we don't try to validate them here.
    is_url = source.startswith(("http://", "https://"))
    if not is_url:
        src_path = Path(source).expanduser()
        if not src_path.exists():
            raise click.ClickException(
                f"No such file: {source}\n"
                f"Check the path and try again. Tip: drag the file into "
                f"your terminal to get the exact path."
            )
        if src_path.is_dir():
            raise click.ClickException(
                f"Path is a directory, not a file: {source}\n"
                f"Ingest takes one file at a time. To process a folder, "
                f"loop over its files in your shell."
            )
        source = str(src_path)

    _require_clean_tree(project, "ingest")
    _warn_if_dirty(project)
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
    except FileNotFoundError as exc:
        # Backstop — the pre-flight check above covers the common case,
        # but an extractor could still hit a missing path (e.g. for a
        # nested resource). Keep the error friendly rather than leaking
        # the extractor's backend exception.
        raise click.ClickException(
            f"File not found during ingest: {exc}\nCheck the path and try again."
        ) from exc
    except PermissionError as exc:
        raise click.ClickException(
            f"Permission denied reading source: {exc}\n"
            f"Check file permissions (chmod / sudo)."
        ) from exc

    # Summary
    created = len(result.pages_created)
    updated = len(result.pages_updated)
    total_tok = result.total_tokens_in + result.total_tokens_out
    click.echo("")
    if created or updated:
        from wikiloom.cli_output import (
            done_summary,
            format_tokens as _fmt_tok,
        )

        click.echo(
            done_summary(
                [
                    f"{created} created",
                    f"{updated} updated",
                    f"{_fmt_tok(total_tok)} tok",
                    f"${result.total_cost_usd:.2f}",
                ]
            )
        )
    else:
        click.echo("  Done. no pages synthesized.")

    if result.notes:
        click.echo("")
        click.echo(click.style("Notes", bold=True))
        for note in result.notes:
            click.echo(f"  {_dim('•')} {note}")
    _post_flight_budget_warning(project)
    click.echo("")


@main.command()
@click.option(
    "--fix",
    is_flag=True,
    default=False,
    help="Apply auto-fixes for broken links and missing frontmatter.",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root. Defaults to walking upward from the current directory.",
)
def lint(fix: bool, project: Path | None) -> None:
    """Run health checks over a WikiLoom project.

    Default behavior prints a report and exits 1 if issues are found.
    Pass --fix to apply mechanical repairs (respecting human-edit
    protection).
    """
    from wikiloom.lint import WikiLinter
    from wikiloom.locking import FileLock

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
        import time as _time

        _require_clean_tree(project, "lint --fix")
        start = _time.monotonic()
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
        summary_parts = [f"{fixes.total_fixed} fixed"]
        if fixes.broken_links_fixed:
            summary_parts.append(f"{fixes.broken_links_fixed} broken links")
        if fixes.frontmatter_repaired:
            summary_parts.append(f"{fixes.frontmatter_repaired} frontmatter")
        if fixes.skipped_human_edited:
            summary_parts.append(
                f"{fixes.skipped_human_edited} human-edited skipped"
            )
        click.echo(
            done_summary(summary_parts, elapsed=_time.monotonic() - start)
        )
        click.echo("")
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
    with git and prints a report. --sync applies the fix: updates
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
                wiki_dir = project / "wiki"
                _sync_cache(
                    project,
                    changed_files=[
                        wiki_dir / f"{d.page_id}.md" for d in drifted
                    ],
                )
                _auto_commit(
                    project,
                    "protect",
                    f"reclassified {len(drifted)} page(s)",
                )
    else:
        drifted = pp.scan()

    if not drifted:
        click.echo("")
        click.echo(_dim("Human-edit flags are in sync with git."))
        click.echo("")
        return

    click.echo("")
    heading = "Reclassified" if sync else "Drift detected"
    click.echo(
        click.style(heading, bold=True)
        + f"  {_dim('(' + str(len(drifted)) + ')')}"
    )
    for page in drifted:
        arrow = "→" if page.git_says else "←"
        click.echo(
            f"  {click.style(page.page_id, fg='cyan')} {arrow} "
            f"human_edited={page.git_says}  "
            f"{_dim(f'(last commit: {page.last_commit_type})')}"
        )
    click.echo("")
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
    click.echo("")
    click.echo(
        f"{_check()} Rebuilt {len(written)} index file(s)."
    )
    click.echo("")


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
    total_pages = len(all_pages)
    click.echo("")
    click.echo(f"Re-linking {total_pages} page(s)...")
    click.echo("")

    import time as _time
    start = _time.monotonic()
    # Report every ~25 pages, or every 10% for smaller wikis, whichever
    # is coarser. Avoids flooding output on small wikis while keeping
    # reassurance visible on 500+ page wikis.
    step = max(25, max(1, total_pages // 10))

    def _progress(done: int, total: int) -> None:
        if done == total or done % step == 0:
            click.echo(f"  {_check()} {done}/{total} pages linked")

    with FileLock(project):
        registry = Registry(project / "_registry")
        # Linking requires an embedder — fail fast with a friendly
        # message when the project has embeddings disabled.
        from wikiloom.cache import SQLiteCache
        from wikiloom.embeddings import load_embedder

        _relink_embedder = load_embedder(project)
        if _relink_embedder is None:
            raise click.ClickException(
                "Linking requires an embedder. Enable [embeddings] in "
                "wikiloom.toml (provider = 'fastembed' is the local "
                "default) and try again."
            )
        linker = LinkingEngine(
            registry,
            embedder=_relink_embedder,
            cache=SQLiteCache(project / "_registry" / "wiki.db"),
            config=linking_cfg,
        )
        linked = linker.link_all(all_pages, progress=_progress)

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

    click.echo("")
    click.echo(
        done_summary(
            [f"{total_pages} pages", f"{len(linked)} updated"],
            elapsed=_time.monotonic() - start,
        )
    )
    click.echo("")


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

    Default output shows just the answer. Use --detail to include
    sources, confidence, cost, and suggested follow-ups. Use
    --last-detail to view detail from the most recent query. Use
    --save-last to save the most recent answer as a synthesis page.

    \b
    Examples:
      \x1b[36mwikiloom query "What is Chase Bank's history?"\x1b[0m
      \x1b[36mwikiloom query "What is Chase Bank's history?" --detail\x1b[0m
      \x1b[36mwikiloom query --last-detail\x1b[0m
      \x1b[36mwikiloom query --save-last\x1b[0m
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
        prev_question = data.get("question", "")
        click.echo("")
        if prev_question:
            click.echo(
                f"{click.style('Question:', bold=True)} {prev_question}"
            )
            click.echo("")
        click.echo(click.style("Answer:", bold=True))
        click.echo("")
        click.echo(data.get("answer", ""))
        click.echo("")
        _print_query_detail(data, project)
        click.echo("")
        return

    if not question:
        raise click.UsageError(
            "Missing argument 'QUESTION'. Use --last-detail or --save-last for the previous result.")

    import threading
    import sys

    # Echo the question back IMMEDIATELY — before config load,
    # embedder init, or any other slow setup — so the user sees
    # confirmation the command is running instead of staring at a
    # blank terminal while fastembed's ONNX model loads.
    click.echo("")
    click.echo(f"{click.style('Question:', bold=True)} {question}")
    click.echo("")

    stop_spinner = threading.Event()

    def _spinner() -> None:
        frames = [
            "Initializing...",
            "Searching wiki...",
            "Reading pages...",
            "Expanding linked context...",
            "Thinking...",
        ]
        # ``\033[2K`` clears the entire current line; ``\r`` returns
        # the cursor to column 0. ``\n\033[A`` reserves a blank row
        # below the spinner (newline drops us to the next row, then
        # cursor-up returns so the next frame rewrites the message
        # line). The visual effect: while the spinner animates, the
        # row beneath it stays empty — breathing room so the terminal
        # prompt below doesn't feel crammed against the moving line.
        ERASE_LINE = "\r\033[2K"
        i = 0
        while not stop_spinner.is_set():
            msg = frames[min(i, len(frames) - 1)]
            sys.stderr.write(
                f"{ERASE_LINE}  {click.style(msg, fg='cyan')}\n\033[A"
            )
            sys.stderr.flush()
            i += 1
            stop_spinner.wait(timeout=2.0)
        sys.stderr.write(ERASE_LINE)
        sys.stderr.flush()

    spinner_thread = threading.Thread(target=_spinner, daemon=True)
    spinner_thread.start()

    try:
        # Heavy setup lives inside the spinner-wrapped block so the
        # user sees the "Initializing..." phase while fastembed's
        # ONNX model loads instead of staring at a blank terminal.
        cfg = _load_config(project)
        if cfg is None:
            raise click.ClickException(
                "Could not load wikiloom.toml. Run inside a project directory."
            )

        llm_client = LLMClient(cfg, model=cfg.llm.for_query())

        embedder = None
        if cfg.embeddings.enabled:
            try:
                from wikiloom.embeddings import get_embedder
                embedder = get_embedder(cfg.embeddings)
            except (ImportError, ValueError):
                pass  # embedding provider not installed; FTS5-only

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

    # Print the answer with a bold section header so it visually
    # separates from the question block above.
    click.echo(click.style("Answer:", bold=True))
    click.echo("")
    click.echo(answer.answer)

    # --detail: show metadata inline
    if detail:
        click.echo("")
        _print_query_detail(result_data, project)

    if not detail:
        click.echo("")
        click.echo(_dim("Next:"))
        click.echo(
            _dim("  --last-detail  show sources and metadata for this answer")
        )
        if answer.suggest_synthesis:
            click.echo(
                _dim("  --save-last    save this answer as a synthesis page")
            )
    click.echo("")


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

    # Regenerate category and root indexes so the new synthesis page
    # appears in wiki/syntheses/index.md and wiki/index.md. Every
    # other writer path (ingest, relink) rebuilds indexes after a
    # manifest change — this one was missing that call, leaving the
    # indexes out of sync until the next ingest or relink.
    from wikiloom.search import IndexUpdater

    index_paths = IndexUpdater(
        project / "wiki", registry=registry
    ).rebuild_all()
    changed = [page_path, *index_paths]
    _sync_cache(project, changed_files=changed)
    title_snippet = question[:60]
    _auto_commit(project, "query", f'saved synthesis "{title_snippet}"')

    rel_path = page_path.relative_to(project)
    click.echo("")
    click.echo(
        f"{_check()} Saved synthesis  "
        f"{click.style(str(rel_path), fg='cyan')}"
    )
    click.echo("")


def _print_query_detail(data: dict, project: Path) -> None:
    """Print the detail view for a query result."""
    from wikiloom.cli_output import (
        check as _qcheck,
        dim as _qdim,
        done_summary as _qdone,
        format_tokens as _qfmt,
    )
    from wikiloom.frontmatter import read_page
    from wikiloom.registry import Registry

    sources = data.get("sources_consulted", [])
    if sources:
        registry = Registry(project / "_registry")
        click.echo(click.style("Sources", bold=True))
        click.echo("")
        for src in sources:
            page_path = src.get("page_path", "")
            relevance = src.get("relevance", "")

            title = page_path
            source_file = ""
            modified = ""
            page_status = ""
            entry = registry.get_page(page_path) if page_path else None
            if entry:
                title = entry.title
                modified = (entry.modified or "")[:10]
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

            rel_tag = _qdim(f"[{relevance:>6}]")
            headline = f"  {_qcheck()} {rel_tag}  {title}"
            bits: list[str] = []
            if source_file:
                bits.append(f"from {source_file}")
            if modified:
                bits.append(modified)
            if page_status:
                bits.append(page_status)
            if bits:
                headline += f"  {_qdim('• ' + ' • '.join(bits))}"
            click.echo(headline)
            click.echo(f"      {_qdim('→ ' + page_path + '.md')}")
        click.echo("")

    confidence = data.get("confidence", "")
    tokens_in = data.get("tokens_in", 0)
    tokens_out = data.get("tokens_out", 0)
    cost = data.get("cost_usd", 0.0)
    total_tok = tokens_in + tokens_out
    conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
        confidence, None
    )
    conf_display = (
        click.style(confidence, fg=conf_color) if conf_color else confidence
    )
    click.echo(
        _qdone(
            [
                f"{_qfmt(total_tok)} tok",
                f"${cost:.4f}",
                f"confidence: {conf_display}",
            ]
        )
    )

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

    # Read-only command. Stats come from whatever's in the SQLite cache,
    # which writer commands keep current.
    registry_dir = project / "_registry"
    cache = SQLiteCache(registry_dir / "wiki.db")
    stats = cache.get_stats()

    sep = _dim("•")

    click.echo("")
    click.echo(
        f"WikiLoom project: {click.style(project.name, fg='cyan', bold=True)}"
    )
    click.echo("")

    # Content section: page-type/status breakdown.
    click.echo(click.style("Content", bold=True))
    total_pages = stats["total_pages"]
    by_type = stats.get("by_type") or {}
    type_parts = [f"{c} {t}" for t, c in sorted(by_type.items())]
    type_suffix = f"  {_dim('(' + ', '.join(type_parts) + ')')}" if type_parts else ""
    click.echo(f"  Pages: {total_pages}{type_suffix}")

    by_status = stats.get("by_status") or {}
    active_n = by_status.get("active", 0)
    dormant_n = by_status.get("dormant", 0)
    deprecated_n = by_status.get("deprecated", 0)
    click.echo(
        f"  Status: {active_n} active  {sep}  "
        f"{dormant_n} dormant  {sep}  {deprecated_n} deprecated"
    )
    click.echo("")

    # Graph section: linkage health. Uses the shared orphan definition
    # so status, orphans, and lint all report the same count.
    from wikiloom.backlinks import BacklinkRegistry
    from wikiloom.lint import find_orphan_page_ids
    from wikiloom.registry import Registry

    registry_obj = Registry(registry_dir)
    bl = BacklinkRegistry(registry_dir)
    orphan_count = len(find_orphan_page_ids(registry_obj, bl))

    click.echo(click.style("Graph", bold=True))
    click.echo(
        f"  {stats['backlinks']} backlinks  {sep}  "
        f"{stats['aliases']} aliases  {sep}  {orphan_count} orphans"
    )
    click.echo("")

    # Storage section: source + chunk provenance.
    chunk_store = ChunkStore(registry_dir / "wiki.db")
    source_count = 0
    if registry_dir.exists():
        catalog = SourceCatalog(registry_dir)
        source_count = len(catalog._entries)  # noqa: SLF001

    click.echo(click.style("Storage", bold=True))
    click.echo(
        f"  {chunk_store.count()} chunks stored  {sep}  "
        f"{source_count} sources ingested"
    )

    # Incomplete-ingest warning, if any — stays prominent.
    from wikiloom.ingest.state import IngestState

    incomplete = IngestState.load(registry_dir)
    if incomplete is not None:
        pending = incomplete.pending_indices()
        total_chunks = len(incomplete.chunks)
        done = total_chunks - len(pending)
        click.echo("")
        click.echo(
            click.style(
                f"  ⚠ Incomplete ingest for {incomplete.source_name} "
                f"({done}/{total_chunks} chunks — re-run with --force)",
                fg="yellow",
            )
        )

    events = parse_log(project / "wiki" / "log.md")
    if events:
        click.echo("")
        click.echo(click.style("Last event", bold=True))
        last = events[0]
        click.echo(
            f"  {last['event_type']} {sep} {last['description']}"
        )
        click.echo(f"  {_dim(str(last['timestamp']))}")

        total_tokens = sum(int(e.get("tokens_used", 0)) for e in events)
        total_cost = sum(float(e.get("cost_usd", 0.0)) for e in events)
        click.echo("")
        click.echo(click.style("Usage", bold=True))
        click.echo(
            f"  {total_tokens:,} tokens  {sep}  ${total_cost:.2f}"
        )
    click.echo("")


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
    click.echo("")
    click.echo(click.style(f"Recent events ({len(shown)})", bold=True))
    click.echo("")
    for event in shown:
        ts = _format_event_timestamp(str(event.get("timestamp", "?")))
        etype = str(event.get("event_type", "?"))
        desc = event.get("description", "")
        tokens = event.get("tokens_used", 0)
        cost = event.get("cost_usd", 0.0)
        commit = event.get("commit", "")

        line = (
            f"  {_dim(ts)}  {click.style(etype, fg='cyan')}  "
            f"{_dim('•')}  {desc}"
        )
        extras: list[str] = []
        if tokens:
            extras.append(f"{int(tokens):,}t")
        if cost:
            extras.append(f"${float(cost):.2f}")
        if commit:
            extras.append(str(commit)[:8])
        if extras:
            line += f"  {_dim('(' + ', '.join(extras) + ')')}"
        click.echo(line)

    if len(events) > limit:
        click.echo("")
        click.echo(
            _dim(f"... {len(events) - limit} more event(s). "
                 f"Use -n to see more.")
        )
    click.echo("")


@main.command("edits")
@click.option("--limit", "-n", type=int, default=20, help="Number of recent edits to show.")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def edits(limit: int, project: Path | None) -> None:
    """Show recent human edits committed via `wikiloom save`.

    Complements `wikiloom log` (LLM / system activity) by surfacing
    the git history of human-edit commits. Useful in multi-user
    wikis to see who edited what, when. `git log` remains the
    exhaustive source of truth.
    """
    from wikiloom.git_ops import GitOps

    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    try:
        gitops = GitOps(project)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # iter_commits with --grep returns commits whose subject matches the
    # regex. Anchored to the start of the subject so we don't catch
    # anything that merely mentions "human-edit:" in the body.
    commits = list(
        gitops.repo.iter_commits(
            all=False, grep=r"^human-edit:", max_count=limit + 1)
    )
    if not commits:
        click.echo(
            "No human edits yet — use `wikiloom save` to commit manual changes.")
        return

    shown = commits[:limit]
    author_width = max((len(c.author.name or "") for c in shown), default=6)
    click.echo("")
    click.echo(
        click.style("Recent human edits", bold=True)
        + f"  {_dim('(' + str(len(shown)) + ')')}"
    )
    click.echo("")
    for c in shown:
        when = c.authored_datetime.strftime("%Y-%m-%d %H:%M:%S")
        author = (c.author.name or "?").ljust(author_width)
        subject = c.message.splitlines()[0]
        short = c.hexsha[:8]
        click.echo(
            f"  {_dim(when)}  "
            f"{click.style(author, fg='cyan')}  "
            f"{subject}  {_dim('(' + short + ')')}"
        )

    if len(commits) > limit:
        click.echo("")
        click.echo(_dim("... more edits exist. Use -n to see more."))
    click.echo("")


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
        bucket = by_type.setdefault(
            etype, {"tokens": 0, "cost": 0.0, "count": 0})
        bucket["tokens"] += tokens
        bucket["cost"] += cost_usd
        bucket["count"] += 1

    total_tokens = 0
    total_cost = 0.0
    total_events = 0

    click.echo("")
    click.echo(click.style("Usage by event type", bold=True))
    click.echo("")
    header = f"  {'Event':<16} {'Count':>8} {'Tokens':>12} {'Cost':>10}"
    click.echo(_dim(header))
    for etype in sorted(by_type):
        b = by_type[etype]
        t = int(b["tokens"])
        c = b["cost"]
        n = int(b["count"])
        total_tokens += t
        total_cost += c
        total_events += n
        click.echo(
            f"  {click.style(etype, fg='cyan'):<25} "
            f"{n:>8} {t:>12,} {_dim(f'${c:>7.2f}')}"
        )

    click.echo("")
    click.echo(
        done_summary(
            [
                f"{total_events} events",
                f"{total_tokens:,} tokens",
                f"${total_cost:.2f}",
            ]
        )
    )

    cfg = _load_config(project)
    if cfg is not None:
        budget = cfg.llm.monthly_budget_usd
        pct = (total_cost / budget * 100) if budget > 0 else 0
        bar_color = "green" if pct < 50 else "yellow" if pct < 90 else "red"
        click.echo("")
        click.echo(click.style("Monthly budget", bold=True))
        click.echo(
            f"  ${total_cost:.2f} of ${budget:.2f}  "
            f"{_dim('•')}  "
            f"{click.style(f'{pct:.1f}% used', fg=bar_color)}"
        )
    click.echo("")


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

    \b
    Examples:
      \x1b[36mwikiloom show concepts/transformer\x1b[0m
      \x1b[36mwikiloom show sources/chase-bank --field chunk_ids\x1b[0m
      \x1b[36mwikiloom show entities/openai --json\x1b[0m
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
        raise click.ClickException(_page_not_found_message(page_id))

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
            # JSON output stays unstyled so it's round-trip parseable.
            # Blank-line framing is still useful for visual parsing on
            # a terminal; Click auto-strips nothing from JSON so the
            # output is still round-trippable through ``jq``.
            click.echo("")
            click.echo(json_mod.dumps(value, indent=2, ensure_ascii=False))
            click.echo("")
            return
        # Pretty --field output: cyan for identifier-like field values
        # (chunk_ids, aliases, sources) so the console reads
        # consistently with the shared styling. Click auto-strips
        # ANSI when the stream isn't a TTY (piped to a file, CI, etc.)
        # so scripts still see the raw strings.
        cyan_fields = {"title", "type", "aliases", "chunk_ids"}
        style_value = field in cyan_fields

        click.echo("")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    click.echo(json_mod.dumps(item, ensure_ascii=False))
                else:
                    text = str(item)
                    click.echo(click.style(text, fg="cyan") if style_value else text)
        elif value is None:
            click.echo(_dim("—"))
        else:
            text = str(value)
            click.echo(click.style(text, fg="cyan") if style_value else text)
        click.echo("")
        return

    if as_json:
        click.echo(json_mod.dumps(data, indent=2, ensure_ascii=False))
        return

    # Pretty default — mirrors the shared styling used by status,
    # links, and source: leading/trailing blank lines, bold section
    # headers, cyan for identifier-ish values, dim labels and
    # placeholder dashes, plain values for everything else.
    def _label(text: str) -> str:
        return _dim(f"  {text:<22}")

    def _placeholder() -> str:
        return _dim("—")

    click.echo("")
    click.echo(
        f"Page  {click.style(page_id, fg='cyan', bold=True)}"
    )
    click.echo("")
    click.echo(click.style("Frontmatter", bold=True))

    # Identifier-ish keys rendered in cyan so page_ids, titles, and
    # aliases read consistently with the rest of the CLI. Everything
    # else keeps its natural formatting.
    cyan_keys = {"title", "type", "aliases", "chunk_ids"}

    for key, value in data.items():
        if value in ([], {}, None):
            click.echo(f"{_label(key + ':')} {_placeholder()}")
            continue
        if isinstance(value, list):
            click.echo(f"{_label(key + ':')} {_dim(f'({len(value)})')}")
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
                        extra = _dim(
                            f"  (chunks: {len(item.get('chunk_ids') or [])})"
                        )
                    click.echo(
                        f"    {_dim('-')} {click.style(name, fg='cyan')}{extra}"
                    )
                else:
                    item_str = str(item)
                    if key in cyan_keys:
                        item_str = click.style(item_str, fg="cyan")
                    click.echo(f"    {_dim('-')} {item_str}")
        elif isinstance(value, dict):
            click.echo(
                f"{_label(key + ':')} {json_mod.dumps(value, ensure_ascii=False)}"
            )
        else:
            value_str = str(value)
            if key in cyan_keys:
                value_str = click.style(value_str, fg="cyan")
            click.echo(f"{_label(key + ':')} {value_str}")
    click.echo("")


@main.command("links")
@click.argument("page_id", required=False)
@click.option(
    "--list",
    "list_mode",
    is_flag=True,
    default=False,
    help="List every pending link candidate across the project.",
)
@click.option(
    "--review",
    is_flag=True,
    default=False,
    help="Interactively walk one page's pending candidates (y/n/q). "
         "Requires a page_id.",
)
@click.option(
    "--accept-all",
    is_flag=True,
    default=False,
    help="Accept every pending link without prompting. Implies --list.",
)
@click.option(
    "--clear",
    is_flag=True,
    default=False,
    help="Discard all pending links without inserting any. Implies --list.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap how many rows --list prints.",
)
@click.option(
    "--min-similarity",
    type=float,
    default=None,
    help="Only --list rows whose cosine score is >= this value (0-1).",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def links(
    page_id: str | None,
    list_mode: bool,
    review: bool,
    accept_all: bool,
    clear: bool,
    limit: int | None,
    min_similarity: float | None,
    project: Path | None,
) -> None:
    """Inspect links, review pending candidates, or see a project summary.

    \b
    Modes:
      wikiloom links                       project summary
      wikiloom links <page_id>             inbound + outbound for one page
      wikiloom links <page_id> --review    interactive y/n/q walkthrough
      wikiloom links --list                dump every pending candidate
      wikiloom links --accept-all          accept all pending (batch)
      wikiloom links --clear               discard all pending (batch)

    \b
    --list filters (optional):
      --limit N              cap rows printed
      --min-similarity 0.65  cosine floor (hybrid pending entries only)
    """
    if project is None:
        project = _find_project_root(Path.cwd())
        if project is None:
            raise click.ClickException(
                "Could not find a WikiLoom project (no wikiloom.toml found)."
            )

    if list_mode and review:
        raise click.UsageError("--list and --review are mutually exclusive.")

    # --accept-all and --clear imply --list (they operate on the whole
    # pending queue). Users don't have to type both.
    batch_action = accept_all or clear
    if batch_action and review:
        raise click.UsageError(
            "--accept-all / --clear cannot be combined with --review."
        )

    if list_mode or batch_action:
        if page_id:
            raise click.UsageError(
                "--list / --accept-all / --clear operate on the whole "
                "project; don't pass a page_id."
            )
        _run_pending_list(
            project,
            accept_all=accept_all,
            clear=clear,
            limit=limit,
            min_similarity=min_similarity,
        )
        return

    if review:
        if not page_id:
            raise click.UsageError(
                "--review requires a page_id. "
                "Run `wikiloom links` to see pages with pending candidates."
            )
        _run_pending_review(project, page_id=_normalize_page_id(page_id))
        return

    if limit is not None or min_similarity is not None:
        raise click.UsageError(
            "--limit and --min-similarity only apply with --list."
        )

    if page_id:
        _run_page_links(project, _normalize_page_id(page_id))
        return

    _run_links_summary(project)


def _normalize_page_id(page_id: str) -> str:
    page_id = page_id.replace(".md", "").strip("/")
    if page_id.startswith("wiki/"):
        page_id = page_id[len("wiki/"):]
    return page_id


def _run_page_links(project: Path, page_id: str) -> None:
    """Show inbound and outbound wikilinks for a single page."""
    from wikiloom.backlinks import BacklinkRegistry
    from wikiloom.registry import Registry

    _warn_if_dirty(project)

    registry = Registry(project / "_registry")
    page = registry.get_page(page_id)
    if page is None:
        raise click.ClickException(_page_not_found_message(page_id))

    bl = BacklinkRegistry(project / "_registry")

    outbound = [e for e in bl.edges if e.source == page_id]
    inbound = [e for e in bl.edges if e.target == page_id]

    click.echo("")
    click.echo(
        f"Links for {click.style(page_id, fg='cyan', bold=True)}  "
        f"{_dim(f'({page.title})')}"
    )
    click.echo("")

    click.echo(click.style(f"Outbound ({len(outbound)})", bold=True))
    if outbound:
        for edge in outbound:
            target = registry.get_page(edge.target)
            title = target.title if target else edge.target
            click.echo(
                f"  → {click.style(edge.target, fg='cyan')}  "
                f"{_dim(f'({title})')}"
            )
    else:
        click.echo(_dim("  (none)"))
    click.echo("")

    click.echo(click.style(f"Inbound ({len(inbound)})", bold=True))
    if inbound:
        for edge in inbound:
            source = registry.get_page(edge.source)
            title = source.title if source else edge.source
            click.echo(
                f"  ← {click.style(edge.source, fg='cyan')}  "
                f"{_dim(f'({title})')}"
            )
    else:
        click.echo(_dim("  (none)"))
    click.echo("")

    total = len(outbound) + len(inbound)
    click.echo(_dim(f"Total: {total} link(s)"))
    click.echo("")


def _run_links_summary(project: Path) -> None:
    """Project-level links summary: edges, pending, top source pages.

    The default when ``wikiloom links`` is invoked without a page_id
    or any flag. Mirrors ``wikiloom status`` in shape — headline
    numbers, a short drill-in list, and tips pointing at the other
    modes.
    """
    import json
    from collections import Counter

    from wikiloom.backlinks import BacklinkRegistry

    sep = _dim("•")

    bl = BacklinkRegistry(project / "_registry")
    edge_count = len(bl.edges)
    linked_pages = {e.source for e in bl.edges} | {e.target for e in bl.edges}

    pending_items: list[dict] = []
    pending_path = project / "_registry" / "pending.json"
    if pending_path.exists():
        try:
            data = json.loads(pending_path.read_text(encoding="utf-8"))
            pending_items = (
                data.get("pending", []) if isinstance(data, dict) else data
            )
        except json.JSONDecodeError:
            pending_items = []

    click.echo("")
    click.echo(click.style("Links", bold=True))
    click.echo(
        f"  Backlinks: {edge_count} edges  {sep}  "
        f"{len(linked_pages)} pages linked"
    )

    band_label = _format_pending_band(pending_items)
    click.echo(f"  Pending:   {len(pending_items)} candidates{band_label}")

    # Top source pages with pending — gives the user an actionable
    # starting point for `links --review <page_id>`.
    if pending_items:
        counts = Counter(
            item.get("source_page", "") for item in pending_items
            if item.get("source_page")
        )
        top = counts.most_common(5)
        if top:
            click.echo("")
            click.echo(click.style("Top pages with pending", bold=True))
            for pid, n in top:
                click.echo(
                    f"  {click.style(pid, fg='cyan')}  {_dim(f'({n} candidates)')}"
                )

    click.echo("")
    click.echo(click.style("Tips", bold=True))
    click.echo(
        f"  {_dim('wikiloom links <page_id>')}          "
        f"inspect one page's links"
    )
    click.echo(
        f"  {_dim('wikiloom links <page_id> --review')} "
        f"walk a page's pending candidates"
    )
    click.echo(
        f"  {_dim('wikiloom links --list')}             "
        f"dump all pending candidates"
    )
    click.echo("")


def _format_pending_band(items: list[dict]) -> str:
    """Return a ``  (cosine 0.50–0.74)`` suffix, or empty when unavailable."""
    scores = [
        float(i["cosine_score"])
        for i in items
        if isinstance(i.get("cosine_score"), (int, float))
    ]
    if scores:
        return _dim(f"  (cosine {min(scores):.2f}–{max(scores):.2f})")
    return ""


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

    \b
    Examples:
      \x1b[36mwikiloom related concepts/transformer\x1b[0m
      \x1b[36mwikiloom related concepts/transformer -n 10\x1b[0m
      \x1b[36mwikiloom related concepts/transformer --save --link\x1b[0m
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
        raise click.ClickException(_page_not_found_message(page_id))

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

    results = cache.semantic_search(
        page_vec, limit=limit + len(linked_ids) + 1)

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
        click.echo("")
        return

    click.echo("")
    click.echo(
        click.style("Related to", bold=True)
        + f" {click.style(page['title'], fg='cyan')}  "
        + _dim(f"({page_id})")
    )
    click.echo("")
    for pid, title, sim in related_pages:
        click.echo(
            f"  {_dim(f'{sim:.0%}')}  "
            f"{click.style(title, fg='cyan')}"
        )
        click.echo(f"      {_dim('→ ' + pid + '.md')}")

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
        _sync_cache(project, changed_files=[page_path])
        _auto_commit(
            project, "related", f"updated {page_id} with related pages"
        )

        actions = []
        if save:
            actions.append(
                f"{len(related_pages)} related page(s) to frontmatter")
        if link:
            actions.append(f"{len(new_links)} wikilink(s) to page body")
        click.echo("")
        click.echo(f"{_check()} Saved: {', '.join(actions)}.")
        click.echo("")

    if not save and not link and related_pages:
        click.echo("")
        click.echo(
            _dim(
                "Tip: --save records these in frontmatter, --link appends "
                "wikilinks in the page body, or pass both."
            )
        )
        click.echo("")


@main.command("orphans")
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root.",
)
def orphans(project: Path | None) -> None:
    """List pages with zero inbound wikilinks.

    An orphan is a page nothing in the wiki links to — outbound links
    don't count. Excludes sources (provenance, not meant to be linked
    to), index pages (derived), and deprecated pages (out of flow).
    Shares the same definition with `wikiloom lint`.
    """
    from wikiloom.backlinks import BacklinkRegistry
    from wikiloom.lint import find_orphan_page_ids
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
    orphan_ids = find_orphan_page_ids(registry, bl)

    if not orphan_ids:
        click.echo("No orphan pages found.")
        click.echo("")
        return

    # Hydrate display info (title + type) from the manifest.
    orphan_list = [
        (pid, registry.pages[pid].title, registry.pages[pid].type)
        for pid in orphan_ids
    ]

    click.echo("")
    click.echo(
        click.style("Orphan pages", bold=True)
        + f"  {_dim('(' + str(len(orphan_list)) + ')')}"
    )
    click.echo("")
    for pid, title, ptype in orphan_list:
        click.echo(
            f"  {_dim('[' + ptype + ']')} "
            f"{click.style(title, fg='cyan')}"
        )
        click.echo(f"      {_dim('→ ' + pid + '.md')}")

    example_pid = orphan_list[0][0]
    click.echo("")
    click.echo(
        _dim(
            f"Tip: `wikiloom related {example_pid}` to find connections, "
            f"or `wikiloom relink` to re-run the linker."
        )
    )
    click.echo("")


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
        click.echo(
            "No dormant candidates — all active pages are within their windows.")
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
        raise click.ClickException(_page_not_found_message(page_id))

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

        _sync_cache(project, changed_files=[page_path])
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

    import time as _time

    total = len(candidates)
    marked = 0
    skipped = 0
    check_mark = _check()
    cross_mark = _cross()
    skip_glyph = _skip_mark()
    click.echo(f"Reviewing {total} candidate(s).")
    click.echo("For each: [m]ark dormant / [n]ext (skip) / [q]uit")
    click.echo("")
    start = _time.monotonic()

    for i, candidate in enumerate(sorted(candidates, key=lambda x: -x.age_days), start=1):
        registry = Registry(project / "_registry")
        entry = registry.get_page(candidate.page_id)
        if entry is None:
            continue
        click.echo(
            f"  {_dim(f'--- {i}/{total}')}  {candidate.page_id}  "
            f"{_dim(f'({entry.type})')}"
        )
        click.echo(f"    title:   {entry.title}")
        click.echo(
            f"    age:     {candidate.age_days}d "
            f"{_dim(f'(window {candidate.window_days}d)')}"
        )
        if entry.summary:
            click.echo(f"    summary: {entry.summary[:100]}")

        choice = click.prompt(
            "  Action",
            type=click.Choice(["m", "n", "q"], case_sensitive=False),
            default="n",
            show_choices=True,
            show_default=True,
        ).lower()

        if choice == "q":
            click.echo("  Quit.")
            break
        if choice == "n":
            click.echo(f"  {skip_glyph} skipped\n")
            skipped += 1
            continue

        page_path = project / "wiki" / f"{candidate.page_id}.md"
        if not page_path.exists():
            click.echo(f"  {cross_mark} file missing\n")
            skipped += 1
            continue
        with FileLock(project):
            text = page_path.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(text)
            if fm is None:
                click.echo(f"  {cross_mark} no frontmatter\n")
                skipped += 1
                continue
            fm.status = "dormant"
            page_path.write_text(
                render_frontmatter(fm) + "\n" + body, encoding="utf-8"
            )
            entry.status = "dormant"
            registry.save()
            _sync_cache(project, changed_files=[page_path])
            _auto_commit(project, "dormant", f"mark {candidate.page_id}")
        click.echo(f"  {check_mark} marked dormant\n")
        marked += 1

    click.echo(
        done_summary(
            [f"{marked} marked", f"{skipped} skipped"],
            elapsed=_time.monotonic() - start,
        )
    )


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
        raise click.UsageError(
            "--review and --auto-merge are mutually exclusive.")

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

    # Split by what the user can act on: safe-to-auto pairs go through
    # `--auto-merge`, the rest need `--review`. Two sections in one
    # view means the user doesn't have to mentally partition the list.
    safe_pairs: list[tuple] = []
    review_pairs: list[tuple] = []
    for pair in pairs:
        sug = suggest_winner(pair)
        if sug.is_safe_to_auto:
            safe_pairs.append((pair, sug))
        else:
            review_pairs.append((pair, sug))

    def _render_pair(pair: Any, sug: Any) -> None:
        emb_display = (
            f"emb {pair.embedding_score:.2f}"
            if pair.embedding_score >= 0
            else "emb n/a"
        )
        scores = _dim(f"slug {pair.slug_score:.0f}%  •  {emb_display}")
        click.echo(f"  {scores}")
        click.echo(
            f"    {click.style(pair.page_a, fg='cyan')}  "
            f"{_dim('(' + pair.title_a + ')')}"
        )
        click.echo(
            f"    {click.style(pair.page_b, fg='cyan')}  "
            f"{_dim('(' + pair.title_b + ')')}"
        )
        click.echo(
            f"    {_dim('→')} "
            f"{click.style(sug.loser_page_id, fg='cyan')}  →  "
            f"{click.style(sug.winner_page_id, fg='cyan')}  "
            f"{_dim('(' + sug.reason + ')')}"
        )
        click.echo("")

    click.echo("")
    click.echo(
        click.style("Suspected duplicate pairs", bold=True)
        + f"  {_dim('(' + str(len(pairs)) + ')')}"
    )
    click.echo("")

    if safe_pairs:
        click.echo(
            click.style("Safe to auto-merge", bold=True)
            + f"  {_dim('(' + str(len(safe_pairs)) + ')')}  "
            + _dim("— run `wikiloom duplicates --auto-merge`")
        )
        click.echo("")
        for pair, sug in safe_pairs[:limit]:
            _render_pair(pair, sug)
        if len(safe_pairs) > limit:
            click.echo(_dim(f"  … and {len(safe_pairs) - limit} more"))
            click.echo("")

    if review_pairs:
        remaining_limit = max(0, limit - len(safe_pairs[:limit]))
        click.echo(
            click.style("Needs review", bold=True)
            + f"  {_dim('(' + str(len(review_pairs)) + ')')}  "
            + _dim("— run `wikiloom duplicates --review`")
        )
        click.echo("")
        shown_review = review_pairs[:remaining_limit] if remaining_limit else review_pairs
        for pair, sug in shown_review:
            _render_pair(pair, sug)
        if len(review_pairs) > len(shown_review):
            click.echo(
                _dim(
                    f"  … and {len(review_pairs) - len(shown_review)} more. "
                    f"Pass --limit N to see more."
                )
            )
            click.echo("")
    click.echo("")


def _finalize_batch_merge(
    project: Path,
    applied: list[tuple[str, str]],
    commit_subject: str,
) -> None:
    """Run _sync_cache once and _auto_commit once for a batch of merges.

    ``applied`` is the list of (winner, loser) pairs whose merges
    succeeded on disk. The commit body lists every pair so
    ``git log`` and ``git show`` can be used to audit or revert the
    batch as a whole. No-ops when ``applied`` is empty.

    Passes only the winner/loser page paths to the cache sync so
    the incremental path re-embeds just those rows. Losers are
    handled via the "file gone" branch of the incremental sync —
    their rows get dropped since the file has moved to archive/.
    """
    if not applied:
        return
    from wikiloom.registry import Registry
    from wikiloom.search import IndexUpdater

    wiki_dir = project / "wiki"
    touched_paths: list[Path] = []
    for winner, loser in applied:
        touched_paths.append(wiki_dir / f"{winner}.md")
        touched_paths.append(wiki_dir / f"{loser}.md")
    # Rebuild indexes so archived losers disappear from category
    # indexes and the root index.
    registry = Registry(project / "_registry")
    index_paths = IndexUpdater(wiki_dir, registry=registry).rebuild_all()
    _sync_cache(project, changed_files=touched_paths + list(index_paths))
    body = "\n".join(f"  {loser} → {winner}" for winner, loser in applied)
    description = f"{commit_subject}\n\n{body}"
    commit_hash = _auto_commit(project, "merge", description)

    # Emit a MERGE event per pair, all carrying this batch's commit
    # hash so `wikiloom log` can show them alongside ingest events.
    from wikiloom.merge import MergeResult, emit_merge_event

    for winner, loser in applied:
        emit_merge_event(
            project,
            MergeResult(
                winner_page_id=winner,
                loser_page_id=loser,
                rewrote_links_in=[],
                archive_path=None,
            ),
            commit_hash,
        )
    _commit_merge_log_tail(
        project, f"merge: log {len(applied)} event(s)"
    )


def _run_review_mode(project: Path, pairs: list) -> None:
    """Interactive walkthrough: prompt y/s/n/q for each pair."""
    import time as _time

    from wikiloom.duplicates import suggest_winner
    from wikiloom.locking import FileLock
    from wikiloom.merge import merge_pages

    merged = 0
    skipped = 0
    total = len(pairs)
    applied: list[tuple[str, str]] = []
    check_mark = _check()
    cross_mark = _cross()
    skip_glyph = _skip_mark()

    click.echo(f"Reviewing {total} suspected duplicate pair(s).")
    click.echo(
        "For each pair: [y]es merge / [s]wap winner-loser / [n]o skip / [q]uit")
    click.echo("")
    start = _time.monotonic()

    with FileLock(project):
        for i, pair in enumerate(pairs, start=1):
            suggestion = suggest_winner(pair)
            emb = f"{pair.embedding_score:.2f}" if pair.embedding_score >= 0 else "n/a"
            winner_title = (
                pair.title_a
                if suggestion.winner_page_id == pair.page_a
                else pair.title_b
            )
            loser_title = (
                pair.title_b
                if suggestion.winner_page_id == pair.page_a
                else pair.title_a
            )
            click.echo(
                f"  {_dim(f'--- {i}/{total}')}  "
                f"{_dim(f'(slug {pair.slug_score:.0f}%, emb {emb})')}"
            )
            click.echo(
                f"    winner: {suggestion.winner_page_id}  "
                f"{_dim(f'({winner_title})')}"
            )
            click.echo(
                f"    loser:  {suggestion.loser_page_id}  "
                f"{_dim(f'({loser_title})')}"
            )
            click.echo(f"    reason: {suggestion.reason}")

            choice = click.prompt(
                "  Action",
                type=click.Choice(["y", "s", "n", "q"], case_sensitive=False),
                default="n",
                show_choices=True,
                show_default=True,
            ).lower()

            if choice == "q":
                click.echo("  Quit.")
                break
            if choice == "n":
                click.echo(f"  {skip_glyph} skipped\n")
                skipped += 1
                continue

            winner = suggestion.winner_page_id
            loser = suggestion.loser_page_id
            if choice == "s":
                winner, loser = loser, winner

            try:
                merge_pages(project, winner, loser)
                applied.append((winner, loser))
                click.echo(f"  {check_mark} merged {loser} → {winner}\n")
                merged += 1
            except ValueError as exc:
                click.echo(f"  {cross_mark} failed: {exc}\n")
                skipped += 1

        _finalize_batch_merge(
            project, applied, f"reviewed {len(applied)} pair(s)"
        )

    click.echo(
        done_summary(
            [f"{merged} merged", f"{skipped} skipped"],
            elapsed=_time.monotonic() - start,
        )
    )


def _run_auto_merge_mode(project: Path, pairs: list, dry_run: bool) -> None:
    """Batch-merge only pairs flagged is_safe_to_auto by suggest_winner."""
    import time as _time

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

    click.echo(f"Plan ({len(safe_plan)} safe merge(s)):")
    click.echo("")
    for pair, suggestion in safe_plan:
        emb = f"{pair.embedding_score:.2f}" if pair.embedding_score >= 0 else "n/a"
        click.echo(
            f"  {suggestion.loser_page_id}  →  {suggestion.winner_page_id}"
            f"  {_dim(f'(slug {pair.slug_score:.0f}%, emb {emb}, {suggestion.reason})')}"
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

    click.echo("")
    click.echo("Merging...")
    click.echo("")

    merged = 0
    skipped = 0
    applied: list[tuple[str, str]] = []
    start = _time.monotonic()
    check_mark = _check()
    cross_mark = _cross()

    with FileLock(project):
        for pair, suggestion in safe_plan:
            try:
                merge_pages(
                    project,
                    suggestion.winner_page_id,
                    suggestion.loser_page_id,
                )
                applied.append(
                    (suggestion.winner_page_id, suggestion.loser_page_id)
                )
                click.echo(
                    f"  {check_mark} {suggestion.loser_page_id}  →  "
                    f"{suggestion.winner_page_id}"
                )
                merged += 1
            except ValueError as exc:
                click.echo(
                    f"  {cross_mark} {suggestion.loser_page_id}: {exc}"
                )
                skipped += 1

        _finalize_batch_merge(
            project, applied, f"auto-merged {len(applied)} pair(s)"
        )

    click.echo("")
    click.echo(
        done_summary(
            [f"{merged} merged", f"{skipped} skipped"],
            elapsed=_time.monotonic() - start,
        )
    )


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

    \b
    Examples:
      \x1b[36mwikiloom merge concepts/transformer concepts/transformers\x1b[0m
      \x1b[36mwikiloom merge entities/openai entities/open-ai --yes\x1b[0m
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
            wiki_dir = project / "wiki"
            # Rebuild indexes so the archived loser disappears.
            from wikiloom.merge import emit_merge_event
            from wikiloom.registry import Registry
            from wikiloom.search import IndexUpdater

            registry = Registry(project / "_registry")
            index_paths = IndexUpdater(
                wiki_dir, registry=registry
            ).rebuild_all()
            _sync_cache(
                project,
                changed_files=[
                    wiki_dir / f"{winner}.md",
                    wiki_dir / f"{loser}.md",
                    *index_paths,
                ],
            )
            commit_hash = _auto_commit(
                project,
                "merge",
                f"{loser} into {winner}",
            )
            emit_merge_event(project, result, commit_hash)
            _commit_merge_log_tail(
                project, f"merge: log event for {loser} → {winner}"
            )
    except ValueError as exc:
        msg = str(exc)
        # Surface a recovery hint when the ValueError is a missing-
        # page lookup — the most common merge failure in practice.
        if "not found" in msg.lower():
            msg = (
                f"{msg}\n"
                f"Tip: run `wikiloom search <keyword>` to find the right "
                f"page_id, or `wikiloom duplicates` to see candidate pairs."
            )
        raise click.ClickException(msg) from exc

    click.echo("")
    click.echo(
        f"{_check()} Merged  "
        f"{click.style(loser, fg='cyan')}  →  "
        f"{click.style(winner, fg='cyan')}"
    )
    detail_bits: list[str] = []
    if result.rewrote_links_in:
        detail_bits.append(
            f"wikilinks rewritten in {len(result.rewrote_links_in)} page(s)"
        )
    if result.archive_path is not None:
        detail_bits.append(
            f"archived to {result.archive_path.relative_to(project)}"
        )
    for bit in detail_bits:
        click.echo(_dim(f"  {bit}"))
    click.echo("")


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

    \b
    Examples:
      \x1b[36mwikiloom deprecate concepts/old-page\x1b[0m
      \x1b[36mwikiloom deprecate concepts/old-page --superseded-by concepts/new-page\x1b[0m
      \x1b[36mwikiloom deprecate concepts/old-page --yes\x1b[0m
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
        raise click.ClickException(_page_not_found_message(page_id))
    if entry.status == "deprecated":
        raise click.ClickException(f"{page_id} is already deprecated.")
    if superseded_by and registry.get_page(superseded_by) is None:
        raise click.ClickException(
            f"--superseded-by target not found: {superseded_by}\n"
            f"Tip: run `wikiloom search <keyword>` to find the right "
            f"page_id for the replacement page."
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
        from wikiloom.search import IndexUpdater

        original_path = project / "wiki" / f"{page_id}.md"
        archive_path = registry.deprecate_page(
            page_id,
            superseded_by=superseded_by,
            move_to_archive=True,
            emit_event=True,
        )
        registry.save()
        # Rebuild indexes so the deprecated page disappears from the
        # category/root indexes.
        index_paths = IndexUpdater(
            project / "wiki", registry=registry
        ).rebuild_all()
        # Original path is gone (file moved) — incremental sync will
        # drop the old row. The archive file is a new page_id so it
        # gets upserted too.
        touched = [original_path]
        if archive_path is not None:
            touched.append(archive_path)
        touched.extend(index_paths)
        _sync_cache(project, changed_files=touched)
        suffix = f" (superseded by {superseded_by})" if superseded_by else ""
        _auto_commit(project, "deprecate", f"{page_id}{suffix}")

    click.echo("")
    click.echo(
        f"{_check()} Deprecated  {click.style(page_id, fg='cyan')}"
    )
    if archive_path is not None:
        click.echo(
            _dim(f"  archived to {archive_path.relative_to(project)}")
        )
    click.echo(_dim("  to undo: `git revert HEAD`"))
    click.echo("")


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

    \b
    Examples:
      \x1b[36mwikiloom purge concepts/old-page\x1b[0m
      \x1b[36mwikiloom purge concepts/old-page --yes\x1b[0m
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
        raise click.ClickException(_page_not_found_message(page_id))
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
            click.echo(
                "  - delete the manifest entry (archive file already missing)")
        click.echo(f"  - remove the manifest entry for {page_id}")
        click.echo("  This cannot be undone via wikiloom (only via git revert).")
        typed = click.prompt(f"\nType the page_id to confirm",
                             default="", show_default=False)
        if typed.strip() != page_id:
            click.echo("Aborted: confirmation did not match.")
            return

    with FileLock(project):
        from wikiloom.search import IndexUpdater

        if archive_file.exists():
            archive_file.unlink()
        # Remove manifest entry directly — there's no Registry method for this
        # because nothing else needs it. Purge is the only consumer.
        if page_id in registry._pages:  # noqa: SLF001
            del registry._pages[page_id]  # noqa: SLF001
            registry.save()
        index_paths = IndexUpdater(
            project / "wiki", registry=registry
        ).rebuild_all()
        _sync_cache(
            project, changed_files=[archive_file, *index_paths]
        )
        _auto_commit(project, "deprecate", f"purge {page_id}")

    click.echo("")
    click.echo(
        f"{_check()} Purged  {click.style(page_id, fg='cyan')}"
    )
    click.echo("")


def _load_pending(project: Path) -> tuple[dict, list[dict]]:
    """Return ``(full_data, items_list)`` from ``_registry/pending.json``.

    Tolerates a missing file (returns an empty shell) and a bare-list
    form (legacy shape). Keeping the load shared means every pending
    helper agrees on what "no pending" means.
    """
    import json

    pending_path = project / "_registry" / "pending.json"
    if not pending_path.exists():
        return {"version": 1, "pending": []}, []
    try:
        data = json.loads(pending_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "pending": []}, []
    if isinstance(data, list):
        data = {"version": 1, "pending": data}
    items = data.get("pending", []) if isinstance(data, dict) else []
    return data, items


def _write_pending(project: Path, data: dict) -> None:
    import json

    pending_path = project / "_registry" / "pending.json"
    pending_path.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


def _format_pending_row(item: dict) -> str:
    """Render one pending.json entry as a single line with both scores.

    Shows cosine as the decision metric and fuzzy as a breadcrumb for
    context. Rows without a cosine score (written by a pre-hybrid link
    pass or a project with embeddings disabled at the time) omit the
    cosine cell.
    """
    source = item.get("source_page", "?")
    target = item.get("candidate_page_id", "?")
    matched = item.get("matched_text", "?")
    fuzzy = item.get("score", "?")
    cosine = item.get("cosine_score")
    cosine_bit = f"cos {cosine:.2f}, " if isinstance(cosine, (int, float)) else ""
    return (
        f"  {click.style(source, fg='cyan')} → "
        f"[[{click.style(target, fg='cyan')}]]  "
        f"{_dim(f'({cosine_bit}fuzzy {fuzzy}, matched {matched!r})')}"
    )


def _apply_pending_link(wiki_dir: Path, item: dict) -> bool:
    """Insert one pending link into its source page. Returns True on change.

    Skips silently on missing fields or a missing file so a batch
    operation can continue through partial breakage.
    """
    from wikiloom.frontmatter import parse_frontmatter, render_frontmatter

    source_page = item.get("source_page", "")
    target = item.get("candidate_page_id", "")
    matched_text = item.get("matched_text", "")
    if not source_page or not target or not matched_text:
        return False

    page_path = wiki_dir / f"{source_page}.md"
    if not page_path.exists():
        return False

    text = page_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    wikilink = f"[[{target}|{matched_text}]]"
    new_body = body.replace(matched_text, wikilink, 1)
    if new_body == body:
        return False
    if fm is not None:
        page_path.write_text(
            render_frontmatter(fm) + "\n" + new_body,
            encoding="utf-8",
        )
    else:
        page_path.write_text(new_body, encoding="utf-8")
    return True


def _run_pending_list(
    project: Path,
    *,
    accept_all: bool,
    clear: bool,
    limit: int | None,
    min_similarity: float | None,
) -> None:
    """Back-end for ``wikiloom links --list``.

    Dumps the full pending queue with optional filters, or batch-
    actions via ``--accept-all`` / ``--clear``. ``--min-similarity``
    filters rows whose cosine score falls below the threshold; rows
    without a cosine score (pre-hybrid) always pass through so the
    filter never silently hides them.
    """
    from wikiloom.locking import FileLock

    if accept_all and clear:
        raise click.UsageError(
            "--accept-all and --clear are mutually exclusive."
        )

    data, items = _load_pending(project)
    if not items:
        click.echo("")
        click.echo(_dim("No pending links."))
        click.echo("")
        return

    if clear:
        _require_clean_tree(project, "links --list --clear")
        with FileLock(project):
            data["pending"] = []
            _write_pending(project, data)
            _auto_commit(
                project, "review", f"cleared {len(items)} pending link(s)"
            )
        click.echo("")
        click.echo(done_summary([f"{len(items)} cleared"]))
        click.echo("")
        return

    if accept_all:
        _require_clean_tree(project, "links --list --accept-all")
        wiki_dir = project / "wiki"
        inserted = 0
        with FileLock(project):
            for item in items:
                if _apply_pending_link(wiki_dir, item):
                    inserted += 1
            data["pending"] = []
            _write_pending(project, data)
            if inserted:
                _sync_cache(project)
                _auto_commit(
                    project, "review", f"accepted {inserted} pending link(s)"
                )
        click.echo("")
        click.echo(
            done_summary([f"{inserted} accepted", f"{len(items)} cleared"])
        )
        click.echo("")
        return

    # Plain --list: print rows with optional filters.
    filtered = items
    if min_similarity is not None:
        filtered = [
            i for i in filtered
            if not isinstance(i.get("cosine_score"), (int, float))
            or float(i["cosine_score"]) >= min_similarity
        ]
    total_filtered = len(filtered)
    truncated = False
    if limit is not None and total_filtered > limit:
        filtered = filtered[:limit]
        truncated = True

    click.echo("")
    header = f"Pending links ({total_filtered}"
    if min_similarity is not None:
        header += f", cosine ≥ {min_similarity:.2f}"
    header += ")"
    click.echo(click.style(header, bold=True))
    for item in filtered:
        click.echo(_format_pending_row(item))
    if truncated:
        click.echo("")
        click.echo(
            _dim(
                f"Showing {len(filtered)} of {total_filtered}. "
                f"Raise --limit to see more."
            )
        )
    click.echo("")
    click.echo(
        _dim(
            "Tip: `--accept-all` to insert every row, `--clear` to "
            "discard, or `wikiloom links <page_id> --review` to walk "
            "one page's candidates."
        )
    )
    click.echo("")


def _run_pending_review(project: Path, page_id: str) -> None:
    """Interactive y/s/n/q walkthrough of one page's pending candidates.

    Scope is a single page so every decision is made in the context
    of the reviewer's mental model of that page. Changes land in one
    commit after the walkthrough, so a quit mid-review commits only
    the candidates already accepted.
    """
    import time as _time

    from wikiloom.locking import FileLock
    from wikiloom.registry import Registry

    registry = Registry(project / "_registry")
    if registry.get_page(page_id) is None:
        raise click.ClickException(_page_not_found_message(page_id))

    data, items = _load_pending(project)
    scoped = [i for i in items if i.get("source_page") == page_id]
    if not scoped:
        click.echo("")
        click.echo(_dim(f"No pending candidates for {page_id}."))
        click.echo("")
        return

    _require_clean_tree(project, "links --review")

    accepted = 0
    skipped = 0
    quit_early = False
    index_map: dict[int, int] = {}  # scoped_index -> items_index
    for i, it in enumerate(items):
        if it.get("source_page") == page_id:
            index_map[len(index_map)] = i

    wiki_dir = project / "wiki"
    check_mark = _check()
    skip_glyph = _skip_mark()
    cross_mark = _cross()

    click.echo("")
    click.echo(
        f"Reviewing {len(scoped)} pending candidate(s) for "
        f"{click.style(page_id, fg='cyan')}"
    )
    click.echo(_dim("  [y] accept / [n] skip / [q] quit"))
    click.echo("")

    start = _time.monotonic()
    accepted_scoped_indices: set[int] = set()

    with FileLock(project):
        for scoped_idx, item in enumerate(scoped):
            target = item.get("candidate_page_id", "?")
            matched = item.get("matched_text", "?")
            fuzzy = item.get("score", "?")
            cosine = item.get("cosine_score")
            cosine_bit = (
                f"cos {cosine:.2f}, " if isinstance(cosine, (int, float)) else ""
            )
            click.echo(
                f"  {_dim(f'--- {scoped_idx + 1}/{len(scoped)}')}  "
                f"{_dim(f'({cosine_bit}fuzzy {fuzzy})')}"
            )
            click.echo(
                f"    → [[{click.style(target, fg='cyan')}]]  "
                f"{_dim(f'matched {matched!r}')}"
            )
            choice = click.prompt(
                "  Action",
                type=click.Choice(["y", "n", "q"], case_sensitive=False),
                default="n",
                show_choices=True,
                show_default=True,
            ).lower()

            if choice == "q":
                click.echo(f"  {_dim('quit')}")
                quit_early = True
                break
            if choice == "n":
                click.echo(f"  {skip_glyph} skipped\n")
                skipped += 1
                continue

            if _apply_pending_link(wiki_dir, item):
                accepted += 1
                accepted_scoped_indices.add(scoped_idx)
                click.echo(f"  {check_mark} accepted\n")
            else:
                click.echo(f"  {cross_mark} could not apply (missing file or text)\n")
                skipped += 1

        # Rewrite pending.json without the accepted rows. Skipped rows
        # stay — running --review again picks up where you left off.
        if accepted_scoped_indices:
            accepted_items_indices = {
                index_map[s] for s in accepted_scoped_indices
            }
            data["pending"] = [
                it for i, it in enumerate(items)
                if i not in accepted_items_indices
            ]
            _write_pending(project, data)
            _sync_cache(project)
            _auto_commit(
                project,
                "review",
                f"accepted {accepted} link(s) on {page_id}",
            )

    tail = done_summary(
        [f"{accepted} accepted", f"{skipped} skipped"],
        elapsed=_time.monotonic() - start,
    )
    if quit_early:
        tail = f"{tail}  {_dim('(quit early)')}"
    click.echo("")
    click.echo(tail)
    click.echo("")




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

    A chunk_id is a 16-char identifier for one slice of an ingested
    source, recorded in the synthesized page's frontmatter.

    \b
    Getting a chunk_id to pass in:
      \x1b[36mwikiloom show <page_id> --field chunk_ids\x1b[0m

    \b
    Examples:
      \x1b[36mwikiloom source 9f2e1c4a3b8d7e02\x1b[0m
      \x1b[36mwikiloom source 9f2e1c4a3b8d7e02 --project ~/projects/kb\x1b[0m
      \x1b[36mwikiloom source --help\x1b[0m

    \b
    Output:
      metadata header (source, origin path or URL, chunk index,
      tokens, timestamp) followed by the exact chunk text.
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
    # URL sources carry no raw_path (nothing on disk) but do carry
    # ``url``. Surface whichever the catalog has so the click-through
    # points to the real origin rather than the em-dash placeholder.
    if source_entry and source_entry.raw_path:
        origin_label = "raw_path"
        origin_value = source_entry.raw_path
    elif source_entry and source_entry.url:
        origin_label = "url"
        origin_value = source_entry.url
    else:
        origin_label = "raw_path"
        origin_value = "—"

    # Labels are always the same width so values align into a clean
    # right column. Identifier-ish fields (chunk_id, source, url/path)
    # use the shared cyan "this is something you can type or point at"
    # color; plain-text fields (chunk index, content type, tokens,
    # timestamp) stay unstyled so they don't compete for attention.
    def _label(text: str) -> str:
        return _dim(f"  {text:<13}")

    created_display = _format_event_timestamp(chunk.created_at)

    click.echo("")
    click.echo(click.style("Chunk", bold=True))
    click.echo(f"{_label('chunk_id:')} {click.style(chunk.chunk_id, fg='cyan')}")
    click.echo(f"{_label('source:')} {click.style(source_name, fg='cyan')}")
    click.echo(
        f"{_label(origin_label + ':')} "
        f"{click.style(origin_value, fg='cyan') if origin_value != '—' else _dim(origin_value)}"
    )
    click.echo(
        f"{_label('chunk:')} {chunk.chunk_index + 1} of {chunk.chunk_total}"
    )
    click.echo(f"{_label('content_type:')} {chunk.content_type}")
    click.echo(f"{_label('tokens:')} {chunk.token_estimate:,}")
    click.echo(f"{_label('created_at:')} {created_display}")
    click.echo("")
    click.echo(click.style("Content", bold=True))
    # Indent every body line by two spaces so the block is visually
    # separated from the metadata above. Blank lines stay blank — no
    # trailing indent artifact.
    for line in chunk.text.splitlines():
        click.echo(f"  {line}" if line else "")
    click.echo("")


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
    """Commit manual edits with a human-edit: prefix.

    Picks up edits to pages under `wiki/`, plus `wikiloom.toml` and
    `.wikiloom/prompts/*.md`. The resulting commit is classified as
    human-authored so auto-tools (lint --fix, re-ingest) leave the
    affected pages alone.
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

    dirty = git.dirty_human_paths()
    if not dirty:
        click.echo("")
        click.echo(_dim("Nothing to save — working tree is clean."))
        click.echo("")
        return

    if dry_run:
        click.echo("")
        click.echo(
            click.style("Would commit", bold=True)
            + f"  {_dim('(' + str(len(dirty)) + ')')}"
        )
        for p in dirty:
            click.echo(f"  {click.style(str(p), fg='cyan')}")
        default_msg = message or f"human-edit: {len(dirty)} file(s) [protected]"
        click.echo("")
        click.echo(f"{_dim('Message:')} {default_msg}")
        click.echo("")
        return

    commit_msg = message or f"human-edit: {len(dirty)} file(s) [protected]"
    if not commit_msg.startswith("human-edit:"):
        commit_msg = f"human-edit: {commit_msg}"

    with FileLock(project):
        from wikiloom.registry import Registry
        from wikiloom.search import IndexUpdater

        # Auto-bump frontmatter.modified on each saved page so manual
        # edits don't silently roll into dormant. Also flip dormant →
        # active since the user just touched the page. `_bump_...`
        # self-skips anything that isn't a wiki/*.md page, so config
        # and prompt edits flow through untouched.
        freshened = _bump_modified_and_freshen(project, dirty)
        # Rebuild indexes so newly-created pages (human edits that
        # added whole new files) show up in category/root indexes.
        registry = Registry(project / "_registry")
        index_paths = IndexUpdater(
            project / "wiki", registry=registry
        ).rebuild_all()
        # `git add -A -- <paths>` handles adds, modifications, and
        # deletions in one call, which matters for pages the user
        # removed via their editor.
        staged = list(dirty) + list(index_paths)
        git.repo.git.add("-A", "--", *[str(p) for p in staged])
        git.commit([], commit_msg)
        _sync_cache(project, changed_files=staged)

    parts = [f"{len(dirty)} file(s) saved"]
    if freshened:
        parts.append(f"{freshened} freshened from dormant")
    click.echo("")
    click.echo(done_summary(parts))
    click.echo("")


def _bump_modified_and_freshen(project: Path, paths: list[Path]) -> int:
    """Bump modified on each page; flag real human edits on the way.

    For every wiki page in ``paths``:

    1. Always bumps ``frontmatter.modified`` and flips dormant →
       active (the "freshen" part).
    2. Flags ``human_edited = True`` (frontmatter + manifest) **only
       when the working-tree change actually touches the human region
       above the ``wikiloom:auto`` marker**. A relink that only
       rewrites wikilinks below the marker is a mechanical edit, not
       a human one, and shouldn't trigger the flag.

    New files (no HEAD version) are always flagged — they're
    human-created by definition. Non-markered files (no auto region
    yet) are also flagged — the whole body is human content.

    Returns the count of pages freshened from dormant.
    """
    from wikiloom.frontmatter import parse_frontmatter, render_frontmatter
    from wikiloom.git_ops import GitOps
    from wikiloom.protection import HumanEditProtection
    from wikiloom.registry import Registry
    from wikiloom.utils import now_iso, page_id_from_path

    freshened = 0
    timestamp = now_iso()
    registry: Registry | None = None
    manifest_changed = False

    try:
        git = GitOps(project)
    except ValueError:
        git = None

    def _head_body(rel_posix: str) -> str | None:
        """Return the file's body as of HEAD, or None if absent."""
        if git is None:
            return None
        try:
            head_text = git.repo.git.show(f"HEAD:{rel_posix}")
        except Exception:
            return None
        _, body_at_head = parse_frontmatter(head_text)
        return body_at_head

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

        # Decide whether this edit actually touches the human region.
        # If not, leave human_edited alone — it's a mechanical change
        # (e.g. linker rewrote wikilinks below the auto marker) that
        # happened to land in the save batch.
        head_body = _head_body(rel.as_posix())
        human_now, _ = HumanEditProtection.split(body)
        human_edit = (
            head_body is None  # new file — human-created
            or HumanEditProtection.split(head_body)[0] != human_now
        )

        original_status = fm.status
        fm.modified = timestamp
        if human_edit:
            fm.human_edited = True
            fm.human_edited_at = timestamp
        if fm.status == "dormant":
            fm.status = "active"
            freshened += 1
        new_text = render_frontmatter(fm) + "\n" + body
        if new_text != text or original_status != fm.status:
            full.write_text(new_text, encoding="utf-8")

        if human_edit:
            # Mirror the flag into the manifest so status / lint / the
            # HumanEditProtection scanner all see the same truth.
            if registry is None:
                registry = Registry(project / "_registry")
            page_id = page_id_from_path(project / "wiki", full)
            entry = registry.get_page(page_id)
            if entry is not None and not entry.human_edited:
                entry.human_edited = True
                manifest_changed = True

    if manifest_changed and registry is not None:
        registry.save()
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

    The cache at `_registry/wiki.db` is a git-ignored derived index.
    Run this if it's missing, corrupt, or suspected to be out of sync.
    """
    import time as _time

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

    click.echo("")
    click.echo("Rebuilding cache...")
    start = _time.monotonic()

    progress_fn: Callable[[int, int], None] | None = None
    if embedder is not None:
        click.echo("")
        click.echo("Computing embeddings...")
        click.echo("")
        check_mark = _check()

        def _progress(done: int, total: int) -> None:
            step = max(32, max(1, total // 10))
            if done == total or done % step == 0:
                click.echo(f"  {check_mark} {done}/{total} pages embedded")

        progress_fn = _progress

    with FileLock(project):
        cache = SQLiteCache(project / "_registry" / "wiki.db")
        count = cache.full_rebuild(
            project, embedder=embedder, progress=progress_fn
        )
    stats = cache.get_stats()

    click.echo("")
    click.echo(
        done_summary(
            [
                f"{count} pages",
                f"{stats['aliases']} aliases",
                f"{stats['backlinks']} backlinks",
            ],
            elapsed=_time.monotonic() - start,
        )
    )
    click.echo("")


def _print_report(report) -> None:
    """Render a ``LintReport`` to stdout with the shared CLI style."""
    click.echo("")
    if report.is_healthy and not report.promoted_from_update:
        click.echo(f"{_check()} Wiki is healthy.")
        click.echo("")
        return

    _CAP = 10  # how many items to show per category before the trailing note.

    def _section(name: str, count: int, note: str | None = None) -> None:
        header = (
            click.style(name, bold=True)
            + f"  {_dim('(' + str(count) + ')')}"
        )
        if note:
            header += f"  {_dim('— ' + note)}"
        click.echo(header)

    def _item(text: str) -> None:
        click.echo(f"  {_dim('•')} {text}")

    def _more(total: int, cap: int = _CAP) -> None:
        if total > cap:
            click.echo(f"  {_dim(f'… and {total - cap} more')}")

    click.echo(
        click.style("Lint report", bold=True)
        + f"  {_dim(f'({report.total_issues} issue(s))')}"
    )
    click.echo("")

    if report.broken_links:
        _section("Broken links", len(report.broken_links))
        for b in report.broken_links[:_CAP]:
            _item(
                f"{click.style(b.source, fg='cyan')} → {b.target}  "
                f"{_dim('(' + b.reason + ')')}"
            )
        _more(len(report.broken_links))
        click.echo("")

    if report.orphans:
        _section("Orphans", len(report.orphans))
        for pid in report.orphans[:_CAP]:
            _item(click.style(pid, fg="cyan"))
        _more(len(report.orphans))
        click.echo("")

    if report.dormant:
        _section(
            "Dormant candidates",
            len(report.dormant),
            note="informational — `wikiloom dormant <page>` to mark",
        )
        for d in report.dormant[:_CAP]:
            _item(
                f"{click.style(d.page_id, fg='cyan')}  "
                f"{_dim(f'({d.age_days}d > {d.window_days}d)')}"
            )
        _more(len(report.dormant))
        click.echo("")

    if report.duplicates:
        total = len(report.duplicates)
        safe = getattr(report, "duplicates_auto_safe", 0)
        review = total - safe
        split_note_parts = []
        if safe:
            split_note_parts.append(f"{safe} safe to auto-merge")
        if review:
            split_note_parts.append(f"{review} need review")
        split_note = (
            ", ".join(split_note_parts) if split_note_parts else None
        )
        _section("Duplicates", total, note=split_note)
        for d in report.duplicates[:_CAP]:
            pages = " ~ ".join(click.style(p, fg="cyan") for p in d.pages)
            emb_display = (
                f"emb {d.embedding_score:.2f}"
                if d.embedding_score >= 0
                else "emb n/a"
            )
            _item(
                f"{pages}  "
                f"{_dim(f'(slug {d.slug_score}%  •  {emb_display})')}"
            )
        _more(total)
        click.echo("")

    if report.frontmatter_issues:
        _section("Frontmatter issues", len(report.frontmatter_issues))
        for pid in report.frontmatter_issues[:_CAP]:
            _item(click.style(pid, fg="cyan"))
        _more(len(report.frontmatter_issues))
        click.echo("")

    if report.index_drift:
        _section("Index drift", len(report.index_drift))
        for name in report.index_drift[:_CAP]:
            _item(click.style(name, fg="cyan"))
        _more(len(report.index_drift))
        click.echo("")

    if report.contradictions:
        _section("Contradictions", len(report.contradictions))
        for c in report.contradictions[:_CAP]:
            _item(
                f"{click.style(c.page_id, fg='cyan')}  "
                f"{_dim(f'{c.existing[:40]} vs {c.new[:40]}')}"
            )
        _more(len(report.contradictions))
        click.echo("")

    if report.stubs:
        _section("Stubs", len(report.stubs))
        for pid in report.stubs[:_CAP]:
            _item(click.style(pid, fg="cyan"))
        _more(len(report.stubs))
        click.echo("")

    if report.promoted_from_update:
        _section(
            "Promoted from update",
            len(report.promoted_from_update),
            note=(
                "informational — LLM proposed an update for a nonexistent "
                "page; content preserved as a new page for review"
            ),
        )
        for pid in report.promoted_from_update[:_CAP]:
            _item(click.style(pid, fg="cyan"))
        _more(len(report.promoted_from_update))
        click.echo("")


if __name__ == "__main__":
    main()
