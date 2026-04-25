---
title: "Stub Title (already filled by the linker — do not change)"
type: entity   # or concept; matches the auto-created stub
status: stub   # change to "active" after you add real content
summary: "One-sentence summary that explains the entity or concept."
aliases: []
tags: []
related_pages: []
---

# Stub Title

## What it is

One short paragraph explaining what this entity or concept is in the
context of your wiki's domain. Aim for grounded, specific language —
"the Federal Deposit Insurance Corporation, the US agency that
insures bank deposits up to a per-account limit" rather than "FDIC."

## Why it matters here

Why does this page exist in *your* wiki? Which sources mention it?
What role does it play in the topics this wiki covers? Two or three
sentences.

## Key facts

- A bullet list of concrete facts you want a reader to come away with.
- Keep each bullet self-contained so it makes sense in isolation.
- Cite the source if a fact is non-obvious — `(see [[sources/foo]])`.

## Related concepts

- `[[concepts/related-thing]]` — how it connects.
- `[[entities/related-org]]` — how it connects.

---

## How to fill in a stub manually

1. Open the file in your editor (path printed by `wikiloom stubs`).
2. Replace the placeholder body with real content.
3. Update the frontmatter:
   - Change `status: stub` to `status: active`.
   - Replace the placeholder `summary:` with a real one-sentence summary.
   - Add any `aliases` the entity is also known by.
4. Save and run `wikiloom save` to commit the edit. The status flip
   to `active` is then locked in via the human-edit protection trail.

If the stub turns out to not be worth a real page:
- `wikiloom deprecate <page-id>` retires it (kept in archive).
- `wikiloom merge <winner> <stub-id>` folds it into an existing page
  and rewrites every inbound `[[stub-id]]` wikilink to `[[winner]]`.

*This file was copied from `.wikiloom/templates/stub.md`. The auto-
created stub on disk has the title and slug populated; this template
shows the body shape and the conventional set of sections.*
