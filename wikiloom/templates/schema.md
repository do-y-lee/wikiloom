# Wiki Schema & Conventions

## Page Types

- **entities/**: People, organizations, tools, projects, products
- **concepts/**: Ideas, methods, patterns, algorithms, techniques
- **sources/**: One summary per raw source document ingested
- **syntheses/**: Cross-cutting analyses that connect multiple entities/concepts
- **decisions/**: Decision records with rationale and alternatives considered

## Naming Conventions

- File names use **lowercase hyphenated slugs**: `attention-mechanism.md`, `google-brain.md`
- No abbreviations in file names (use `reinforcement-learning.md`, not `rl.md`)
- No special characters beyond hyphens
- Page ID = relative path minus `.md` (e.g., `entities/google-brain`)

## Frontmatter Requirements

Every wiki page MUST have YAML frontmatter with these required fields:
- `title`: Human-readable title
- `type`: One of `entity`, `concept`, `source`, `synthesis`, `decision`
- `status`: One of `active`, `stub`, `deprecated`
- `created`: ISO 8601 timestamp
- `modified`: ISO 8601 timestamp
- `summary`: One-line description

## Writing Rules

1. Write **prose only** — do NOT include `[[wikilinks]]`. The linking engine handles all cross-references deterministically after you write.
2. Use markdown formatting (headings, lists, bold, code blocks) as appropriate.
3. When information conflicts with existing pages, **flag the contradiction explicitly** — never silently overwrite.
4. Each page should be self-contained enough to be useful on its own.
5. Attribute claims to their sources whenever possible.

## Contradiction Handling

When new information contradicts existing content:
- Add a "Contradictions" section noting both the existing and new claims
- Include the source for each claim
- Do NOT resolve the contradiction — flag it for human review
