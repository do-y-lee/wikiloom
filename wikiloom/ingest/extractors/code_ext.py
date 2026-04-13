"""Source code extractor with language context."""

from __future__ import annotations

from pathlib import Path

from wikiloom.ingest.extractors.base import BaseExtractor, ExtractedContent
from wikiloom.utils import estimate_tokens

CODE_CONTEXT: dict[str, str] = {
    ".py":         "Python",
    ".js":         "JavaScript",
    ".ts":         "TypeScript",
    ".tsx":        "React TypeScript component",
    ".jsx":        "React JavaScript component",
    ".go":         "Go",
    ".rs":         "Rust",
    ".java":       "Java",
    ".rb":         "Ruby",
    ".cs":         "C#",
    ".cpp":        "C++",
    ".c":          "C",
    ".sh":         "Shell/Bash script",
    ".sql":        "SQL database query or schema definition",
    ".tf":         "Terraform infrastructure-as-code",
    ".hcl":        "HashiCorp Configuration Language",
    ".proto":      "Protocol Buffers schema definition",
    ".graphql":    "GraphQL schema or query",
    ".yaml":       "YAML configuration",
    ".yml":        "YAML configuration",
    ".json":       "JSON configuration or data",
    ".toml":       "TOML configuration",
    ".dockerfile": "Docker container definition",
}


class CodeExtractor(BaseExtractor):
    """Reads source code files as plain text and prepends language context.

    The language hint significantly improves wiki page quality — the LLM
    knows to extract resource definitions from .tf files, table schemas
    from .sql files, component props from .tsx files, etc.
    """

    def can_handle(self, path: Path) -> bool:
        if path.name.lower() == "dockerfile":
            return True
        return path.suffix.lower() in CODE_CONTEXT

    def extract(self, path: Path) -> ExtractedContent:
        text = path.read_text(encoding="utf-8", errors="replace")

        if path.name.lower() == "dockerfile":
            lang = "Docker container definition"
        else:
            lang = CODE_CONTEXT.get(path.suffix.lower(), "source code")

        contextualized = f"[File: {path.name} | Language: {lang}]\n\n{text}"

        return ExtractedContent(
            text=contextualized,
            metadata={"language": lang, "filename": path.name},
            source_path=path,
            content_type="code",
            extraction_method="plain-text-with-context",
            token_estimate=estimate_tokens(contextualized),
        )
