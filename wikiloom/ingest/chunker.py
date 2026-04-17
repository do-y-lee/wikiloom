"""Chunker for splitting oversized sources into LLM-sized pieces.

Splits by markdown headings, PDF page boundaries, or paragraph
groups with overlap. Most files fit in one chunk without splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wikiloom.ingest.extractors.base import ExtractedContent
from wikiloom.utils import estimate_tokens


@dataclass
class BudgetPlan:
    """Token budget for an ingest operation.

    Produced by the context budget manager (Component 14) ahead of chunking.
    For now we use a simple structure with the two fields the chunker needs.
    """

    available_for_source: int
    needs_chunking: bool


class Chunker:
    """Split oversized ExtractedContent into coherent chunks."""

    def split(self, content: ExtractedContent, budget: BudgetPlan) -> list[ExtractedContent]:
        """Split content into chunks that fit the budget.

        Tries structural boundaries in preference order. Always returns
        at least one chunk.
        """
        max_tokens = budget.available_for_source

        # Strategy 1: split on markdown headings
        sections = self._split_on_headings(content.text)
        if sections and all(estimate_tokens(s) <= max_tokens for s in sections):
            return self._to_chunks(sections, content)

        # Strategy 2: PDF page boundaries
        if content.content_type == "pdf" and content.metadata.get("pages"):
            page_groups = self._group_pages(content.metadata["pages"], max_tokens)
            if page_groups:
                return self._to_chunks(page_groups, content)

        # Strategy 3: paragraph groups with overlap (last resort)
        paragraphs = content.text.split("\n\n")
        para_groups = self._group_by_token_budget(
            paragraphs, max_tokens, overlap_tokens=200
        )
        return self._to_chunks(para_groups, content)

    # ------------------------------------------------------------------
    # Boundary strategies
    # ------------------------------------------------------------------

    def _split_on_headings(self, text: str) -> list[str] | None:
        """Split text on markdown level-1 or level-2 headings.

        Returns None if no headings are found.
        """
        heading_pattern = r"^#{1,2}\s+"
        sections = re.split(heading_pattern, text, flags=re.MULTILINE)
        if len(sections) <= 1:
            return None
        return [s.strip() for s in sections if s.strip()]

    def _group_pages(self, pages: list[str], max_tokens: int) -> list[str]:
        """Group consecutive PDF pages into chunks within the budget.

        Keeps a 1-page overlap between groups for continuity.
        """
        groups: list[str] = []
        current_group: list[str] = []
        current_tokens = 0

        for page_text in pages:
            page_tokens = estimate_tokens(page_text)
            if current_tokens + page_tokens > max_tokens and current_group:
                groups.append("\n\n".join(current_group))
                # Overlap: keep the last page of the previous group
                current_group = [current_group[-1]]
                current_tokens = estimate_tokens(current_group[0])
            current_group.append(page_text)
            current_tokens += page_tokens

        if current_group:
            groups.append("\n\n".join(current_group))
        return groups

    def _group_by_token_budget(
        self,
        blocks: list[str],
        max_tokens: int,
        overlap_tokens: int = 200,
    ) -> list[str]:
        """Group text blocks into chunks within the budget, with overlap."""
        groups: list[str] = []
        current_blocks: list[str] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = estimate_tokens(block)
            if current_tokens + block_tokens > max_tokens and current_blocks:
                groups.append("\n\n".join(current_blocks))
                # Build overlap from the tail of the current group
                overlap_blocks: list[str] = []
                overlap_count = 0
                for b in reversed(current_blocks):
                    overlap_count += estimate_tokens(b)
                    if overlap_count > overlap_tokens:
                        break
                    overlap_blocks.insert(0, b)
                current_blocks = overlap_blocks
                current_tokens = sum(estimate_tokens(b) for b in current_blocks)
            current_blocks.append(block)
            current_tokens += block_tokens

        if current_blocks:
            groups.append("\n\n".join(current_blocks))
        return groups

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _to_chunks(
        self,
        chunk_texts: list[str],
        content: ExtractedContent,
    ) -> list[ExtractedContent]:
        """Wrap chunk text strings as ExtractedContent objects.

        Each chunk inherits the parent's source_path, content_type, and
        extraction_method, but has its own token estimate. The parent's
        metadata is shallow-copied with chunk_index/chunk_total added.
        """
        chunks: list[ExtractedContent] = []
        total = len(chunk_texts)
        for idx, text in enumerate(chunk_texts):
            chunk_meta = dict(content.metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_total"] = total
            chunks.append(
                ExtractedContent(
                    text=text,
                    metadata=chunk_meta,
                    source_path=content.source_path,
                    content_type=content.content_type,
                    extraction_method=content.extraction_method,
                    token_estimate=estimate_tokens(text),
                )
            )
        return chunks


def plan_budget(
    content: ExtractedContent,
    max_tokens_per_operation: int,
    manifest_token_overhead: int = 0,
) -> BudgetPlan:
    """Produce a simple BudgetPlan for the given content.

    This is a placeholder for the full Context Budget Manager (Component 14).
    It reserves space for the manifest and prompt overhead and decides
    whether the source needs chunking based on the remaining budget.
    """
    # Reserve roughly 25% of the budget for prompts + manifest + response.
    overhead = manifest_token_overhead + max(1, max_tokens_per_operation // 4)
    available = max(1, max_tokens_per_operation - overhead)
    return BudgetPlan(
        available_for_source=available,
        needs_chunking=content.token_estimate > available,
    )
