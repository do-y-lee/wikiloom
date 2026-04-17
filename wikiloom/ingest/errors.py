"""Exceptions raised by the ingest pipeline boundary guards."""

from __future__ import annotations


class IngestError(Exception):
    """Base class for ingest-boundary failures."""


class FileTooLargeError(IngestError):
    """Source file exceeds ``[ingest] max_file_size_mb``."""

    def __init__(self, path: str, size_mb: float, limit_mb: int) -> None:
        super().__init__(
            f"Source {path!r} is {size_mb:.1f} MB, which exceeds the "
            f"configured limit of {limit_mb} MB. Raise "
            f"[ingest] max_file_size_mb in wikiloom.toml to allow it."
        )
        self.path = path
        self.size_mb = size_mb
        self.limit_mb = limit_mb


class EmptyExtractionError(IngestError):
    """Extractor returned no usable text.

    Typically a scanned PDF with no text layer, a blank document, or
    a URL whose fetched body had no extractable content.
    """

    def __init__(self, path: str, content_type: str, extracted_chars: int) -> None:
        super().__init__(
            f"Extractor produced only {extracted_chars} character(s) of "
            f"text from {path!r} (content_type={content_type!r}). This "
            f"often means a scanned PDF with no text layer or an empty "
            f"document. Skipping ingest to avoid wasting LLM tokens on "
            f"empty input."
        )
        self.path = path
        self.content_type = content_type
        self.extracted_chars = extracted_chars


class BudgetExceededError(IngestError):
    """Pre-flight estimate exceeded the configured monthly budget.

    Raised by the ingest pre-flight budget check when the estimated
    token cost of the run would push the project past
    ``config.llm.monthly_budget_usd``. Disable the check by setting
    ``[ingest] enable_budget_check = false`` or raise the budget.
    """

    def __init__(self, estimated_usd: float, budget_usd: float) -> None:
        super().__init__(
            f"Pre-flight cost estimate of ${estimated_usd:.4f} exceeds "
            f"the monthly budget of ${budget_usd:.2f}. Raise "
            f"[llm] monthly_budget_usd in wikiloom.toml, disable the "
            f"check via [ingest] enable_budget_check = false, or "
            f"reduce the input size."
        )
        self.estimated_usd = estimated_usd
        self.budget_usd = budget_usd
