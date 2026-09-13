from __future__ import annotations

import hashlib

from backend.app.ingestion.models.content import TextContent
from backend.app.ingestion.models.source_element import SourceElement


def content_hash(text: str) -> str:
    """
    Return a deterministic SHA-256 hash for the given content.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_deduplication_content(element: SourceElement) -> str | None:
    """
    Return the content used for exact deduplication.

    Currently text and table content can be compared directly.
    Other multimodal elements are preserved without deduplication.
    """
    if isinstance(element.content, TextContent):
        return element.content.text

    # Tables may be handled separately once table-specific
    # preprocessing is introduced.
    return None


def deduplicate_elements(
    elements: list[SourceElement],
) -> list[SourceElement]:
    """
    Remove exact duplicate elements while preserving the first occurrence.

    The input list and its elements are not modified.
    """
    seen_hashes: set[str] = set()
    unique_elements: list[SourceElement] = []

    for element in elements:
        content = _get_deduplication_content(element)

        # Non-text/multimodal elements are preserved as-is for now.
        if content is None:
            unique_elements.append(element)
            continue

        normalized_content = content.strip()
        content_fingerprint = content_hash(normalized_content)

        if content_fingerprint in seen_hashes:
            continue

        seen_hashes.add(content_fingerprint)
        unique_elements.append(element)

    return unique_elements