from __future__ import annotations

from copy import deepcopy

from backend.app.ingestion.models.content import TableContent, TextContent
from backend.app.ingestion.models.ingestion_result import IngestionResult
from backend.app.ingestion.models.source_element import SourceElement

from backend.app.ingestion.preprocessing.deduplication import (
    deduplicate_elements,
)
from backend.app.ingestion.preprocessing.metadata import normalize_metadata
from backend.app.ingestion.preprocessing.relationship_reconciliation import (
    reconcile_relationships,
)
from backend.app.ingestion.preprocessing.text import normalize_text
from backend.app.ingestion.preprocessing.validation import validate_element


def _preprocess_element(element: SourceElement) -> SourceElement:
    """
    Return a preprocessed copy of an element.

    The original SourceElement is never modified.
    """
    processed = deepcopy(element)

    # Normalize metadata first.
    processed.metadata = normalize_metadata(processed.metadata)

    # Normalize text-like content.
    if isinstance(processed.content, TextContent):
        processed.content = TextContent(
            text=normalize_text(processed.content.text),
        )

    elif isinstance(processed.content, TableContent):
        processed.content = TableContent(
            text=normalize_text(processed.content.text),
        )

    return processed


def _preprocess_elements(
    elements: list[SourceElement],
) -> list[SourceElement]:
    processed_elements: list[SourceElement] = []

    for element in elements:
        processed = _preprocess_element(element)

        validation = validate_element(processed)

        if not validation.is_valid:
            continue

        processed_elements.append(processed)

    return processed_elements


def preprocess_ingestion(
    ingestion_result: IngestionResult,
) -> IngestionResult:
    """
    Run the complete preprocessing pipeline.

    Pipeline order:

        normalize
            ↓
        validate
            ↓
        deduplicate
            ↓
        reconcile relationships

    The original IngestionResult is not modified.
    """
    processed_elements = _preprocess_elements(
        ingestion_result.elements,
    )

    processed_elements = deduplicate_elements(
        processed_elements,
    )

    processed_relationships = reconcile_relationships(
        processed_elements,
        ingestion_result.relationships,
    )

    return IngestionResult(
        document_id=ingestion_result.document_id,
        elements=processed_elements,
        relationships=processed_relationships,
    )