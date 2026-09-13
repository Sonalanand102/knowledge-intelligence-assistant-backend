from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.ingestion.models.content import TableContent, TextContent
from backend.app.ingestion.models.ingestion_result import IngestionResult
from backend.app.ingestion.models.source_element import SourceElement


# Relationships strong enough to influence chunk grouping.
GROUPING_RELATIONSHIPS = {
    "contains",
    "parent_of",
    "child_of",
    "caption_of",
    "derived_from",
    "refers_to",
    "belongs_to",
}

# These relationships provide ordering/context, but do not merge
# otherwise unrelated elements into the same group.
CONTEXT_RELATIONSHIPS = {
    "follows",
    "precedes",
    "spatially_adjacent",
    "temporally_adjacent",
}


def _get_text(element: SourceElement) -> str | None:
    """Return chunkable textual content."""
    if isinstance(element.content, TextContent):
        return element.content.text

    if isinstance(element.content, TableContent):
        return element.content.text

    return None


def _is_chunkable(element: SourceElement) -> bool:
    return _get_text(element) is not None


def _build_relationship_maps(
    relationships: list,
) -> tuple[dict[str, set[str]], dict[str, list]]:
    """
    Build:
    1. grouping adjacency
    2. all relationship metadata by element
    """
    grouping_graph: dict[str, set[str]] = defaultdict(set)
    relationships_by_element: dict[str, list] = defaultdict(list)

    for relationship in relationships:
        source_id = relationship.source_element_id
        target_id = relationship.target_element_id
        relationship_type = relationship.relationship_type

        relationships_by_element[source_id].append(relationship)
        relationships_by_element[target_id].append(relationship)

        if relationship_type in GROUPING_RELATIONSHIPS:
            grouping_graph[source_id].add(target_id)
            grouping_graph[target_id].add(source_id)

    return grouping_graph, relationships_by_element


def _build_groups(
    elements: list[SourceElement],
    relationships: list,
) -> list[list[SourceElement]]:
    """
    Build connected groups using only strong semantic relationships.

    Original element order is preserved.
    """
    element_by_id = {
        element.element_id: element
        for element in elements
    }

    grouping_graph, _ = _build_relationship_maps(relationships)

    visited: set[str] = set()
    groups: list[list[SourceElement]] = []

    for element in elements:
        element_id = element.element_id

        if element_id in visited:
            continue

        if not _is_chunkable(element):
            visited.add(element_id)
            continue

        stack = [element_id]
        group_ids: set[str] = set()

        while stack:
            current_id = stack.pop()

            if current_id in visited:
                continue

            current = element_by_id.get(current_id)

            if current is None:
                continue

            visited.add(current_id)

            if not _is_chunkable(current):
                continue

            group_ids.add(current_id)

            for neighbor_id in grouping_graph.get(current_id, set()):
                if neighbor_id not in visited:
                    stack.append(neighbor_id)

        if not group_ids:
            continue

        # Restore original source order.
        group = [
            current_element
            for current_element in elements
            if current_element.element_id in group_ids
        ]

        groups.append(group)

    return groups


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split oversized text while preserving a simple overlap.

    This remains a deterministic fallback for groups that exceed
    the configured chunk size.
    """
    if len(text) <= chunk_size:
        return [text]

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def _group_relationship_metadata(
    group: list[SourceElement],
    relationships: list,
) -> list[dict[str, Any]]:
    """Return relationships touching elements in this chunk group."""
    group_ids = {element.element_id for element in group}

    metadata: list[dict[str, Any]] = []

    for relationship in relationships:
        if (
            relationship.source_element_id in group_ids
            or relationship.target_element_id in group_ids
        ):
            metadata.append(
                {
                    "source_element_id": relationship.source_element_id,
                    "relationship_type": relationship.relationship_type,
                    "target_element_id": relationship.target_element_id,
                    "metadata": deepcopy(relationship.metadata),
                }
            )

    return metadata


def _build_chunk_metadata(
    group: list[SourceElement],
    relationships: list,
) -> dict[str, Any]:
    """Build provenance and relationship metadata for a chunk."""
    group_ids = {element.element_id for element in group}

    related_element_ids: set[str] = set()

    for relationship in relationships:
        if relationship.source_element_id in group_ids:
            related_element_ids.add(relationship.target_element_id)

        if relationship.target_element_id in group_ids:
            related_element_ids.add(relationship.source_element_id)

    related_element_ids -= group_ids

    metadata: dict[str, Any] = {
        "element_ids": [
            element.element_id
            for element in group
        ],
        "element_types": [
            element.element_type
            for element in group
        ],
        "relationships": _group_relationship_metadata(
            group,
            relationships,
        ),
        "related_element_ids": sorted(related_element_ids),
    }

    # Preserve metadata from source elements.
    #
    # For a group, values that exist on all elements and are identical
    # are promoted to the chunk level.
    keys: set[str] = set()

    for element in group:
        keys.update(element.metadata.keys())

    for key in keys:
        values = [
            element.metadata[key]
            for element in group
            if key in element.metadata
        ]

        if len(values) == 1:
            metadata[key] = deepcopy(values[0])

        elif values and all(value == values[0] for value in values):
            metadata[key] = deepcopy(values[0])

    return metadata


def _create_group_chunks(
    group: list[SourceElement],
    relationships: list,
    chunk_size: int,
    chunk_overlap: int,
    document_id: str,
    starting_index: int,
) -> list[ChunkDocument]:
    """
    Create chunks from one relationship group.

    Elements are combined until the chunk-size budget is exceeded.
    """
    chunks: list[ChunkDocument] = []

    current_elements: list[SourceElement] = []
    current_length = 0

    def flush() -> None:
        nonlocal current_elements, current_length

        if not current_elements:
            return

        content_parts = [
            _get_text(element)
            for element in current_elements
        ]

        content = "\n\n".join(
            part
            for part in content_parts
            if part is not None
        ).strip()

        if not content:
            current_elements = []
            current_length = 0
            return

        chunk_index = starting_index + len(chunks)

        chunks.append(
            ChunkDocument(
                content=content,
                document_id=document_id,
                chunk_index=chunk_index,
                metadata=_build_chunk_metadata(
                    current_elements,
                    relationships,
                ),
            )
        )

        current_elements = []
        current_length = 0

    for element in group:
        text = _get_text(element)

        if text is None:
            continue

        text = text.strip()

        if not text:
            continue

        element_length = len(text)

        # Oversized individual element.
        if element_length > chunk_size:
            flush()

            pieces = _split_text(
                text,
                chunk_size,
                chunk_overlap,
            )

            for piece_index, piece in enumerate(pieces):
                metadata = _build_chunk_metadata(
                    [element],
                    relationships,
                )

                chunks.append(
                    ChunkDocument(
                        content=piece,
                        document_id=document_id,
                        chunk_index=starting_index + len(chunks),
                        metadata={
                            **metadata,
                            "split_from_element": True,
                            "split_index": piece_index,
                            "split_count": len(pieces),
                        },
                    )
                )

            continue

        separator_length = 2 if current_elements else 0
        proposed_length = (
            current_length
            + separator_length
            + element_length
        )

        if current_elements and proposed_length > chunk_size:
            flush()

        current_elements.append(element)
        current_length = (
            element_length
            if len(current_elements) == 1
            else current_length + 2 + element_length
        )

    flush()

    return chunks


def chunk_documents(
    ingestion_result: IngestionResult,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[ChunkDocument]:
    """
    Convert a preprocessed IngestionResult into relationship-aware chunks.

    Strong semantic relationships can group elements together.
    Weak/contextual relationships are preserved as metadata but do not
    force unrelated elements into the same textual chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    groups = _build_groups(
        ingestion_result.elements,
        ingestion_result.relationships,
    )

    chunks: list[ChunkDocument] = []

    for group in groups:
        group_chunks = _create_group_chunks(
            group=group,
            relationships=ingestion_result.relationships,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=ingestion_result.document_id,
            starting_index=len(chunks),
        )

        chunks.extend(group_chunks)

    return chunks