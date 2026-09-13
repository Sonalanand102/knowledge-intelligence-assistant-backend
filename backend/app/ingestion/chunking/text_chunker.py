from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.ingestion.models.content import TableContent, TextContent
from backend.app.ingestion.models.ingestion_result import IngestionResult
from backend.app.ingestion.models.source_element import SourceElement


# These relationships are strong enough to influence chunk grouping.
GROUPING_RELATIONSHIPS = {
    "contains",
    "parent_of",
    "child_of",
    "caption_of",
    "derived_from",
    "refers_to",
    "belongs_to",
}


# These relationships provide context but should not force two otherwise
# unrelated textual elements into the same chunk.
CONTEXT_RELATIONSHIPS = {
    "follows",
    "precedes",
    "spatially_adjacent",
    "temporally_adjacent",
}


def _get_text(element: SourceElement) -> str | None:
    """Return text that can be included in a textual chunk."""
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

    1. A graph containing only strong grouping relationships.
    2. A mapping of all relationships touching each element.
    """
    grouping_graph: dict[str, set[str]] = defaultdict(set)
    relationships_by_element: dict[str, list] = defaultdict(list)

    for relationship in relationships:
        source_id = relationship.source_element_id
        target_id = relationship.target_element_id

        relationships_by_element[source_id].append(relationship)
        relationships_by_element[target_id].append(relationship)

        if relationship.relationship_type in GROUPING_RELATIONSHIPS:
            grouping_graph[source_id].add(target_id)
            grouping_graph[target_id].add(source_id)

    return grouping_graph, relationships_by_element


def _build_groups(
    elements: list[SourceElement],
    relationships: list,
) -> list[list[SourceElement]]:
    """
    Group chunkable elements using strong semantic relationships.

    Original source order is preserved.
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

        # Non-textual elements are not textual chunk groups.
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


def _create_text_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Create the recursive splitter used when an individual textual group
    exceeds the configured size.

    Separator priority:
        paragraph → line → sentence → word → character
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
        keep_separator=True,
    )


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Recursively split oversized text while preferring natural boundaries.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    splitter = _create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if chunk.strip()
    ]


def _group_relationship_metadata(
    group: list[SourceElement],
    relationships: list,
) -> list[dict[str, Any]]:
    """Return all relationships touching this chunk's elements."""
    group_ids = {
        element.element_id
        for element in group
    }

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
    """
    Build provenance and relationship metadata for a chunk.
    """
    group_ids = {
        element.element_id
        for element in group
    }

    related_element_ids: set[str] = set()

    for relationship in relationships:
        if relationship.source_element_id in group_ids:
            related_element_ids.add(
                relationship.target_element_id
            )

        if relationship.target_element_id in group_ids:
            related_element_ids.add(
                relationship.source_element_id
            )

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
        "related_element_ids": sorted(
            related_element_ids
        ),
    }

    # Preserve source metadata.
    #
    # A metadata key is promoted to chunk level only when:
    # - it appears on exactly one element, or
    # - all elements have the same value.
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

        elif values and all(
            value == values[0]
            for value in values
        ):
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
    Build chunks from one relationship group.

    Related elements are kept together as long as the size budget allows.
    Oversized individual elements are recursively split.
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

        # An individual element is larger than the configured budget.
        if element_length > chunk_size:
            flush()

            pieces = _split_text(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
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

        if len(current_elements) == 1:
            current_length = element_length
        else:
            current_length += 2 + element_length

    flush()

    return chunks


def chunk_documents(
    ingestion_result: IngestionResult,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[ChunkDocument]:
    """
    Convert an IngestionResult into relationship-aware chunks.

    Strategy:

        1. Group semantically related elements.
        2. Preserve source order.
        3. Keep related elements together when possible.
        4. Recursively split oversized textual elements.
        5. Preserve relationships and provenance in metadata.

    Contextual relationships such as spatial or temporal adjacency do not
    force unrelated textual elements into the same text chunk.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

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
        chunks.extend(
            _create_group_chunks(
                group=group,
                relationships=ingestion_result.relationships,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_id=ingestion_result.document_id,
                starting_index=len(chunks),
            )
        )

    return chunks