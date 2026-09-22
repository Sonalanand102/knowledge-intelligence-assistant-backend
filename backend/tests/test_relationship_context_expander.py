from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.relationship_context_expander import (
    RelationshipContextExpander,
)


@dataclass(frozen=True)
class FakeRelationship:
    source_element_id: str
    relationship_type: str
    target_element_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FakeElement:
    element_id: str
    document_id: str
    element_type: str
    content_type: str
    text_content: str | None
    asset_path: str | None
    metadata: dict[str, Any]


class FakeRelationshipStore:
    def __init__(
        self,
        relationships: list[FakeRelationship],
        elements: list[FakeElement],
    ) -> None:
        self.relationships = relationships
        self.elements = elements

    async def get_relationships(
        self,
        element_ids: set[str],
        document_id: str,
    ) -> list[FakeRelationship]:
        return [
            relationship
            for relationship in self.relationships
            if (
                relationship.source_element_id in element_ids
                or relationship.target_element_id in element_ids
            )
        ]

    async def get_elements(
        self,
        element_ids: set[str],
        document_id: str,
    ) -> list[FakeElement]:
        return [
            element
            for element in self.elements
            if (
                element.element_id in element_ids
                and element.document_id == document_id
            )
        ]


def make_result(
    *,
    document_id: str = "doc-1",
    element_id: str = "doc-1:image-1:semantic",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="chunk-1",
        document_id=document_id,
        chunk_index=0,
        content="Image semantic description.",
        score=0.95,
        metadata={
            "element_ids": [element_id],
            "element_types": ["image_semantic"],
        },
    )


@pytest.mark.asyncio
async def test_expander_adds_related_text_context():
    relationships = [
        FakeRelationship(
            source_element_id="doc-1:image-1:semantic",
            relationship_type="derived_from",
            target_element_id="doc-1:image-1",
            metadata={
                "derivation_type": "visual_semantic_representation",
            },
        ),
        FakeRelationship(
            source_element_id="doc-1:paragraph-1",
            relationship_type="refers_to",
            target_element_id="doc-1:image-1",
            metadata={},
        ),
    ]

    elements = [
        FakeElement(
            element_id="doc-1:image-1",
            document_id="doc-1",
            element_type="image",
            content_type="image",
            text_content=None,
            asset_path="/tmp/image.webp",
            metadata={},
        ),
        FakeElement(
            element_id="doc-1:paragraph-1",
            document_id="doc-1",
            element_type="paragraph",
            content_type="text",
            text_content="Figure 1 shows the SWOT analysis.",
            asset_path=None,
            metadata={},
        ),
    ]

    store = FakeRelationshipStore(
        relationships=relationships,
        elements=elements,
    )

    expander = RelationshipContextExpander(
        store=store,
        max_hops=2,
    )

    results = await expander.expand(
        [make_result()],
    )

    assert len(results) == 1

    result = results[0]

    assert "Figure 1 shows the SWOT analysis." in result.content

    assert result.metadata["relationship_context"]

    related_ids = {
        item["element_id"]
        for item in result.metadata["relationship_context"]
    }

    assert "doc-1:paragraph-1" in related_ids


@pytest.mark.asyncio
async def test_expander_does_not_cross_document_boundaries():
    relationships = [
        FakeRelationship(
            source_element_id="doc-1:image-1:semantic",
            relationship_type="derived_from",
            target_element_id="doc-2:image-1",
            metadata={},
        ),
    ]

    elements = [
        FakeElement(
            element_id="doc-2:image-1",
            document_id="doc-2",
            element_type="image",
            content_type="image",
            text_content="Other document content.",
            asset_path="/tmp/other.webp",
            metadata={},
        ),
    ]

    store = FakeRelationshipStore(
        relationships=relationships,
        elements=elements,
    )

    expander = RelationshipContextExpander(
        store=store,
        max_hops=2,
    )

    results = await expander.expand(
        [make_result()],
    )

    assert "Other document content." not in results[0].content
    assert results[0].metadata["relationship_context"] == []


@pytest.mark.asyncio
async def test_expander_ignores_contextual_relationships():
    relationships = [
        FakeRelationship(
            source_element_id="doc-1:image-1:semantic",
            relationship_type="spatially_adjacent",
            target_element_id="doc-1:paragraph-1",
            metadata={},
        ),
    ]

    elements = [
        FakeElement(
            element_id="doc-1:paragraph-1",
            document_id="doc-1",
            element_type="paragraph",
            content_type="text",
            text_content="Unrelated nearby text.",
            asset_path=None,
            metadata={},
        ),
    ]

    store = FakeRelationshipStore(
        relationships=relationships,
        elements=elements,
    )

    expander = RelationshipContextExpander(
        store=store,
        max_hops=2,
    )

    results = await expander.expand(
        [make_result()],
    )

    assert "Unrelated nearby text." not in results[0].content


@pytest.mark.asyncio
async def test_expander_preserves_original_result_when_no_relationships():
    store = FakeRelationshipStore(
        relationships=[],
        elements=[],
    )

    result = make_result()

    expander = RelationshipContextExpander(
        store=store,
    )

    results = await expander.expand([result])

    assert results[0].content == result.content
    assert results[0].metadata["relationship_context"] == []


def test_expander_rejects_invalid_max_hops():
    store = FakeRelationshipStore(
        relationships=[],
        elements=[],
    )

    with pytest.raises(ValueError):
        RelationshipContextExpander(
            store=store,
            max_hops=0,
        )