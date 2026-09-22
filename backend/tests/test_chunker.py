from backend.app.ingestion.chunking.text_chunker import chunk_documents
from backend.app.ingestion.models.content import (
    ImageContent,
    TableContent,
    TextContent,
)
from backend.app.ingestion.models.element_relationship import ElementRelationship
from backend.app.ingestion.models.ingestion_result import IngestionResult
from backend.app.ingestion.models.source_element import SourceElement
import pytest

from backend.app.ingestion.models.chunk_document import ChunkDocument

def make_text_element(
    element_id: str,
    text: str,
    element_type: str = "paragraph",
    metadata: dict | None = None,
) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        document_id="document-1",
        element_type=element_type,
        content=TextContent(text),
        metadata=metadata or {},
    )


def make_table_element(
    element_id: str,
    text: str,
    metadata: dict | None = None,
) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        document_id="document-1",
        element_type="table",
        content=TableContent(text),
        metadata=metadata or {},
    )


def make_relationship(
    source_id: str,
    relationship_type: str,
    target_id: str,
) -> ElementRelationship:
    return ElementRelationship(
        source_element_id=source_id,
        relationship_type=relationship_type,
        target_element_id=target_id,
        metadata={},
    )


def test_single_element_creates_single_chunk():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("element-1", "Hello world"),
        ],
        relationships=[],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert len(chunks) == 1
    assert chunks[0].content == "Hello world"
    assert chunks[0].chunk_index == 0


def test_related_heading_and_paragraph_stay_in_same_chunk():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "heading-1",
                "Introduction",
                element_type="heading",
            ),
            make_text_element(
                "paragraph-1",
                "This is the introduction.",
            ),
        ],
        relationships=[
            make_relationship(
                "heading-1",
                "parent_of",
                "paragraph-1",
            ),
        ],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert len(chunks) == 1
    assert "Introduction" in chunks[0].content
    assert "This is the introduction." in chunks[0].content


def test_related_elements_are_recorded_in_metadata():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("heading-1", "Introduction", "heading"),
            make_text_element("paragraph-1", "Some explanation."),
        ],
        relationships=[
            make_relationship(
                "heading-1",
                "parent_of",
                "paragraph-1",
            ),
        ],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert chunks[0].metadata["element_ids"] == [
        "document-1:heading-1",
        "document-1:paragraph-1",
    ]


def test_relationship_metadata_is_preserved():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("heading-1", "Introduction", "heading"),
            make_text_element("paragraph-1", "Some explanation."),
        ],
        relationships=[
            ElementRelationship(
                source_element_id="heading-1",
                relationship_type="parent_of",
                target_element_id="paragraph-1",
                metadata={"confidence": 1.0},
            ),
        ],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    relationships = chunks[0].metadata["relationships"]

    assert len(relationships) == 1
    assert relationships[0]["source_element_id"] == "document-1:heading-1"
    assert relationships[0]["target_element_id"] == "document-1:paragraph-1"
    assert relationships[0]["relationship_type"] == "parent_of"
    assert relationships[0]["metadata"]["confidence"] == 1.0


def test_unrelated_elements_remain_separate():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("element-1", "First"),
            make_text_element("element-2", "Second"),
        ],
        relationships=[],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert len(chunks) == 2


def test_derived_table_relationship_keeps_table_with_chart_context():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "chart-1",
                "Sales by month",
                element_type="chart",
            ),
            make_table_element(
                "table-1",
                "Month | Sales\nJan | 100",
            ),
        ],
        relationships=[
            make_relationship(
                "chart-1",
                "derived_from",
                "table-1",
            ),
        ],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert len(chunks) == 1
    assert "Sales by month" in chunks[0].content
    assert "Month | Sales" in chunks[0].content


def test_caption_relationship_keeps_caption_with_related_content():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "caption-1",
                "Figure 1: System architecture",
                element_type="caption",
            ),
            make_text_element(
                "description-1",
                "The system contains three layers.",
            ),
        ],
        relationships=[
            make_relationship(
                "caption-1",
                "caption_of",
                "description-1",
            ),
        ],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert len(chunks) == 1


def test_spatial_relationship_is_preserved_but_does_not_force_text_grouping():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("element-1", "First"),
            make_text_element("element-2", "Second"),
        ],
        relationships=[
            make_relationship(
                "element-1",
                "spatially_adjacent",
                "element-2",
            ),
        ],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert len(chunks) == 2


def test_related_multimodal_element_is_preserved_as_context():
    image = SourceElement(
        element_id="image-1",
        document_id="document-1",
        element_type="image",
        content=ImageContent("/tmp/system.png"),
        metadata={},
    )

    paragraph = make_text_element(
        "paragraph-1",
        "The architecture is shown below.",
    )

    result = IngestionResult(
        document_id="document-1",
        elements=[paragraph, image],
        relationships=[
            make_relationship(
                "paragraph-1",
                "refers_to",
                "image-1",
            ),
        ],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert len(chunks) == 1
    assert chunks[0].content == "The architecture is shown below."
    assert "document-1:image-1" in chunks[0].metadata["related_element_ids"]


def test_long_related_group_is_split_when_chunk_size_is_exceeded():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "heading-1",
                "Introduction",
                element_type="heading",
            ),
            make_text_element(
                "paragraph-1",
                "This is a long paragraph that should exceed the limit.",
            ),
        ],
        relationships=[
            make_relationship(
                "heading-1",
                "parent_of",
                "paragraph-1",
            ),
        ],
    )

    chunks = chunk_documents(
        result,
        chunk_size=30,
        chunk_overlap=0,
    )

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks))
    )


def test_metadata_from_elements_is_preserved():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                "Hello",
                metadata={
                    "page_number": 3,
                    "bbox": {
                        "x0": 10,
                        "y0": 20,
                        "x1": 100,
                        "y1": 50,
                    },
                },
            ),
        ],
        relationships=[],
    )

    chunks = chunk_documents(result, chunk_size=100, chunk_overlap=0,)

    assert chunks[0].metadata["page_number"] == 3
    assert chunks[0].metadata["bbox"] == {
        "x0": 10,
        "y0": 20,
        "x1": 100,
        "y1": 50,
    }


def test_empty_result_returns_empty_list():
    result = IngestionResult(
        document_id="document-1",
        elements=[],
        relationships=[],
    )

    assert chunk_documents(result) == []


def test_non_chunkable_elements_without_text_are_skipped():
    image = SourceElement(
        element_id="image-1",
        document_id="document-1",
        element_type="image",
        content=ImageContent("/tmp/image.png"),
        metadata={},
    )

    result = IngestionResult(
        document_id="document-1",
        elements=[image],
        relationships=[],
    )

    assert chunk_documents(result) == []


def test_chunk_indices_are_global_and_sequential():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                "one two three four five six seven eight nine ten",
            ),
            make_text_element(
                "element-2",
                "another long piece of content here",
            ),
        ],
        relationships=[],
    )

    chunks = chunk_documents(
        result,
        chunk_size=20,
        chunk_overlap=0,
    )

    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks))
    )

def test_oversized_text_prefers_natural_boundaries():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                (
                    "This is the first sentence. "
                    "This is the second sentence. "
                    "This is the third sentence. "
                    "This is the fourth sentence."
                ),
            ),
        ],
        relationships=[],
    )

    chunks = chunk_documents(
        result,
        chunk_size=45,
        chunk_overlap=0,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.content) <= 45


def test_oversized_element_preserves_split_metadata():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                "one two three four five six seven eight nine ten",
            ),
        ],
        relationships=[],
    )

    chunks = chunk_documents(
        result,
        chunk_size=20,
        chunk_overlap=0,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["element_ids"] == ["document-1:element-1"]
        assert chunk.metadata["split_from_element"] is True
        assert "split_index" in chunk.metadata
        assert "split_count" in chunk.metadata


def test_chunk_overlap_is_supported():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                (
                    "This is a long piece of text that should "
                    "be split into overlapping chunks."
                ),
            ),
        ],
        relationships=[],
    )

    chunks = chunk_documents(
        result,
        chunk_size=40,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.content) <= 40


def test_invalid_chunk_size_raises_error():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("element-1", "Hello"),
        ],
        relationships=[],
    )

    try:
        chunk_documents(
            result,
            chunk_size=0,
            chunk_overlap=0,
        )
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_invalid_overlap_raises_error():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("element-1", "Hello"),
        ],
        relationships=[],
    )

    try:
        chunk_documents(
            result,
            chunk_size=20,
            chunk_overlap=20,
        )
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

def test_chunk_document_has_stable_chunk_id():
    chunk = ChunkDocument(
        content="This is a test chunk.",
        document_id="doc_123",
        chunk_index=0,
    )

    assert chunk.chunk_id
    assert isinstance(chunk.chunk_id, str)

def test_chunk_ids_are_unique_for_different_chunks():
    chunk_1 = ChunkDocument(
        content="First chunk.",
        document_id="doc_123",
        chunk_index=0,
    )

    chunk_2 = ChunkDocument(
        content="Second chunk.",
        document_id="doc_123",
        chunk_index=1,
    )

    assert chunk_1.chunk_id != chunk_2.chunk_id