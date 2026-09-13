from copy import deepcopy

from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)
from backend.app.ingestion.models.element_relationship import ElementRelationship
from backend.app.ingestion.models.ingestion_result import IngestionResult
from backend.app.ingestion.models.source_element import SourceElement
from backend.app.ingestion.preprocessing.pipeline import preprocess_ingestion


def make_text_element(
    element_id: str,
    text: str,
    metadata: dict | None = None,
) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        document_id="document-1",
        element_type="paragraph",
        content=TextContent(text),
        metadata=metadata or {},
    )


def test_pipeline_normalizes_text():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("element-1", "   Hello   world   "),
        ],
        relationships=[],
    )

    processed = preprocess_ingestion(result)

    assert len(processed.elements) == 1
    assert processed.elements[0].content.text == "Hello   world"


def test_pipeline_normalizes_metadata():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                "Hello",
                metadata={
                    "page_number": "3",
                    "source_type": "  pdf  ",
                    "custom_field": "preserve-me",
                },
            ),
        ],
        relationships=[],
    )

    processed = preprocess_ingestion(result)

    metadata = processed.elements[0].metadata

    assert metadata["page_number"] == 3
    assert metadata["source_type"] == "pdf"
    assert metadata["custom_field"] == "preserve-me"


def test_pipeline_removes_empty_elements():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("valid", "Hello"),
            make_text_element("empty", "   "),
        ],
        relationships=[],
    )

    processed = preprocess_ingestion(result)

    assert [element.element_id for element in processed.elements] == [
        "valid",
    ]


def test_pipeline_removes_exact_duplicates():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("first", "Hello world"),
            make_text_element("duplicate", "Hello world"),
            make_text_element("different", "Different content"),
        ],
        relationships=[],
    )

    processed = preprocess_ingestion(result)

    assert [element.element_id for element in processed.elements] == [
        "first",
        "different",
    ]


def test_pipeline_reconciles_relationships_after_filtering():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element("a", "Hello"),
            make_text_element("b", "World"),
        ],
        relationships=[
            ElementRelationship(
                source_element_id="a",
                relationship_type="refers_to",
                target_element_id="b",
            ),
            ElementRelationship(
                source_element_id="a",
                relationship_type="refers_to",
                target_element_id="missing",
            ),
        ],
    )

    processed = preprocess_ingestion(result)

    assert len(processed.relationships) == 1
    assert processed.relationships[0].source_element_id == "a"
    assert processed.relationships[0].target_element_id == "b"


def test_pipeline_preserves_multimodal_elements():
    image = SourceElement(
        element_id="image-1",
        document_id="document-1",
        element_type="image",
        content=ImageContent("/tmp/image.png"),
        metadata={
            "page_number": "2",
            "custom_field": "keep-me",
        },
    )

    result = IngestionResult(
        document_id="document-1",
        elements=[image],
        relationships=[],
    )

    processed = preprocess_ingestion(result)

    assert len(processed.elements) == 1
    assert processed.elements[0].content.path == "/tmp/image.png"
    assert processed.elements[0].metadata["page_number"] == 2
    assert processed.elements[0].metadata["custom_field"] == "keep-me"


def test_pipeline_does_not_modify_original_result():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                "   Hello   ",
                metadata={"page_number": "3"},
            ),
            make_text_element(
                "element-2",
                "Hello",
                metadata={"page_number": "4"},
            ),
        ],
        relationships=[],
    )

    original = deepcopy(result)

    processed = preprocess_ingestion(result)

    assert result == original
    assert processed is not result


def test_pipeline_preserves_document_id():
    result = IngestionResult(
        document_id="my-document",
        elements=[
            make_text_element("element-1", "Hello"),
        ],
        relationships=[],
    )

    processed = preprocess_ingestion(result)

    assert processed.document_id == "my-document"


def test_pipeline_handles_empty_result():
    result = IngestionResult(
        document_id="document-1",
        elements=[],
        relationships=[],
    )

    processed = preprocess_ingestion(result)

    assert processed.document_id == "document-1"
    assert processed.elements == []
    assert processed.relationships == []


def test_pipeline_is_idempotent():
    result = IngestionResult(
        document_id="document-1",
        elements=[
            make_text_element(
                "element-1",
                "   Hello   world   ",
                metadata={"page_number": "3"},
            ),
            make_text_element(
                "element-2",
                "Hello   world",
                metadata={"page_number": "4"},
            ),
        ],
        relationships=[],
    )

    first_pass = preprocess_ingestion(result)
    second_pass = preprocess_ingestion(first_pass)

    assert second_pass == first_pass