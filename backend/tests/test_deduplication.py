from backend.app.db.models.ingestion_run import IngestionRun
from backend.app.ingestion.models.content import TextContent
from backend.app.ingestion.models.source_element import SourceElement
from backend.app.ingestion.preprocessing.deduplication import (
    deduplicate_elements,
    content_hash,
)


def make_element(element_id: str, text: str) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        document_id="document-1",
        element_type="paragraph",
        content=TextContent(text),
        metadata={},
    )


def test_content_hash_is_deterministic():
    text = "Hello world"

    first = content_hash(text)
    second = content_hash(text)

    assert first == second


def test_different_content_has_different_hash():
    first = content_hash("Hello world")
    second = content_hash("Goodbye world")

    assert first != second


def test_duplicate_elements_are_removed():
    elements = [
        make_element("element-1", "Hello world"),
        make_element("element-2", "Hello world"),
        make_element("element-3", "Different text"),
    ]

    result = deduplicate_elements(elements)

    assert [element.element_id for element in result] == [
        "element-1",
        "element-3",
    ]


def test_first_occurrence_is_preserved():
    elements = [
        make_element("first", "Same content"),
        make_element("second", "Same content"),
    ]

    result = deduplicate_elements(elements)

    assert len(result) == 1
    assert result[0].element_id == "first"


def test_unique_elements_are_preserved():
    elements = [
        make_element("element-1", "Hello"),
        make_element("element-2", "World"),
        make_element("element-3", "Python"),
    ]

    result = deduplicate_elements(elements)

    assert len(result) == 3
    assert [element.element_id for element in result] == [
        "element-1",
        "element-2",
        "element-3",
    ]


def test_empty_input_returns_empty_list():
    assert deduplicate_elements([]) == []


def test_non_text_elements_are_preserved():
    image = SourceElement(
        element_id="image-1",
        document_id="document-1",
        element_type="image",
        content=__import__(
            "backend.app.ingestion.models.content",
            fromlist=["ImageContent"],
        ).ImageContent("/tmp/image.png"),
        metadata={},
    )

    result = deduplicate_elements([image])

    assert len(result) == 1
    assert result[0].element_id == "image-1"


def test_deduplication_does_not_modify_original_list():
    elements = [
        make_element("element-1", "Hello"),
        make_element("element-2", "Hello"),
    ]

    original_ids = [element.element_id for element in elements]

    deduplicate_elements(elements)

    assert [element.element_id for element in elements] == original_ids

def test_ingestion_run_document_id_is_nullable():
    column = IngestionRun.__table__.columns[
        "document_id"
    ]

    assert column.nullable is True