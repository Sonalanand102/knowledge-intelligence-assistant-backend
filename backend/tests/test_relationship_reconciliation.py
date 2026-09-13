from backend.app.ingestion.models.element_relationship import ElementRelationship
from backend.app.ingestion.models.source_element import SourceElement
from backend.app.ingestion.models.content import TextContent
from backend.app.ingestion.preprocessing.relationship_reconciliation import (
    reconcile_relationships,
)


def make_element(element_id: str) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        document_id="document-1",
        element_type="paragraph",
        content=TextContent(f"Content {element_id}"),
        metadata={},
    )


def make_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str = "refers_to",
) -> ElementRelationship:
    return ElementRelationship(
        source_element_id=source_id,
        relationship_type=relationship_type,
        target_element_id=target_id,
        metadata={},
    )


def test_valid_relationship_is_preserved():
    elements = [
        make_element("a"),
        make_element("b"),
    ]

    relationships = [
        make_relationship("a", "b"),
    ]

    result = reconcile_relationships(elements, relationships)

    assert result == relationships


def test_relationship_with_missing_source_is_removed():
    elements = [
        make_element("b"),
    ]

    relationships = [
        make_relationship("a", "b"),
    ]

    result = reconcile_relationships(elements, relationships)

    assert result == []


def test_relationship_with_missing_target_is_removed():
    elements = [
        make_element("a"),
    ]

    relationships = [
        make_relationship("a", "b"),
    ]

    result = reconcile_relationships(elements, relationships)

    assert result == []


def test_relationship_with_both_missing_elements_is_removed():
    elements = []

    relationships = [
        make_relationship("a", "b"),
    ]

    result = reconcile_relationships(elements, relationships)

    assert result == []


def test_multiple_valid_relationships_are_preserved():
    elements = [
        make_element("a"),
        make_element("b"),
        make_element("c"),
    ]

    relationships = [
        make_relationship("a", "b"),
        make_relationship("b", "c"),
        make_relationship(
            "c",
            "a",
            relationship_type="spatially_adjacent",
        ),
    ]

    result = reconcile_relationships(elements, relationships)

    assert result == relationships


def test_mixed_valid_and_invalid_relationships():
    elements = [
        make_element("a"),
        make_element("b"),
    ]

    relationships = [
        make_relationship("a", "b"),
        make_relationship("a", "missing"),
        make_relationship("missing", "b"),
    ]

    result = reconcile_relationships(elements, relationships)

    assert result == [
        make_relationship("a", "b"),
    ]


def test_original_relationship_list_is_not_modified():
    elements = [
        make_element("a"),
        make_element("b"),
    ]

    relationships = [
        make_relationship("a", "b"),
        make_relationship("a", "missing"),
    ]

    original = relationships.copy()

    reconcile_relationships(elements, relationships)

    assert relationships == original


def test_empty_relationships_return_empty_list():
    elements = [
        make_element("a"),
    ]

    result = reconcile_relationships(elements, [])

    assert result == []