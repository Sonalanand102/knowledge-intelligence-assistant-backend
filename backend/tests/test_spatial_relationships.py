from backend.app.ingestion.models.content import (TextContent, ImageContent)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)
from backend.app.ingestion.relationships.resolver import (
    resolve_spatial_relationships,
)


def test_resolve_spatial_relationships():
    text = SourceElement(
        element_id="text-1",
        document_id="doc-1",
        element_type="paragraph",
        content=TextContent(
            text="Figure 1: System Architecture"
        ),
        metadata={
            "page_number": 1,
            "bbox": {
                "x0": 100,
                "y0": 100,
                "x1": 400,
                "y1": 140,
            },
        },
    )

    image = SourceElement(
        element_id="image-1",
        document_id="doc-1",
        element_type="image",
        content=TextContent(
            text=""
        ),
        metadata={
            "page_number": 1,
            "bbox": {
                "x0": 100,
                "y0": 160,
                "x1": 400,
                "y1": 400,
            },
        },
    )

    relationships = resolve_spatial_relationships(
        [text, image]
    )

    assert len(relationships) == 2

    pairs = {
        (
            relationship.source_element_id,
            relationship.target_element_id,
        )
        for relationship in relationships
    }

    assert (
        "text-1",
        "image-1",
    ) in pairs

    assert (
        "image-1",
        "text-1",
    ) in pairs


def test_different_pages_are_not_related():
    text = SourceElement(
        element_id="text-1",
        document_id="doc-1",
        element_type="paragraph",
        content=TextContent(text="hello"),
        metadata={
            "page_number": 1,
            "bbox": {
                "x0": 0,
                "y0": 0,
                "x1": 100,
                "y1": 100,
            },
        },
    )

    image = SourceElement(
        element_id="image-1",
        document_id="doc-1",
        element_type="image",
        content=ImageContent(path="image.png"),
        metadata={
            "page_number": 2,
            "bbox": {
                "x0": 0,
                "y0": 0,
                "x1": 100,
                "y1": 100,
            },
        },
    )

    relationships = resolve_spatial_relationships(
        [text, image]
    )

    assert relationships == []


def test_distant_elements_are_not_related():
    text = SourceElement(
        element_id="text-1",
        document_id="doc-1",
        element_type="paragraph",
        content=TextContent(text="hello"),
        metadata={
            "page_number": 1,
            "bbox": {
                "x0": 0,
                "y0": 0,
                "x1": 100,
                "y1": 100,
            },
        },
    )

    image = SourceElement(
        element_id="image-1",
        document_id="doc-1",
        element_type="image",
        content=ImageContent(path="image.png"),
        metadata={
            "page_number": 1,
            "bbox": {
                "x0": 500,
                "y0": 500,
                "x1": 600,
                "y1": 600,
            },
        },
    )

    relationships = resolve_spatial_relationships(
        [text, image]
    )

    assert relationships == []