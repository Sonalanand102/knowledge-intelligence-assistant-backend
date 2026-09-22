from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.ingestion.enrichment.base import (
    ImageSemanticResult,
)
from backend.app.ingestion.enrichment.image_enricher import (
    ImageEnricher,
)
from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)
from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)


class FakeImageSemanticProvider:
    def describe_image(
        self,
        image_path: str,
    ) -> ImageSemanticResult:
        assert Path(image_path).exists()

        return ImageSemanticResult(
            text=(
                "Architecture diagram showing "
                "a FastAPI API layer and PostgreSQL."
            ),
            metadata={
                "provider": "fake",
            },
        )


def make_image_element(
    image_path: str,
) -> SourceElement:
    return SourceElement(
        element_id="image-1",
        document_id="document-1",
        element_type="image",
        content=ImageContent(
            path=image_path,
        ),
        metadata={
            "page_number": 2,
            "content_type": "image",
        },
    )


def test_image_enricher_supports_images(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")

    enricher = ImageEnricher(
        provider=FakeImageSemanticProvider()
    )

    element = make_image_element(
        str(image_path)
    )

    assert enricher.supports(element) is True


def test_image_enricher_does_not_support_text() -> None:
    element = SourceElement(
        element_id="text-1",
        document_id="document-1",
        element_type="paragraph",
        content=TextContent(
            text="hello",
        ),
        metadata={},
    )

    enricher = ImageEnricher(
        provider=FakeImageSemanticProvider()
    )

    assert enricher.supports(element) is False


def test_image_enricher_creates_semantic_element_and_relationship(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")

    enricher = ImageEnricher(
        provider=FakeImageSemanticProvider()
    )

    element = make_image_element(
        str(image_path)
    )

    elements, relationships = (
        enricher.enrich(element)
    )

    assert len(elements) == 1
    assert len(relationships) == 1

    semantic_element = elements[0]

    assert (
        semantic_element.element_id
        == "image-1:semantic"
    )

    assert (
        semantic_element.element_type
        == "image_semantic"
    )

    assert isinstance(
        semantic_element.content,
        TextContent,
    )

    assert (
        "FastAPI"
        in semantic_element.content.text
    )

    assert (
        semantic_element.metadata[
            "derived_from"
        ]
        == "image-1"
    )

    relationship = relationships[0]

    assert (
        relationship.source_element_id
        == "image-1:semantic"
    )

    assert (
        relationship.target_element_id
        == "image-1"
    )

    assert (
        relationship.relationship_type
        == "derived_from"
    )


def test_image_enricher_rejects_missing_asset(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "missing.png"

    enricher = ImageEnricher(
        provider=FakeImageSemanticProvider()
    )

    element = make_image_element(
        str(image_path)
    )

    with pytest.raises(FileNotFoundError):
        enricher.enrich(element)


def test_image_enricher_rejects_empty_semantic_output(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")

    class EmptyProvider:
        def describe_image(
            self,
            image_path: str,
        ) -> ImageSemanticResult:
            return ImageSemanticResult(
                text="",
                metadata={},
            )

    enricher = ImageEnricher(
        provider=EmptyProvider()
    )

    element = make_image_element(
        str(image_path)
    )

    with pytest.raises(ValueError):
        enricher.enrich(element)