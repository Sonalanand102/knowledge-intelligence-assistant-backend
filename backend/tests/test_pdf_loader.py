import base64
from pathlib import Path

import pymupdf
import pytest

from backend.app.ingestion.loaders.pdf_loader import (
    load_pdf,
)
from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    "CAQAAAC1HAwCAAAAC0lEQVR42mNk+A8"
    "AAQUBAScY42YAAAAASUVORK5CYII="
)


def create_test_pdf(
    pdf_path: Path,
) -> None:
    document = pymupdf.open()

    page = document.new_page()

    # ---------------------------------------------------------
    # Paragraph referencing the image
    # ---------------------------------------------------------
    page.insert_text(
        (72, 72),
        "Figure 1 shows the system architecture.",
    )

    # ---------------------------------------------------------
    # Actual embedded image
    # ---------------------------------------------------------
    image_path = (
        pdf_path.parent / "test-image.png"
    )

    image_path.write_bytes(
        PNG_BYTES
    )

    page.insert_image(
        pymupdf.Rect(
            72,
            150,
            250,
            260,
        ),
        filename=str(
            image_path
        ),
    )

    # ---------------------------------------------------------
    # Image caption
    # ---------------------------------------------------------
    page.insert_text(
        (72, 285),
        "Figure 1: System Architecture",
    )

    # ---------------------------------------------------------
    # Hyperlink
    # ---------------------------------------------------------
    page.insert_text(
        (72, 340),
        "OpenAI",
    )

    page.insert_link(
        {
            "kind": pymupdf.LINK_URI,
            "from": pymupdf.Rect(
                72,
                325,
                130,
                345,
            ),
            "uri": "https://openai.com",
        }
    )

    document.save(
        str(pdf_path)
    )

    document.close()


def test_load_pdf_extracts_elements(
    tmp_path,
):
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "output"

    create_test_pdf(
        pdf_path
    )

    result = load_pdf(
        file_path=str(pdf_path),
        document_id="pdf-123",
        output_dir=str(output_dir),
    )

    assert result.document_id == "pdf-123"
    assert result.elements

    element_types = {
        element.element_type
        for element in result.elements
    }

    assert "page" in element_types
    assert "paragraph" in element_types
    assert "image" in element_types
    assert "link" in element_types


def test_pdf_extracts_embedded_image(
    tmp_path,
):
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "output"

    create_test_pdf(
        pdf_path
    )

    result = load_pdf(
        file_path=str(pdf_path),
        document_id="pdf-123",
        output_dir=str(output_dir),
    )

    images = [
        element
        for element in result.elements
        if element.element_type == "image"
    ]

    assert len(images) == 1

    image = images[0]

    assert isinstance(
        image.content,
        ImageContent,
    )

    assert Path(
        image.content.path
    ).exists()

    assert (
        image.metadata["page_number"]
        == 1
    )

    assert "bbox" in image.metadata


def test_pdf_preserves_page_relationships(
    tmp_path,
):
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "output"

    create_test_pdf(
        pdf_path
    )

    result = load_pdf(
        file_path=str(pdf_path),
        document_id="pdf-123",
        output_dir=str(output_dir),
    )

    page_relationships = [
        relationship
        for relationship in result.relationships
        if (
            relationship.source_element_id
            == "page-1"
            and relationship.relationship_type
            == "contains"
        )
    ]

    assert page_relationships

    target_ids = {
        relationship.target_element_id
        for relationship in page_relationships
    }

    assert any(
        target_id.startswith(
            "page-1-text-"
        )
        for target_id in target_ids
    )

    assert any(
        target_id.startswith(
            "page-1-image-"
        )
        for target_id in target_ids
    )

    assert any(
        target_id.startswith(
            "page-1-link-"
        )
        for target_id in target_ids
    )


def test_pdf_preserves_link_metadata(
    tmp_path,
):
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "output"

    create_test_pdf(
        pdf_path
    )

    result = load_pdf(
        file_path=str(pdf_path),
        document_id="pdf-123",
        output_dir=str(output_dir),
    )

    links = [
        element
        for element in result.elements
        if element.element_type == "link"
    ]

    assert len(links) == 1

    link = links[0]

    assert isinstance(
        link.content,
        TextContent,
    )

    assert link.content.text == "OpenAI"

    assert (
        link.metadata["url"]
        == "https://openai.com"
    )


def test_pdf_creates_image_relationships(
    tmp_path,
):
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "output"

    create_test_pdf(
        pdf_path
    )

    result = load_pdf(
        file_path=str(pdf_path),
        document_id="pdf-123",
        output_dir=str(output_dir),
    )

    caption_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == "caption_of"
    ]

    reference_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == "refers_to"
    ]

    assert caption_relationships
    assert reference_relationships

    image_ids = {
        element.element_id
        for element in result.elements
        if element.element_type == "image"
    }

    assert (
        caption_relationships[0].target_element_id
        in image_ids
    )

    assert (
        reference_relationships[0].target_element_id
        in image_ids
    )


def test_pdf_missing_file(
    tmp_path,
):
    missing_path = (
        tmp_path / "missing.pdf"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_pdf(
            file_path=str(
                missing_path
            ),
            document_id="pdf-123",
            output_dir=str(
                tmp_path / "output"
            ),
        )


def test_pdf_rejects_non_pdf(
    tmp_path,
):
    file_path = (
        tmp_path / "sample.txt"
    )

    file_path.write_text(
        "hello"
    )

    with pytest.raises(
        ValueError
    ):
        load_pdf(
            file_path=str(
                file_path
            ),
            document_id="pdf-123",
            output_dir=str(
                tmp_path / "output"
            ),
        )