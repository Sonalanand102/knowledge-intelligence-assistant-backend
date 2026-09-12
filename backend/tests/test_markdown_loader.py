# from backend.app.ingestion.loaders.markdown_loader import load_markdown
# from backend.app.ingestion.models.content import TextContent

# def test_load_markdown():
#     documents = load_markdown(
#         file_path="backend/tests/data/sample.md",
#         document_id="md-1",
#         file_name="sample.md",
#     )

#     assert len(documents) == 1
#     assert isinstance(documents[0].content, TextContent)
#     assert documents[0].document_id == "md-1"
#     assert documents[0].source_type == "markdown"
#     assert documents[0].metadata["file_name"] == "sample.md"
#     assert ("# Knowledge Intelligence Assistant" in documents[0].content.text)
#     assert ("## Supported Sources" in documents[0].content.text)

from pathlib import Path

import pytest

from backend.app.ingestion.loaders.markdown_loader import (
    load_markdown,
)
from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)


def test_load_markdown_extracts_elements(
    tmp_path,
):
    markdown_path = (
        tmp_path / "sample.md"
    )

    image_path = (
        tmp_path / "architecture.png"
    )

    image_path.write_bytes(
        b"fake-image"
    )

    markdown_path.write_text(
        """# Knowledge Assistant

This is a paragraph.

![Architecture](architecture.png)

[OpenAI](https://openai.com)
""",
        encoding="utf-8",
    )

    result = load_markdown(
        file_path=str(markdown_path),
        document_id="md-123",
    )

    element_types = {
        element.element_type
        for element in result.elements
    }

    assert "document" in element_types
    assert "heading" in element_types
    assert "paragraph" in element_types
    assert "image" in element_types
    assert "link" in element_types


def test_load_markdown_extracts_image(
    tmp_path,
):
    markdown_path = (
        tmp_path / "sample.md"
    )

    image_path = (
        tmp_path / "architecture.png"
    )

    image_path.write_bytes(
        b"fake-image"
    )

    markdown_path.write_text(
        "![Architecture](architecture.png)",
        encoding="utf-8",
    )

    result = load_markdown(
        file_path=str(markdown_path),
        document_id="md-123",
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

    assert image.metadata[
        "alt_text"
    ] == "Architecture"

    assert image.metadata[
        "exists"
    ] is True


def test_load_markdown_extracts_links(
    tmp_path,
):
    markdown_path = (
        tmp_path / "sample.md"
    )

    markdown_path.write_text(
        "Read [OpenAI](https://openai.com)",
        encoding="utf-8",
    )

    result = load_markdown(
        file_path=str(markdown_path),
        document_id="md-123",
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


def test_load_markdown_creates_relationships(
    tmp_path,
):
    markdown_path = (
        tmp_path / "sample.md"
    )

    markdown_path.write_text(
        """# Title

Some content.

[OpenAI](https://openai.com)
""",
        encoding="utf-8",
    )

    result = load_markdown(
        file_path=str(markdown_path),
        document_id="md-123",
    )

    relationships = result.relationships

    assert relationships

    assert any(
        relationship.source_element_id
        == "document"
        and relationship.relationship_type
        == "contains"
        for relationship in relationships
    )


def test_load_markdown_supports_remote_images(
    tmp_path,
):
    markdown_path = (
        tmp_path / "sample.md"
    )

    markdown_path.write_text(
        "![Logo](https://example.com/logo.png)",
        encoding="utf-8",
    )

    result = load_markdown(
        file_path=str(markdown_path),
        document_id="md-123",
    )

    images = [
        element
        for element in result.elements
        if element.element_type == "image"
    ]

    assert len(images) == 1

    assert images[0].metadata[
        "remote"
    ] is True


def test_load_markdown_missing_file(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        load_markdown(
            file_path=str(
                tmp_path / "missing.md"
            ),
            document_id="md-123",
        )


def test_load_markdown_rejects_wrong_extension(
    tmp_path,
):
    file_path = (
        tmp_path / "sample.txt"
    )

    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        load_markdown(
            file_path=str(file_path),
            document_id="md-123",
        )