# from backend.app.ingestion.loaders.html_loader import load_html
# from backend.app.ingestion.models.content import TextContent


# def test_load_html():
#     documents = load_html(
#         "backend/tests/data/sample.html",
#         document_id="html-1",
#         file_name="sample.html",
#     )

#     assert len(documents) == 1

#     document = documents[0]

#     assert isinstance(document.content, TextContent)
#     assert document.document_id == "html-1"
#     assert document.source_type == "html"
#     assert document.metadata["file_name"] == "sample.html"
#     assert document.metadata["content_type"] == "webpage"

#     assert ("Knowledge Intelligence Assistant" in document.content.text)
#     assert ("Supported Sources" in document.content.text)
#     assert ("PDF documents" in document.content.text)
#     assert ("Excel files" in document.content.text)
#     assert ("Retrieval" in document.content.text)

#     # Script/style content should not be included
#     assert ("This should not be extracted" not in document.content.text)
#     assert ("font-family" not in document.content.text)

from pathlib import Path

import pytest

from backend.app.ingestion.loaders.html_loader import (
    load_html,
)
from backend.app.ingestion.models.content import (
    ImageContent,
    TableContent,
    TextContent,
)


def test_load_html_extracts_elements(
    tmp_path,
):
    html_path = (
        tmp_path / "sample.html"
    )

    image_path = (
        tmp_path / "architecture.png"
    )

    image_path.write_bytes(
        b"fake-image"
    )

    html_path.write_text(
        """
        <html>
            <body>
                <article>
                    <h1>Knowledge Assistant</h1>

                    <p>
                        This is a paragraph with
                        <a href="https://openai.com">
                            OpenAI
                        </a>.
                    </p>

                    <img
                        src="architecture.png"
                        alt="Architecture"
                    />

                    <table>
                        <tr>
                            <th>Component</th>
                            <th>Technology</th>
                        </tr>
                        <tr>
                            <td>Backend</td>
                            <td>FastAPI</td>
                        </tr>
                    </table>
                </article>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = load_html(
        file_path=str(
            html_path
        ),
        document_id="html-123",
    )

    element_types = {
        element.element_type
        for element in result.elements
    }

    assert "document" in element_types
    assert "heading" in element_types
    assert "paragraph" in element_types
    assert "image" in element_types
    assert "table" in element_types
    assert "link" in element_types


def test_load_html_extracts_image(
    tmp_path,
):
    html_path = (
        tmp_path / "sample.html"
    )

    image_path = (
        tmp_path / "architecture.png"
    )

    image_path.write_bytes(
        b"fake-image"
    )

    html_path.write_text(
        """
        <img
            src="architecture.png"
            alt="Architecture"
        />
        """,
        encoding="utf-8",
    )

    result = load_html(
        file_path=str(
            html_path
        ),
        document_id="html-123",
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

    assert (
        image.metadata["alt_text"]
        == "Architecture"
    )

    assert (
        image.metadata["remote"]
        is False
    )


def test_load_html_extracts_table(
    tmp_path,
):
    html_path = (
        tmp_path / "sample.html"
    )

    html_path.write_text(
        """
        <table>
            <tr>
                <th>Component</th>
                <th>Technology</th>
            </tr>
            <tr>
                <td>Backend</td>
                <td>FastAPI</td>
            </tr>
        </table>
        """,
        encoding="utf-8",
    )

    result = load_html(
        file_path=str(
            html_path
        ),
        document_id="html-123",
    )

    tables = [
        element
        for element in result.elements
        if element.element_type == "table"
    ]

    assert len(tables) == 1

    table = tables[0]

    assert isinstance(
        table.content,
        TableContent,
    )

    assert "Component" in table.content.text
    assert "FastAPI" in table.content.text


def test_load_html_preserves_links(
    tmp_path,
):
    html_path = (
        tmp_path / "sample.html"
    )

    html_path.write_text(
        """
        <p>
            Read
            <a href="https://openai.com">
                OpenAI
            </a>
        </p>
        """,
        encoding="utf-8",
    )

    result = load_html(
        file_path=str(
            html_path
        ),
        document_id="html-123",
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


def test_html_preserves_dom_relationships(
    tmp_path,
):
    html_path = (
        tmp_path / "sample.html"
    )

    html_path.write_text(
        """
        <article>
            <h1>Title</h1>
            <p>Hello world.</p>
        </article>
        """,
        encoding="utf-8",
    )

    result = load_html(
        file_path=str(
            html_path
        ),
        document_id="html-123",
    )

    heading = next(
        element
        for element in result.elements
        if element.element_type == "heading"
    )

    paragraph = next(
        element
        for element in result.elements
        if element.element_type == "paragraph"
    )

    article_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == "contains"
    ]

    assert article_relationships

    assert any(
        relationship.target_element_id
        == heading.element_id
        for relationship in article_relationships
    )

    assert any(
        relationship.target_element_id
        == paragraph.element_id
        for relationship in article_relationships
    )


def test_load_html_removes_script_and_style(
    tmp_path,
):
    html_path = (
        tmp_path / "sample.html"
    )

    html_path.write_text(
        """
        <html>
            <head>
                <style>
                    body { color: red; }
                </style>
            </head>
            <body>
                <script>
                    console.log("secret");
                </script>

                <p>Visible content</p>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = load_html(
        file_path=str(
            html_path
        ),
        document_id="html-123",
    )

    text_elements = [
        element
        for element in result.elements
        if element.element_type
        == "paragraph"
    ]

    assert any(
        element.content.text
        == "Visible content"
        for element in text_elements
    )

    assert not any(
        "console.log"
        in element.content.text
        for element in text_elements
    )


def test_load_html_missing_file(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        load_html(
            file_path=str(
                tmp_path / "missing.html"
            ),
            document_id="html-123",
        )


def test_load_html_rejects_wrong_extension(
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
        load_html(
            file_path=str(
                file_path
            ),
            document_id="html-123",
        )