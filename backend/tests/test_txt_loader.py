# from backend.app.ingestion.loaders.txt_loader import load_txt
# from backend.app.ingestion.models.content import TextContent

# def test_load_txt():
#     documents = load_txt(
#         file_path="backend/tests/data/sample.txt",
#         document_id="txt-1",
#         file_name="sample.txt",
#     )

#     assert len(documents) == 1
#     assert isinstance(documents[0].content, TextContent)
#     assert documents[0].document_id == "txt-1"
#     assert documents[0].source_type == "txt"
#     assert documents[0].metadata["file_name"] == "sample.txt"
#     assert (
#         "Knowledge Intelligence Assistant"
#         in documents[0].content.text
#     )

from pathlib import Path

import pytest

from backend.app.ingestion.loaders.txt_loader import (
    load_txt,
)
from backend.app.ingestion.models.content import (
    TextContent,
)


def test_load_txt(tmp_path):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "This is a sample text document.",
        encoding="utf-8",
    )

    result = load_txt(
        file_path=str(file_path),
        document_id="txt-123",
    )

    assert result.document_id == "txt-123"

    assert len(result.elements) == 2

    root = result.elements[0]
    text_element = result.elements[1]

    assert root.element_id == "document"
    assert root.element_type == "document"

    assert text_element.element_id == "text-1"
    assert text_element.element_type == "text"

    assert isinstance(
        text_element.content,
        TextContent,
    )

    assert (
        text_element.content.text
        == "This is a sample text document."
    )

    assert (
        text_element.metadata["file_name"]
        == "sample.txt"
    )

    assert (
        text_element.metadata["content_type"]
        == "text"
    )


def test_load_txt_creates_relationship(
    tmp_path,
):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Hello world.",
        encoding="utf-8",
    )

    result = load_txt(
        file_path=str(file_path),
        document_id="txt-123",
    )

    relationships = result.relationships

    assert len(relationships) == 1

    relationship = relationships[0]

    assert (
        relationship.source_element_id
        == "document"
    )

    assert (
        relationship.relationship_type
        == "contains"
    )

    assert (
        relationship.target_element_id
        == "text-1"
    )


def test_load_txt_empty_file(tmp_path):
    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    result = load_txt(
        file_path=str(file_path),
        document_id="txt-123",
    )

    assert result.document_id == "txt-123"

    assert len(result.elements) == 2

    text_element = result.elements[1]

    assert text_element.element_type == "text"
    assert text_element.content.text == ""


def test_load_txt_missing_file(tmp_path):
    file_path = (
        tmp_path / "missing.txt"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_txt(
            file_path=str(file_path),
            document_id="txt-123",
        )


def test_load_txt_rejects_directory(
    tmp_path,
):
    directory = tmp_path / "documents"
    directory.mkdir()

    with pytest.raises(
        ValueError
    ):
        load_txt(
            file_path=str(directory),
            document_id="txt-123",
        )


def test_load_txt_rejects_wrong_extension(
    tmp_path,
):
    file_path = tmp_path / "sample.pdf"

    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        load_txt(
            file_path=str(file_path),
            document_id="txt-123",
        )