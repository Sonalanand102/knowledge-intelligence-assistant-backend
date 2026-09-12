# from backend.app.ingestion.loaders.csv_loader import load_csv
# from backend.app.ingestion.models.content import TableContent


# def test_load_csv():
#     documents = load_csv(
#         "backend/tests/data/sample.csv",
#         document_id="csv-1",
#         file_name="sample.csv",
#     )

#     assert len(documents) == 1

#     document = documents[0]

#     assert isinstance(document.content, TableContent)
#     assert document.document_id == "csv-1"
#     assert document.source_type == "csv"
#     assert document.metadata["file_name"] == "sample.csv"
#     assert document.metadata["content_type"] == "table"
#     assert document.metadata["row_count"] == 5

#     assert "Component: Backend" in document.content.text
#     assert "Status: Done" in document.content.text
#     assert "Owner: Sonal" in document.content.text

from pathlib import Path

import pytest

from backend.app.ingestion.loaders.csv_loader import (
    load_csv,
)
from backend.app.ingestion.models.content import (
    TableContent,
)


def create_test_csv(
    csv_path: Path,
) -> None:
    csv_path.write_text(
        """Name,Role,Experience
Sonal,Developer,2
Rahul,Engineer,3
""",
        encoding="utf-8",
    )


def test_load_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"

    create_test_csv(csv_path)

    result = load_csv(
        file_path=str(csv_path),
        document_id="csv-123",
    )

    assert result.document_id == "csv-123"

    assert len(result.elements) == 2

    root = result.elements[0]
    table = result.elements[1]

    assert root.element_id == "document"
    assert root.element_type == "document"

    assert table.element_id == "table-1"
    assert table.element_type == "table"

    assert isinstance(
        table.content,
        TableContent,
    )

    assert "Name | Role | Experience" in (
        table.content.text
    )

    assert "Sonal | Developer | 2" in (
        table.content.text
    )

    assert (
        table.metadata["file_name"]
        == "sample.csv"
    )

    assert (
        table.metadata["content_type"]
        == "table"
    )

    assert (
        table.metadata["row_count"]
        == 3
    )

    assert (
        table.metadata["column_count"]
        == 3
    )


def test_load_csv_creates_relationship(
    tmp_path,
):
    csv_path = tmp_path / "sample.csv"

    create_test_csv(csv_path)

    result = load_csv(
        file_path=str(csv_path),
        document_id="csv-123",
    )

    assert len(result.relationships) == 1

    relationship = result.relationships[0]

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
        == "table-1"
    )


def test_load_csv_empty_file(
    tmp_path,
):
    csv_path = tmp_path / "empty.csv"

    csv_path.write_text(
        "",
        encoding="utf-8",
    )

    result = load_csv(
        file_path=str(csv_path),
        document_id="csv-123",
    )

    assert result.document_id == "csv-123"

    assert len(result.elements) == 2

    table = result.elements[1]

    assert table.element_type == "table"
    assert table.content.text == ""

    assert (
        table.metadata["row_count"]
        == 0
    )


def test_load_csv_missing_file(
    tmp_path,
):
    csv_path = (
        tmp_path / "missing.csv"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_csv(
            file_path=str(csv_path),
            document_id="csv-123",
        )


def test_load_csv_rejects_directory(
    tmp_path,
):
    directory = tmp_path / "data"
    directory.mkdir()

    with pytest.raises(
        ValueError
    ):
        load_csv(
            file_path=str(directory),
            document_id="csv-123",
        )


def test_load_csv_rejects_wrong_extension(
    tmp_path,
):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "a,b\n1,2",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        load_csv(
            file_path=str(file_path),
            document_id="csv-123",
        )