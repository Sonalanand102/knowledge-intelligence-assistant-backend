# from backend.app.ingestion.loaders.excel_loader import load_excel
# from backend.app.ingestion.models.content import TableContent


# def test_load_excel():
#     documents = load_excel(
#         "backend/tests/data/sample.xlsx",
#         document_id="excel-1",
#         file_name="sample.xlsx",
#     )

#     assert len(documents) == 2

#     projects_document = documents[0]

#     assert isinstance(projects_document.content, TableContent)
#     assert projects_document.document_id == "excel-1"
#     assert projects_document.source_type == "excel"
#     assert projects_document.metadata["file_name"] == "sample.xlsx"
#     assert projects_document.metadata["content_type"] == "table"
#     assert projects_document.metadata["sheet_name"] == "Projects"
#     assert projects_document.metadata["row_count"] == 4
#     assert projects_document.metadata["column_count"] == 4

#     assert ("Component: Backend" in projects_document.content.text)
#     assert ("Owner: Sonal" in projects_document.content.text)

#     team_document = documents[1]

#     assert isinstance(team_document.content, TableContent)
#     assert team_document.metadata["sheet_name"] == "Team"
#     assert team_document.metadata["row_count"] == 3
#     assert team_document.metadata["column_count"] == 3

#     assert ("Name: Sonal" in team_document.content.text)
#     assert ("Role: Full Stack Developer" in team_document.content.text)

from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image as XLImage
from PIL import Image

from backend.app.ingestion.loaders.excel_loader import (
    load_excel,
)
from backend.app.ingestion.models.content import (
    ImageContent,
    TableContent,
)


def create_test_excel(
    excel_path: Path,
) -> None:
    workbook = Workbook()

    worksheet = workbook.active
    worksheet.title = "Sales"

    rows = [
        ["Product", "Revenue"],
        ["Laptop", 100000],
        ["Phone", 80000],
        ["Tablet", 50000],
    ]

    for row in rows:
        worksheet.append(row)

    # ---------------------------------------------------------
    # Chart
    # ---------------------------------------------------------
    chart = BarChart()

    chart.title = "Revenue by Product"

    data = Reference(
        worksheet,
        min_col=2,
        min_row=1,
        max_row=4,
    )

    categories = Reference(
        worksheet,
        min_col=1,
        min_row=2,
        max_row=4,
    )

    chart.add_data(
        data,
        titles_from_data=True,
    )

    chart.set_categories(
        categories
    )

    worksheet.add_chart(
        chart,
        "D2",
    )

    # ---------------------------------------------------------
    # Embedded image
    # ---------------------------------------------------------
    image_path = (
        excel_path.parent
        / "test-image.png"
    )

    image = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    image.save(
        image_path,
        format="PNG",
    )

    worksheet_image = XLImage(
        str(image_path)
    )

    worksheet_image.width = 50
    worksheet_image.height = 50

    worksheet.add_image(
        worksheet_image,
        "D20",
    )

    workbook.save(
        excel_path
    )


def test_load_excel_extracts_elements(
    tmp_path,
):
    excel_path = (
        tmp_path / "sample.xlsx"
    )

    output_dir = (
        tmp_path / "output"
    )

    create_test_excel(
        excel_path
    )

    result = load_excel(
        file_path=str(
            excel_path
        ),
        document_id="excel-123",
        output_dir=str(
            output_dir
        ),
    )

    element_types = {
        element.element_type
        for element in result.elements
    }

    assert "workbook" in element_types
    assert "sheet" in element_types
    assert "table" in element_types
    assert "image" in element_types
    assert "chart" in element_types


def test_load_excel_extracts_table(
    tmp_path,
):
    excel_path = (
        tmp_path / "sample.xlsx"
    )

    output_dir = (
        tmp_path / "output"
    )

    create_test_excel(
        excel_path
    )

    result = load_excel(
        file_path=str(
            excel_path
        ),
        document_id="excel-123",
        output_dir=str(
            output_dir
        ),
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

    assert "Product" in table.content.text
    assert "Laptop" in table.content.text
    assert "100000" in table.content.text


def test_load_excel_extracts_image(
    tmp_path,
):
    excel_path = (
        tmp_path / "sample.xlsx"
    )

    output_dir = (
        tmp_path / "output"
    )

    create_test_excel(
        excel_path
    )

    result = load_excel(
        file_path=str(
            excel_path
        ),
        document_id="excel-123",
        output_dir=str(
            output_dir
        ),
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

    assert image.metadata[
        "sheet_name"
    ] == "Sales"


def test_load_excel_extracts_chart(
    tmp_path,
):
    excel_path = (
        tmp_path / "sample.xlsx"
    )

    output_dir = (
        tmp_path / "output"
    )

    create_test_excel(
        excel_path
    )

    result = load_excel(
        file_path=str(
            excel_path
        ),
        document_id="excel-123",
        output_dir=str(
            output_dir
        ),
    )

    charts = [
        element
        for element in result.elements
        if element.element_type == "chart"
    ]

    assert len(charts) == 1

    chart = charts[0]

    assert (
        chart.metadata["chart_title"]
        == "Revenue by Product"
    )

    assert chart.metadata[
        "source_references"
    ]


def test_load_excel_creates_relationships(
    tmp_path,
):
    excel_path = (
        tmp_path / "sample.xlsx"
    )

    output_dir = (
        tmp_path / "output"
    )

    create_test_excel(
        excel_path
    )

    result = load_excel(
        file_path=str(
            excel_path
        ),
        document_id="excel-123",
        output_dir=str(
            output_dir
        ),
    )

    contains_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == "contains"
    ]

    assert contains_relationships

    derived_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == "derived_from"
    ]

    assert derived_relationships

    chart_relationship = (
        derived_relationships[0]
    )

    assert chart_relationship.metadata[
        "source_reference"
    ]


def test_load_excel_missing_file(
    tmp_path,
):
    missing_path = (
        tmp_path / "missing.xlsx"
    )

    try:
        load_excel(
            file_path=str(
                missing_path
            ),
            document_id="excel-123",
            output_dir=str(
                tmp_path / "output"
            ),
        )
    except FileNotFoundError:
        return

    raise AssertionError(
        "Expected FileNotFoundError"
    )


def test_load_excel_rejects_unsupported_format(
    tmp_path,
):
    file_path = (
        tmp_path / "sample.csv"
    )

    file_path.write_text(
        "a,b\n1,2"
    )

    try:
        load_excel(
            file_path=str(
                file_path
            ),
            document_id="excel-123",
            output_dir=str(
                tmp_path / "output"
            ),
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )