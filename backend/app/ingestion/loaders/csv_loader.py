# import csv

# from backend.app.ingestion.models.source_document import SourceDocument
# from backend.app.ingestion.models.content import TableContent

# def load_csv(
#     file_path: str,
#     document_id: str,
#     file_name: str,
# ) -> list[SourceDocument]:
#     documents = []

#     with open(file_path, "r", encoding="utf-8", newline="") as file:
#         reader = csv.DictReader(file)

#         rows = list(reader)

#     if not rows:
#         return []

#     content_parts = []

#     for row in rows:
#         row_content = " | ".join(
#             f"{key}: {value}"
#             for key, value in row.items()
#             if value is not None
#         )

#         if row_content.strip():
#             content_parts.append(row_content)

#     if not content_parts:
#         return []

#     content = "\n".join(content_parts)

#     documents.append(
#         SourceDocument(
#             content=TableContent(text=content),
#             document_id=document_id,
#             source_type="csv",
#             metadata={
#                 "file_name": file_name,
#                 "content_type": "table",
#                 "row_count": len(rows),
#             },
#         )
#     )

#     return documents

from __future__ import annotations

import csv
from pathlib import Path

from backend.app.ingestion.models.content import (
    TableContent,
)
from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)


def load_csv(
    file_path: str,
    document_id: str,
) -> IngestionResult:
    csv_path = Path(file_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    if not csv_path.is_file():
        raise ValueError(
            f"CSV path is not a file: {file_path}"
        )

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported CSV format: "
            f"{csv_path.suffix}"
        )

    rows = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)

        for row in reader:
            rows.append(
                [
                    value.strip()
                    for value in row
                ]
            )

    table_text = "\n".join(
        " | ".join(row)
        for row in rows
    )

    table_element = SourceElement(
        element_id="table-1",
        document_id=document_id,
        element_type="table",
        content=TableContent(
            text=table_text
        ),
        metadata={
            "file_name": csv_path.name,
            "content_type": "table",
            "row_count": len(rows),
            "column_count": max(
                (
                    len(row)
                    for row in rows
                ),
                default=0,
            ),
        },
    )

    root_element = SourceElement(
        element_id="document",
        document_id=document_id,
        element_type="document",
        content=TableContent(
            text=""
        ),
        metadata={
            "file_name": csv_path.name,
            "content_type": "document",
        },
    )

    relationship = ElementRelationship(
        source_element_id="document",
        relationship_type="contains",
        target_element_id="table-1",
    )

    return IngestionResult(
        document_id=document_id,
        elements=[
            root_element,
            table_element,
        ],
        relationships=[
            relationship,
        ],
    )