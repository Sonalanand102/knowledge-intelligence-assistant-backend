from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf

from backend.app.ingestion.models.content import (
    ImageContent,
    TableContent,
    TextContent,
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


FIGURE_REFERENCE_PATTERN = re.compile(
    r"\b(?:figure|fig\.)\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

TABLE_REFERENCE_PATTERN = re.compile(
    r"\btable\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


def _bbox_to_dict(bbox: Any) -> dict[str, float]:
    return {
        "x0": float(bbox[0]),
        "y0": float(bbox[1]),
        "x1": float(bbox[2]),
        "y1": float(bbox[3]),
    }


def _bbox_area(
    bbox: dict[str, float],
) -> float:
    return max(
        0.0,
        bbox["x1"] - bbox["x0"],
    ) * max(
        0.0,
        bbox["y1"] - bbox["y0"],
    )


def _intersection_area(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
) -> float:
    x0 = max(
        bbox_a["x0"],
        bbox_b["x0"],
    )
    y0 = max(
        bbox_a["y0"],
        bbox_b["y0"],
    )
    x1 = min(
        bbox_a["x1"],
        bbox_b["x1"],
    )
    y1 = min(
        bbox_a["y1"],
        bbox_b["y1"],
    )

    if x1 <= x0 or y1 <= y0:
        return 0.0

    return (x1 - x0) * (y1 - y0)


def _overlap_ratio(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
) -> float:
    area = _bbox_area(bbox_a)

    if area == 0:
        return 0.0

    return _intersection_area(
        bbox_a,
        bbox_b,
    ) / area


def _is_inside_table(
    bbox: dict[str, float],
    table_bboxes: list[dict[str, float]],
) -> bool:
    return any(
        _overlap_ratio(
            bbox,
            table_bbox,
        )
        >= 0.5
        for table_bbox in table_bboxes
    )


def _extract_table_elements(
    page,
    document_id: str,
    page_number: int,
    file_name: str,
) -> tuple[
    list[SourceElement],
    list[dict[str, Any]],
]:
    elements: list[SourceElement] = []
    table_info: list[dict[str, Any]] = []

    try:
        table_finder = page.find_tables()
    except Exception:
        return elements, table_info

    for table_index, table in enumerate(
        table_finder.tables,
        start=1,
    ):
        rows = table.extract()

        normalized_rows = []

        for row in rows:
            normalized_rows.append(
                [
                    str(cell).strip()
                    if cell is not None
                    else ""
                    for cell in row
                ]
            )

        table_text = "\n".join(
            " | ".join(row)
            for row in normalized_rows
        )

        element_id = (
            f"page-{page_number}-"
            f"table-{table_index}"
        )

        bbox = _bbox_to_dict(
            table.bbox
        )

        elements.append(
            SourceElement(
                element_id=element_id,
                document_id=document_id,
                element_type="table",
                content=TableContent(
                    text=table_text,
                ),
                metadata={
                    "file_name": file_name,
                    "page_number": page_number,
                    "element_index": table_index,
                    "bbox": bbox,
                    "content_type": "table",
                    "row_count": len(
                        normalized_rows
                    ),
                    "column_count": max(
                        (
                            len(row)
                            for row in normalized_rows
                        ),
                        default=0,
                    ),
                },
            )
        )

        table_info.append(
            {
                "element_id": element_id,
                "bbox": bbox,
                "table_number": table_index,
            }
        )

    return elements, table_info


def _extract_text_and_images(
    page,
    document_id: str,
    page_number: int,
    file_name: str,
    output_dir: Path,
    table_bboxes: list[dict[str, float]],
) -> tuple[
    list[SourceElement],
    list[dict[str, Any]],
]:
    elements: list[SourceElement] = []
    image_info: list[dict[str, Any]] = []

    page_dict = page.get_text(
        "dict",
        sort=True,
    )

    text_index = 0
    image_index = 0

    for block in page_dict.get(
        "blocks",
        [],
    ):
        block_type = block.get("type")
        bbox = block.get("bbox")

        if not bbox:
            continue

        bbox_dict = _bbox_to_dict(
            bbox
        )

        # -----------------------------------------------------
        # Text
        # -----------------------------------------------------
        if block_type == 0:
            lines = []

            for line in block.get(
                "lines",
                [],
            ):
                spans = []

                for span in line.get(
                    "spans",
                    [],
                ):
                    text = span.get(
                        "text",
                        "",
                    )

                    if text.strip():
                        spans.append(text)

                line_text = "".join(
                    spans
                ).strip()

                if line_text:
                    lines.append(
                        line_text
                    )

            text = "\n".join(
                lines
            ).strip()

            if not text:
                continue

            # Table content is already represented
            # by TableContent.
            if _is_inside_table(
                bbox_dict,
                table_bboxes,
            ):
                continue

            text_index += 1

            element_id = (
                f"page-{page_number}-"
                f"text-{text_index}"
            )

            elements.append(
                SourceElement(
                    element_id=element_id,
                    document_id=document_id,
                    element_type="paragraph",
                    content=TextContent(
                        text=text,
                    ),
                    metadata={
                        "file_name": file_name,
                        "page_number": page_number,
                        "element_index": text_index,
                        "bbox": bbox_dict,
                        "content_type": "text",
                    },
                )
            )

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------
        elif block_type == 1:
            image_bytes = block.get(
                "image"
            )

            if not image_bytes:
                continue

            image_index += 1

            extension = (
                block.get("ext")
                or "png"
            ).lower()

            image_path = (
                output_dir
                / (
                    f"{document_id}_"
                    f"page_{page_number}_"
                    f"image_{image_index}."
                    f"{extension}"
                )
            )

            image_path.write_bytes(
                image_bytes
            )

            element_id = (
                f"page-{page_number}-"
                f"image-{image_index}"
            )

            elements.append(
                SourceElement(
                    element_id=element_id,
                    document_id=document_id,
                    element_type="image",
                    content=ImageContent(
                        path=str(
                            image_path
                        )
                    ),
                    metadata={
                        "file_name": file_name,
                        "page_number": page_number,
                        "element_index": image_index,
                        "bbox": bbox_dict,
                        "content_type": "image",
                        "image_path": str(
                            image_path
                        ),
                        "image_extension": extension,
                        "width": block.get(
                            "width"
                        ),
                        "height": block.get(
                            "height"
                        ),
                    },
                )
            )

            image_info.append(
                {
                    "element_id": element_id,
                    "bbox": bbox_dict,
                    "image_number": image_index,
                }
            )

    return elements, image_info


def _extract_link_elements(
    page,
    document_id: str,
    page_number: int,
    file_name: str,
) -> list[SourceElement]:
    elements: list[SourceElement] = []

    for link_index, link in enumerate(
        page.get_links(),
        start=1,
    ):
        uri = link.get(
            "uri"
        )

        if not uri:
            continue

        bbox = link.get(
            "from"
        )

        if bbox is None:
            continue

        bbox_dict = _bbox_to_dict(
            bbox
        )

        visible_text_parts = []

        for word in page.get_text(
            "words",
            sort=True,
        ):
            word_bbox = {
                "x0": word[0],
                "y0": word[1],
                "x1": word[2],
                "y1": word[3],
            }

            if _intersection_area(
                bbox_dict,
                word_bbox,
            ) > 0:
                visible_text_parts.append(
                    word[4]
                )

        visible_text = " ".join(
            visible_text_parts
        ).strip()

        element_id = (
            f"page-{page_number}-"
            f"link-{link_index}"
        )

        elements.append(
            SourceElement(
                element_id=element_id,
                document_id=document_id,
                element_type="link",
                content=TextContent(
                    text=visible_text,
                ),
                metadata={
                    "file_name": file_name,
                    "page_number": page_number,
                    "element_index": link_index,
                    "bbox": bbox_dict,
                    "content_type": "link",
                    "url": uri,
                },
            )
        )

    return elements


def _create_page_element(
    document_id: str,
    page_number: int,
    file_name: str,
) -> SourceElement:
    return SourceElement(
        element_id=f"page-{page_number}",
        document_id=document_id,
        element_type="page",
        content=TextContent(
            text=""
        ),
        metadata={
            "file_name": file_name,
            "page_number": page_number,
            "content_type": "page",
        },
    )


def _is_near(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
    max_vertical_distance: float = 60.0,
    max_horizontal_distance: float = 80.0,
) -> bool:
    horizontal_distance = max(
        0.0,
        bbox_a["x0"] - bbox_b["x1"],
        bbox_b["x0"] - bbox_a["x1"],
    )

    vertical_distance = max(
        0.0,
        bbox_a["y0"] - bbox_b["y1"],
        bbox_b["y0"] - bbox_a["y1"],
    )

    return (
        horizontal_distance
        <= max_horizontal_distance
        and vertical_distance
        <= max_vertical_distance
    )


def _extract_reference_number(
    text: str,
    pattern: re.Pattern[str],
) -> str | None:
    match = pattern.search(
        text
    )

    if not match:
        return None

    return match.group(1)


def _build_relationships(
    page_element_id: str,
    page_elements: list[SourceElement],
) -> list[ElementRelationship]:
    relationships: list[ElementRelationship] = []

    # ---------------------------------------------------------
    # Structural relationships
    # ---------------------------------------------------------
    for element in page_elements:
        relationships.append(
            ElementRelationship(
                source_element_id=page_element_id,
                relationship_type="contains",
                target_element_id=element.element_id,
                metadata={
                    "page_number": element.metadata.get(
                        "page_number"
                    )
                },
            )
        )

    text_elements = [
        element
        for element in page_elements
        if element.element_type == "paragraph"
    ]

    image_elements = [
        element
        for element in page_elements
        if element.element_type == "image"
    ]

    table_elements = [
        element
        for element in page_elements
        if element.element_type == "table"
    ]

    # ---------------------------------------------------------
    # Figure captions
    # ---------------------------------------------------------
    image_captions: dict[
        str,
        SourceElement,
    ] = {}

    for image_element in image_elements:
        image_bbox = image_element.metadata[
            "bbox"
        ]

        candidates = []

        for text_element in text_elements:
            text_bbox = text_element.metadata[
                "bbox"
            ]

            if not _is_near(
                image_bbox,
                text_bbox,
            ):
                continue

            figure_number = (
                _extract_reference_number(
                    text_element.content.text,
                    FIGURE_REFERENCE_PATTERN,
                )
            )

            if figure_number is not None:
                candidates.append(
                    (
                        text_element,
                        figure_number,
                    )
                )

        if not candidates:
            continue

        caption, figure_number = min(
            candidates,
            key=lambda item: abs(
                item[0].metadata["bbox"]["y0"]
                - image_bbox["y1"]
            ),
        )

        image_captions[
            figure_number
        ] = caption

        image_element.metadata[
            "figure_number"
        ] = figure_number

        relationships.append(
            ElementRelationship(
                source_element_id=caption.element_id,
                relationship_type="caption_of",
                target_element_id=image_element.element_id,
                metadata={
                    "figure_number": figure_number,
                },
            )
        )

    # ---------------------------------------------------------
    # Table captions
    # ---------------------------------------------------------
    table_captions: dict[
        str,
        SourceElement,
    ] = {}

    for table_element in table_elements:
        table_bbox = table_element.metadata[
            "bbox"
        ]

        candidates = []

        for text_element in text_elements:
            text_bbox = text_element.metadata[
                "bbox"
            ]

            if not _is_near(
                table_bbox,
                text_bbox,
            ):
                continue

            table_number = (
                _extract_reference_number(
                    text_element.content.text,
                    TABLE_REFERENCE_PATTERN,
                )
            )

            if table_number is not None:
                candidates.append(
                    (
                        text_element,
                        table_number,
                    )
                )

        if not candidates:
            continue

        caption, table_number = min(
            candidates,
            key=lambda item: abs(
                item[0].metadata["bbox"]["y0"]
                - table_bbox["y1"]
            ),
        )

        table_captions[
            table_number
        ] = caption

        table_element.metadata[
            "table_number"
        ] = table_number

        relationships.append(
            ElementRelationship(
                source_element_id=caption.element_id,
                relationship_type="caption_of",
                target_element_id=table_element.element_id,
                metadata={
                    "table_number": table_number,
                },
            )
        )

    # ---------------------------------------------------------
    # Explicit figure/table references
    # ---------------------------------------------------------
    for text_element in text_elements:
        text = text_element.content.text

        figure_number = (
            _extract_reference_number(
                text,
                FIGURE_REFERENCE_PATTERN,
            )
        )

        if figure_number:
            caption = image_captions.get(
                figure_number
            )

            if caption:
                target_image = next(
                    (
                        image
                        for image in image_elements
                        if image.metadata.get(
                            "figure_number"
                        )
                        == figure_number
                    ),
                    None,
                )

                if target_image:
                    relationships.append(
                        ElementRelationship(
                            source_element_id=(
                                text_element.element_id
                            ),
                            relationship_type="refers_to",
                            target_element_id=(
                                target_image.element_id
                            ),
                            metadata={
                                "reference": (
                                    f"Figure "
                                    f"{figure_number}"
                                )
                            },
                        )
                    )

        table_number = (
            _extract_reference_number(
                text,
                TABLE_REFERENCE_PATTERN,
            )
        )

        if table_number:
            caption = table_captions.get(
                table_number
            )

            if caption:
                target_table = next(
                    (
                        table
                        for table in table_elements
                        if table.metadata.get(
                            "table_number"
                        )
                        == table_number
                    ),
                    None,
                )

                if target_table:
                    relationships.append(
                        ElementRelationship(
                            source_element_id=(
                                text_element.element_id
                            ),
                            relationship_type="refers_to",
                            target_element_id=(
                                target_table.element_id
                            ),
                            metadata={
                                "reference": (
                                    f"Table "
                                    f"{table_number}"
                                )
                            },
                        )
                    )

    return relationships


def load_pdf(
    file_path: str,
    document_id: str,
    output_dir: str,
) -> IngestionResult:
    pdf_path = Path(
        file_path
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"PDF path is not a file: {file_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported PDF format: "
            f"{pdf_path.suffix}"
        )

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    elements: list[SourceElement] = []
    relationships: list[
        ElementRelationship
    ] = []

    with pymupdf.open(
        str(pdf_path)
    ) as pdf:
        for page_index, page in enumerate(
            pdf,
            start=1,
        ):
            page_element = _create_page_element(
                document_id=document_id,
                page_number=page_index,
                file_name=pdf_path.name,
            )

            elements.append(
                page_element
            )

            (
                table_elements,
                table_info,
            ) = _extract_table_elements(
                page=page,
                document_id=document_id,
                page_number=page_index,
                file_name=pdf_path.name,
            )

            table_bboxes = [
                info["bbox"]
                for info in table_info
            ]

            (
                content_elements,
                _,
            ) = _extract_text_and_images(
                page=page,
                document_id=document_id,
                page_number=page_index,
                file_name=pdf_path.name,
                output_dir=output_path,
                table_bboxes=table_bboxes,
            )

            link_elements = _extract_link_elements(
                page=page,
                document_id=document_id,
                page_number=page_index,
                file_name=pdf_path.name,
            )

            page_elements = (
                table_elements
                + content_elements
                + link_elements
            )

            elements.extend(
                page_elements
            )

            relationships.extend(
                _build_relationships(
                    page_element_id=(
                        page_element.element_id
                    ),
                    page_elements=page_elements,
                )
            )

    return IngestionResult(
        document_id=document_id,
        elements=elements,
        relationships=relationships,
    )