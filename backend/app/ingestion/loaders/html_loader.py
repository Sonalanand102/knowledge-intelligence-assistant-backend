# from pathlib import Path

# from bs4 import BeautifulSoup

# from backend.app.ingestion.models.source_document import SourceDocument
# from backend.app.ingestion.models.content import TextContent

# def load_html(
#     file_path: str,
#     document_id: str,
#     file_name: str,
# ) -> list[SourceDocument]:

#     path = Path(file_path)

#     html_content = path.read_text(encoding="utf-8")

#     if not html_content.strip():
#         return []

#     soup = BeautifulSoup(html_content, "html.parser")

#     # Remove elements that don't contain useful knowledge

#     for element in soup(["script", "style", "noscript"]):
#         element.decompose()

#     content = soup.get_text(separator="\n", strip=True)

#     if not content.strip():
#         return []

#     return [
#         SourceDocument(
#             content=TextContent(text=content),
#             document_id=document_id,
#             source_type="html",
#             metadata={
#                 "file_name" : file_name,
#                 "content_type": "webpage",
#             },
#         )
#     ]

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

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


TEXT_TAGS = {
    "p",
    "span",
    "li",
    "blockquote",
    "pre",
    "code",
}

HEADING_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
}


def _tag_text(
    tag: Tag,
) -> str:
    return " ".join(
        tag.stripped_strings
    ).strip()


def _resolve_url(
    source: str,
    target: str,
) -> str:
    return urljoin(
        source,
        target,
    )


def _create_element(
    *,
    element_id: str,
    document_id: str,
    element_type: str,
    content,
    metadata: dict,
) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        document_id=document_id,
        element_type=element_type,
        content=content,
        metadata=metadata,
    )


def _is_http_url(
    value: str,
) -> bool:
    parsed = urlparse(value)

    return parsed.scheme in {
        "http",
        "https",
    }


def _extract_table(
    table: Tag,
) -> str:
    rows: list[str] = []

    for tr in table.find_all(
        "tr"
    ):
        cells = []

        for cell in tr.find_all(
            ["th", "td"]
        ):
            cells.append(
                _tag_text(cell)
            )

        if cells:
            rows.append(
                " | ".join(cells)
            )

    return "\n".join(rows)


def load_html(
    file_path: str,
    document_id: str,
    base_url: str | None = None,
) -> IngestionResult:
    html_path = Path(
        file_path
    )

    if not html_path.exists():
        raise FileNotFoundError(
            f"HTML file not found: {file_path}"
        )

    if not html_path.is_file():
        raise ValueError(
            f"HTML path is not a file: {file_path}"
        )

    if html_path.suffix.lower() not in {
        ".html",
        ".htm",
    }:
        raise ValueError(
            "Unsupported HTML format: "
            f"{html_path.suffix}"
        )

    html = html_path.read_text(
        encoding="utf-8"
    )

    return _load_html_content(
        html=html,
        document_id=document_id,
        file_name=html_path.name,
        base_url=base_url,
    )


def _load_html_content(
    html: str,
    document_id: str,
    file_name: str,
    base_url: str | None = None,
) -> IngestionResult:
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup(
        list(SKIP_TAGS)
    ):
        tag.decompose()

    elements: list[SourceElement] = []
    relationships: list[
        ElementRelationship
    ] = []

    document_element = _create_element(
        element_id="document",
        document_id=document_id,
        element_type="document",
        content=TextContent(
            text=""
        ),
        metadata={
            "file_name": file_name,
            "content_type": "document",
            "base_url": base_url,
        },
    )

    elements.append(
        document_element
    )

    element_counter = 0

    # Map Tag object identity to element ID.
    tag_element_ids: dict[int, str] = {}

    # ---------------------------------------------------------
    # Build elements from the DOM
    # ---------------------------------------------------------
    for tag in soup.find_all(True):
        if tag.name in SKIP_TAGS:
            continue

        if tag.name == "html":
            continue

        text = _tag_text(tag)

        # -----------------------------------------------------
        # Heading
        # -----------------------------------------------------
        if tag.name in HEADING_TAGS:
            if not text:
                continue

            element_counter += 1

            element_id = (
                f"heading-{element_counter}"
            )

            element = _create_element(
                element_id=element_id,
                document_id=document_id,
                element_type="heading",
                content=TextContent(
                    text=text
                ),
                metadata={
                    "file_name": file_name,
                    "content_type": "heading",
                    "tag": tag.name,
                    "level": int(
                        tag.name[1]
                    ),
                },
            )

            elements.append(
                element
            )

            tag_element_ids[
                id(tag)
            ] = element_id

        # -----------------------------------------------------
        # Text
        # -----------------------------------------------------
        elif tag.name in TEXT_TAGS:
            if not text:
                continue

            element_counter += 1

            element_id = (
                f"text-{element_counter}"
            )

            element = _create_element(
                element_id=element_id,
                document_id=document_id,
                element_type="paragraph",
                content=TextContent(
                    text=text
                ),
                metadata={
                    "file_name": file_name,
                    "content_type": "text",
                    "tag": tag.name,
                },
            )

            elements.append(
                element
            )

            tag_element_ids[
                id(tag)
            ] = element_id

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------
        elif tag.name == "img":
            src = (
                tag.get("src")
                or tag.get("data-src")
            )

            if not src:
                continue

            resolved_src = (
                _resolve_url(
                    base_url,
                    src,
                )
                if base_url
                else src
            )

            element_counter += 1

            element_id = (
                f"image-{element_counter}"
            )

            image_element = _create_element(
                element_id=element_id,
                document_id=document_id,
                element_type="image",
                content=ImageContent(
                    path=resolved_src
                ),
                metadata={
                    "file_name": file_name,
                    "content_type": "image",
                    "src": src,
                    "resolved_src": resolved_src,
                    "alt_text": (
                        tag.get("alt")
                        or ""
                    ).strip(),
                    "title": (
                        tag.get("title")
                        or ""
                    ).strip(),
                    "remote": _is_http_url(
                        resolved_src
                    ),
                },
            )

            elements.append(
                image_element
            )

            tag_element_ids[
                id(tag)
            ] = element_id

        # -----------------------------------------------------
        # Table
        # -----------------------------------------------------
        elif tag.name == "table":
            table_text = _extract_table(
                tag
            )

            if not table_text:
                continue

            element_counter += 1

            element_id = (
                f"table-{element_counter}"
            )

            rows = tag.find_all(
                "tr"
            )

            column_count = max(
                (
                    len(
                        row.find_all(
                            ["th", "td"]
                        )
                    )
                    for row in rows
                ),
                default=0,
            )

            table_element = _create_element(
                element_id=element_id,
                document_id=document_id,
                element_type="table",
                content=TableContent(
                    text=table_text
                ),
                metadata={
                    "file_name": file_name,
                    "content_type": "table",
                    "row_count": len(
                        rows
                    ),
                    "column_count": column_count,
                },
            )

            elements.append(
                table_element
            )

            tag_element_ids[
                id(tag)
            ] = element_id

        # -----------------------------------------------------
        # Links
        # -----------------------------------------------------
        elif tag.name == "a":
            href = tag.get(
                "href"
            )

            if not href:
                continue

            link_text = _tag_text(tag)

            resolved_href = (
                _resolve_url(
                    base_url,
                    href,
                )
                if base_url
                else href
            )

            element_counter += 1

            element_id = (
                f"link-{element_counter}"
            )

            link_element = _create_element(
                element_id=element_id,
                document_id=document_id,
                element_type="link",
                content=TextContent(
                    text=link_text
                ),
                metadata={
                    "file_name": file_name,
                    "content_type": "link",
                    "url": resolved_href,
                    "href": href,
                    "title": (
                        tag.get("title")
                        or ""
                    ).strip(),
                },
            )

            elements.append(
                link_element
            )

            tag_element_ids[
                id(tag)
            ] = element_id

    # ---------------------------------------------------------
    # DOM parent → child relationships
    # ---------------------------------------------------------
    for tag in soup.find_all(True):
        source_id = tag_element_ids.get(
            id(tag)
        )

        if source_id is None:
            continue

        for child in tag.children:
            if not isinstance(
                child,
                Tag,
            ):
                continue

            target_id = tag_element_ids.get(
                id(child)
            )

            if target_id is None:
                continue

            relationships.append(
                ElementRelationship(
                    source_element_id=source_id,
                    relationship_type="contains",
                    target_element_id=target_id,
                )
            )

    # ---------------------------------------------------------
    # Attach top-level elements to document
    # ---------------------------------------------------------
    child_element_ids = {
        relationship.target_element_id
        for relationship in relationships
    }

    for element in elements:
        if element.element_id == "document":
            continue

        if element.element_id in child_element_ids:
            continue

        relationships.append(
            ElementRelationship(
                source_element_id="document",
                relationship_type="contains",
                target_element_id=element.element_id,
            )
        )

    return IngestionResult(
        document_id=document_id,
        elements=elements,
        relationships=relationships,
    )