# from pathlib import Path

# from backend.app.ingestion.models.source_document import SourceDocument
# from backend.app.ingestion.models.content import TextContent

# def load_markdown(
#     file_path: str,
#     document_id: str,
#     file_name: str,
# ) -> list[SourceDocument]:

#     path = Path(file_path)
    
#     content = path.read_text(encoding="utf-8")
    
#     if not content.strip():
#         return []

#     return [SourceDocument(content=TextContent(text=content), document_id=document_id, source_type="markdown", metadata={"file_name": file_name})]

from __future__ import annotations

import re
from pathlib import Path

from backend.app.ingestion.models.content import (
    ImageContent,
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


HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.*)$"
)

IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)"
)

LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)"
)


def _resolve_path(
    source_path: Path,
    target: str,
) -> str:
    if target.startswith(("http://", "https://")):
        return target

    return str(
        (source_path.parent / target).resolve()
    )


def load_markdown(
    file_path: str,
    document_id: str,
) -> IngestionResult:
    markdown_path = Path(file_path)

    if not markdown_path.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {file_path}"
        )

    if not markdown_path.is_file():
        raise ValueError(
            f"Markdown path is not a file: {file_path}"
        )

    if markdown_path.suffix.lower() not in {
        ".md",
        ".markdown",
    }:
        raise ValueError(
            "Unsupported Markdown format: "
            f"{markdown_path.suffix}"
        )

    text = markdown_path.read_text(
        encoding="utf-8"
    )

    elements: list[SourceElement] = []
    relationships: list[
        ElementRelationship
    ] = []

    root = SourceElement(
        element_id="document",
        document_id=document_id,
        element_type="document",
        content=TextContent(text=""),
        metadata={
            "file_name": markdown_path.name,
            "content_type": "document",
        },
    )

    elements.append(root)

    element_index = 0

    lines = text.splitlines()

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------
        image_match = IMAGE_PATTERN.fullmatch(
            line
        )

        if image_match:
            alt_text = (
                image_match.group(1).strip()
            )

            target = image_match.group(2).strip()

            source = _resolve_path(
                markdown_path,
                target,
            )

            element_index += 1

            element_id = (
                f"image-{element_index}"
            )

            image_is_remote = target.startswith(
                (
                    "http://",
                    "https://",
                )
            )

            if image_is_remote:
                image_element = SourceElement(
                    element_id=element_id,
                    document_id=document_id,
                    element_type="image",
                    content=ImageContent(
                        path=source
                    ),
                    metadata={
                        "file_name": markdown_path.name,
                        "content_type": "image",
                        "alt_text": alt_text,
                        "source": source,
                        "remote": True,
                        "line_number": line_number,
                    },
                )
            else:
                image_element = SourceElement(
                    element_id=element_id,
                    document_id=document_id,
                    element_type="image",
                    content=ImageContent(
                        path=source
                    ),
                    metadata={
                        "file_name": markdown_path.name,
                        "content_type": "image",
                        "alt_text": alt_text,
                        "source": source,
                        "remote": False,
                        "exists": Path(source).exists(),
                        "line_number": line_number,
                    },
                )

            elements.append(
                image_element
            )

            relationships.append(
                ElementRelationship(
                    source_element_id="document",
                    relationship_type="contains",
                    target_element_id=element_id,
                )
            )

            continue

        # -----------------------------------------------------
        # Heading
        # -----------------------------------------------------
        heading_match = HEADING_PATTERN.match(
            line
        )

        if heading_match:
            level = len(
                heading_match.group(1)
            )

            heading_text = (
                heading_match.group(2).strip()
            )

            element_index += 1

            element_id = (
                f"heading-{element_index}"
            )

            heading_element = SourceElement(
                element_id=element_id,
                document_id=document_id,
                element_type="heading",
                content=TextContent(
                    text=heading_text
                ),
                metadata={
                    "file_name": markdown_path.name,
                    "content_type": "heading",
                    "level": level,
                    "line_number": line_number,
                },
            )

            elements.append(
                heading_element
            )

            relationships.append(
                ElementRelationship(
                    source_element_id="document",
                    relationship_type="contains",
                    target_element_id=element_id,
                )
            )

            continue

        # -----------------------------------------------------
        # Link-only line
        # -----------------------------------------------------
        link_match = LINK_PATTERN.fullmatch(
            line
        )

        if link_match:
            link_text = (
                link_match.group(1).strip()
            )

            url = (
                link_match.group(2).strip()
            )

            element_index += 1

            element_id = (
                f"link-{element_index}"
            )

            link_element = SourceElement(
                element_id=element_id,
                document_id=document_id,
                element_type="link",
                content=TextContent(
                    text=link_text
                ),
                metadata={
                    "file_name": markdown_path.name,
                    "content_type": "link",
                    "url": url,
                    "title": link_match.group(3),
                    "line_number": line_number,
                },
            )

            elements.append(
                link_element
            )

            relationships.append(
                ElementRelationship(
                    source_element_id="document",
                    relationship_type="contains",
                    target_element_id=element_id,
                )
            )

            continue

        # -----------------------------------------------------
        # Regular text
        # -----------------------------------------------------
        element_index += 1

        element_id = (
            f"text-{element_index}"
        )

        text_element = SourceElement(
            element_id=element_id,
            document_id=document_id,
            element_type="paragraph",
            content=TextContent(
                text=line
            ),
            metadata={
                "file_name": markdown_path.name,
                "content_type": "text",
                "line_number": line_number,
            },
        )

        elements.append(
            text_element
        )

        relationships.append(
            ElementRelationship(
                source_element_id="document",
                relationship_type="contains",
                target_element_id=element_id,
            )
        )

        # -----------------------------------------------------
        # Inline links inside paragraph
        # -----------------------------------------------------
        for link_index, link_match in enumerate(
            LINK_PATTERN.finditer(line),
            start=1,
        ):
            link_text = (
                link_match.group(1).strip()
            )

            url = (
                link_match.group(2).strip()
            )

            link_element_id = (
                f"{element_id}-link-{link_index}"
            )

            link_element = SourceElement(
                element_id=link_element_id,
                document_id=document_id,
                element_type="link",
                content=TextContent(
                    text=link_text
                ),
                metadata={
                    "file_name": markdown_path.name,
                    "content_type": "link",
                    "url": url,
                    "title": link_match.group(3),
                    "line_number": line_number,
                },
            )

            elements.append(
                link_element
            )

            relationships.append(
                ElementRelationship(
                    source_element_id=element_id,
                    relationship_type="contains",
                    target_element_id=link_element_id,
                )
            )

    return IngestionResult(
        document_id=document_id,
        elements=elements,
        relationships=relationships,
    )