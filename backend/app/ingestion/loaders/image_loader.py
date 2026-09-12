# from pathlib import Path

# from backend.app.ingestion.models.content import ImageContent
# from backend.app.ingestion.models.source_document import SourceDocument


# SUPPORTED_IMAGE_EXTENSIONS = {
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".webp",
#     ".bmp",
#     ".tiff",
#     ".tif",
# }


# def load_image(
#     file_path: str,
#     document_id: str,
# ) -> list[SourceDocument]:
#     image_path = Path(file_path)

#     if not image_path.exists():
#         raise FileNotFoundError(
#             f"Image file not found: {file_path}"
#         )

#     if not image_path.is_file():
#         raise ValueError(
#             f"Image path is not a file: {file_path}"
#         )

#     if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
#         raise ValueError(
#             f"Unsupported image format: {image_path.suffix}"
#         )

#     document = SourceDocument(
#         document_id=document_id,
#         source_type="image",
#         content=ImageContent(
#             path=str(image_path)
#         ),
#         metadata={
#             "file_name": image_path.name,
#             "content_type": "image",
#             "file_extension": image_path.suffix.lower(),
#         },
#     )

#     return [document]

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


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


def load_image(
    file_path: str,
    document_id: str,
) -> IngestionResult:
    image_path = Path(file_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {file_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Image path is not a file: {file_path}"
        )

    extension = image_path.suffix.lower()

    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {extension}"
        )

    root = SourceElement(
        element_id="document",
        document_id=document_id,
        element_type="document",
        content=TextContent(text=""),
        metadata={
            "file_name": image_path.name,
            "content_type": "document",
        },
    )

    image = SourceElement(
        element_id="image-1",
        document_id=document_id,
        element_type="image",
        content=ImageContent(
            path=str(image_path)
        ),
        metadata={
            "file_name": image_path.name,
            "content_type": "image",
            "file_extension": extension,
        },
    )

    relationship = ElementRelationship(
        source_element_id="document",
        relationship_type="contains",
        target_element_id="image-1",
    )

    return IngestionResult(
        document_id=document_id,
        elements=[
            root,
            image,
        ],
        relationships=[
            relationship,
        ],
    )