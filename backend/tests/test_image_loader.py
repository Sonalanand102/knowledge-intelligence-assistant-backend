# from pathlib import Path

# import pytest

# from backend.app.ingestion.loaders.image_loader import (
#     load_image,
# )
# from backend.app.ingestion.models.content import ImageContent


# def test_load_image(tmp_path):
#     image_path = tmp_path / "sample.png"

#     image_path.write_bytes(
#         b"fake image data"
#     )

#     documents = load_image(
#         file_path=str(image_path),
#         document_id="image-123",
#     )

#     assert len(documents) == 1

#     document = documents[0]

#     assert document.document_id == "image-123"
#     assert document.source_type == "image"

#     assert isinstance(
#         document.content,
#         ImageContent,
#     )

#     assert (
#         document.content.path
#         == str(image_path)
#     )

#     assert (
#         document.metadata["file_name"]
#         == "sample.png"
#     )

#     assert (
#         document.metadata["content_type"]
#         == "image"
#     )

#     assert (
#         document.metadata["file_extension"]
#         == ".png"
#     )


# def test_load_image_supports_multiple_formats(
#     tmp_path,
# ):
#     for extension in (
#         ".jpg",
#         ".jpeg",
#         ".png",
#         ".webp",
#         ".bmp",
#         ".tiff",
#         ".tif",
#     ):
#         image_path = (
#             tmp_path / f"sample{extension}"
#         )

#         image_path.write_bytes(
#             b"fake image data"
#         )

#         documents = load_image(
#             file_path=str(image_path),
#             document_id=f"image-{extension}",
#         )

#         assert len(documents) == 1

#         assert isinstance(
#             documents[0].content,
#             ImageContent,
#         )


# def test_load_image_missing_file(tmp_path):
#     missing_path = (
#         tmp_path / "missing.png"
#     )

#     with pytest.raises(FileNotFoundError):
#         load_image(
#             file_path=str(missing_path),
#             document_id="image-123",
#         )


# def test_load_image_rejects_directory(tmp_path):
#     image_dir = tmp_path / "images"
#     image_dir.mkdir()

#     with pytest.raises(ValueError):
#         load_image(
#             file_path=str(image_dir),
#             document_id="image-123",
#         )


# def test_load_image_rejects_unsupported_format(
#     tmp_path,
# ):
#     file_path = tmp_path / "sample.pdf"

#     file_path.write_bytes(
#         b"not an image"
#     )

#     with pytest.raises(ValueError):
#         load_image(
#             file_path=str(file_path),
#             document_id="image-123",
#         )

import pytest

from backend.app.ingestion.loaders.image_loader import (
    load_image,
)
from backend.app.ingestion.models.content import (
    ImageContent,
)


SUPPORTED_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
)


def test_load_image(tmp_path):
    image_path = (
        tmp_path / "sample.png"
    )

    image_path.write_bytes(
        b"fake image data"
    )

    result = load_image(
        file_path=str(image_path),
        document_id="image-123",
    )

    assert result.document_id == "image-123"
    assert len(result.elements) == 2

    root = result.elements[0]
    image = result.elements[1]

    assert root.element_id == "document"
    assert root.element_type == "document"

    assert image.element_id == "image-1"
    assert image.element_type == "image"

    assert isinstance(
        image.content,
        ImageContent,
    )

    assert (
        image.content.path
        == str(image_path)
    )

    assert (
        image.metadata["file_name"]
        == "sample.png"
    )

    assert (
        image.metadata["content_type"]
        == "image"
    )

    assert (
        image.metadata["file_extension"]
        == ".png"
    )


def test_load_image_supports_multiple_formats(
    tmp_path,
):
    for extension in SUPPORTED_FORMATS:
        image_path = (
            tmp_path
            / f"sample{extension}"
        )

        image_path.write_bytes(
            b"fake image data"
        )

        result = load_image(
            file_path=str(image_path),
            document_id=(
                f"image-{extension}"
            ),
        )

        image_elements = [
            element
            for element in result.elements
            if element.element_type
            == "image"
        ]

        assert len(image_elements) == 1

        image = image_elements[0]

        assert isinstance(
            image.content,
            ImageContent,
        )

        assert (
            image.content.path
            == str(image_path)
        )


def test_load_image_creates_relationship(
    tmp_path,
):
    image_path = (
        tmp_path / "sample.png"
    )

    image_path.write_bytes(
        b"fake image data"
    )

    result = load_image(
        file_path=str(image_path),
        document_id="image-123",
    )

    assert len(result.relationships) == 1

    relationship = (
        result.relationships[0]
    )

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
        == "image-1"
    )


def test_load_image_missing_file(
    tmp_path,
):
    image_path = (
        tmp_path / "missing.png"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_image(
            file_path=str(image_path),
            document_id="image-123",
        )


def test_load_image_rejects_directory(
    tmp_path,
):
    directory = tmp_path / "images"
    directory.mkdir()

    with pytest.raises(
        ValueError
    ):
        load_image(
            file_path=str(directory),
            document_id="image-123",
        )


def test_load_image_rejects_unsupported_format(
    tmp_path,
):
    file_path = (
        tmp_path / "sample.pdf"
    )

    file_path.write_bytes(
        b"not an image"
    )

    with pytest.raises(
        ValueError
    ):
        load_image(
            file_path=str(file_path),
            document_id="image-123",
        )