# from pathlib import Path

# import pytest

# from backend.app.ingestion.loaders.audio_loader import (
#     load_audio,
# )
# from backend.app.ingestion.models.content import AudioContent


# def test_load_audio(tmp_path):
#     audio_path = tmp_path / "sample.wav"

#     audio_path.write_bytes(
#         b"fake audio data"
#     )

#     documents = load_audio(
#         file_path=str(audio_path),
#         document_id="audio-123",
#     )

#     assert len(documents) == 1

#     document = documents[0]

#     assert document.document_id == "audio-123"
#     assert document.source_type == "audio"

#     assert isinstance(
#         document.content,
#         AudioContent,
#     )

#     assert (
#         document.content.path
#         == str(audio_path)
#     )

#     assert (
#         document.metadata["file_name"]
#         == "sample.wav"
#     )

#     assert (
#         document.metadata["content_type"]
#         == "audio"
#     )

#     assert (
#         document.metadata["file_extension"]
#         == ".wav"
#     )


# def test_load_audio_supports_multiple_formats(
#     tmp_path,
# ):
#     for extension in (
#         ".mp3",
#         ".wav",
#         ".m4a",
#         ".flac",
#         ".aac",
#         ".ogg",
#     ):
#         audio_path = (
#             tmp_path / f"sample{extension}"
#         )

#         audio_path.write_bytes(
#             b"fake audio data"
#         )

#         documents = load_audio(
#             file_path=str(audio_path),
#             document_id=f"audio-{extension}",
#         )

#         assert len(documents) == 1

#         assert isinstance(
#             documents[0].content,
#             AudioContent,
#         )


# def test_load_audio_missing_file(tmp_path):
#     missing_path = (
#         tmp_path / "missing.wav"
#     )

#     with pytest.raises(FileNotFoundError):
#         load_audio(
#             file_path=str(missing_path),
#             document_id="audio-123",
#         )


# def test_load_audio_rejects_directory(tmp_path):
#     audio_dir = tmp_path / "audio"
#     audio_dir.mkdir()

#     with pytest.raises(ValueError):
#         load_audio(
#             file_path=str(audio_dir),
#             document_id="audio-123",
#         )


# def test_load_audio_rejects_unsupported_format(
#     tmp_path,
# ):
#     file_path = tmp_path / "sample.pdf"

#     file_path.write_bytes(
#         b"not audio"
#     )

#     with pytest.raises(ValueError):
#         load_audio(
#             file_path=str(file_path),
#             document_id="audio-123",
#         )

from pathlib import Path

import pytest

from backend.app.ingestion.loaders.audio_loader import (
    load_audio,
)
from backend.app.ingestion.models.content import (
    AudioContent,
)


SUPPORTED_FORMATS = (
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
)


def test_load_audio(tmp_path):
    audio_path = (
        tmp_path / "sample.wav"
    )

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
    )

    assert result.document_id == "audio-123"

    assert len(result.elements) == 2

    root = result.elements[0]
    audio = result.elements[1]

    assert root.element_id == "document"
    assert root.element_type == "document"

    assert audio.element_id == "audio-1"
    assert audio.element_type == "audio"

    assert isinstance(
        audio.content,
        AudioContent,
    )

    assert (
        audio.content.path
        == str(audio_path)
    )

    assert (
        audio.metadata["file_name"]
        == "sample.wav"
    )

    assert (
        audio.metadata["content_type"]
        == "audio"
    )

    assert (
        audio.metadata["file_extension"]
        == ".wav"
    )


def test_load_audio_supports_multiple_formats(
    tmp_path,
):
    for extension in SUPPORTED_FORMATS:
        audio_path = (
            tmp_path
            / f"sample{extension}"
        )

        audio_path.write_bytes(
            b"fake audio data"
        )

        result = load_audio(
            file_path=str(audio_path),
            document_id=(
                f"audio-{extension}"
            ),
        )

        audio_elements = [
            element
            for element in result.elements
            if element.element_type
            == "audio"
        ]

        assert len(audio_elements) == 1

        assert isinstance(
            audio_elements[0].content,
            AudioContent,
        )


def test_load_audio_creates_relationship(
    tmp_path,
):
    audio_path = (
        tmp_path / "sample.wav"
    )

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
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
        == "audio-1"
    )


def test_load_audio_missing_file(
    tmp_path,
):
    audio_path = (
        tmp_path / "missing.wav"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_audio(
            file_path=str(audio_path),
            document_id="audio-123",
        )


def test_load_audio_rejects_directory(
    tmp_path,
):
    directory = tmp_path / "audio"
    directory.mkdir()

    with pytest.raises(
        ValueError
    ):
        load_audio(
            file_path=str(directory),
            document_id="audio-123",
        )


def test_load_audio_rejects_unsupported_format(
    tmp_path,
):
    file_path = (
        tmp_path / "sample.pdf"
    )

    file_path.write_bytes(
        b"not audio"
    )

    with pytest.raises(
        ValueError
    ):
        load_audio(
            file_path=str(file_path),
            document_id="audio-123",
        )