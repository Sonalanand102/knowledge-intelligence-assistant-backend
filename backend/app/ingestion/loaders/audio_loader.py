# from pathlib import Path

# from backend.app.ingestion.models.content import AudioContent
# from backend.app.ingestion.models.source_document import SourceDocument


# SUPPORTED_AUDIO_EXTENSIONS = {
#     ".mp3",
#     ".wav",
#     ".m4a",
#     ".flac",
#     ".aac",
#     ".ogg",
# }


# def load_audio(
#     file_path: str,
#     document_id: str,
# ) -> list[SourceDocument]:
#     audio_path = Path(file_path)

#     if not audio_path.exists():
#         raise FileNotFoundError(
#             f"Audio file not found: {file_path}"
#         )

#     if not audio_path.is_file():
#         raise ValueError(
#             f"Audio path is not a file: {file_path}"
#         )

#     if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
#         raise ValueError(
#             f"Unsupported audio format: {audio_path.suffix}"
#         )

#     document = SourceDocument(
#         document_id=document_id,
#         source_type="audio",
#         content=AudioContent(
#             path=str(audio_path)
#         ),
#         metadata={
#             "file_name": audio_path.name,
#             "content_type": "audio",
#             "file_extension": audio_path.suffix.lower(),
#         },
#     )

#     return [document]

from pathlib import Path

from backend.app.ingestion.models.content import (
    AudioContent,
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


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
}


def load_audio(
    file_path: str,
    document_id: str,
) -> IngestionResult:
    audio_path = Path(file_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    if not audio_path.is_file():
        raise ValueError(
            f"Audio path is not a file: {file_path}"
        )

    extension = audio_path.suffix.lower()

    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {extension}"
        )

    root = SourceElement(
        element_id="document",
        document_id=document_id,
        element_type="document",
        content=TextContent(text=""),
        metadata={
            "file_name": audio_path.name,
            "content_type": "document",
        },
    )

    audio = SourceElement(
        element_id="audio-1",
        document_id=document_id,
        element_type="audio",
        content=AudioContent(
            path=str(audio_path)
        ),
        metadata={
            "file_name": audio_path.name,
            "content_type": "audio",
            "file_extension": extension,
        },
    )

    relationship = ElementRelationship(
        source_element_id="document",
        relationship_type="contains",
        target_element_id="audio-1",
    )

    return IngestionResult(
        document_id=document_id,
        elements=[
            root,
            audio,
        ],
        relationships=[
            relationship,
        ],
    )