from pathlib import Path

from backend.app.ingestion.media.transcribers.whisper_transcriber import (
    WhisperTranscriber,
)
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
    whisper_transcriber: WhisperTranscriber | None = None,
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

    # Root document element
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

    # Original audio element
    audio = SourceElement(
        element_id="audio-1",
        document_id=document_id,
        element_type="audio",
        content=AudioContent(
            path=str(audio_path),
        ),
        metadata={
            "file_name": audio_path.name,
            "content_type": "audio",
            "file_extension": extension,
        },
    )

    # Reuse existing Whisper transcription infrastructure.
    if whisper_transcriber is None:
        whisper_transcriber = WhisperTranscriber()

    transcript_segments = (
        whisper_transcriber.transcribe_segments(
            str(audio_path)
        )
    )

    transcript_elements: list[SourceElement] = []
    transcript_relationships: list[ElementRelationship] = []

    # Create one transcript element per Whisper segment.
    for index, segment in enumerate(
        transcript_segments,
        start=1,
    ):
        transcript_element_id = (
            f"transcript-{index}"
        )

        transcript_element = SourceElement(
            element_id=transcript_element_id,
            document_id=document_id,
            element_type="transcript",
            content=TextContent(
                text=segment.text,
            ),
            metadata={
                "file_name": audio_path.name,
                "content_type": "transcript",
                "transcript_source": "whisper",
                "segment_index": index,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
            },
        )

        transcript_elements.append(
            transcript_element
        )

        # audio -> transcript
        transcript_relationships.append(
            ElementRelationship(
                source_element_id="audio-1",
                relationship_type="contains",
                target_element_id=transcript_element_id,
                metadata={
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                },
            )
        )

    # document -> audio
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
            *transcript_elements,
        ],
        relationships=[
            relationship,
            *transcript_relationships,
        ],
    )