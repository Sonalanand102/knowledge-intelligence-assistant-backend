from pathlib import Path

import pytest

from backend.app.ingestion.loaders.audio_loader import load_audio
from backend.app.ingestion.media.transcribers.whisper_transcriber import (
    TranscriptSegment,
)
from backend.app.ingestion.models.content import (
    AudioContent,
    TextContent,
)


SUPPORTED_FORMATS = (
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
)


class FakeWhisperTranscriber:
    def transcribe_segments(
        self,
        audio_path: str,
    ) -> list[TranscriptSegment]:
        assert Path(audio_path).exists()

        return [
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=3.5,
                text="Hello, how are you?",
            ),
            TranscriptSegment(
                start_seconds=3.5,
                end_seconds=7.0,
                text="I am doing well, thank you.",
            ),
        ]


@pytest.fixture
def whisper_transcriber():
    return FakeWhisperTranscriber()


def test_load_audio(
    tmp_path,
    whisper_transcriber,
):
    audio_path = tmp_path / "sample.wav"

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
        whisper_transcriber=whisper_transcriber,
    )

    assert result.document_id == "audio-123"

    # document + audio + 2 transcript segments
    assert len(result.elements) == 4

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
    whisper_transcriber,
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
            document_id=f"audio-{extension}",
            whisper_transcriber=whisper_transcriber,
        )

        audio_elements = [
            element
            for element in result.elements
            if element.element_type == "audio"
        ]

        assert len(audio_elements) == 1

        assert isinstance(
            audio_elements[0].content,
            AudioContent,
        )


def test_load_audio_creates_relationship(
    tmp_path,
    whisper_transcriber,
):
    audio_path = tmp_path / "sample.wav"

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
        whisper_transcriber=whisper_transcriber,
    )

    # document -> audio
    relationship = result.relationships[0]

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


def test_load_audio_creates_transcript_elements(
    tmp_path,
    whisper_transcriber,
):
    audio_path = tmp_path / "sample.wav"

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
        whisper_transcriber=whisper_transcriber,
    )

    transcript_elements = [
        element
        for element in result.elements
        if element.element_type == "transcript"
    ]

    assert len(transcript_elements) == 2

    first = transcript_elements[0]
    second = transcript_elements[1]

    assert first.element_id == "transcript-1"
    assert second.element_id == "transcript-2"

    assert isinstance(
        first.content,
        TextContent,
    )

    assert (
        first.content.text
        == "Hello, how are you?"
    )

    assert (
        second.content.text
        == "I am doing well, thank you."
    )

    assert (
        first.metadata["content_type"]
        == "transcript"
    )

    assert (
        first.metadata["transcript_source"]
        == "whisper"
    )


def test_load_audio_preserves_transcript_timestamps(
    tmp_path,
    whisper_transcriber,
):
    audio_path = tmp_path / "sample.wav"

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
        whisper_transcriber=whisper_transcriber,
    )

    transcript_elements = [
        element
        for element in result.elements
        if element.element_type == "transcript"
    ]

    first = transcript_elements[0]
    second = transcript_elements[1]

    assert (
        first.metadata["start_seconds"]
        == 0.0
    )

    assert (
        first.metadata["end_seconds"]
        == 3.5
    )

    assert (
        second.metadata["start_seconds"]
        == 3.5
    )

    assert (
        second.metadata["end_seconds"]
        == 7.0
    )


def test_load_audio_creates_transcript_relationships(
    tmp_path,
    whisper_transcriber,
):
    audio_path = tmp_path / "sample.wav"

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
        whisper_transcriber=whisper_transcriber,
    )

    # document -> audio
    assert any(
        relationship.source_element_id == "document"
        and relationship.relationship_type == "contains"
        and relationship.target_element_id == "audio-1"
        for relationship in result.relationships
    )

    # audio -> transcript-1
    assert any(
        relationship.source_element_id == "audio-1"
        and relationship.relationship_type == "contains"
        and relationship.target_element_id == "transcript-1"
        for relationship in result.relationships
    )

    # audio -> transcript-2
    assert any(
        relationship.source_element_id == "audio-1"
        and relationship.relationship_type == "contains"
        and relationship.target_element_id == "transcript-2"
        for relationship in result.relationships
    )

    # document -> audio + 2 transcript relationships
    assert len(result.relationships) == 3


def test_load_audio_transcript_relationships_preserve_timestamps(
    tmp_path,
    whisper_transcriber,
):
    audio_path = tmp_path / "sample.wav"

    audio_path.write_bytes(
        b"fake audio data"
    )

    result = load_audio(
        file_path=str(audio_path),
        document_id="audio-123",
        whisper_transcriber=whisper_transcriber,
    )

    transcript_relationship = next(
        relationship
        for relationship in result.relationships
        if relationship.target_element_id
        == "transcript-1"
    )

    assert (
        transcript_relationship.metadata["start_seconds"]
        == 0.0
    )

    assert (
        transcript_relationship.metadata["end_seconds"]
        == 3.5
    )


def test_load_audio_missing_file(
    whisper_transcriber,
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
            whisper_transcriber=whisper_transcriber,
        )


def test_load_audio_rejects_directory(
    tmp_path,
    whisper_transcriber,
):
    directory = tmp_path / "audio"
    directory.mkdir()

    with pytest.raises(
        ValueError
    ):
        load_audio(
            file_path=str(directory),
            document_id="audio-123",
            whisper_transcriber=whisper_transcriber,
        )


def test_load_audio_rejects_unsupported_format(
    tmp_path,
    whisper_transcriber,
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
            whisper_transcriber=whisper_transcriber,
        )