from pathlib import Path

import pytest

from backend.app.ingestion.loaders.video_loader import (
    _parse_caption_text,
    load_video,
)
from backend.app.ingestion.media.transcribers.whisper_transcriber import (
    TranscriptSegment,
    WhisperTranscriber
)
from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)
from backend.app.ingestion.models.source_document import (
    SourceDocument,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)

def test_parse_vtt_caption_segments():
    caption_text = """WEBVTT

00:00.000 --> 00:02.000
Hello, this is a test.

00:02.000 --> 00:05.000
We are testing captions.
"""

    segments = _parse_caption_text(
        caption_text
    )

    assert len(segments) == 2

    assert (
        segments[0].start_seconds
        == 0.0
    )

    assert (
        segments[0].end_seconds
        == 2.0
    )

    assert (
        segments[0].text
        == "Hello, this is a test."
    )

    assert (
        segments[1].start_seconds
        == 2.0
    )

    assert (
        segments[1].end_seconds
        == 5.0
    )

    assert (
        segments[1].text
        == "We are testing captions."
    )


def test_parse_srt_caption_segments():
    caption_text = """1
00:00:01,000 --> 00:00:03,000
Hello world.

2
00:00:03,000 --> 00:00:06,000
This is another segment.
"""

    segments = _parse_caption_text(
        caption_text
    )

    assert len(segments) == 2

    assert (
        segments[0].start_seconds
        == 1.0
    )

    assert (
        segments[0].end_seconds
        == 3.0
    )

    assert (
        segments[1].start_seconds
        == 3.0
    )

    assert (
        segments[1].end_seconds
        == 6.0
    )


def test_parse_caption_with_cue_settings():
    caption_text = """WEBVTT

00:00.000 --> 00:02.000 align:start
Hello world.
"""

    segments = _parse_caption_text(
        caption_text
    )

    assert len(segments) == 1

    assert (
        segments[0].start_seconds
        == 0.0
    )

    assert (
        segments[0].end_seconds
        == 2.0
    )

    assert (
        segments[0].text
        == "Hello world."
    )


def test_load_video_with_captions(
    tmp_path,
):
    video_path = (
        tmp_path / "test_video.mp4"
    )

    # Only path validation is needed here.
    # Frame extraction and Whisper are injected.
    video_path.write_bytes(
        b"fake-video"
    )

    caption_path = (
        tmp_path / "test_video.vtt"
    )

    caption_path.write_text(
        """WEBVTT

00:00.000 --> 00:02.000
Hello, this is a test video.

00:02.000 --> 00:04.000
We are testing temporal relationships.
""",
        encoding="utf-8",
    )

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return [
            SourceDocument(
                document_id=document_id,
                source_type="video",
                content=ImageContent(
                    path="frame-1.jpg"
                ),
                metadata={
                    "content_type": "video_frame",
                    "timestamp_seconds": 1.0,
                    "frame_path": "frame-1.jpg",
                    "selection_method": (
                        "initial_frame"
                    ),
                },
            ),
            SourceDocument(
                document_id=document_id,
                source_type="video",
                content=ImageContent(
                    path="frame-2.jpg"
                ),
                metadata={
                    "content_type": "video_frame",
                    "timestamp_seconds": 3.0,
                    "frame_path": "frame-2.jpg",
                    "selection_method": (
                        "scene_change"
                    ),
                },
            ),
        ]

    def fail_whisper_creator(
        **kwargs,
    ):
        raise AssertionError(
            "Whisper should not be called "
            "when captions exist."
        )

    result = load_video(
        file_path=str(video_path),
        document_id="video-123",
        output_dir=str(
            tmp_path / "output"
        ),
        whisper_transcriber=None,
        frame_extractor=(
            fake_frame_extractor
        ),
        whisper_transcript_creator=(
            fail_whisper_creator
        ),
    )

    assert result.document_id == "video-123"

    transcripts = [
        element
        for element in result.elements
        if element.element_type
        == "transcript"
    ]

    frames = [
        element
        for element in result.elements
        if element.element_type
        == "video_frame"
    ]

    assert len(transcripts) == 2
    assert len(frames) == 2

    assert (
        transcripts[0].content.text
        == "Hello, this is a test video."
    )

    assert (
        transcripts[1].content.text
        == "We are testing temporal relationships."
    )

    assert (
        transcripts[0].metadata[
            "transcript_source"
        ]
        == "sidecar"
    )

    assert (
        transcripts[0].metadata[
            "start_seconds"
        ]
        == 0.0
    )

    assert (
        transcripts[0].metadata[
            "end_seconds"
        ]
        == 2.0
    )

    assert (
        transcripts[1].metadata[
            "start_seconds"
        ]
        == 2.0
    )

    assert (
        transcripts[1].metadata[
            "end_seconds"
        ]
        == 4.0
    )

    temporal_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == "temporally_adjacent"
    ]

    assert temporal_relationships


def test_load_video_without_captions_uses_whisper(
    tmp_path,
):
    video_path = (
        tmp_path / "test_video.mp4"
    )

    video_path.write_bytes(
        b"fake-video"
    )

    whisper_called = False

    def fake_whisper_creator(
        video_path: str,
        output_dir: str,
        document_id: str,
        file_name: str,
        whisper_transcriber,
    ) -> list[SourceElement]:
        nonlocal whisper_called

        whisper_called = True

        return [
            SourceElement(
                element_id=(
                    "transcript-segment-1"
                ),
                document_id=document_id,
                element_type="transcript",
                content=TextContent(
                    text="Hello from Whisper."
                ),
                metadata={
                    "file_name": file_name,
                    "content_type": "transcript",
                    "transcript_source": "whisper",
                    "segment_index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                },
            )
        ]

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return []

    result = load_video(
        file_path=str(video_path),
        document_id="video-456",
        output_dir=str(
            tmp_path / "output"
        ),
        whisper_transcriber=None,
        frame_extractor=(
            fake_frame_extractor
        ),
        whisper_transcript_creator=(
            fake_whisper_creator
        ),
    )

    assert whisper_called is True

    transcripts = [
        element
        for element in result.elements
        if element.element_type
        == "transcript"
    ]

    assert len(transcripts) == 1

    transcript = transcripts[0]

    assert (
        transcript.content.text
        == "Hello from Whisper."
    )

    assert (
        transcript.metadata[
            "transcript_source"
        ]
        == "whisper"
    )

    assert (
        transcript.metadata[
            "start_seconds"
        ]
        == 0.0
    )

    assert (
        transcript.metadata[
            "end_seconds"
        ]
        == 2.0
    )


def test_load_video_creates_root_relationships(
    tmp_path,
):
    video_path = (
        tmp_path / "test_video.mp4"
    )

    video_path.write_bytes(
        b"fake-video"
    )

    def fake_whisper_creator(
        **kwargs,
    ) -> list[SourceElement]:
        return [
            SourceElement(
                element_id=(
                    "transcript-segment-1"
                ),
                document_id="video-789",
                element_type="transcript",
                content=TextContent(
                    text="Hello"
                ),
                metadata={
                    "content_type": "transcript",
                    "transcript_source": "whisper",
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                },
            )
        ]

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return [
            SourceDocument(
                document_id=document_id,
                source_type="video",
                content=ImageContent(
                    path="frame.jpg"
                ),
                metadata={
                    "content_type": "video_frame",
                    "timestamp_seconds": 1.0,
                },
            )
        ]

    result = load_video(
        file_path=str(video_path),
        document_id="video-789",
        output_dir=str(
            tmp_path / "output"
        ),
        whisper_transcriber=None,
        frame_extractor=(
            fake_frame_extractor
        ),
        whisper_transcript_creator=(
            fake_whisper_creator
        ),
    )

    assert any(
        relationship.source_element_id
        == "document"
        and relationship.relationship_type
        == "contains"
        and relationship.target_element_id
        == "transcript-segment-1"
        for relationship
        in result.relationships
    )

    assert any(
        relationship.source_element_id
        == "document"
        and relationship.relationship_type
        == "contains"
        and relationship.target_element_id
        == "frame-1"
        for relationship
        in result.relationships
    )


def test_load_video_creates_temporal_relationships(
    tmp_path,
):
    video_path = (
        tmp_path / "test_video.mp4"
    )

    video_path.write_bytes(
        b"fake-video"
    )

    def fake_whisper_creator(
        **kwargs,
    ) -> list[SourceElement]:
        return [
            SourceElement(
                element_id=(
                    "transcript-segment-1"
                ),
                document_id="video-999",
                element_type="transcript",
                content=TextContent(
                    text="Architecture"
                ),
                metadata={
                    "content_type": "transcript",
                    "transcript_source": "whisper",
                    "start_seconds": 10.0,
                    "end_seconds": 15.0,
                },
            )
        ]

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return [
            SourceDocument(
                document_id=document_id,
                source_type="video",
                content=ImageContent(
                    path="frame.jpg"
                ),
                metadata={
                    "content_type": "video_frame",
                    "timestamp_seconds": 12.0,
                },
            )
        ]

    result = load_video(
        file_path=str(video_path),
        document_id="video-999",
        output_dir=str(
            tmp_path / "output"
        ),
        whisper_transcriber=None,
        frame_extractor=(
            fake_frame_extractor
        ),
        whisper_transcript_creator=(
            fake_whisper_creator
        ),
    )

    temporal_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == "temporally_adjacent"
    ]

    assert len(temporal_relationships) == 2

    assert any(
        relationship.source_element_id
        == "transcript-segment-1"
        and relationship.target_element_id
        == "frame-1"
        for relationship
        in temporal_relationships
    )

    assert any(
        relationship.source_element_id
        == "frame-1"
        and relationship.target_element_id
        == "transcript-segment-1"
        for relationship
        in temporal_relationships
    )


def test_load_video_missing_file(
    tmp_path,
):
    missing_path = (
        tmp_path / "missing.mp4"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_video(
            file_path=str(missing_path),
            document_id="video-123",
            output_dir=str(
                tmp_path / "output"
            ),
            whisper_transcriber=None,
        )


def test_load_video_rejects_directory(
    tmp_path,
):
    directory = (
        tmp_path / "video"
    )

    directory.mkdir()

    with pytest.raises(
        ValueError
    ):
        load_video(
            file_path=str(directory),
            document_id="video-123",
            output_dir=str(
                tmp_path / "output"
            ),
            whisper_transcriber=None,
        )