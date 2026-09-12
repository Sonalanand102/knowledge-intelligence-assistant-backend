from types import SimpleNamespace

import pytest

from backend.app.ingestion.loaders.youtube_loader import (
    extract_video_id,
    load_youtube,
    load_youtube_transcript,
)
from backend.app.ingestion.media.transcribers.whisper_transcriber import (
    TranscriptSegment,
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


def test_extract_video_id_watch_url():
    assert (
        extract_video_id(
            "https://www.youtube.com/watch?v=abc123"
        )
        == "abc123"
    )


def test_extract_video_id_watch_url_with_params():
    assert (
        extract_video_id(
            "https://www.youtube.com/watch?v=abc123&t=30"
        )
        == "abc123"
    )


def test_extract_video_id_short_url():
    assert (
        extract_video_id(
            "https://youtu.be/abc123"
        )
        == "abc123"
    )


def test_extract_video_id_shorts_url():
    assert (
        extract_video_id(
            "https://www.youtube.com/shorts/abc123"
        )
        == "abc123"
    )


def test_extract_video_id_invalid_url():
    with pytest.raises(ValueError):
        extract_video_id(
            "https://example.com/video"
        )


def test_load_youtube_transcript():
    transcript_items = [
        SimpleNamespace(
            start=0.0,
            duration=4.5,
            text="Hello world",
        ),
        SimpleNamespace(
            start=4.5,
            duration=3.0,
            text="This is a test",
        ),
    ]

    def fake_transcript_fetcher(
        video_id: str,
    ):
        assert video_id == "abc123"
        return transcript_items

    elements = load_youtube_transcript(
        url=(
            "https://www.youtube.com/"
            "watch?v=abc123"
        ),
        document_id="youtube-123",
        file_name="youtube-video",
        transcript_fetcher=fake_transcript_fetcher,
    )

    assert len(elements) == 2

    first = elements[0]
    second = elements[1]

    assert (
        first.element_id
        == "transcript-segment-1"
    )

    assert first.document_id == "youtube-123"
    assert first.element_type == "transcript"

    assert isinstance(
        first.content,
        TextContent,
    )

    assert (
        first.content.text
        == "Hello world"
    )

    assert (
        first.metadata["transcript_source"]
        == "youtube_captions"
    )

    assert (
        first.metadata["start_seconds"]
        == 0.0
    )

    assert (
        first.metadata["end_seconds"]
        == 4.5
    )

    assert (
        second.element_id
        == "transcript-segment-2"
    )

    assert (
        second.content.text
        == "This is a test"
    )

    assert (
        second.metadata["start_seconds"]
        == 4.5
    )

    assert (
        second.metadata["end_seconds"]
        == 7.5
    )


def test_load_youtube_uses_captions():
    def fake_downloader(
        url: str,
        output_dir: str,
    ) -> str:
        assert url.endswith(
            "watch?v=abc123"
        )

        return (
            f"{output_dir}/video.mp4"
        )

    def fake_transcript_fetcher(
        video_id: str,
    ):
        assert video_id == "abc123"

        return [
            SimpleNamespace(
                start=0.0,
                duration=2.0,
                text="Hello world",
            )
        ]

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return []

    def fail_whisper_creator(
        **kwargs,
    ):
        raise AssertionError(
            "Whisper should not be called "
            "when YouTube captions exist."
        )

    result = load_youtube(
        url=(
            "https://www.youtube.com/"
            "watch?v=abc123"
        ),
        document_id="youtube-123",
        file_name="video",
        output_dir="/tmp/youtube-test",
        whisper_transcriber=None,
        transcript_fetcher=(
            fake_transcript_fetcher
        ),
        video_downloader=(
            fake_downloader
        ),
        frame_extractor=(
            fake_frame_extractor
        ),
        whisper_transcript_creator=(
            fail_whisper_creator
        ),
    )

    assert result.document_id == "youtube-123"

    transcripts = [
        element
        for element in result.elements
        if element.element_type
        == "transcript"
    ]

    assert len(transcripts) == 1

    assert (
        transcripts[0].content.text
        == "Hello world"
    )

    assert (
        transcripts[0].metadata[
            "transcript_source"
        ]
        == "youtube_captions"
    )


def test_load_youtube_falls_back_to_whisper():
    def fake_downloader(
        url: str,
        output_dir: str,
    ) -> str:
        return (
            f"{output_dir}/video.mp4"
        )

    def failing_transcript_fetcher(
        video_id: str,
    ):
        raise RuntimeError(
            "Captions unavailable"
        )

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return []

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
                    text="Hello from Whisper"
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

    result = load_youtube(
        url=(
            "https://www.youtube.com/"
            "watch?v=abc123"
        ),
        document_id="youtube-456",
        file_name="video",
        output_dir="/tmp/youtube-test",
        whisper_transcriber=None,
        transcript_fetcher=(
            failing_transcript_fetcher
        ),
        video_downloader=(
            fake_downloader
        ),
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
        == "Hello from Whisper"
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


def test_load_youtube_extracts_frames():
    def fake_downloader(
        url: str,
        output_dir: str,
    ) -> str:
        return (
            f"{output_dir}/video.mp4"
        )

    def fake_transcript_fetcher(
        video_id: str,
    ):
        return []

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return [
            SourceDocument(
                document_id=document_id,
                source_type="youtube",
                content=ImageContent(
                    path="frame-1.jpg"
                ),
                metadata={
                    "content_type": "video_frame",
                    "timestamp_seconds": 5.0,
                    "frame_path": "frame-1.jpg",
                    "selection_method": (
                        "scene_change"
                    ),
                },
            )
        ]

    def fake_whisper_creator(
        **kwargs,
    ):
        return []

    result = load_youtube(
        url=(
            "https://www.youtube.com/"
            "watch?v=abc123"
        ),
        document_id="youtube-789",
        file_name="video",
        output_dir="/tmp/youtube-test",
        whisper_transcriber=None,
        transcript_fetcher=(
            fake_transcript_fetcher
        ),
        video_downloader=(
            fake_downloader
        ),
        frame_extractor=(
            fake_frame_extractor
        ),
        whisper_transcript_creator=(
            fake_whisper_creator
        ),
    )

    frames = [
        element
        for element in result.elements
        if element.element_type
        == "video_frame"
    ]

    assert len(frames) == 1

    frame = frames[0]

    assert isinstance(
        frame.content,
        ImageContent,
    )

    assert (
        frame.metadata[
            "timestamp_seconds"
        ]
        == 5.0
    )


def test_load_youtube_creates_root_relationships():
    def fake_downloader(
        url: str,
        output_dir: str,
    ) -> str:
        return (
            f"{output_dir}/video.mp4"
        )

    def fake_transcript_fetcher(
        video_id: str,
    ):
        return [
            SimpleNamespace(
                start=0.0,
                duration=3.0,
                text="Hello",
            )
        ]

    def fake_frame_extractor(
        video_path: str,
        output_dir: str,
        document_id: str,
    ) -> list[SourceDocument]:
        return []

    def fake_whisper_creator(
        **kwargs,
    ):
        return []

    result = load_youtube(
        url=(
            "https://www.youtube.com/"
            "watch?v=abc123"
        ),
        document_id="youtube-999",
        file_name="video",
        output_dir="/tmp/youtube-test",
        whisper_transcriber=None,
        transcript_fetcher=(
            fake_transcript_fetcher
        ),
        video_downloader=(
            fake_downloader
        ),
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


def test_transcript_segment_conversion():
    from backend.app.ingestion.loaders.youtube_loader import (
        _transcript_segment_to_element,
    )

    segment = TranscriptSegment(
        start_seconds=2.5,
        end_seconds=6.0,
        text="RAG combines retrieval and generation.",
    )

    element = (
        _transcript_segment_to_element(
            segment=segment,
            document_id="doc-1",
            file_name="video",
            segment_index=1,
        )
    )

    assert (
        element.element_id
        == "transcript-segment-1"
    )

    assert element.element_type == "transcript"

    assert (
        element.content.text
        == "RAG combines retrieval and generation."
    )

    assert (
        element.metadata["start_seconds"]
        == 2.5
    )

    assert (
        element.metadata["end_seconds"]
        == 6.0
    )

    assert (
        element.metadata[
            "transcript_source"
        ]
        == "whisper"
    )