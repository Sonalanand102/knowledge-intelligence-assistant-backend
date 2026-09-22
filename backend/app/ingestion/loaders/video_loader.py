from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable


from backend.app.ingestion.media.video_frame_extractor import (
    extract_key_frames,
)
from backend.app.ingestion.media.transcribers.whisper_transcriber import (
    TranscriptSegment,
    WhisperTranscriber,
)
from backend.app.ingestion.models.content import (
    TextContent,
    VideoContent,
)
from backend.app.ingestion.models.element_relationship import (
    ElementRelationship,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.models.source_document import (
    SourceDocument,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)
from backend.app.ingestion.relationships.temporal import (
    resolve_temporal_relationships,
)

import logging
import time

logger = logging.getLogger(__name__)


FrameExtractor = Callable[
    [str, str, str],
    list[SourceDocument],
]


TIMESTAMP_PATTERN = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def find_sidecar_caption(
    video_path: Path,
) -> Path | None:
    for extension in (
        ".srt",
        ".vtt",
    ):
        caption_path = (
            video_path.with_suffix(
                extension
            )
        )

        if caption_path.exists():
            return caption_path

    return None


def _parse_timestamp(
    value: str,
) -> float:
    value = value.strip().replace(
        ",",
        ".",
    )

    parts = value.split(":")

    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])

        return (
            minutes * 60
            + seconds
        )

    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    raise ValueError(
        f"Invalid caption timestamp: {value}"
    )


def _parse_caption_text(
    caption_text: str,
) -> list[TranscriptSegment]:
    lines = caption_text.splitlines()

    segments: list[TranscriptSegment] = []

    current_start: float | None = None
    current_end: float | None = None
    text_lines: list[str] = []

    def flush_segment() -> None:
        nonlocal current_start
        nonlocal current_end
        nonlocal text_lines

        if (
            current_start is None
            or current_end is None
        ):
            text_lines = []
            return

        text = " ".join(
            line.strip()
            for line in text_lines
            if line.strip()
        ).strip()

        if text:
            segments.append(
                TranscriptSegment(
                    start_seconds=current_start,
                    end_seconds=current_end,
                    text=text,
                )
            )

        current_start = None
        current_end = None
        text_lines = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush_segment()
            continue

        if line.upper() == "WEBVTT":
            continue

        # Skip numeric SRT cue identifiers.
        if line.isdigit():
            continue

        if "-->" in line:
            flush_segment()

            start_text, end_text = (
                line.split(
                    "-->",
                    1,
                )
            )

            # Remove optional cue settings:
            # 00:00:01.000 align:start
            end_text = (
                end_text.strip()
                .split()[0]
            )

            current_start = (
                _parse_timestamp(
                    start_text
                )
            )

            current_end = (
                _parse_timestamp(
                    end_text
                )
            )

            continue

        text_lines.append(
            line
        )

    flush_segment()

    return segments


def extract_embedded_captions(
    video_path: str,
) -> str | None:
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        video_path,
    ]

    try:
        probe_result = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "[VIDEO] subtitle stream detection timed out file=%s",
            video_path,
        )
        return None

    if probe_result.returncode != 0:
        logger.warning(
            "[VIDEO] subtitle stream detection failed file=%s",
            video_path,
        )
        return None

    stream_indexes = [
        line.strip()
        for line in probe_result.stdout.splitlines()
        if line.strip()
    ]

    if not stream_indexes:
        logger.info(
            "[VIDEO] no embedded subtitle stream file=%s",
            video_path,
        )
        return None

    subtitle_index = stream_indexes[0]

    logger.info(
        "[VIDEO] embedded subtitle stream found index=%s",
        subtitle_index,
    )

    extract_command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-map",
        f"0:{subtitle_index}",
        "-f",
        "webvtt",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            extract_command,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "[VIDEO] embedded subtitle extraction timed out file=%s",
            video_path,
        )
        return None

    if result.returncode != 0:
        logger.warning(
            "[VIDEO] embedded subtitle extraction failed file=%s",
            video_path,
        )
        return None

    captions = result.stdout.strip()

    return captions or None

def load_caption_segments(
    video_path: Path,
) -> tuple[
    list[TranscriptSegment],
    str | None,
]:
    logger.info(
        "[VIDEO] load_caption_segments ENTER file=%s",
        video_path.name,
    )

    sidecar_caption = find_sidecar_caption(
        video_path
    )

    logger.info(
        "[VIDEO] sidecar check completed result=%s",
        sidecar_caption,
    )

    if sidecar_caption:
        caption_text = sidecar_caption.read_text(
            encoding="utf-8"
        )

        segments = _parse_caption_text(
            caption_text
        )

        logger.info(
            "[VIDEO] sidecar captions parsed segments=%d",
            len(segments),
        )

        if segments:
            return segments, "sidecar"

    logger.info(
        "[VIDEO] starting embedded caption extraction"
    )

    embedded_caption = extract_embedded_captions(
        str(video_path)
    )

    logger.info(
        "[VIDEO] embedded caption extraction returned"
    )

    if embedded_caption:
        segments = _parse_caption_text(
            embedded_caption
        )

        if segments:
            return segments, "embedded"

    logger.info(
        "[VIDEO] no captions found, returning empty"
    )

    return [], None

def _caption_segment_to_element(
    segment: TranscriptSegment,
    document_id: str,
    file_name: str,
    segment_index: int,
    transcript_source: str,
) -> SourceElement:
    return SourceElement(
        element_id=(
            f"transcript-segment-"
            f"{segment_index}"
        ),
        document_id=document_id,
        element_type="transcript",
        content=TextContent(
            text=segment.text
        ),
        metadata={
            "file_name": file_name,
            "content_type": "transcript",
            "transcript_source": transcript_source,
            "segment_index": segment_index,
            "start_seconds": (
                segment.start_seconds
            ),
            "end_seconds": (
                segment.end_seconds
            ),
        },
    )


def create_whisper_transcript(
    video_path: str,
    output_dir: str,
    document_id: str,
    file_name: str,
    whisper_transcriber: WhisperTranscriber,
) -> list[SourceElement]:
    whisper_started_at = time.perf_counter()

    logger.info(
        "[VIDEO] Whisper transcription started file=%s",
        file_name,
    )

    transcript_segments = (
        whisper_transcriber.transcribe_segments(
            video_path
        )
    )

    logger.info(
        "[VIDEO] Whisper transcription completed "
        "duration=%.2fs segments=%d",
        time.perf_counter() - whisper_started_at,
        len(transcript_segments),
    )

    return [
        _caption_segment_to_element(
            segment=segment,
            document_id=document_id,
            file_name=file_name,
            segment_index=index,
            transcript_source="whisper",
        )
        for index, segment in enumerate(
            transcript_segments,
            start=1,
        )
    ]

def _frame_documents_to_elements(
    frame_documents: list[SourceDocument],
) -> list[SourceElement]:
    elements: list[SourceElement] = []

    for index, document in enumerate(
        frame_documents,
        start=1,
    ):
        metadata = (
            document.metadata.copy()
        )

        metadata[
            "element_index"
        ] = index

        elements.append(
            SourceElement(
                element_id=f"frame-{index}",
                document_id=document.document_id,
                element_type="video_frame",
                content=document.content,
                metadata=metadata,
            )
        )

    return elements


def load_video(
    file_path: str,
    document_id: str,
    output_dir: str,
    whisper_transcriber: WhisperTranscriber | None = None,
    frame_extractor: FrameExtractor | None = None,
    whisper_transcript_creator=None,
) -> IngestionResult:

    if frame_extractor is None:
        frame_extractor = extract_key_frames

    if whisper_transcript_creator is None:
        whisper_transcript_creator = create_whisper_transcript

    video_path = Path(file_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {file_path}"
        )

    if not video_path.is_file():
        raise ValueError(
            f"Video path is not a file: {file_path}"
        )

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if frame_extractor is None:
        frame_extractor = extract_key_frames

    elements: list[SourceElement] = []
    relationships: list[
        ElementRelationship
    ] = []

    # ---------------------------------------------------------
    # Root video
    # ---------------------------------------------------------
    root = SourceElement(
        element_id="document",
        document_id=document_id,
        element_type="video",
        content=VideoContent(
            path=str(video_path)
        ),
        metadata={
            "file_name": video_path.name,
            "content_type": "video",
        },
    )

    elements.append(root)

    # ---------------------------------------------------------
    # Captions first
    # ---------------------------------------------------------
    caption_segments, caption_source = (
        load_caption_segments(
            video_path
        )
    )

    if caption_segments:
        transcript_elements = [
            _caption_segment_to_element(
                segment=segment,
                document_id=document_id,
                file_name=video_path.name,
                segment_index=index,
                transcript_source=(
                    caption_source
                    or "caption"
                ),
            )
            for index, segment in enumerate(
                caption_segments,
                start=1,
            )
        ]

    else:
        if whisper_transcriber is None:
            whisper_transcriber = WhisperTranscriber()
    
        whisper_started_at = time.perf_counter()

        logger.info(
            "[VIDEO] transcription started file=%s",
            video_path.name,
        )

        transcript_elements = whisper_transcript_creator(
            video_path=str(video_path),
            output_dir=str(output_path),
            document_id=document_id,
            file_name=video_path.name,
            whisper_transcriber=whisper_transcriber,
        )

        logger.info(
            "[VIDEO] transcription completed duration=%.2fs segments=%d",
            time.perf_counter() - whisper_started_at,
            len(transcript_elements),
        )

    elements.extend(
        transcript_elements
    )

    for element in transcript_elements:
        relationships.append(
            ElementRelationship(
                source_element_id="document",
                relationship_type="contains",
                target_element_id=(
                    element.element_id
                ),
            )
        )

    # ---------------------------------------------------------
    # Intelligent frames
    # ---------------------------------------------------------
    frame_output_dir = (
        output_path
        / f"{document_id}_frames"
    )

    frames_started_at = time.perf_counter()

    logger.info(
        "[VIDEO] frame extraction started file=%s",
        video_path.name,
    )

    frame_documents = frame_extractor(
        str(video_path),
        str(frame_output_dir),
        document_id,
    )

    logger.info(
        "[VIDEO] frame extraction completed duration=%.2fs frames=%d",
        time.perf_counter() - frames_started_at,
        len(frame_documents),
    )

    frame_elements = (
        _frame_documents_to_elements(
            frame_documents
        )
    )

    elements.extend(
        frame_elements
    )

    for element in frame_elements:
        relationships.append(
            ElementRelationship(
                source_element_id="document",
                relationship_type="contains",
                target_element_id=(
                    element.element_id
                ),
            )
        )

    # ---------------------------------------------------------
    # Transcript ↔ frame relationships
    # ---------------------------------------------------------
    relationship_started_at = time.perf_counter()

    logger.info("[VIDEO] temporal relationship resolution started")

    temporal_relationships = resolve_temporal_relationships(
        elements
    )

    relationships.extend(temporal_relationships)

    logger.info(
        "[VIDEO] temporal relationship resolution completed duration=%.2fs relationships=%d",
        time.perf_counter() - relationship_started_at,
        len(temporal_relationships),
    )

    return IngestionResult(
        document_id=document_id,
        elements=elements,
        relationships=relationships,
    )