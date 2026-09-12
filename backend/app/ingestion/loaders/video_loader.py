# from pathlib import Path
# import subprocess

# from backend.app.ingestion.media.audio_extractor import extract_audio
# from backend.app.ingestion.media.video_frame_extractor import (
#     extract_key_frames,
# )
# from backend.app.ingestion.media.transcribers.whisper_transcriber import (
#     WhisperTranscriber,
# )
# from backend.app.ingestion.models.content import (
#     AudioContent,
#     ImageContent,
#     TextContent,
#     VideoContent,
# )
# from backend.app.ingestion.models.source_document import SourceDocument


# def find_sidecar_caption(video_path: Path) -> Path | None:
#     """
#     Look for a subtitle file next to the video.
#     """

#     for extension in (".srt", ".vtt"):
#         caption_path = video_path.with_suffix(extension)

#         if caption_path.exists():
#             return caption_path

#     return None


# def extract_embedded_captions(
#     video_path: str,
# ) -> str | None:
#     """
#     Extract the first embedded subtitle stream using FFmpeg.
#     Returns None if the video has no subtitle stream.
#     """

#     command = [
#         "ffmpeg",
#         "-i",
#         video_path,
#         "-map",
#         "0:s:0",
#         "-f",
#         "webvtt",
#         "pipe:1",
#     ]

#     result = subprocess.run(
#         command,
#         capture_output=True,
#         text=True,
#     )

#     if result.returncode != 0:
#         return None

#     captions = result.stdout.strip()

#     return captions or None


# def load_caption_text(
#     video_path: Path,
# ) -> tuple[str | None, str | None]:
#     """
#     Try sidecar captions first, then embedded captions.

#     Returns:
#         (caption_text, caption_source)
#     """

#     sidecar_caption = find_sidecar_caption(video_path)

#     if sidecar_caption:
#         text = sidecar_caption.read_text(
#             encoding="utf-8"
#         ).strip()

#         if text:
#             return text, "sidecar"

#     embedded_caption = extract_embedded_captions(
#         str(video_path)
#     )

#     if embedded_caption:
#         return embedded_caption, "embedded"

#     return None, None


# def create_whisper_transcript(
#     video_path: str,
#     output_dir: str,
#     document_id: str,
#     file_name: str,
#     whisper_transcriber: WhisperTranscriber,
# ) -> SourceDocument:

#     audio_dir = Path(output_dir) / "audio"

#     audio_path = extract_audio(
#         video_path=video_path,
#         output_dir=str(audio_dir),
#     )

#     transcript = whisper_transcriber.transcribe(
#         audio_path
#     )

#     return SourceDocument(
#         document_id=document_id,
#         source_type="video",
#         content=TextContent(text=transcript),
#         metadata={
#             "file_name": file_name,
#             "content_type": "transcript",
#             "transcript_source": "whisper",
#         },
#     )


# def load_video(
#     file_path: str,
#     document_id: str,
#     output_dir: str,
#     whisper_transcriber: WhisperTranscriber,
# ) -> list[SourceDocument]:

#     video_path = Path(file_path)

#     if not video_path.exists():
#         raise FileNotFoundError(
#             f"Video file not found: {file_path}"
#         )

#     if not video_path.is_file():
#         raise ValueError(
#             f"Video path is not a file: {file_path}"
#         )

#     output_path = Path(output_dir)
#     output_path.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     documents: list[SourceDocument] = []

#     # --------------------------------------------------
#     # 1. Original video
#     # --------------------------------------------------

#     video_document = SourceDocument(
#         document_id=document_id,
#         source_type="video",
#         content=VideoContent(
#             path=str(video_path)
#         ),
#         metadata={
#             "file_name": video_path.name,
#             "content_type": "video",
#         },
#     )

#     documents.append(video_document)

#     # --------------------------------------------------
#     # 2. Captions first, Whisper fallback
#     # --------------------------------------------------

#     caption_text, caption_source = load_caption_text(
#         video_path
#     )

#     if caption_text:
#         caption_document = SourceDocument(
#             document_id=document_id,
#             source_type="video",
#             content=TextContent(
#                 text=caption_text
#             ),
#             metadata={
#                 "file_name": video_path.name,
#                 "content_type": "transcript",
#                 "transcript_source": caption_source,
#             },
#         )

#         documents.append(caption_document)

#     else:
#         transcript_document = create_whisper_transcript(
#             video_path=str(video_path),
#             output_dir=str(output_path),
#             document_id=document_id,
#             file_name=video_path.name,
#             whisper_transcriber=whisper_transcriber,
#         )

#         documents.append(transcript_document)

#     # --------------------------------------------------
#     # 3. Intelligent key frames
#     # --------------------------------------------------

#     frame_output_dir = (
#         output_path / f"{document_id}_frames"
#     )

#     frame_documents = extract_key_frames(
#         video_path=str(video_path),
#         output_dir=str(frame_output_dir),
#         document_id=document_id,
#     )

#     documents.extend(frame_documents)

#     return documents

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

from backend.app.ingestion.media.audio_extractor import (
    extract_audio,
)
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
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-map",
        "0:s:0",
        "-f",
        "webvtt",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    captions = result.stdout.strip()

    return captions or None


def load_caption_segments(
    video_path: Path,
) -> tuple[
    list[TranscriptSegment],
    str | None,
]:
    sidecar_caption = (
        find_sidecar_caption(
            video_path
        )
    )

    if sidecar_caption:
        caption_text = (
            sidecar_caption.read_text(
                encoding="utf-8"
            )
        )

        segments = _parse_caption_text(
            caption_text
        )

        if segments:
            return (
                segments,
                "sidecar",
            )

    embedded_caption = (
        extract_embedded_captions(
            str(video_path)
        )
    )

    if embedded_caption:
        segments = _parse_caption_text(
            embedded_caption
        )

        if segments:
            return (
                segments,
                "embedded",
            )

    return (
        [],
        None,
    )


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
    audio_dir = (
        Path(output_dir)
        / "audio"
    )

    audio_path = extract_audio(
        video_path=video_path,
        output_dir=str(audio_dir),
    )

    segments = (
        whisper_transcriber
        .transcribe_segments(
            audio_path
        )
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
            segments,
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
    whisper_transcriber: WhisperTranscriber,
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
        transcript_elements = whisper_transcript_creator(
            video_path=str(video_path),
            output_dir=str(output_path),
            document_id=document_id,
            file_name=video_path.name,
            whisper_transcriber=whisper_transcriber,
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

    frame_documents = frame_extractor(
        str(video_path),
        str(frame_output_dir),
        document_id,
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
    relationships.extend(
        resolve_temporal_relationships(
            elements
        )
    )

    return IngestionResult(
        document_id=document_id,
        elements=elements,
        relationships=relationships,
    )