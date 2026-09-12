# from pathlib import Path
# from urllib.parse import parse_qs, urlparse

# from yt_dlp import YoutubeDL
# from youtube_transcript_api import YouTubeTranscriptApi

# from backend.app.ingestion.media.audio_extractor import extract_audio
# from backend.app.ingestion.media.video_frame_extractor import (
#     extract_key_frames,
# )
# from backend.app.ingestion.media.transcribers.whisper_transcriber import (
#     WhisperTranscriber,
# )
# from backend.app.ingestion.models.content import TextContent
# from backend.app.ingestion.models.source_document import SourceDocument


# def extract_video_id(url: str) -> str:
#     parsed_url = urlparse(url)

#     if parsed_url.hostname in {"www.youtube.com", "youtube.com"}:
#         query = parse_qs(parsed_url.query)

#         if "v" in query and query["v"]:
#             return query["v"][0]

#         if parsed_url.path.startswith("/shorts/"):
#             video_id = parsed_url.path.split("/shorts/")[1].split("/")[0]

#             if video_id:
#                 return video_id

#     if parsed_url.hostname == "youtu.be":
#         video_id = parsed_url.path.strip("/").split("/")[0]

#         if video_id:
#             return video_id

#     raise ValueError(f"Invalid YouTube URL: {url}")


# def load_youtube_transcript(
#     url: str,
#     document_id: str,
#     file_name: str,
# ) -> SourceDocument:
#     video_id = extract_video_id(url)

#     transcript = YouTubeTranscriptApi().fetch(video_id)

#     transcript_parts = [
#         f"[{item.start:.2f}s] {item.text}"
#         for item in transcript
#         if item.text.strip()
#     ]

#     text = "\n".join(transcript_parts)

#     return SourceDocument(
#         document_id=document_id,
#         source_type="youtube",
#         content=TextContent(text=text),
#         metadata={
#             "file_name": file_name,
#             "content_type": "transcript",
#             "transcript_source": "youtube_captions",
#             "video_id": video_id,
#             "url": url,
#         },
#     )


# def download_youtube_video(
#     url: str,
#     output_dir: str,
# ) -> str:
#     output_path = Path(output_dir)
#     output_path.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     ydl_opts = {
#         "format": "bv*+ba/b",
#         "outtmpl": str(output_path / "%(id)s.%(ext)s"),
#         "merge_output_format": "mp4",
#         "quiet": True,
#     }

#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(
#             url,
#             download=True,
#         )

#         video_id = info["id"]

#         prepared_path = Path(
#             ydl.prepare_filename(info)
#         )

#     mp4_path = output_path / f"{video_id}.mp4"

#     if mp4_path.exists():
#         return str(mp4_path)

#     if prepared_path.exists():
#         return str(prepared_path)

#     raise FileNotFoundError(
#         f"Downloaded video was not found: {video_id}"
#     )


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
#         source_type="youtube",
#         content=TextContent(text=transcript),
#         metadata={
#             "file_name": file_name,
#             "content_type": "transcript",
#             "transcript_source": "whisper",
#         },
#     )


# def load_youtube(
#     url: str,
#     document_id: str,
#     file_name: str,
#     output_dir: str,
#     whisper_transcriber: WhisperTranscriber,
# ) -> list[SourceDocument]:

#     video_id = extract_video_id(url)

#     documents: list[SourceDocument] = []

#     # Download only once.
#     video_path = download_youtube_video(
#         url=url,
#         output_dir=output_dir,
#     )

#     # --------------------------------
#     # Transcript
#     # --------------------------------

#     try:
#         transcript_document = load_youtube_transcript(
#             url=url,
#             document_id=document_id,
#             file_name=file_name,
#         )

#     except Exception:
#         transcript_document = create_whisper_transcript(
#             video_path=video_path,
#             output_dir=output_dir,
#             document_id=document_id,
#             file_name=file_name,
#             whisper_transcriber=whisper_transcriber,
#         )

#     documents.append(transcript_document)

#     # --------------------------------
#     # Video frames
#     # --------------------------------

#     frame_output_dir = (
#         Path(output_dir) / f"{video_id}_frames"
#     )

#     frame_documents = extract_key_frames(
#         video_path=video_path,
#         output_dir=str(frame_output_dir),
#         document_id=document_id,
#     )

#     documents.extend(frame_documents)

#     return documents

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from youtube_transcript_api import (
    YouTubeTranscriptApi,
)

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
    ImageContent,
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


TranscriptFetcher = Callable[
    [str],
    Iterable,
]

VideoDownloader = Callable[
    [str, str],
    str,
]

FrameExtractor = Callable[
    [str, str, str],
    list[SourceDocument],
]


def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)

    if parsed_url.hostname in {
        "www.youtube.com",
        "youtube.com",
    }:
        query = parse_qs(parsed_url.query)

        if "v" in query and query["v"]:
            return query["v"][0]

        if parsed_url.path.startswith("/shorts/"):
            video_id = (
                parsed_url.path
                .split("/shorts/")[1]
                .split("/")[0]
            )

            if video_id:
                return video_id

    if parsed_url.hostname == "youtu.be":
        video_id = (
            parsed_url.path
            .strip("/")
            .split("/")[0]
        )

        if video_id:
            return video_id

    raise ValueError(
        f"Invalid YouTube URL: {url}"
    )


def _youtube_caption_to_element(
    item,
    document_id: str,
    file_name: str,
    segment_index: int,
    video_id: str,
    url: str,
) -> SourceElement:
    start_seconds = float(
        getattr(item, "start", 0.0)
    )

    duration_seconds = float(
        getattr(item, "duration", 0.0)
        or 0.0
    )

    end_seconds = (
        start_seconds
        + duration_seconds
    )

    text = str(
        getattr(item, "text", "")
    ).strip()

    return SourceElement(
        element_id=(
            f"transcript-segment-"
            f"{segment_index}"
        ),
        document_id=document_id,
        element_type="transcript",
        content=TextContent(
            text=text
        ),
        metadata={
            "file_name": file_name,
            "content_type": "transcript",
            "transcript_source": "youtube_captions",
            "video_id": video_id,
            "url": url,
            "segment_index": segment_index,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        },
    )


def load_youtube_transcript(
    url: str,
    document_id: str,
    file_name: str,
    transcript_fetcher: TranscriptFetcher | None = None,
) -> list[SourceElement]:
    video_id = extract_video_id(url)

    if transcript_fetcher is None:
        api = YouTubeTranscriptApi()

        def transcript_fetcher(
            current_video_id: str,
        ):
            return api.fetch(
                current_video_id
            )

    transcript_items = transcript_fetcher(
        video_id
    )

    elements: list[SourceElement] = []

    for index, item in enumerate(
        transcript_items,
        start=1,
    ):
        text = str(
            getattr(item, "text", "")
        ).strip()

        if not text:
            continue

        elements.append(
            _youtube_caption_to_element(
                item=item,
                document_id=document_id,
                file_name=file_name,
                segment_index=index,
                video_id=video_id,
                url=url,
            )
        )

    return elements


def download_youtube_video(
    url: str,
    output_dir: str,
) -> str:
    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(
            output_path / "%(id)s.%(ext)s"
        ),
        "merge_output_format": "mp4",
        "quiet": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            url,
            download=True,
        )

        video_id = info["id"]

        prepared_path = Path(
            ydl.prepare_filename(info)
        )

    mp4_path = (
        output_path
        / f"{video_id}.mp4"
    )

    if mp4_path.exists():
        return str(mp4_path)

    if prepared_path.exists():
        return str(prepared_path)

    raise FileNotFoundError(
        "Downloaded video was not "
        f"found: {video_id}"
    )


def _transcript_segment_to_element(
    segment: TranscriptSegment,
    document_id: str,
    file_name: str,
    segment_index: int,
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
            "transcript_source": "whisper",
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
        _transcript_segment_to_element(
            segment=segment,
            document_id=document_id,
            file_name=file_name,
            segment_index=index,
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
        element_id = f"frame-{index}"

        metadata = (
            document.metadata.copy()
        )

        metadata["element_index"] = index

        elements.append(
            SourceElement(
                element_id=element_id,
                document_id=document.document_id,
                element_type="video_frame",
                content=ImageContent(
                    path=document.content.path
                ),
                metadata=metadata,
            )
        )

    return elements


def load_youtube(
    url: str,
    document_id: str,
    file_name: str,
    output_dir: str,
    whisper_transcriber: WhisperTranscriber,
    transcript_fetcher: TranscriptFetcher | None = None,
    video_downloader: VideoDownloader | None = None,
    frame_extractor: FrameExtractor | None = None,
    whisper_transcript_creator=None,
) -> IngestionResult:

    if video_downloader is None:
        video_downloader = download_youtube_video

    if frame_extractor is None:
        frame_extractor = extract_key_frames

    if whisper_transcript_creator is None:
        whisper_transcript_creator = create_whisper_transcript

    video_id = extract_video_id(url)

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if video_downloader is None:
        video_downloader = (
            download_youtube_video
        )

    if frame_extractor is None:
        frame_extractor = (
            extract_key_frames
        )

    video_path = video_downloader(
        url,
        output_dir,
    )

    elements: list[SourceElement] = []
    relationships: list[
        ElementRelationship
    ] = []

    # ---------------------------------------------------------
    # Root YouTube source
    # ---------------------------------------------------------
    root = SourceElement(
        element_id="document",
        document_id=document_id,
        element_type="youtube",
        content=VideoContent(
            path=video_path
        ),
        metadata={
            "file_name": file_name,
            "content_type": "youtube",
            "video_id": video_id,
            "url": url,
        },
    )

    elements.append(root)

    # ---------------------------------------------------------
    # Captions → Whisper fallback
    # ---------------------------------------------------------
    try:
        transcript_elements = (
            load_youtube_transcript(
                url=url,
                document_id=document_id,
                file_name=file_name,
                transcript_fetcher=(
                    transcript_fetcher
                ),
            )
        )
    except Exception:
        transcript_elements = whisper_transcript_creator(
            video_path=video_path,
            output_dir=str(output_path),
            document_id=document_id,
            file_name=file_name,
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
    # Key frames
    # ---------------------------------------------------------
    frame_output_dir = (
        output_path
        / f"{video_id}_frames"
    )

    frame_documents = frame_extractor(
        video_path,
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
    # Transcript ↔ frame temporal relationships
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