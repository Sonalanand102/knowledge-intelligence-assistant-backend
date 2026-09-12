from pathlib import Path

import cv2

from backend.app.ingestion.models.content import ImageContent
from backend.app.ingestion.models.source_document import SourceDocument


def calculate_frame_difference(
    previous_frame,
    current_frame,
) -> float:
    """
    Calculate the average visual difference between two frames.

    Higher value = larger visual change.
    """

    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(previous_gray, current_gray)

    return float(difference.mean())


def extract_key_frames(
    video_path: str,
    output_dir: str,
    document_id: str,
    difference_threshold: float = 25.0,
    minimum_interval_seconds: float = 2.0,
    analysis_fps: float = 2.0,
) -> list[SourceDocument]:
    """
    Extract representative frames from a video based on
    significant visual changes.

    The video is sampled at `analysis_fps` instead of processing
    every decoded frame.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        video.release()
        raise ValueError("Could not determine video FPS.")

    frame_interval = max(int(fps / analysis_fps), 1)

    documents: list[SourceDocument] = []

    previous_frame = None
    last_saved_timestamp = -minimum_interval_seconds
    frame_number = 0

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            if frame_number % frame_interval != 0:
                frame_number += 1
                continue

            timestamp_seconds = frame_number / fps

            # Always keep the first analyzed frame.
            if previous_frame is None:
                frame_path = output_path / (
                    f"{document_id}_frame_{timestamp_seconds:.2f}s.jpg"
                )

                cv2.imwrite(str(frame_path), frame)

                documents.append(
                    SourceDocument(
                        document_id=document_id,
                        source_type="youtube",
                        content=ImageContent(path=str(frame_path)),
                        metadata={
                            "content_type": "video_frame",
                            "timestamp_seconds": timestamp_seconds,
                            "frame_path": str(frame_path),
                            "selection_method": "initial_frame",
                        },
                    )
                )

                previous_frame = frame
                last_saved_timestamp = timestamp_seconds

                frame_number += 1
                continue

            difference = calculate_frame_difference(
                previous_frame,
                frame,
            )

            enough_time_passed = (
                timestamp_seconds - last_saved_timestamp
                >= minimum_interval_seconds
            )

            if difference >= difference_threshold and enough_time_passed:
                frame_path = output_path / (
                    f"{document_id}_frame_{timestamp_seconds:.2f}s.jpg"
                )

                cv2.imwrite(str(frame_path), frame)

                documents.append(
                    SourceDocument(
                        document_id=document_id,
                        source_type="youtube",
                        content=ImageContent(path=str(frame_path)),
                        metadata={
                            "content_type": "video_frame",
                            "timestamp_seconds": timestamp_seconds,
                            "frame_path": str(frame_path),
                            "selection_method": "scene_change",
                            "frame_difference": difference,
                        },
                    )
                )

                last_saved_timestamp = timestamp_seconds

            previous_frame = frame
            frame_number += 1

    finally:
        video.release()

    return documents