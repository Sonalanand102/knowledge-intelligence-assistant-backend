from pathlib import Path

import cv2
import numpy as np

from backend.app.ingestion.media.video_frame_extractor import (
    calculate_frame_difference,
    extract_key_frames,
)


def create_test_video(
    video_path: Path,
    fps: int = 10,
) -> None:
    """
    Create a deterministic synthetic video:

    0-2 sec   -> black
    2-4 sec   -> white
    4-6 sec   -> gray
    """

    width = 320
    height = 240

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    black = np.zeros((height, width, 3), dtype=np.uint8)

    white = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    gray = np.full(
        (height, width, 3),
        128,
        dtype=np.uint8,
    )

    for _ in range(fps * 2):
        writer.write(black)

    for _ in range(fps * 2):
        writer.write(white)

    for _ in range(fps * 2):
        writer.write(gray)

    writer.release()


def test_calculate_frame_difference():
    black = np.zeros((100, 100, 3), dtype=np.uint8)

    white = np.full(
        (100, 100, 3),
        255,
        dtype=np.uint8,
    )

    difference = calculate_frame_difference(
        black,
        white,
    )

    assert difference == 255.0


def test_extract_key_frames(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    output_dir = tmp_path / "frames"

    create_test_video(video_path)

    documents = extract_key_frames(
        video_path=str(video_path),
        output_dir=str(output_dir),
        document_id="test-video",
        difference_threshold=20.0,
        minimum_interval_seconds=1.0,
        analysis_fps=2.0,
    )

    assert len(documents) >= 3

    assert documents[0].metadata["selection_method"] == "initial_frame"

    for document in documents:
        frame_path = document.metadata["frame_path"]

        assert Path(frame_path).exists()
        assert document.metadata["content_type"] == "video_frame"
        assert document.metadata["timestamp_seconds"] >= 0