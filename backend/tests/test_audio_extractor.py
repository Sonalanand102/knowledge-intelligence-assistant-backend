from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.ingestion.media.audio_extractor import extract_audio


@patch(
    "backend.app.ingestion.media.audio_extractor.subprocess.run"
)
def test_extract_audio(mock_run, tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.touch()

    output_dir = tmp_path / "audio"

    def create_audio_file(*args, **kwargs):
        audio_path = output_dir / "sample.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path.touch()

        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stderr": "",
            },
        )()

    mock_run.side_effect = create_audio_file

    result = extract_audio(
        video_path=str(video_path),
        output_dir=str(output_dir),
    )

    assert result.endswith("sample.wav")
    assert Path(result).exists()

    mock_run.assert_called_once()


@patch(
    "backend.app.ingestion.media.audio_extractor.subprocess.run"
)
def test_extract_audio_failure(mock_run, tmp_path):
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "FFmpeg error"

    video_path = tmp_path / "sample.mp4"
    video_path.touch()

    with pytest.raises(RuntimeError, match="Audio extraction failed"):
        extract_audio(
            video_path=str(video_path),
            output_dir=str(tmp_path / "audio"),
        )