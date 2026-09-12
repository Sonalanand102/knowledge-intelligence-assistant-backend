from pathlib import Path
import subprocess


def extract_audio(
    video_path: str,
    output_dir: str,
) -> str:
    """
    Extract audio from a video and save it as a WAV file.

    Returns the path to the extracted audio.
    """

    video_path = Path(video_path)
    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_path = output_path / f"{video_path.stem}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Audio extraction failed:\n{result.stderr}"
        )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file was not created: {audio_path}"
        )

    return str(audio_path)