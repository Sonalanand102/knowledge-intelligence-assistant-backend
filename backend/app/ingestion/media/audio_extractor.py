from pathlib import Path
import logging
import subprocess
import time


logger = logging.getLogger(__name__)


AUDIO_EXTRACTION_TIMEOUT_SECONDS = 120


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

    audio_path = (
        output_path / f"{video_path.stem}.wav"
    )

    command = [
        "ffmpeg",
        "-nostdin",
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

    logger.info(
        "[AUDIO_EXTRACTOR] started video=%s output=%s",
        video_path.name,
        audio_path,
    )

    started_at = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=AUDIO_EXTRACTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started_at

        logger.error(
            "[AUDIO_EXTRACTOR] timed out "
            "duration=%.2fs video=%s",
            duration,
            video_path,
        )

        raise RuntimeError(
            "Audio extraction timed out after "
            f"{AUDIO_EXTRACTION_TIMEOUT_SECONDS} seconds: "
            f"{video_path}"
        ) from exc

    duration = time.perf_counter() - started_at

    if result.returncode != 0:
        logger.error(
            "[AUDIO_EXTRACTOR] failed "
            "duration=%.2fs stderr=%s",
            duration,
            result.stderr[-2000:],
        )

        raise RuntimeError(
            f"Audio extraction failed:\n{result.stderr}"
        )

    if not audio_path.exists():
        logger.error(
            "[AUDIO_EXTRACTOR] output missing "
            "duration=%.2fs output=%s",
            duration,
            audio_path,
        )

        raise FileNotFoundError(
            f"Audio file was not created: {audio_path}"
        )

    logger.info(
        "[AUDIO_EXTRACTOR] completed "
        "duration=%.2fs output=%s size_mb=%.2f",
        duration,
        audio_path,
        audio_path.stat().st_size / (1024 * 1024),
    )

    return str(audio_path)