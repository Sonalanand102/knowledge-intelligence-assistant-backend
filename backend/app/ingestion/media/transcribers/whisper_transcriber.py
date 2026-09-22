from __future__ import annotations

from dataclasses import dataclass

from faster_whisper import WhisperModel

import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        logger.info(
            "[WHISPER] model loading started model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )

        started_at = time.perf_counter()

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        logger.info(
            "[WHISPER] model loading completed duration=%.2fs",
            time.perf_counter() - started_at,
        )
        
    def transcribe_segments(
        self,
        audio_path: str,
    ) -> list[TranscriptSegment]:

        started_at = time.perf_counter()

        logger.info(
            "[WHISPER] transcription started file=%s",
            audio_path,
        )

        segments, _ = self.model.transcribe(
            audio_path,
            vad_filter=True,
        )

        logger.info(
            "[WHISPER] transcription completed duration=%.2fs",
            time.perf_counter() - started_at,
        )

        transcript_segments = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            transcript_segments.append(
                TranscriptSegment(
                    start_seconds=float(
                        segment.start
                    ),
                    end_seconds=float(
                        segment.end
                    ),
                    text=text,
                )
            )

        return transcript_segments

    def transcribe(
        self,
        audio_path: str,
    ) -> str:
        segments = self.transcribe_segments(
            audio_path
        )

        return "\n".join(
            (
                f"[{segment.start_seconds:.2f}s] "
                f"{segment.text}"
            )
            for segment in segments
        )