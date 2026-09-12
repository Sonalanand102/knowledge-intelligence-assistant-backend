# from faster_whisper import WhisperModel


# class WhisperTranscriber:
#     def __init__(
#         self,
#         model_size: str = "small",
#         device: str = "cpu",
#         compute_type: str = "int8",
#     ):
#         self.model = WhisperModel(
#             model_size,
#             device=device,
#             compute_type=compute_type,
#         )

#     def transcribe(self, audio_path: str) -> str:
#         segments, _ = self.model.transcribe(
#             audio_path,
#             vad_filter=True,
#         )

#         transcript_parts = []

#         for segment in segments:
#             text = segment.text.strip()

#             if not text:
#                 continue

#             transcript_parts.append(
#                 f"[{segment.start:.2f}s] {text}"
#             )

#         return "\n".join(transcript_parts)

from __future__ import annotations

from dataclasses import dataclass

from faster_whisper import WhisperModel


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
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe_segments(
        self,
        audio_path: str,
    ) -> list[TranscriptSegment]:
        segments, _ = self.model.transcribe(
            audio_path,
            vad_filter=True,
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