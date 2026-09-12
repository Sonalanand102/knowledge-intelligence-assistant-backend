from backend.app.ingestion.media.transcribers.whisper_transcriber import (
    WhisperTranscriber,
)


AUDIO_PATH = "tmp/youtube/audio/harvard.wav"


transcriber = WhisperTranscriber(
    model_size="small",
    device="cpu",
    compute_type="int8",
)

transcript = transcriber.transcribe(AUDIO_PATH)

print("\n===== TRANSCRIPT =====\n")
print(transcript)