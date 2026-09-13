from backend.app.ingestion.loaders.youtube_loader import load_youtube
from backend.app.ingestion.media.transcribers.whisper_transcriber import (
    WhisperTranscriber,
)


YOUTUBE_URL = "https://www.youtube.com/watch?v=W6yQ88FpMOs"


whisper = WhisperTranscriber(
    model_size="small",
    device="cpu",
    compute_type="int8",
)


documents = load_youtube(
    url=YOUTUBE_URL,
    document_id="youtube-smoke-test",
    file_name="youtube-smoke-test",
    output_dir="tmp/youtube",
    whisper_transcriber=whisper,
)


print(f"\nTotal documents: {len(documents.elements)}")

for document in documents.elements:
    print("\n---")

    print(
        "Content type:",
        document.metadata.get("content_type"),
    )

    if document.metadata.get("content_type") == "transcript":
        print(
            "Transcript source:",
            document.metadata.get("transcript_source"),
        )

        print("\nTranscript preview:")
        print(document.content.text[:1000])

    elif document.metadata.get("content_type") == "video_frame":
        print(
            "Timestamp:",
            document.metadata["timestamp_seconds"],
        )

        print(
            "Frame:",
            document.metadata["frame_path"],
        )