# from pathlib import Path

# from backend.app.ingestion.models.source_document import SourceDocument
# from backend.app.ingestion.models.content import TextContent

# def load_txt(
#     file_path: str,
#     document_id: str,
#     file_name: str,
# ) -> list[SourceDocument]:

#     path = Path(file_path)

#     content = path.read_text(encoding="utf-8")

#     if not content.strip():
#         return []


#     return [SourceDocument(content=TextContent(text=content), document_id=document_id, source_type="txt", metadata={"file_name": file_name})]

from pathlib import Path

from backend.app.ingestion.loaders._utils import (
    wrap_source_document,
)
from backend.app.ingestion.models.content import (
    TextContent,
)
from backend.app.ingestion.models.source_document import (
    SourceDocument,
)


def load_txt(
    file_path: str,
    document_id: str,
):
    file = Path(file_path)

    if not file.exists():
        raise FileNotFoundError(
            f"Text file not found: {file_path}"
        )

    if not file.is_file():
        raise ValueError(
            f"Text path is not a file: {file_path}"
        )

    if file.suffix.lower() != ".txt":
        raise ValueError(
            f"Unsupported text format: {file.suffix}"
        )

    text = file.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return wrap_source_document(
            SourceDocument(
                document_id=document_id,
                source_type="txt",
                content=TextContent(
                    text=""
                ),
                metadata={
                    "file_name": file.name,
                    "content_type": "text",
                },
            ),
            "text",
        )

    return wrap_source_document(
        SourceDocument(
            document_id=document_id,
            source_type="txt",
            content=TextContent(
                text=text
            ),
            metadata={
                "file_name": file.name,
                "content_type": "text",
            },
        ),
        "text",
    )