from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from backend.app.api.dependencies import get_document_upload_service
from backend.app.ingestion.loader_registry import UnsupportedFileTypeError
from backend.app.schemas.document_upload import MultiDocumentUploadResponse
from backend.app.services.document_upload_service import (
    ChatNotFoundError,
    DocumentUploadService,
)

router = APIRouter(
    prefix="/chats",
    tags=["documents"],
)


@router.post(
    "/{chat_id}/documents",
    response_model=MultiDocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_documents(
    chat_id: str,
    files: list[UploadFile] = File(
        ...,
        description="Upload one or more supported documents",
    ),
    upload_service: DocumentUploadService = Depends(
        get_document_upload_service
    ),
) -> MultiDocumentUploadResponse:
    try:
        return await upload_service.upload_documents(
            chat_id=chat_id,
            files=files,
        )

    except ChatNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc