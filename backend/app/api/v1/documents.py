from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from backend.app.api.dependencies import (
    get_document_management_service,
    get_document_upload_service,
)

from backend.app.ingestion.loader_registry import (
    UnsupportedFileTypeError,
)

from backend.app.schemas.document_management import (
    ChatDocumentListResponse,
    DocumentResponse,
    DocumentRetryResponse,
)

from backend.app.schemas.document_upload import (
    MultiDocumentUploadResponse,
)

from backend.app.services.document_management_service import (
    ChatNotFoundError as ManagementChatNotFoundError,
    DocumentManagementService,
    DocumentNotFoundError,
    DocumentNotRetryableError,
)

from backend.app.services.document_upload_service import (
    ChatNotFoundError as UploadChatNotFoundError,
    DocumentUploadService,
)


# ============================================================
# CHAT-SCOPED DOCUMENT ROUTER
# ============================================================

router = APIRouter(
    prefix="/chats",
    tags=["documents"],
)


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

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
        get_document_upload_service,
    ),
) -> MultiDocumentUploadResponse:
    try:
        return await upload_service.upload_documents(
            chat_id=chat_id,
            files=files,
        )

    except UploadChatNotFoundError as exc:
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


# ============================================================
# LIST CHAT DOCUMENTS
# ============================================================

@router.get(
    "/{chat_id}/documents",
    response_model=ChatDocumentListResponse,
)
async def list_chat_documents(
    chat_id: str,
    document_service: DocumentManagementService = Depends(
        get_document_management_service,
    ),
) -> ChatDocumentListResponse:
    try:
        return await document_service.list_chat_documents(
            chat_id=chat_id,
        )

    except ManagementChatNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# ============================================================
# DETACH DOCUMENT FROM CHAT
# ============================================================

@router.delete(
    "/{chat_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_document(
    chat_id: str,
    document_id: str,
    document_service: DocumentManagementService = Depends(
        get_document_management_service,
    ),
) -> None:
    try:
        await document_service.detach_document(
            chat_id=chat_id,
            document_id=document_id,
        )

    except (
        ManagementChatNotFoundError,
        DocumentNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# ============================================================
# GLOBAL DOCUMENT ROUTER
# ============================================================

document_router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


# ============================================================
# GET DOCUMENT
# ============================================================

@document_router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_document(
    document_id: str,
    document_service: DocumentManagementService = Depends(
        get_document_management_service,
    ),
) -> DocumentResponse:
    try:
        return await document_service.get_document(
            document_id=document_id,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# ============================================================
# RETRY DOCUMENT
# ============================================================

@document_router.post(
    "/{document_id}/retry",
    response_model=DocumentRetryResponse,
)
async def retry_document(
    document_id: str,
    document_service: DocumentManagementService = Depends(
        get_document_management_service,
    ),
) -> DocumentRetryResponse:
    try:
        return await document_service.retry_document(
            document_id=document_id,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except DocumentNotRetryableError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc