"""
Knowledge Base API Router
All endpoints are org-scoped: /api/v1/orgs/{org_id}/knowledge-base/...
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import get_current_user
from api.db.models import User
from api.db.session import get_session
from api.knowledge_base.schemas import (
    DatasetListResponse,
    DatasetOut,
    FieldMappingRequest,
    FileMetaOut,
    UploadRequest,
    UploadResponse,
    ValidationReport,
)
from api.knowledge_base.service import KnowledgeBaseService

logger = logging.getLogger(__name__)

kb_router = APIRouter(
    prefix="/orgs/{org_id}/knowledge-base",
    tags=["knowledge-base"],
)


def _check_org_membership(user: User, org_id: str) -> None:
    """Verify user belongs to the org."""
    for m in user.memberships:
        if str(m.org_id) == org_id:
            return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have access to this organization",
    )

@kb_router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_files(
    org_id: str,
    body: UploadRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = KnowledgeBaseService(db)

    try:
        upload = await svc.create_upload(
            org_id=org_id,
            user_id=str(current_user.id),
            files=[f.model_dump() for f in body.files],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return UploadResponse(
        upload_id=str(upload.id),
        document_count=upload.document_count,
        file_metadata=[FileMetaOut(**fm) for fm in upload.file_metadata],
        preview=upload.raw_data[:5],
    )

@kb_router.post("/uploads/{upload_id}/validate", response_model=ValidationReport)
async def validate_upload(
    org_id: str,
    upload_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = KnowledgeBaseService(db)

    report = await svc.validate_upload(upload_id, org_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload session not found")

    return ValidationReport(**report)

@kb_router.post("/uploads/{upload_id}/field-mapping", response_model=ValidationReport)
async def apply_field_mapping(
    org_id: str,
    upload_id: str,
    body: FieldMappingRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = KnowledgeBaseService(db)

    report = await svc.apply_field_mapping(upload_id, org_id, body.mapping)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload session not found")

    return ValidationReport(**report)

@kb_router.post("/uploads/{upload_id}/ingest")
async def ingest(
    org_id: str,
    upload_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = KnowledgeBaseService(db)

    async def event_stream():
        async for event in svc.start_ingestion(upload_id, org_id, str(current_user.id)):
            event_type = event.get("event", "message")
            data = json.dumps(event.get("data", {}))
            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@kb_router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    org_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = KnowledgeBaseService(db)
    datasets = await svc.list_datasets(org_id)

    return DatasetListResponse(
        datasets=[
            DatasetOut(
                id=str(d.id),
                version_number=d.version_number,
                collection_name=d.collection_name,
                document_count=d.document_count,
                segment_count=d.segment_count,
                is_active=d.is_active,
                status=d.status,
                created_at=d.created_at.isoformat(),
            )
            for d in datasets
        ]
    )


@kb_router.post("/datasets/{dataset_id}/activate", response_model=DatasetOut)
async def activate_dataset(
    org_id: str,
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = KnowledgeBaseService(db)

    dataset = await svc.activate_dataset(dataset_id, org_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return DatasetOut(
        id=str(dataset.id),
        version_number=dataset.version_number,
        collection_name=dataset.collection_name,
        document_count=dataset.document_count,
        segment_count=dataset.segment_count,
        is_active=dataset.is_active,
        status=dataset.status,
        created_at=dataset.created_at.isoformat(),
    )


@kb_router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    org_id: str,
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    _check_org_membership(current_user, org_id)
    svc = KnowledgeBaseService(db)

    deleted = await svc.delete_dataset(dataset_id, org_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")