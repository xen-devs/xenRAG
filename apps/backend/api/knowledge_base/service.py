"""
Knowledge Base Service
Handles file parsing, validation, field mapping, and ingestion orchestration.
All operations are org-scoped.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ai_core.ingestion.pipeline import process_batch, process_for_vector_store
from ai_core.retrieval.stores.qdrant import QdrantVectorStore, copy_collection_points
from api.db.models.knowledge_base_dataset import KnowledgeBaseDataset
from api.db.models.knowledge_base_upload import KnowledgeBaseUpload

logger = logging.getLogger(__name__)

TARGET_FIELDS = {
    "text": "The main text content (review, feedback body, description)",
    "id": "Unique identifier for the document",
    "rating": "Numeric rating (optional)",
    "user_id": "User/customer identifier (optional)",
    "title": "Document title (optional)",
    "category": "Category or topic (optional)",
}

REQUIRED_TARGET_FIELDS = {"text"}

AUTO_FIELD_MAP: Dict[str, str] = {
    # text
    "text": "text",
    "content": "text",
    "body": "text",
    "review": "text",
    "feedback": "text",
    "comment": "text",
    "description": "text",
    "review_body": "text",
    "feedback_text": "text",
    "message": "text",
    # id
    "id": "id",
    "review_id": "id",
    "feedback_id": "id",
    "document_id": "id",
    "_id": "id",
    # rating
    "rating": "rating",
    "score": "rating",
    "stars": "rating",
    "overall": "rating",
    # user_id
    "user_id": "user_id",
    "customer_id": "user_id",
    "reviewer_id": "user_id",
    "author_id": "user_id",
    # title
    "title": "title",
    "summary": "title",
    "subject": "title",
    "headline": "title",
    # category
    "category": "category",
    "topic": "category",
    "type": "category",
    "product": "category",
    "asin": "category",
}

INGEST_BATCH_SIZE = 50
MAX_UPLOAD_FILE_BYTES = 100 * 1024 * 1024


def parse_file_content(filename: str, content: str) -> List[Dict[str, Any]]:
    """Parse JSON or JSONL file content into a list of records."""
    filename_lower = filename.lower()

    if filename_lower.endswith(".jsonl"):
        records = []
        for i, line in enumerate(content.strip().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {filename}: {e}")
        return records

    if filename_lower.endswith(".json"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    return val
            return [data]
        raise ValueError(f"Unexpected JSON structure in {filename}")

    raise ValueError(f"Unsupported file type: {filename}. Use .json or .jsonl")


def detect_fields(records: List[Dict[str, Any]]) -> List[str]:
    """Detect all unique fields across records."""
    fields: set[str] = set()
    for rec in records[:200]:  # sample first 200
        if isinstance(rec, dict):
            fields.update(rec.keys())
    return sorted(fields)


def auto_detect_mapping(source_fields: List[str]) -> Dict[str, str]:
    """Auto-detect field mapping from source fields to target fields."""
    mapping: Dict[str, str] = {}
    for field in source_fields:
        key = field.lower().strip()
        if key in AUTO_FIELD_MAP:
            target = AUTO_FIELD_MAP[key]
            if target not in mapping.values():
                mapping[field] = target
    return mapping


def apply_mapping(records: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Apply field mapping to records, producing documents with target field names."""
    mapped = []
    for rec in records:
        doc: Dict[str, Any] = {}
        for source, target in mapping.items():
            if source in rec:
                doc[target] = rec[source]
        for key, val in rec.items():
            if key not in mapping and key not in doc:
                doc[key] = val
        if "id" not in doc:
            doc["id"] = str(uuid.uuid4())
        mapped.append(doc)
    return mapped


def validate_records(
    records: List[Dict[str, Any]],
    mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate records against required fields. Returns a validation report."""
    source_fields = detect_fields(records)

    effective_mapping = mapping or auto_detect_mapping(source_fields)

    mapped_targets = set(effective_mapping.values())
    missing_required = REQUIRED_TARGET_FIELDS - mapped_targets

    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    if missing_required:
        errors.append({
            "type": "missing_required_fields",
            "message": f"No source field maps to required field(s): {', '.join(missing_required)}. "
            f"Use field mapping to assign a source field.",
            "fields": list(missing_required),
        })

    text_field = None
    for src, tgt in effective_mapping.items():
        if tgt == "text":
            text_field = src
            break

    empty_count = 0
    if text_field:
        for rec in records:
            val = rec.get(text_field, "")
            if not val or (isinstance(val, str) and not val.strip()):
                empty_count += 1
        if empty_count > 0:
            warnings.append({
                "type": "empty_text",
                "message": f"{empty_count} record(s) have empty text content and will be skipped.",
                "count": str(empty_count),
            })

    id_field = None
    for src, tgt in effective_mapping.items():
        if tgt == "id":
            id_field = src
            break

    if id_field:
        ids = [rec.get(id_field) for rec in records if rec.get(id_field)]
        dupes = len(ids) - len(set(ids))
        if dupes > 0:
            warnings.append({
                "type": "duplicate_ids",
                "message": f"{dupes} duplicate ID(s) found. Later records will overwrite earlier ones.",
                "count": str(dupes),
            })

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "total_records": len(records),
        "errors": errors,
        "warnings": warnings,
        "source_fields": source_fields,
        "detected_mapping": effective_mapping,
        "target_fields": {k: v for k, v in TARGET_FIELDS.items()},
    }


class KnowledgeBaseService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_upload(
        self,
        org_id: str,
        user_id: str,
        files: List[Dict[str, Any]],
    ) -> KnowledgeBaseUpload:
        """Parse uploaded files and create an upload session."""
        all_records: List[Dict[str, Any]] = []
        file_meta: List[Dict[str, Any]] = []

        for f in files:
            filename = f["filename"]
            content = f["content"]
            size = f.get("size", len(content))
            if size > MAX_UPLOAD_FILE_BYTES or len(content) > MAX_UPLOAD_FILE_BYTES:
                raise ValueError(
                    f"File {filename} exceeds the maximum size of "
                    f"{MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB"
                )

            records = parse_file_content(filename, content)
            file_meta.append({
                "filename": filename,
                "size": size,
                "row_count": len(records),
            })
            all_records.extend(records)

        upload = KnowledgeBaseUpload(
            org_id=org_id,
            uploaded_by=user_id,
            status="pending",
            file_metadata=file_meta,
            raw_data=all_records,
            document_count=len(all_records),
        )
        self.db.add(upload)
        await self.db.flush()
        return upload

    async def get_upload(self, upload_id: str, org_id: str) -> Optional[KnowledgeBaseUpload]:
        result = await self.db.execute(
            select(KnowledgeBaseUpload).where(
                KnowledgeBaseUpload.id == upload_id,
                KnowledgeBaseUpload.org_id == org_id,
            )
        )
        return result.scalar_one_or_none()

    async def validate_upload(self, upload_id: str, org_id: str) -> Optional[Dict[str, Any]]:
        upload = await self.get_upload(upload_id, org_id)
        if not upload:
            return None

        report = validate_records(upload.raw_data, upload.field_mapping)
        upload.validation_report = report
        upload.status = "validated"
        await self.db.flush()
        return report

    async def apply_field_mapping(
        self,
        upload_id: str,
        org_id: str,
        mapping: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        upload = await self.get_upload(upload_id, org_id)
        if not upload:
            return None

        upload.field_mapping = mapping
        # Re-validate with the new mapping
        report = validate_records(upload.raw_data, mapping)
        upload.validation_report = report
        upload.status = "validated"
        await self.db.flush()
        return report

    async def start_ingestion(
        self,
        upload_id: str,
        org_id: str,
        user_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run ingestion pipeline and yield SSE progress events.
        Creates an org-scoped Qdrant collection.
        """
        upload = await self.get_upload(upload_id, org_id)
        if not upload:
            yield {"event": "error", "data": {"message": "Upload session not found"}}
            return

        upload.status = "ingesting"
        await self.db.flush()
        await self.db.commit()

        try:
            # Previous active dataset = full corpus to carry forward (append semantics)
            active_result = await self.db.execute(
                select(KnowledgeBaseDataset)
                .where(
                    KnowledgeBaseDataset.org_id == org_id,
                    KnowledgeBaseDataset.is_active.is_(True),
                    KnowledgeBaseDataset.status == "active",
                )
                .order_by(KnowledgeBaseDataset.version_number.desc())
                .limit(1)
            )
            active_dataset = active_result.scalar_one_or_none()
            prev_doc_count = int(active_dataset.document_count) if active_dataset else 0
            prev_seg_count = int(active_dataset.segment_count) if active_dataset else 0

            # Determine version number
            result = await self.db.execute(
                select(KnowledgeBaseDataset)
                .where(KnowledgeBaseDataset.org_id == org_id)
                .order_by(KnowledgeBaseDataset.version_number.desc())
                .limit(1)
            )
            latest = result.scalar_one_or_none()
            version_number = (latest.version_number + 1) if latest else 1

            # Org-scoped collection name
            org_short = str(org_id).replace("-", "")[:12]
            collection_name = f"org_{org_short}_kb_v{version_number}"

            # Create dataset record
            dataset = KnowledgeBaseDataset(
                org_id=org_id,
                version_number=version_number,
                collection_name=collection_name,
                created_by=user_id,
                status="creating",
            )
            self.db.add(dataset)
            await self.db.flush()
            await self.db.commit()

            yield {
                "event": "progress",
                "data": {
                    "batch": 0,
                    "totalBatches": 0,
                    "message": f"Preparing v{version_number}...",
                },
            }

            # Apply field mapping to raw records
            mapping = upload.field_mapping or auto_detect_mapping(
                detect_fields(upload.raw_data)
            )
            documents = apply_mapping(upload.raw_data, mapping)

            # Filter out empty text
            documents = [d for d in documents if d.get("text") and str(d["text"]).strip()]

            if not documents:
                upload.status = "failed"
                upload.error_message = "No documents with text content after mapping"
                dataset.status = "failed"
                await self.db.flush()
                await self.db.commit()
                yield {
                    "event": "error",
                    "data": {"message": "No documents with text content found after field mapping."},
                }
                return

            total_batches = (len(documents) + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE

            yield {
                "event": "progress",
                "data": {
                    "batch": 0,
                    "totalBatches": total_batches,
                    "message": f"Processing {len(documents)} new documents in {total_batches} batches...",
                },
            }

            # Initialize Qdrant store with the org-scoped collection (empty)
            vector_store = QdrantVectorStore(collection_name=collection_name)
            total_segments = 0

            # Carry forward all vectors from the currently active dataset, then add new upload
            if active_dataset and active_dataset.collection_name != collection_name:
                yield {
                    "event": "progress",
                    "data": {
                        "batch": 0,
                        "totalBatches": total_batches,
                        "message": f"Copying v{active_dataset.version_number} ({prev_seg_count} segments) into v{version_number}...",
                    },
                }
                copied = await asyncio.to_thread(
                    copy_collection_points,
                    active_dataset.collection_name,
                    collection_name,
                )
                if copied != prev_seg_count and prev_seg_count > 0:
                    logger.warning(
                        "Copied %s points but DB had segment_count=%s for %s",
                        copied,
                        prev_seg_count,
                        active_dataset.collection_name,
                    )

            for i in range(0, len(documents), INGEST_BATCH_SIZE):
                batch_num = (i // INGEST_BATCH_SIZE) + 1
                batch = documents[i : i + INGEST_BATCH_SIZE]

                yield {
                    "event": "progress",
                    "data": {
                        "batch": batch_num,
                        "totalBatches": total_batches,
                        "message": f"Processing batch {batch_num}/{total_batches} ({len(batch)} docs)...",
                    },
                }

                # Run through ingestion pipeline: normalize -> segment -> enrich -> extract
                segments = process_batch(batch)
                vector_docs = process_for_vector_store(segments)

                if vector_docs:
                    # Add org_id to each document payload for filtering
                    for doc in vector_docs:
                        doc["org_id"] = str(org_id)

                    vector_store.add_documents(vector_docs)
                    total_segments += len(vector_docs)

                yield {
                    "event": "progress",
                    "data": {
                        "batch": batch_num,
                        "totalBatches": total_batches,
                        "message": f"Batch {batch_num}/{total_batches} complete. {total_segments} segments created.",
                    },
                }

            # Deactivate previous active version for this org
            await self.db.execute(
                update(KnowledgeBaseDataset)
                .where(
                    KnowledgeBaseDataset.org_id == org_id,
                    KnowledgeBaseDataset.is_active.is_(True),
                )
                .values(is_active=False, status="archived")
            )

            # Activate the new version (cumulative document + segment counts)
            dataset.document_count = prev_doc_count + len(documents)
            dataset.segment_count = prev_seg_count + total_segments
            dataset.is_active = True
            dataset.status = "active"

            upload.status = "completed"
            await self.db.flush()
            await self.db.commit()

            yield {
                "event": "complete",
                "data": {
                    "version_number": version_number,
                    "collection_name": collection_name,
                    "document_count": dataset.document_count,
                    "segment_count": dataset.segment_count,
                    "message": (
                        f"Ingestion complete! v{version_number} activated with "
                        f"{dataset.document_count} documents and {dataset.segment_count} segments "
                        f"(includes previous KB + {len(documents)} new source documents)."
                    ),
                },
            }

        except Exception as e:
            logger.exception("Ingestion failed")
            upload.status = "failed"
            upload.error_message = str(e)
            await self.db.flush()
            await self.db.commit()
            yield {
                "event": "error",
                "data": {"message": f"Ingestion failed: {e}"},
            }

    async def list_datasets(self, org_id: str) -> List[KnowledgeBaseDataset]:
        result = await self.db.execute(
            select(KnowledgeBaseDataset)
            .where(KnowledgeBaseDataset.org_id == org_id)
            .order_by(KnowledgeBaseDataset.version_number.desc())
        )
        return list(result.scalars().all())

    async def activate_dataset(self, dataset_id: str, org_id: str) -> Optional[KnowledgeBaseDataset]:
        result = await self.db.execute(
            select(KnowledgeBaseDataset).where(
                KnowledgeBaseDataset.id == dataset_id,
                KnowledgeBaseDataset.org_id == org_id,
            )
        )
        dataset = result.scalar_one_or_none()
        if not dataset or dataset.status == "failed":
            return None

        # Deactivate current
        await self.db.execute(
            update(KnowledgeBaseDataset)
            .where(
                KnowledgeBaseDataset.org_id == org_id,
                KnowledgeBaseDataset.is_active.is_(True),
            )
            .values(is_active=False, status="archived")
        )

        dataset.is_active = True
        dataset.status = "active"
        await self.db.flush()
        return dataset

    async def delete_dataset(self, dataset_id: str, org_id: str) -> bool:
        result = await self.db.execute(
            select(KnowledgeBaseDataset).where(
                KnowledgeBaseDataset.id == dataset_id,
                KnowledgeBaseDataset.org_id == org_id,
            )
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            return False

        # Delete Qdrant collection
        try:
            from qdrant_client import QdrantClient
            from ai_core.config.settings import QDRANT_URL

            client = QdrantClient(url=QDRANT_URL)
            collections = client.get_collections().collections
            if any(c.name == dataset.collection_name for c in collections):
                client.delete_collection(dataset.collection_name)
                logger.info(f"Deleted Qdrant collection: {dataset.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant collection {dataset.collection_name}: {e}")

        await self.db.delete(dataset)
        await self.db.flush()
        return True