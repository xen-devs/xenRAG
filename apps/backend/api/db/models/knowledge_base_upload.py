import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID

from api.db.base import Base


class KnowledgeBaseUpload(Base):
    __tablename__ = "knowledge_base_uploads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status = Column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
    )  # pending | validated | mapping | ingesting | completed | failed
    file_metadata = Column(JSON, nullable=False, default=list)  # [{filename, size, row_count}]
    raw_data = Column(JSON, nullable=False, default=list)  # parsed records
    document_count = Column(Integer, nullable=False, default=0)
    validation_report = Column(JSON, nullable=True)  # {errors, warnings, file_fields}
    field_mapping = Column(JSON, nullable=True)  # {source_field: target_field}
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), default=func.now(), onupdate=func.now())