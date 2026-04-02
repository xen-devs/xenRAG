import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID

from api.db.base import Base


class KnowledgeBaseDataset(Base):
    __tablename__ = "knowledge_base_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    collection_name = Column(String(255), nullable=False)  # Qdrant collection name
    document_count = Column(Integer, nullable=False, default=0)
    segment_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=False, server_default="false")
    status = Column(
        String(32),
        nullable=False,
        default="creating",
        server_default="creating",
    )  # creating | active | archived | failed
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=func.now())
