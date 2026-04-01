import uuid
from sqlalchemy import Column, String, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from api.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(Text, nullable=False)
    created_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        default=func.now(),
    )

    memberships = relationship("UserOrganization", back_populates="user", cascade="all, delete-orphan", lazy="selectin")
