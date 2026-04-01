from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from api.db.base import Base


class UserOrganization(Base):
    __tablename__ = "user_organizations"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), primary_key=True)
    role = Column(String(50), default="owner", nullable=False)

    user = relationship("User", back_populates="memberships", lazy="selectin")
    organization = relationship("Organization", back_populates="members", lazy="selectin")
