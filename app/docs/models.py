# app/docs/models.py
from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.auth.db import Base  # same Base used by auth models

class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Display / provenance
    title = Column(String, nullable=False)         # Usually file name or page title
    source = Column(String, nullable=False)        # Path/URL
    chunk_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Optional ORM relationships (not strictly required)
    tenant = relationship("Tenant", backref="documents")
    user = relationship("User", backref="uploaded_documents")