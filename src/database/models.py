from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, BigInteger, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Document(Base):
    """Document metadata table."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False, unique=True)
    content_type = Column(String(100))
    file_size = Column(BigInteger)
    content_hash = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime)
    status = Column(String(50), default="pending")
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class DocumentChunk(Base):
    """Document text chunks with embeddings."""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768))  # nomic-text-embed dimension
    token_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to document
    document = relationship("Document", back_populates="chunks")
    
    # Unique constraint
    __table_args__ = (
        Index("idx_document_chunks_unique", "document_id", "chunk_index", unique=True),
        Index("idx_document_chunks_embedding", "embedding", postgresql_using="ivfflat", 
              postgresql_ops={"embedding": "vector_cosine_ops"}, postgresql_with={"lists": 100}),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class ChatMemory(Base):
    """Conversation history storage."""
    __tablename__ = "chat_memory"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), nullable=False)
    message_type = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    extra_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_chat_memory_session_id", "session_id"),
        Index("idx_chat_memory_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<ChatMemory(id={self.id}, session_id={self.session_id}, type={self.message_type})>"