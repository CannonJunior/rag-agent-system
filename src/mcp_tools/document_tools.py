from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select, func, and_, or_
from loguru import logger

from src.database.connection import db_manager
from src.database.models import Document, DocumentChunk


class DocumentTools:
    """MCP tools for document management operations."""
    
    async def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        filename_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """List documents with filtering and pagination."""
        try:
            async with db_manager.get_async_session() as session:
                # Build query
                query = select(Document)
                
                # Apply filters
                conditions = []
                if filename_filter:
                    conditions.append(Document.filename.ilike(f"%{filename_filter}%"))
                if content_type_filter:
                    conditions.append(Document.content_type.ilike(f"%{content_type_filter}%"))
                if status_filter:
                    conditions.append(Document.status == status_filter)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                # Apply sorting
                sort_column = getattr(Document, sort_by, Document.created_at)
                if sort_order.lower() == "desc":
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())
                
                # Apply pagination
                query = query.offset(offset).limit(limit)
                
                # Execute query
                result = await session.execute(query)
                documents = result.scalars().all()
                
                # Get total count for pagination
                count_query = select(func.count(Document.id))
                if conditions:
                    count_query = count_query.where(and_(*conditions))
                
                count_result = await session.execute(count_query)
                total_count = count_result.scalar()
                
                # Format results
                document_list = []
                for doc in documents:
                    document_list.append({
                        "id": doc.id,
                        "filename": doc.filename,
                        "filepath": doc.filepath,
                        "content_type": doc.content_type,
                        "file_size": doc.file_size,
                        "file_size_human": self._format_file_size(doc.file_size),
                        "status": doc.status,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                        "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                        "content_hash": doc.content_hash
                    })
                
                return {
                    "documents": document_list,
                    "pagination": {
                        "total": total_count,
                        "limit": limit,
                        "offset": offset,
                        "has_more": offset + limit < total_count
                    },
                    "filters_applied": {
                        "filename": filename_filter,
                        "content_type": content_type_filter,
                        "status": status_filter
                    },
                    "sorting": {
                        "sort_by": sort_by,
                        "sort_order": sort_order
                    }
                }
                
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return {
                "error": str(e),
                "documents": [],
                "pagination": {"total": 0, "limit": limit, "offset": offset, "has_more": False}
            }
    
    async def get_document(
        self,
        document_id: int,
        include_content: bool = False,
        include_chunks: bool = False
    ) -> Dict[str, Any]:
        """Get detailed information about a specific document."""
        try:
            async with db_manager.get_async_session() as session:
                # Get document
                doc_query = select(Document).where(Document.id == document_id)
                doc_result = await session.execute(doc_query)
                document = doc_result.scalar_one_or_none()
                
                if not document:
                    return {"error": f"Document {document_id} not found"}
                
                # Base document info
                doc_info = {
                    "id": document.id,
                    "filename": document.filename,
                    "filepath": document.filepath,
                    "content_type": document.content_type,
                    "file_size": document.file_size,
                    "file_size_human": self._format_file_size(document.file_size),
                    "content_hash": document.content_hash,
                    "status": document.status,
                    "created_at": document.created_at.isoformat() if document.created_at else None,
                    "updated_at": document.updated_at.isoformat() if document.updated_at else None,
                    "processed_at": document.processed_at.isoformat() if document.processed_at else None
                }
                
                # Get chunk statistics
                chunk_stats_query = select(
                    func.count(DocumentChunk.id).label('total_chunks'),
                    func.sum(DocumentChunk.token_count).label('total_tokens'),
                    func.count(DocumentChunk.embedding).label('chunks_with_embeddings')
                ).where(DocumentChunk.document_id == document_id)
                
                stats_result = await session.execute(chunk_stats_query)
                stats = stats_result.first()
                
                doc_info["chunk_statistics"] = {
                    "total_chunks": stats.total_chunks or 0,
                    "total_tokens": stats.total_tokens or 0,
                    "chunks_with_embeddings": stats.chunks_with_embeddings or 0
                }
                
                # Include content if requested
                if include_content:
                    try:
                        file_path = Path(document.filepath)
                        if file_path.exists():
                            with open(file_path, 'r', encoding='utf-8') as f:
                                doc_info["file_content"] = f.read()
                        else:
                            doc_info["file_content"] = "File not found at original location"
                    except Exception as e:
                        doc_info["file_content"] = f"Error reading file: {str(e)}"
                
                # Include chunks if requested
                if include_chunks:
                    chunks_query = select(DocumentChunk).where(
                        DocumentChunk.document_id == document_id
                    ).order_by(DocumentChunk.chunk_index)
                    
                    chunks_result = await session.execute(chunks_query)
                    chunks = chunks_result.scalars().all()
                    
                    doc_info["chunks"] = [
                        {
                            "id": chunk.id,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "token_count": chunk.token_count,
                            "has_embedding": chunk.embedding is not None,
                            "created_at": chunk.created_at.isoformat() if chunk.created_at else None
                        }
                        for chunk in chunks
                    ]
                
                return doc_info
                
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return {"error": str(e)}
    
    async def get_file_contents(
        self,
        document_id: int,
        chunk_size: int = 10000
    ) -> str:
        """Get the original file contents of a document."""
        try:
            async with db_manager.get_async_session() as session:
                # Get document
                doc_query = select(Document).where(Document.id == document_id)
                doc_result = await session.execute(doc_query)
                document = doc_result.scalar_one_or_none()
                
                if not document:
                    return f"Error: Document {document_id} not found"
                
                # Try to read original file
                file_path = Path(document.filepath)
                if not file_path.exists():
                    # Fall back to reconstructed content from chunks
                    chunks_query = select(DocumentChunk.content).where(
                        DocumentChunk.document_id == document_id
                    ).order_by(DocumentChunk.chunk_index)
                    
                    chunks_result = await session.execute(chunks_query)
                    chunks = chunks_result.scalars().all()
                    
                    if chunks:
                        content = "\n\n".join(chunks)
                        if len(content) > chunk_size:
                            content = content[:chunk_size] + f"\n\n[Content truncated at {chunk_size} characters]"
                        return content
                    else:
                        return f"Error: File not found and no chunks available for document {document_id}"
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) > chunk_size:
                            content = content[:chunk_size] + f"\n\n[Content truncated at {chunk_size} characters]"
                        return content
                except UnicodeDecodeError:
                    # Try different encodings
                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                                if len(content) > chunk_size:
                                    content = content[:chunk_size] + f"\n\n[Content truncated at {chunk_size} characters]"
                                return content
                        except Exception:
                            continue
                    return f"Error: Could not decode file {file_path} with any encoding"
                
        except Exception as e:
            logger.error(f"Error getting file contents for document {document_id}: {e}")
            return f"Error: {str(e)}"
    
    async def get_document_chunks(
        self,
        document_id: int,
        limit: int = 20,
        offset: int = 0,
        include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """Get text chunks from a specific document."""
        try:
            async with db_manager.get_async_session() as session:
                # Check if document exists
                doc_query = select(Document).where(Document.id == document_id)
                doc_result = await session.execute(doc_query)
                document = doc_result.scalar_one_or_none()
                
                if not document:
                    return {"error": f"Document {document_id} not found"}
                
                # Get chunks
                chunks_query = select(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                ).order_by(DocumentChunk.chunk_index).offset(offset).limit(limit)
                
                chunks_result = await session.execute(chunks_query)
                chunks = chunks_result.scalars().all()
                
                # Get total count
                count_query = select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.document_id == document_id
                )
                count_result = await session.execute(count_query)
                total_count = count_result.scalar()
                
                # Format chunks
                chunk_list = []
                for chunk in chunks:
                    chunk_data = {
                        "id": chunk.id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "token_count": chunk.token_count,
                        "has_embedding": chunk.embedding is not None,
                        "created_at": chunk.created_at.isoformat() if chunk.created_at else None
                    }
                    
                    if include_embeddings and chunk.embedding:
                        chunk_data["embedding"] = chunk.embedding
                    
                    chunk_list.append(chunk_data)
                
                return {
                    "document_id": document_id,
                    "document_filename": document.filename,
                    "chunks": chunk_list,
                    "pagination": {
                        "total": total_count,
                        "limit": limit,
                        "offset": offset,
                        "has_more": offset + limit < total_count
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return {"error": str(e)}
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get overall document collection statistics."""
        try:
            async with db_manager.get_async_session() as session:
                # Document counts by status
                status_query = select(
                    Document.status,
                    func.count(Document.id).label('count')
                ).group_by(Document.status)
                
                status_result = await session.execute(status_query)
                status_counts = {row.status: row.count for row in status_result}
                
                # Content type distribution
                content_type_query = select(
                    Document.content_type,
                    func.count(Document.id).label('count')
                ).group_by(Document.content_type)
                
                content_type_result = await session.execute(content_type_query)
                content_type_counts = {row.content_type: row.count for row in content_type_result}
                
                # Size statistics
                size_stats_query = select(
                    func.count(Document.id).label('total_documents'),
                    func.sum(Document.file_size).label('total_size'),
                    func.avg(Document.file_size).label('avg_size'),
                    func.min(Document.file_size).label('min_size'),
                    func.max(Document.file_size).label('max_size')
                )
                
                size_result = await session.execute(size_stats_query)
                size_stats = size_result.first()
                
                # Chunk statistics
                chunk_stats_query = select(
                    func.count(DocumentChunk.id).label('total_chunks'),
                    func.sum(DocumentChunk.token_count).label('total_tokens'),
                    func.avg(DocumentChunk.token_count).label('avg_tokens_per_chunk'),
                    func.count(DocumentChunk.embedding).label('chunks_with_embeddings')
                )
                
                chunk_result = await session.execute(chunk_stats_query)
                chunk_stats = chunk_result.first()
                
                # Recent activity (last 24 hours)
                recent_query = select(func.count(Document.id)).where(
                    Document.created_at >= datetime.utcnow() - timedelta(days=1)
                )
                recent_result = await session.execute(recent_query)
                recent_count = recent_result.scalar()
                
                return {
                    "total_documents": size_stats.total_documents or 0,
                    "status_distribution": status_counts,
                    "content_type_distribution": content_type_counts,
                    "size_statistics": {
                        "total_size": size_stats.total_size or 0,
                        "total_size_human": self._format_file_size(size_stats.total_size or 0),
                        "average_size": int(size_stats.avg_size or 0),
                        "average_size_human": self._format_file_size(int(size_stats.avg_size or 0)),
                        "min_size": size_stats.min_size or 0,
                        "max_size": size_stats.max_size or 0
                    },
                    "chunk_statistics": {
                        "total_chunks": chunk_stats.total_chunks or 0,
                        "total_tokens": chunk_stats.total_tokens or 0,
                        "average_tokens_per_chunk": int(chunk_stats.avg_tokens_per_chunk or 0),
                        "chunks_with_embeddings": chunk_stats.chunks_with_embeddings or 0,
                        "embedding_coverage": (
                            (chunk_stats.chunks_with_embeddings or 0) / (chunk_stats.total_chunks or 1) * 100
                        ) if chunk_stats.total_chunks else 0
                    },
                    "recent_activity": {
                        "documents_added_last_24h": recent_count
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {"error": str(e)}
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024.0 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"