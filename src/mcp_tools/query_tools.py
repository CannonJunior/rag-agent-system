from typing import Dict, Any, List, Optional
import re

from sqlalchemy import select, func, and_, or_, text, cast, Float, join
from loguru import logger

from src.database.connection import db_manager
from src.database.models import Document, DocumentChunk
from src.ingest.embedder import document_embedder


class QueryTools:
    """MCP tools for document querying and vector search operations."""
    
    async def query_documents(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        document_ids: Optional[List[int]] = None,
        content_types: Optional[List[str]] = None,
        include_content: bool = True
    ) -> Dict[str, Any]:
        """Perform semantic search across documents using vector similarity."""
        try:
            # Generate embedding for query
            logger.info(f"Generating embedding for query: {query[:100]}...")
            query_embedding = await document_embedder.embed_query(query)
            
            if not query_embedding:
                return {
                    "error": "Failed to generate embedding for query",
                    "query": query,
                    "results": []
                }
            
            async with db_manager.get_async_session() as session:
                # Build the vector similarity query
                # Using pgvector's cosine similarity operator (1 - cosine_distance)
                similarity_expr = 1 - DocumentChunk.embedding.cosine_distance(cast(query_embedding, DocumentChunk.embedding.type))
                
                # Base query with joins
                query_stmt = select(
                    DocumentChunk.id.label('chunk_id'),
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                    DocumentChunk.token_count,
                    Document.id.label('document_id'),
                    Document.filename,
                    Document.filepath,
                    Document.content_type,
                    similarity_expr.label('similarity_score')
                ).select_from(
                    join(DocumentChunk, Document, DocumentChunk.document_id == Document.id)
                ).where(
                    and_(
                        DocumentChunk.embedding.is_not(None),
                        similarity_expr >= similarity_threshold
                    )
                )
                
                # Apply document ID filter if provided
                if document_ids:
                    query_stmt = query_stmt.where(Document.id.in_(document_ids))
                
                # Apply content type filter if provided
                if content_types:
                    query_stmt = query_stmt.where(Document.content_type.in_(content_types))
                
                # Order by similarity score and apply limit
                query_stmt = query_stmt.order_by(similarity_expr.desc()).limit(limit)
                
                # Execute query
                result = await session.execute(query_stmt)
                rows = result.fetchall()
                
                # Format results
                search_results = []
                for row in rows:
                    result_data = {
                        "chunk_id": row.chunk_id,
                        "document_id": row.document_id,
                        "document_filename": row.filename,
                        "document_filepath": row.filepath,
                        "content_type": row.content_type,
                        "chunk_index": row.chunk_index,
                        "similarity_score": float(row.similarity_score),
                        "token_count": row.token_count
                    }
                    
                    if include_content:
                        result_data["content"] = row.content
                        result_data["content_preview"] = self._create_content_preview(
                            row.content, query
                        )
                    
                    search_results.append(result_data)
                
                return {
                    "query": query,
                    "results": search_results,
                    "metadata": {
                        "total_results": len(search_results),
                        "similarity_threshold": similarity_threshold,
                        "max_results": limit,
                        "embedding_model": document_embedder.client.model,
                        "filters_applied": {
                            "document_ids": document_ids,
                            "content_types": content_types
                        }
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in vector search for query '{query}': {e}")
            return {
                "error": str(e),
                "query": query,
                "results": []
            }
    
    async def search_chunks(
        self,
        search_text: str,
        limit: int = 10,
        case_sensitive: bool = False,
        whole_words: bool = False,
        document_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Search for specific text within document chunks."""
        try:
            async with db_manager.get_async_session() as session:
                # Build text search query
                query_stmt = select(
                    DocumentChunk.id.label('chunk_id'),
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                    DocumentChunk.token_count,
                    Document.id.label('document_id'),
                    Document.filename,
                    Document.filepath,
                    Document.content_type
                ).select_from(
                    join(DocumentChunk, Document, DocumentChunk.document_id == Document.id)
                )
                
                # Build search condition
                if whole_words:
                    # Use regex for whole word matching
                    if case_sensitive:
                        pattern = rf'\b{re.escape(search_text)}\b'
                    else:
                        pattern = rf'(?i)\b{re.escape(search_text)}\b'
                    search_condition = DocumentChunk.content.op('~')(pattern)
                else:
                    # Simple substring search
                    if case_sensitive:
                        search_condition = DocumentChunk.content.contains(search_text)
                    else:
                        search_condition = DocumentChunk.content.ilike(f'%{search_text}%')
                
                query_stmt = query_stmt.where(search_condition)
                
                # Apply document ID filter if provided
                if document_ids:
                    query_stmt = query_stmt.where(Document.id.in_(document_ids))
                
                # Order by document and chunk index, apply limit
                query_stmt = query_stmt.order_by(
                    Document.id, DocumentChunk.chunk_index
                ).limit(limit)
                
                # Execute query
                result = await session.execute(query_stmt)
                rows = result.fetchall()
                
                # Format results
                search_results = []
                for row in rows:
                    # Find text matches in content
                    matches = self._find_text_matches(
                        row.content, search_text, case_sensitive, whole_words
                    )
                    
                    result_data = {
                        "chunk_id": row.chunk_id,
                        "document_id": row.document_id,
                        "document_filename": row.filename,
                        "document_filepath": row.filepath,
                        "content_type": row.content_type,
                        "chunk_index": row.chunk_index,
                        "token_count": row.token_count,
                        "content": row.content,
                        "matches": matches,
                        "match_count": len(matches),
                        "highlighted_content": self._highlight_matches(
                            row.content, search_text, case_sensitive, whole_words
                        )
                    }
                    
                    search_results.append(result_data)
                
                return {
                    "search_text": search_text,
                    "results": search_results,
                    "metadata": {
                        "total_results": len(search_results),
                        "max_results": limit,
                        "search_options": {
                            "case_sensitive": case_sensitive,
                            "whole_words": whole_words
                        },
                        "filters_applied": {
                            "document_ids": document_ids
                        }
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in text search for '{search_text}': {e}")
            return {
                "error": str(e),
                "search_text": search_text,
                "results": []
            }
    
    async def get_similar_chunks(
        self,
        chunk_id: int,
        limit: int = 5,
        similarity_threshold: float = 0.8,
        exclude_same_document: bool = False
    ) -> Dict[str, Any]:
        """Find chunks similar to a given chunk using vector similarity."""
        try:
            async with db_manager.get_async_session() as session:
                # Get the source chunk and its embedding
                source_query = select(DocumentChunk).where(DocumentChunk.id == chunk_id)
                source_result = await session.execute(source_query)
                source_chunk = source_result.scalar_one_or_none()
                
                if not source_chunk:
                    return {"error": f"Chunk {chunk_id} not found"}
                
                if not source_chunk.embedding:
                    return {"error": f"Chunk {chunk_id} has no embedding"}
                
                # Find similar chunks
                similarity_expr = 1 - DocumentChunk.embedding.cosine_distance(
                    cast(source_chunk.embedding, DocumentChunk.embedding.type)
                )
                
                query_stmt = select(
                    DocumentChunk.id.label('chunk_id'),
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                    DocumentChunk.token_count,
                    Document.id.label('document_id'),
                    Document.filename,
                    Document.content_type,
                    similarity_expr.label('similarity_score')
                ).select_from(
                    join(DocumentChunk, Document, DocumentChunk.document_id == Document.id)
                ).where(
                    and_(
                        DocumentChunk.id != chunk_id,  # Exclude source chunk
                        DocumentChunk.embedding.is_not(None),
                        similarity_expr >= similarity_threshold
                    )
                )
                
                # Exclude same document if requested
                if exclude_same_document:
                    query_stmt = query_stmt.where(
                        Document.id != source_chunk.document_id
                    )
                
                # Order by similarity and apply limit
                query_stmt = query_stmt.order_by(similarity_expr.desc()).limit(limit)
                
                # Execute query
                result = await session.execute(query_stmt)
                rows = result.fetchall()
                
                # Format results
                similar_chunks = []
                for row in rows:
                    similar_chunks.append({
                        "chunk_id": row.chunk_id,
                        "document_id": row.document_id,
                        "document_filename": row.filename,
                        "content_type": row.content_type,
                        "chunk_index": row.chunk_index,
                        "similarity_score": float(row.similarity_score),
                        "token_count": row.token_count,
                        "content": row.content
                    })
                
                return {
                    "source_chunk_id": chunk_id,
                    "source_content_preview": source_chunk.content[:200] + "..." if len(source_chunk.content) > 200 else source_chunk.content,
                    "similar_chunks": similar_chunks,
                    "metadata": {
                        "total_results": len(similar_chunks),
                        "similarity_threshold": similarity_threshold,
                        "max_results": limit,
                        "exclude_same_document": exclude_same_document
                    }
                }
                
        except Exception as e:
            logger.error(f"Error finding similar chunks for {chunk_id}: {e}")
            return {"error": str(e)}
    
    async def get_document_context(
        self,
        chunk_id: int,
        context_chunks: int = 2
    ) -> Dict[str, Any]:
        """Get surrounding chunks for context around a specific chunk."""
        try:
            async with db_manager.get_async_session() as session:
                # Get the target chunk
                target_query = select(DocumentChunk).where(DocumentChunk.id == chunk_id)
                target_result = await session.execute(target_query)
                target_chunk = target_result.scalar_one_or_none()
                
                if not target_chunk:
                    return {"error": f"Chunk {chunk_id} not found"}
                
                # Get surrounding chunks from the same document
                context_query = select(
                    DocumentChunk.id.label('chunk_id'),
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                    DocumentChunk.token_count
                ).where(
                    and_(
                        DocumentChunk.document_id == target_chunk.document_id,
                        DocumentChunk.chunk_index >= target_chunk.chunk_index - context_chunks,
                        DocumentChunk.chunk_index <= target_chunk.chunk_index + context_chunks
                    )
                ).order_by(DocumentChunk.chunk_index)
                
                context_result = await session.execute(context_query)
                context_rows = context_result.fetchall()
                
                # Format context chunks
                context_chunks_list = []
                for row in context_rows:
                    is_target = row.chunk_id == chunk_id
                    context_chunks_list.append({
                        "chunk_id": row.chunk_id,
                        "chunk_index": row.chunk_index,
                        "content": row.content,
                        "token_count": row.token_count,
                        "is_target": is_target,
                        "position_relative_to_target": row.chunk_index - target_chunk.chunk_index
                    })
                
                # Get document info
                doc_query = select(Document).where(Document.id == target_chunk.document_id)
                doc_result = await session.execute(doc_query)
                document = doc_result.scalar_one()
                
                return {
                    "target_chunk_id": chunk_id,
                    "document_id": target_chunk.document_id,
                    "document_filename": document.filename,
                    "context_window": context_chunks,
                    "context_chunks": context_chunks_list,
                    "metadata": {
                        "total_context_chunks": len(context_chunks_list),
                        "target_chunk_index": target_chunk.chunk_index
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting context for chunk {chunk_id}: {e}")
            return {"error": str(e)}
    
    def _create_content_preview(self, content: str, query: str, max_length: int = 300) -> str:
        """Create a content preview highlighting the query terms."""
        # Find the best match position
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find first occurrence of any query word
        query_words = query_lower.split()
        best_pos = len(content)
        
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1 and pos < best_pos:
                best_pos = pos
        
        if best_pos == len(content):
            # No match found, return beginning
            best_pos = 0
        
        # Calculate preview window
        start = max(0, best_pos - max_length // 2)
        end = min(len(content), start + max_length)
        
        # Adjust start if we're at the end
        if end - start < max_length:
            start = max(0, end - max_length)
        
        preview = content[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            preview = "..." + preview
        if end < len(content):
            preview = preview + "..."
        
        return preview
    
    def _find_text_matches(
        self,
        content: str,
        search_text: str,
        case_sensitive: bool,
        whole_words: bool
    ) -> List[Dict[str, Any]]:
        """Find all matches of search text in content."""
        matches = []
        
        if whole_words:
            # Use regex for whole word matching
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = rf'\b{re.escape(search_text)}\b'
            
            for match in re.finditer(pattern, content, flags):
                matches.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "context_before": content[max(0, match.start() - 50):match.start()],
                    "context_after": content[match.end():match.end() + 50]
                })
        else:
            # Simple substring search
            search_in = content if case_sensitive else content.lower()
            search_for = search_text if case_sensitive else search_text.lower()
            
            start = 0
            while True:
                pos = search_in.find(search_for, start)
                if pos == -1:
                    break
                
                matches.append({
                    "start": pos,
                    "end": pos + len(search_text),
                    "text": content[pos:pos + len(search_text)],
                    "context_before": content[max(0, pos - 50):pos],
                    "context_after": content[pos + len(search_text):pos + len(search_text) + 50]
                })
                
                start = pos + 1
        
        return matches
    
    def _highlight_matches(
        self,
        content: str,
        search_text: str,
        case_sensitive: bool,
        whole_words: bool,
        highlight_start: str = "**",
        highlight_end: str = "**"
    ) -> str:
        """Highlight matches in content with markdown-style highlighting."""
        if whole_words:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = rf'\b{re.escape(search_text)}\b'
            return re.sub(
                pattern,
                f"{highlight_start}\\g<0>{highlight_end}",
                content,
                flags=flags
            )
        else:
            if case_sensitive:
                return content.replace(
                    search_text,
                    f"{highlight_start}{search_text}{highlight_end}"
                )
            else:
                # Case-insensitive replacement is more complex
                pattern = re.escape(search_text)
                return re.sub(
                    pattern,
                    f"{highlight_start}\\g<0>{highlight_end}",
                    content,
                    flags=re.IGNORECASE
                )