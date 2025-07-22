import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from config.settings import settings
from src.database.connection import db_manager
from src.database.models import Document, DocumentChunk
from src.ingest.text_extractor import text_extractor, ExtractedContent
from src.ingest.chunker import text_chunker, Chunk
from src.ingest.embedder import document_embedder


class ProcessingStatus(Enum):
    """File processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingResult:
    """Result of file processing."""
    file_path: Path
    status: ProcessingStatus
    document_id: Optional[int]
    chunks_created: int
    embeddings_created: int
    processing_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class DocumentProcessor:
    """Async document processor with queue management."""
    
    def __init__(self, max_concurrent_files: int = 3):
        self.max_concurrent_files = max_concurrent_files
        self.processing_queue = asyncio.Queue()
        self.processing_tasks = set()
        self.shutdown_event = asyncio.Event()
        
        # Metrics
        self.files_processed = 0
        self.files_failed = 0
        self.total_chunks_created = 0
        self.total_processing_time = 0.0
    
    async def start(self):
        """Start the document processor workers."""
        logger.info(f"Starting document processor with {self.max_concurrent_files} workers")
        
        # Start worker tasks
        for i in range(self.max_concurrent_files):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.processing_tasks.add(task)
        
        logger.info("Document processor workers started")
    
    async def stop(self):
        """Stop the document processor gracefully."""
        logger.info("Stopping document processor...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for current processing to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        logger.info("Document processor stopped")
    
    async def queue_file(self, file_path: Path, priority: int = 0) -> bool:
        """Queue a file for processing."""
        try:
            await self.processing_queue.put((priority, file_path))
            logger.info(f"Queued file for processing: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to queue file {file_path}: {e}")
            return False
    
    async def process_file_immediately(self, file_path: Path) -> ProcessingResult:
        """Process a file immediately without queueing."""
        return await self._process_single_file(file_path)
    
    async def _worker(self, worker_name: str):
        """Worker coroutine for processing files."""
        logger.info(f"Document processor worker {worker_name} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for file with timeout
                priority, file_path = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                logger.info(f"Worker {worker_name} processing: {file_path}")
                result = await self._process_single_file(file_path)
                
                # Update metrics
                if result.status == ProcessingStatus.COMPLETED:
                    self.files_processed += 1
                    self.total_chunks_created += result.chunks_created
                elif result.status == ProcessingStatus.FAILED:
                    self.files_failed += 1
                
                self.total_processing_time += result.processing_time
                
                # Mark task done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # No files to process, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Document processor worker {worker_name} stopped")
    
    async def _process_single_file(self, file_path: Path) -> ProcessingResult:
        """Process a single file through the complete pipeline."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check if file exists and is supported
            if not file_path.exists():
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.FAILED,
                    document_id=None,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=0.0,
                    error="File not found"
                )
            
            if not text_extractor.is_supported(file_path):
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.SKIPPED,
                    document_id=None,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=0.0,
                    error="Unsupported file type"
                )
            
            # Check if file already processed (by hash)
            existing_doc = await self._check_existing_document(file_path)
            if existing_doc:
                logger.info(f"File already processed, skipping: {file_path}")
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.SKIPPED,
                    document_id=existing_doc.id,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    error="Already processed"
                )
            
            # Step 1: Extract text
            logger.info(f"Extracting text from: {file_path}")
            extracted_content = await text_extractor.extract(file_path)
            
            if not extracted_content:
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.FAILED,
                    document_id=None,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    error="Text extraction failed"
                )
            
            # Validate extracted content
            if not await text_extractor.validate_content(extracted_content):
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.FAILED,
                    document_id=None,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    error="Content validation failed"
                )
            
            # Step 2: Create document record
            document = await self._create_document_record(file_path, extracted_content)
            if not document:
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.FAILED,
                    document_id=None,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    error="Failed to create document record"
                )
            
            # Step 3: Chunk text
            logger.info(f"Chunking text from: {file_path}")
            chunks = text_chunker.chunk_text(
                extracted_content.text,
                extracted_content.content_type,
                extracted_content.metadata
            )
            
            if not chunks:
                await self._update_document_status(document.id, ProcessingStatus.FAILED, "No chunks created")
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.FAILED,
                    document_id=document.id,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    error="Text chunking failed"
                )
            
            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks from: {file_path}")
            chunk_embedding_pairs = await document_embedder.embed_chunks(chunks)
            
            # Step 5: Store chunks with embeddings
            chunks_created = 0
            embeddings_created = 0
            
            async with db_manager.get_async_session() as session:
                for chunk, embedding_result in chunk_embedding_pairs:
                    chunk_record = DocumentChunk(
                        document_id=document.id,
                        chunk_index=chunk.index,
                        content=chunk.content,
                        embedding=embedding_result.embedding if embedding_result.embedding else None,
                        token_count=chunk.token_count
                    )
                    
                    session.add(chunk_record)
                    chunks_created += 1
                    
                    if embedding_result.embedding:
                        embeddings_created += 1
                
                await session.commit()
            
            # Step 6: Update document status
            await self._update_document_status(document.id, ProcessingStatus.COMPLETED)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(
                f"Successfully processed {file_path}: "
                f"{chunks_created} chunks, {embeddings_created} embeddings "
                f"in {processing_time:.2f}s"
            )
            
            return ProcessingResult(
                file_path=file_path,
                status=ProcessingStatus.COMPLETED,
                document_id=document.id,
                chunks_created=chunks_created,
                embeddings_created=embeddings_created,
                processing_time=processing_time,
                metadata={
                    'extraction_method': extracted_content.extraction_method,
                    'content_hash': extracted_content.content_hash,
                    'word_count': extracted_content.word_count,
                    'char_count': extracted_content.char_count
                }
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error processing file {file_path}: {e}")
            
            return ProcessingResult(
                file_path=file_path,
                status=ProcessingStatus.FAILED,
                document_id=None,
                chunks_created=0,
                embeddings_created=0,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def _check_existing_document(self, file_path: Path) -> Optional[Document]:
        """Check if document already exists based on file path and modification time."""
        try:
            async with db_manager.get_async_session() as session:
                stmt = select(Document).where(Document.filepath == str(file_path))
                result = await session.execute(stmt)
                existing_doc = result.scalar_one_or_none()
                
                if existing_doc:
                    # Check if file was modified since last processing
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if existing_doc.processed_at and file_mtime <= existing_doc.processed_at:
                        return existing_doc
                
                return None
        except Exception as e:
            logger.error(f"Error checking existing document {file_path}: {e}")
            return None
    
    async def _create_document_record(self, file_path: Path, content: ExtractedContent) -> Optional[Document]:
        """Create document record in database."""
        try:
            document = Document(
                filename=file_path.name,
                filepath=str(file_path),
                content_type=content.content_type,
                file_size=content.file_size,
                content_hash=content.content_hash,
                status=ProcessingStatus.PROCESSING.value
            )
            
            async with db_manager.get_async_session() as session:
                session.add(document)
                await session.commit()
                await session.refresh(document)
            
            logger.info(f"Created document record {document.id} for {file_path}")
            return document
            
        except IntegrityError as e:
            logger.warning(f"Document already exists for {file_path}: {e}")
            # Try to get existing document
            async with db_manager.get_async_session() as session:
                stmt = select(Document).where(Document.filepath == str(file_path))
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error creating document record for {file_path}: {e}")
            return None
    
    async def _update_document_status(
        self, 
        document_id: int, 
        status: ProcessingStatus, 
        error: str = None
    ):
        """Update document processing status."""
        try:
            update_data = {
                'status': status.value,
                'updated_at': datetime.utcnow()
            }
            
            if status == ProcessingStatus.COMPLETED:
                update_data['processed_at'] = datetime.utcnow()
            
            async with db_manager.get_async_session() as session:
                stmt = update(Document).where(Document.id == document_id).values(**update_data)
                await session.execute(stmt)
                await session.commit()
            
            logger.info(f"Updated document {document_id} status to {status.value}")
            
        except Exception as e:
            logger.error(f"Error updating document {document_id} status: {e}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        queue_size = self.processing_queue.qsize()
        active_workers = len([task for task in self.processing_tasks if not task.done()])
        
        avg_processing_time = (
            self.total_processing_time / (self.files_processed + self.files_failed)
            if (self.files_processed + self.files_failed) > 0 else 0.0
        )
        
        return {
            'queue_size': queue_size,
            'active_workers': active_workers,
            'total_workers': len(self.processing_tasks),
            'files_processed': self.files_processed,
            'files_failed': self.files_failed,
            'total_chunks_created': self.total_chunks_created,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'is_running': not self.shutdown_event.is_set()
        }
    
    async def process_directory(self, directory: Path, recursive: bool = True) -> List[ProcessingResult]:
        """Process all supported files in a directory."""
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Find all supported files
        files_to_process = []
        
        if recursive:
            for extension in settings.ingest.supported_extensions:
                pattern = f"**/*{extension}"
                files_to_process.extend(directory.glob(pattern))
        else:
            for extension in settings.ingest.supported_extensions:
                pattern = f"*{extension}"
                files_to_process.extend(directory.glob(pattern))
        
        logger.info(f"Found {len(files_to_process)} files to process in {directory}")
        
        # Process files
        results = []
        for file_path in files_to_process:
            result = await self.process_file_immediately(file_path)
            results.append(result)
        
        return results


# Global document processor instance
document_processor = DocumentProcessor()