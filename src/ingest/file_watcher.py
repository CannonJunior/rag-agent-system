import asyncio
from pathlib import Path
from typing import Set, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from loguru import logger

from config.settings import settings
from src.ingest.processor import document_processor, ProcessingResult


class FileEventType(Enum):
    """File system event types."""
    CREATED = "created"
    MODIFIED = "modified" 
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """File system event with metadata."""
    event_type: FileEventType
    file_path: Path
    timestamp: datetime
    is_directory: bool = False
    src_path: Optional[Path] = None  # For move events


class FileWatcherEventHandler(FileSystemEventHandler):
    """Custom file system event handler."""
    
    def __init__(self, callback: Callable[[FileEvent], None]):
        super().__init__()
        self.callback = callback
        self.supported_extensions = set(settings.ingest.supported_extensions)
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        if file_path.is_dir():
            return False
        
        extension = file_path.suffix.lower()
        return extension in self.supported_extensions
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        file_path = Path(event.src_path)
        
        if self._should_process_file(file_path):
            file_event = FileEvent(
                event_type=FileEventType.CREATED,
                file_path=file_path,
                timestamp=datetime.utcnow(),
                is_directory=event.is_directory
            )
            self.callback(file_event)
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        file_path = Path(event.src_path)
        
        if self._should_process_file(file_path):
            file_event = FileEvent(
                event_type=FileEventType.MODIFIED,
                file_path=file_path,
                timestamp=datetime.utcnow(),
                is_directory=event.is_directory
            )
            self.callback(file_event)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events."""
        file_path = Path(event.src_path)
        
        # Process deletion even for unsupported files to clean up database
        file_event = FileEvent(
            event_type=FileEventType.DELETED,
            file_path=file_path,
            timestamp=datetime.utcnow(),
            is_directory=event.is_directory
        )
        self.callback(file_event)
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move events."""
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)
        
        file_event = FileEvent(
            event_type=FileEventType.MOVED,
            file_path=dest_path,
            src_path=src_path,
            timestamp=datetime.utcnow(),
            is_directory=event.is_directory
        )
        self.callback(file_event)


class FileDebouncer:
    """Debounce file events to avoid processing duplicate events."""
    
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.pending_files: Dict[Path, float] = {}
        self._lock = asyncio.Lock()
    
    async def should_process(self, file_path: Path) -> bool:
        """Check if file should be processed (debouncing logic)."""
        async with self._lock:
            current_time = time.time()
            
            if file_path in self.pending_files:
                last_event_time = self.pending_files[file_path]
                if current_time - last_event_time < self.delay:
                    # Update timestamp and skip processing
                    self.pending_files[file_path] = current_time
                    return False
            
            # Mark file for processing
            self.pending_files[file_path] = current_time
            return True
    
    async def cleanup_old_entries(self):
        """Remove old entries from pending files."""
        async with self._lock:
            current_time = time.time()
            expired_files = [
                file_path for file_path, timestamp in self.pending_files.items()
                if current_time - timestamp > self.delay * 2
            ]
            
            for file_path in expired_files:
                del self.pending_files[file_path]
            
            if expired_files:
                logger.debug(f"Cleaned up {len(expired_files)} expired file entries")


class FileWatcher:
    """Async file watcher with processing integration."""
    
    def __init__(self, watch_directory: Path = None):
        self.watch_directory = watch_directory or settings.ingest.watch_directory
        self.observer = None
        self.event_handler = None
        self.debouncer = FileDebouncer(delay=2.0)
        self.is_running = False
        
        # Event processing
        self.event_queue = asyncio.Queue()
        self.processing_task = None
        self.cleanup_task = None
        
        # Statistics
        self.events_received = 0
        self.events_processed = 0
        self.events_skipped = 0
        self.last_event_time = None
        
        # Ensure watch directory exists
        self.watch_directory.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the file watcher."""
        if self.is_running:
            logger.warning("File watcher is already running")
            return
        
        logger.info(f"Starting file watcher for directory: {self.watch_directory}")
        
        # Setup event handler
        self.event_handler = FileWatcherEventHandler(self._on_file_event)
        
        # Setup observer
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            str(self.watch_directory),
            recursive=True
        )
        
        # Start processing task
        self.processing_task = asyncio.create_task(self._process_events())
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Start observer
        self.observer.start()
        self.is_running = True
        
        logger.info("File watcher started successfully")
        
        # Process existing files
        await self._process_existing_files()
    
    async def stop(self):
        """Stop the file watcher."""
        if not self.is_running:
            return
        
        logger.info("Stopping file watcher...")
        
        # Stop observer
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        # Cancel tasks
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.is_running = False
        logger.info("File watcher stopped")
    
    def _on_file_event(self, event: FileEvent):
        """Handle file system events (called from observer thread)."""
        # Queue event for async processing
        try:
            asyncio.create_task(self._queue_event(event))
        except RuntimeError:
            # Handle case where event loop is not running
            logger.warning(f"Could not queue file event: {event.file_path}")
    
    async def _queue_event(self, event: FileEvent):
        """Queue file event for processing."""
        await self.event_queue.put(event)
        self.events_received += 1
        self.last_event_time = datetime.utcnow()
        
        logger.debug(f"Queued {event.event_type.value} event for: {event.file_path}")
    
    async def _process_events(self):
        """Process file events from the queue."""
        logger.info("File event processor started")
        
        while True:
            try:
                # Wait for event
                event = await self.event_queue.get()
                
                # Process the event
                await self._handle_file_event(event)
                
                # Mark task done
                self.event_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("File event processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing file event: {e}")
    
    async def _handle_file_event(self, event: FileEvent):
        """Handle a single file event."""
        try:
            logger.info(f"Processing {event.event_type.value} event: {event.file_path}")
            
            if event.event_type == FileEventType.DELETED:
                await self._handle_file_deletion(event.file_path)
            elif event.event_type == FileEventType.MOVED:
                await self._handle_file_move(event.src_path, event.file_path)
            elif event.event_type in [FileEventType.CREATED, FileEventType.MODIFIED]:
                await self._handle_file_creation_or_modification(event.file_path)
            
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"Error handling file event {event.event_type.value} for {event.file_path}: {e}")
    
    async def _handle_file_creation_or_modification(self, file_path: Path):
        """Handle file creation or modification."""
        # Check if file still exists (might have been deleted quickly)
        if not file_path.exists():
            logger.debug(f"File no longer exists, skipping: {file_path}")
            return
        
        # Apply debouncing
        should_process = await self.debouncer.should_process(file_path)
        if not should_process:
            logger.debug(f"Skipping debounced file: {file_path}")
            self.events_skipped += 1
            return
        
        # Wait a bit for file to stabilize (in case it's still being written)
        await asyncio.sleep(1.0)
        
        # Check again if file exists and is stable
        if not file_path.exists():
            logger.debug(f"File disappeared during stabilization wait: {file_path}")
            return
        
        try:
            # Get file size to check if it's changing
            initial_size = file_path.stat().st_size
            await asyncio.sleep(0.5)
            final_size = file_path.stat().st_size
            
            if initial_size != final_size:
                logger.debug(f"File still changing size, will retry: {file_path}")
                # Re-queue the event for later processing
                await asyncio.sleep(2.0)
                await self._queue_event(FileEvent(
                    event_type=FileEventType.MODIFIED,
                    file_path=file_path,
                    timestamp=datetime.utcnow()
                ))
                return
        except Exception as e:
            logger.warning(f"Could not check file stability for {file_path}: {e}")
        
        # Queue file for processing
        success = await document_processor.queue_file(file_path)
        if success:
            logger.info(f"Queued file for document processing: {file_path}")
        else:
            logger.error(f"Failed to queue file for processing: {file_path}")
    
    async def _handle_file_deletion(self, file_path: Path):
        """Handle file deletion."""
        logger.info(f"File deleted: {file_path}")
        # TODO: Remove document and chunks from database
        # This would require database operations to clean up orphaned records
        pass
    
    async def _handle_file_move(self, src_path: Path, dest_path: Path):
        """Handle file move/rename."""
        logger.info(f"File moved: {src_path} -> {dest_path}")
        # TODO: Update document filepath in database
        # This would require database operations to update the document record
        pass
    
    async def _process_existing_files(self):
        """Process files that already exist in the watch directory."""
        logger.info("Processing existing files in watch directory...")
        
        existing_files = []
        for extension in settings.ingest.supported_extensions:
            pattern = f"**/*{extension}"
            existing_files.extend(self.watch_directory.glob(pattern))
        
        if not existing_files:
            logger.info("No existing files found")
            return
        
        logger.info(f"Found {len(existing_files)} existing files to process")
        
        # Queue existing files for processing
        for file_path in existing_files:
            if file_path.is_file():
                await document_processor.queue_file(file_path)
        
        logger.info("Queued all existing files for processing")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of internal state."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.debouncer.cleanup_old_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get file watcher statistics."""
        queue_size = self.event_queue.qsize()
        
        return {
            'is_running': self.is_running,
            'watch_directory': str(self.watch_directory),
            'events_received': self.events_received,
            'events_processed': self.events_processed,
            'events_skipped': self.events_skipped,
            'queue_size': queue_size,
            'last_event_time': self.last_event_time.isoformat() if self.last_event_time else None,
            'pending_debounced_files': len(self.debouncer.pending_files)
        }
    
    async def force_process_file(self, file_path: Path) -> Optional[ProcessingResult]:
        """Force immediate processing of a specific file."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        logger.info(f"Force processing file: {file_path}")
        result = await document_processor.process_file_immediately(file_path)
        return result


# Global file watcher instance
file_watcher = FileWatcher()