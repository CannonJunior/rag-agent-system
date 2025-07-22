import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import magic

from loguru import logger
from config.settings import settings


class FileValidator:
    """File validation utilities."""
    
    def __init__(self):
        self.max_file_size = settings.ingest.max_file_size_mb * 1024 * 1024
        self.supported_extensions = set(settings.ingest.supported_extensions)
        
        # Initialize python-magic for better MIME type detection
        try:
            self.magic = magic.Magic(mime=True)
            self.magic_available = True
        except Exception:
            logger.warning("python-magic not available, falling back to mimetypes")
            self.magic_available = False
    
    def is_valid_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """Validate if file can be processed."""
        try:
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check if it's a file (not directory)
            if not file_path.is_file():
                return False, "Path is not a file"
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False, "File is empty"
            
            if file_size > self.max_file_size:
                return False, f"File too large ({file_size} bytes, max {self.max_file_size})"
            
            # Check file extension
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                return False, f"Unsupported file extension: {extension}"
            
            # Check file permissions
            if not file_path.is_readable():
                return False, "File is not readable"
            
            # Validate MIME type consistency
            mime_type = self.get_mime_type(file_path)
            if not self._is_mime_type_consistent(extension, mime_type):
                logger.warning(f"MIME type {mime_type} may not match extension {extension} for {file_path}")
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_mime_type(self, file_path: Path) -> Optional[str]:
        """Get MIME type of file."""
        try:
            if self.magic_available:
                return self.magic.from_file(str(file_path))
            else:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                return mime_type
        except Exception as e:
            logger.warning(f"Could not determine MIME type for {file_path}: {e}")
            return None
    
    def _is_mime_type_consistent(self, extension: str, mime_type: Optional[str]) -> bool:
        """Check if MIME type is consistent with file extension."""
        if not mime_type:
            return True  # Can't validate without MIME type
        
        mime_type = mime_type.lower()
        
        # Define expected MIME types for extensions
        expected_mime_types = {
            '.txt': ['text/plain'],
            '.md': ['text/markdown', 'text/plain'],
            '.pdf': ['application/pdf'],
            '.docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
            '.doc': ['application/msword'],
            '.rtf': ['application/rtf', 'text/rtf'],
            '.odt': ['application/vnd.oasis.opendocument.text'],
            '.py': ['text/x-python', 'text/plain'],
            '.js': ['application/javascript', 'text/javascript', 'text/plain'],
            '.json': ['application/json', 'text/plain'],
            '.xml': ['application/xml', 'text/xml', 'text/plain'],
            '.html': ['text/html'],
            '.csv': ['text/csv', 'text/plain'],
        }
        
        expected = expected_mime_types.get(extension, [])
        return not expected or any(expected_type in mime_type for expected_type in expected)


class FileHasher:
    """File hashing utilities."""
    
    @staticmethod
    def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> Optional[str]:
        """Calculate hash of file content."""
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    @staticmethod
    def calculate_content_hash(content: str, algorithm: str = 'sha256') -> str:
        """Calculate hash of text content."""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content.encode('utf-8'))
        return hash_obj.hexdigest()


class FileMetadata:
    """File metadata extraction."""
    
    @staticmethod
    def get_file_metadata(file_path: Path) -> Dict[str, Any]:
        """Extract comprehensive file metadata."""
        try:
            stat = file_path.stat()
            
            metadata = {
                'filename': file_path.name,
                'extension': file_path.suffix.lower(),
                'file_size': stat.st_size,
                'created_time': datetime.fromtimestamp(stat.st_ctime),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'accessed_time': datetime.fromtimestamp(stat.st_atime),
                'is_symlink': file_path.is_symlink(),
                'absolute_path': str(file_path.absolute()),
                'relative_path': str(file_path),
            }
            
            # Add MIME type
            validator = FileValidator()
            metadata['mime_type'] = validator.get_mime_type(file_path)
            
            # Add file hash
            metadata['file_hash'] = FileHasher.calculate_file_hash(file_path)
            
            # Add file size in human readable format
            metadata['file_size_human'] = FileMetadata._format_file_size(stat.st_size)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return {
                'filename': file_path.name if file_path else 'unknown',
                'error': str(e)
            }
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024.0 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"


class FileScanner:
    """Directory scanning utilities."""
    
    def __init__(self, validator: Optional[FileValidator] = None):
        self.validator = validator or FileValidator()
    
    def scan_directory(
        self, 
        directory: Path, 
        recursive: bool = True,
        include_invalid: bool = False
    ) -> Dict[str, List[Path]]:
        """Scan directory for files and categorize them."""
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory not found or not a directory: {directory}")
            return {'valid': [], 'invalid': [], 'errors': []}
        
        results = {
            'valid': [],
            'invalid': [],
            'errors': []
        }
        
        # Get file pattern
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        # Scan files
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    is_valid, error = self.validator.is_valid_file(file_path)
                    
                    if is_valid:
                        results['valid'].append(file_path)
                    else:
                        if include_invalid:
                            results['invalid'].append((file_path, error))
                        logger.debug(f"Invalid file {file_path}: {error}")
                        
                except Exception as e:
                    results['errors'].append((file_path, str(e)))
                    logger.error(f"Error scanning file {file_path}: {e}")
        
        logger.info(
            f"Directory scan complete: {len(results['valid'])} valid, "
            f"{len(results['invalid'])} invalid, {len(results['errors'])} errors"
        )
        
        return results
    
    def get_file_statistics(self, files: List[Path]) -> Dict[str, Any]:
        """Get statistics about a list of files."""
        if not files:
            return {
                'total_files': 0,
                'total_size': 0,
                'total_size_human': '0 B',
                'extensions': {},
                'size_distribution': {}
            }
        
        total_size = 0
        extensions = {}
        size_ranges = {
            'small': 0,      # < 1MB
            'medium': 0,     # 1MB - 10MB
            'large': 0,      # 10MB - 100MB
            'huge': 0        # > 100MB
        }
        
        for file_path in files:
            try:
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Count extensions
                ext = file_path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                
                # Size distribution
                if file_size < 1024 * 1024:  # < 1MB
                    size_ranges['small'] += 1
                elif file_size < 10 * 1024 * 1024:  # < 10MB
                    size_ranges['medium'] += 1
                elif file_size < 100 * 1024 * 1024:  # < 100MB
                    size_ranges['large'] += 1
                else:  # > 100MB
                    size_ranges['huge'] += 1
                    
            except Exception as e:
                logger.warning(f"Could not get size for {file_path}: {e}")
        
        return {
            'total_files': len(files),
            'total_size': total_size,
            'total_size_human': FileMetadata._format_file_size(total_size),
            'extensions': extensions,
            'size_distribution': size_ranges,
            'average_size': total_size // len(files) if files else 0,
            'average_size_human': FileMetadata._format_file_size(total_size // len(files)) if files else '0 B'
        }


# Global utility instances
file_validator = FileValidator()
file_scanner = FileScanner()
file_hasher = FileHasher()
file_metadata = FileMetadata()