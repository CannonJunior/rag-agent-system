from pathlib import Path
from config.settings import settings
from loguru import logger


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.ingest.watch_directory,
        Path("logs"),
        Path("data"),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def validate_settings():
    """Validate configuration settings."""
    errors = []
    
    # Check watch directory
    if not settings.ingest.watch_directory.exists():
        try:
            settings.ingest.watch_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create watch directory: {e}")
    
    # Validate chunk settings
    if settings.ingest.chunk_size <= 0:
        errors.append("Chunk size must be positive")
    
    if settings.ingest.chunk_overlap >= settings.ingest.chunk_size:
        errors.append("Chunk overlap must be less than chunk size")
    
    # Validate Ollama settings
    if not settings.ollama.base_url.startswith("http"):
        errors.append("Ollama base URL must be a valid HTTP(S) URL")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    logger.info("Configuration validation passed")


def initialize_config():
    """Initialize configuration and setup."""
    setup_directories()
    validate_settings()
    logger.info("Configuration initialized successfully")