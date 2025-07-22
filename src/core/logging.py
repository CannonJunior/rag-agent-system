import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


def setup_logging():
    """Configure structured logging with Loguru."""
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format=settings.logging.format,
        level=settings.logging.level,
        colorize=True,
        diagnose=True,
    )
    
    # File handler if specified
    if settings.logging.log_file:
        log_path = Path(settings.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format=settings.logging.format,
            level=settings.logging.level,
            rotation=settings.logging.rotation,
            retention=settings.logging.retention,
            compression="gz",
            serialize=True,  # JSON format for structured logging
        )
    
    logger.info("Logging configured successfully")