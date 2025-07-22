#!/usr/bin/env python3
"""Setup script for RAG Agent System."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.core.config import initialize_config
from src.core.logging import setup_logging
from src.database.connection import db_manager
from src.utils.ollama_manager import ollama_manager


async def setup_environment():
    """Setup the complete environment."""
    logger.info("Starting RAG Agent System setup...")
    
    # 1. Initialize configuration
    logger.info("Initializing configuration...")
    initialize_config()
    
    # 2. Initialize database
    logger.info("Initializing database connections...")
    db_manager.initialize_all()
    
    # 3. Check database connectivity
    logger.info("Checking database connectivity...")
    if not await db_manager.check_connection():
        logger.error("Database connection failed")
        return False
    
    # 4. Check vector extension
    logger.info("Checking pgvector extension...")
    if not await db_manager.check_vector_extension():
        logger.error("pgvector extension not available")
        return False
    
    # 5. Create tables
    logger.info("Creating database tables...")
    db_manager.create_tables()
    
    # 6. Check Ollama service
    logger.info("Checking Ollama service...")
    health = await ollama_manager.health_check()
    if health["status"] == "unhealthy":
        logger.error(f"Ollama service unhealthy: {health.get('error')}")
        return False
    
    # 7. Ensure required models
    logger.info("Ensuring required Ollama models...")
    if not await ollama_manager.ensure_models():
        logger.error("Failed to ensure required models")
        return False
    
    logger.info("✅ RAG Agent System setup completed successfully!")
    return True


async def main():
    """Main setup function."""
    setup_logging()
    
    try:
        success = await setup_environment()
        if not success:
            logger.error("❌ Setup failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        sys.exit(1)
    finally:
        await db_manager.close()
        await ollama_manager.close()


if __name__ == "__main__":
    asyncio.run(main())