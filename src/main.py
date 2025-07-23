#!/usr/bin/env python3
"""Main entry point for RAG Agent System."""

import asyncio
import click
from pathlib import Path

from loguru import logger
from src.core.logging import setup_logging
from src.core.config import initialize_config
from src.database.connection import db_manager
from src.ingest.processor import document_processor
from src.ingest.file_watcher import file_watcher
from src.mcp_tools.server import RAGMCPServer


@click.group()
def cli():
    """RAG Agent System - Document ingestion and querying with local models."""
    pass


@cli.command()
@click.option('--transport', default='stdio', help='Transport type for MCP server')
async def mcp_server(transport: str):
    """Start the MCP server for agent integration."""
    setup_logging()
    logger.info("Starting RAG MCP Server...")
    
    server = RAGMCPServer()
    await server.run(transport)


@cli.command()
@click.option('--directory', type=click.Path(exists=True, path_type=Path), help='Directory to process')
@click.option('--recursive/--no-recursive', default=True, help='Process subdirectories')
async def ingest(directory: Path, recursive: bool):
    """Process documents in a directory."""
    setup_logging()
    initialize_config()
    
    logger.info(f"Starting document ingestion from {directory}")
    
    # Initialize database
    db_manager.initialize_all()
    
    # Start document processor
    await document_processor.start()
    
    try:
        # Process directory
        results = await document_processor.process_directory(directory, recursive)
        
        # Report results
        successful = sum(1 for r in results if r.status.value == "completed")
        failed = sum(1 for r in results if r.status.value == "failed")
        skipped = sum(1 for r in results if r.status.value == "skipped")
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed, {skipped} skipped")
        
        # Show detailed results
        for result in results:
            if result.status.value == "completed":
                logger.info(f"✓ {result.file_path}: {result.chunks_created} chunks, {result.embeddings_created} embeddings")
            elif result.status.value == "failed":
                logger.error(f"✗ {result.file_path}: {result.error}")
            elif result.status.value == "skipped":
                logger.warning(f"⚠ {result.file_path}: {result.error}")
    
    finally:
        await document_processor.stop()
        await db_manager.close()


@cli.command()
async def watch():
    """Start file watcher for automatic document processing."""
    setup_logging()
    initialize_config()
    
    logger.info("Starting file watcher and document processor...")
    
    # Initialize database
    db_manager.initialize_all()
    
    # Start services
    await document_processor.start()
    await file_watcher.start()
    
    try:
        logger.info("File watcher active. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(10)
            
            # Log periodic stats
            watcher_stats = await file_watcher.get_stats()
            processor_stats = await document_processor.get_processing_stats()
            
            logger.info(
                f"Status: {watcher_stats['events_processed']} events processed, "
                f"{processor_stats['queue_size']} files queued, "
                f"{processor_stats['files_processed']} files completed"
            )
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    
    finally:
        await file_watcher.stop()
        await document_processor.stop()
        await db_manager.close()
        logger.info("Shutdown complete")


@cli.command()
@click.option('--query', required=True, help='Search query')
@click.option('--limit', default=5, help='Maximum results to return')
@click.option('--threshold', default=0.7, help='Similarity threshold')
async def search(query: str, limit: int, threshold: float):
    """Search documents using semantic similarity."""
    setup_logging()
    initialize_config()
    
    logger.info(f"Searching for: {query}")
    
    # Initialize database
    db_manager.initialize_all()
    
    try:
        from src.mcp_tools.query_tools import QueryTools
        query_tools = QueryTools()
        
        results = await query_tools.query_documents(
            query=query,
            limit=limit,
            similarity_threshold=threshold
        )
        
        if "error" in results:
            logger.error(f"Search failed: {results['error']}")
            return
        
        logger.info(f"Found {len(results['results'])} results")
        
        for i, result in enumerate(results['results'], 1):
            logger.info(f"\n--- Result {i} ---")
            logger.info(f"Document: {result['document_filename']}")
            logger.info(f"Similarity: {result['similarity_score']:.3f}")
            logger.info(f"Content: {result.get('content_preview', result.get('content', ''))[:200]}...")
    
    finally:
        await db_manager.close()


@cli.command()
async def status():
    """Show system status and health."""
    setup_logging()
    initialize_config()
    
    # Initialize database
    db_manager.initialize_all()
    
    try:
        from src.mcp_tools.health_tools import HealthTools
        health_tools = HealthTools()
        
        # Get health status
        health = await health_tools.health_check(detailed=True)
        
        logger.info("=== System Status ===")
        logger.info(f"Overall Status: {health.get('overall', {}).get('status', 'unknown')}")
        
        for service, status in health.items():
            if service == "overall":
                continue
            
            service_status = status.get('status', 'unknown')
            logger.info(f"{service.title()}: {service_status}")
            
            if service_status == "unhealthy" and 'error' in status:
                logger.error(f"  Error: {status['error']}")
        
        # Get system stats
        stats = await health_tools.get_system_stats()
        
        if 'database_stats' in stats:
            db_stats = stats['database_stats']
            logger.info(f"\nDatabase: {db_stats.get('documents', 0)} documents, {db_stats.get('document_chunks', 0)} chunks")
        
        if 'service_stats' in stats and 'embedding' in stats['service_stats']:
            embed_stats = stats['service_stats']['embedding']
            if 'metrics' in embed_stats:
                metrics = embed_stats['metrics']
                logger.info(f"Embeddings: {metrics.get('successful_requests', 0)} successful, {metrics.get('failed_requests', 0)} failed")
    
    finally:
        await db_manager.close()


def main():
    """Async main wrapper."""
    def run_async(func):
        def wrapper(*args, **kwargs):
            return asyncio.run(func(*args, **kwargs))
        return wrapper
    
    # Make commands async-aware
    for command in [mcp_server, ingest, watch, search, status]:
        command.callback = run_async(command.callback)
    
    cli()


if __name__ == "__main__":
    main()