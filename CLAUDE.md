# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Agent System - A local document ingestion and retrieval system with MCP tools for agent interactions. Uses PostgreSQL+pgvector for storage, Ollama for local LLM/embedding models, and async Python processing.

## Key Commands

### Development & Testing
```bash
# Setup environment (run once)
python scripts/setup.py

# Health check all services
python scripts/health_check.py

# Start PostgreSQL
docker-compose up -d

# Stop services
docker-compose down

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Run application
uv run rag-agent

# Run MCP server
python scripts/mcp_server.py

# Run tests
uv run pytest

# Code formatting
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

### Database Operations
```bash
# Create tables
python -c "from src.database.connection import db_manager; db_manager.initialize_all(); db_manager.create_tables()"

# Reset database (WARNING: destructive)
docker-compose down -v && docker-compose up -d
```

### Ollama Model Management
```bash
# Check available models
curl http://localhost:11434/api/tags

# Pull required models
ollama pull nomic-embed-text
ollama pull incept5/llama3.1-claude:latest
```

## Architecture

### Core Components
- **src/core/**: Configuration, logging, health monitoring
- **src/database/**: SQLAlchemy models for documents, chunks, chat memory
- **src/ingest/**: File watching, text extraction, chunking, embedding
- **src/agent/**: Ollama LLM client and agent logic
- **src/mcp_tools/**: MCP server with document/query/memory tools
- **src/utils/**: Ollama service management

### Database Schema
- `documents`: File metadata (filepath, size, status, timestamps)
- `document_chunks`: Text chunks with embeddings (768-dim vectors)
- `chat_memory`: Conversation persistence by session_id

### Configuration
- Settings in `config/settings.py` using Pydantic
- Environment variables via `.env` file
- Nested config: `database__host`, `ollama__base_url`

## Common Tasks

### Adding New File Types
1. Update `IngestSettings.supported_extensions` in `config/settings.py`
2. Add extraction logic in `src/ingest/text_extractor.py`
3. Test with sample files in `data/` directory

### Adding MCP Tools
1. Create tool in `src/mcp_tools/` (document_tools.py, query_tools.py, etc.)
2. Register in `src/mcp_tools/server.py`
3. Follow MCP tool schema: name, description, parameters, handler

### Database Migrations
1. Modify models in `src/database/models.py`
2. Update `scripts/init.sql` for new installations
3. Create migration script for existing databases

### Debugging Issues
- Check logs in `logs/rag-agent.log` (JSON format)
- Use `python scripts/health_check.py` for service status
- PostgreSQL logs: `docker-compose logs postgres`
- Ollama status: `curl http://localhost:11434/api/tags`

## Required Models

### Ollama Models
- **Embedding**: `nomic-embed-text` (768 dimensions)
- **LLM**: `incept5/llama3.1-claude:latest` 

Models auto-pull during setup if missing. Check with health_check.py.

## File Processing Flow

1. **File Watch**: `src/ingest/file_watcher.py` monitors `data/` directory
2. **Text Extract**: `src/ingest/text_extractor.py` handles PDF/DOCX/TXT
3. **Chunking**: `src/ingest/chunker.py` splits text with overlap
4. **Embedding**: `src/ingest/embedder.py` calls Ollama nomic-embed-text
5. **Storage**: Chunks + embeddings stored in PostgreSQL

## MCP Tools Available

- `list_documents`: Get document metadata with filters
- `get_file_contents`: Retrieve original file content
- `query_documents`: Semantic search using pgvector cosine similarity
- `get_document_chunks`: Access specific text chunks
- `store_chat_memory` / `retrieve_chat_memory`: Session persistence

## Dependencies

### Core Dependencies
- **sqlalchemy**: ORM with async support
- **psycopg2-binary**: PostgreSQL driver
- **pgvector**: Vector extension integration
- **watchdog**: File system monitoring
- **ollama**: API client for local models
- **pydantic**: Configuration and validation
- **loguru**: Structured logging

### Text Processing
- **textract**: Multi-format text extraction
- **pypdf**: PDF processing
- **python-docx**: Word document processing

## Configuration Notes

- Default database: `rag_db` on localhost:5432
- Default Ollama: localhost:11434
- Watch directory: `./data/`
- Chunk size: 1000 chars with 200 char overlap
- Vector dimensions: 768 (nomic-embed-text)
- Connection pool: 10 connections, 20 overflow

## Performance Considerations

- File processing is async to prevent blocking
- Database connection pooling configured
- pgvector indexes for fast similarity search
- Ollama request rate limiting to prevent overload
- Health checks with circuit breaker patterns

## Common Errors

- **Ollama connection failed**: Check `curl http://localhost:11434/api/tags`
- **pgvector not found**: Ensure PostgreSQL container has pgvector extension
- **Model not available**: Run `python scripts/setup.py` to pull models
- **Permission denied on files**: Check file permissions in `data/` directory
- **Database connection refused**: Start PostgreSQL with `docker-compose up -d`