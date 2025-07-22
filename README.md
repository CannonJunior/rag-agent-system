# RAG Agent System

A local RAG (Retrieval-Augmented Generation) agent system with MCP tools, PostgreSQL storage, and Ollama integration.

## Features

- **Document Ingestion**: Automatic monitoring and processing of local documents
- **Vector Storage**: PostgreSQL with pgvector for semantic search
- **Local Models**: Ollama integration with configurable embedding and LLM models
- **MCP Tools**: Standardized agent tools for document operations
- **Async Processing**: Non-blocking file processing and embedding generation
- **Health Monitoring**: Service health checks and automatic recovery

## Quick Start

1. **Prerequisites**:
   - Python 3.11+
   - Docker & Docker Compose
   - Ollama service running locally

2. **Setup**:
   ```bash
   # Install with uv
   uv sync
   
   # Start PostgreSQL
   docker-compose up -d
   
   # Setup environment
   python scripts/setup.py
   
   # Check health
   python scripts/health_check.py
   ```

3. **Configuration**: Copy `.env.example` to `.env` and customize settings

4. **Run**: `uv run rag-agent` or `python scripts/mcp_server.py`

## Architecture

- **src/core/**: Configuration, logging, health monitoring
- **src/database/**: PostgreSQL models and connection management
- **src/ingest/**: File watching, text extraction, embedding generation
- **src/agent/**: LLM client and agent logic
- **src/mcp_tools/**: MCP server and tool implementations
- **src/utils/**: Ollama management and utilities

## Required Models

- **Embedding**: `nomic-embed-text`
- **LLM**: `incept5/llama3.1-claude:latest`

Models are automatically pulled during setup if not available.

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```