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

## Usage Guide

### Ingesting Sample Data

The RAG system supports multiple file formats and provides several ways to ingest documents for retrieval and question answering.

#### Supported File Types
- **Text files**: `.txt`, `.md`, `.py`, `.js`, `.json`, `.xml`, `.html`, `.csv`
- **Documents**: `.pdf`, `.docx`, `.doc`, `.rtf`, `.odt`
- **Structured data**: `.yaml`, `.yml`

#### Method 1: One-time Document Processing

Process a directory of documents immediately:

```bash
# Process all supported files in a directory
uv run rag-agent ingest --directory ./documents

# Process recursively (includes subdirectories)
uv run rag-agent ingest --directory ./documents --recursive

# Process single directory level only
uv run rag-agent ingest --directory ./documents --no-recursive
```

#### Method 2: Real-time File Watching

Start the file watcher for automatic processing of new documents:

```bash
# Start file watcher (monitors ./data directory by default)
uv run rag-agent watch

# The watcher will automatically process any files added to ./data/
# Press Ctrl+C to stop watching
```

#### Method 3: Using the Data Directory

Simply copy files to the `data/` directory and they'll be processed automatically if the watcher is running:

```bash
# Create sample documents
mkdir -p data/sample-docs

# Add some sample files
echo "Python is a programming language." > data/sample-docs/python-intro.txt
echo "Machine learning uses algorithms to find patterns." > data/sample-docs/ml-basics.txt

# Create a markdown file
cat > data/sample-docs/rag-overview.md << 'EOF'
# RAG (Retrieval-Augmented Generation)

RAG combines information retrieval with text generation.

## Key Components
- **Retrieval**: Finding relevant documents
- **Augmentation**: Adding context to prompts  
- **Generation**: Creating responses with LLMs

## Benefits
- More accurate responses
- Up-to-date information
- Reduced hallucinations
EOF

# If file watcher is running, files will be processed automatically
# Otherwise, process them manually:
uv run rag-agent ingest --directory data/sample-docs
```

#### Processing Status

Monitor processing status:

```bash
# Check system health and processing stats
uv run rag-agent status

# View detailed health information
uv run scripts/health_check.py
```

### Querying Your Data

Once documents are ingested, you can search through them:

```bash
# Semantic search using vector similarity
uv run rag-agent search --query "What is machine learning?"

# Limit results and adjust similarity threshold
uv run rag-agent search --query "Python programming" --limit 3 --threshold 0.8
```

### Using with MCP Agents

Start the MCP server to integrate with agents:

```bash
# Start MCP server for agent integration
python scripts/mcp_server.py

# The server provides these tools:
# - list_documents: Browse your document collection
# - query_documents: Semantic search across documents  
# - get_file_contents: Retrieve original file content
# - store_chat_memory: Save conversation context
# - retrieve_chat_memory: Load conversation history
```

### Example Workflow

Complete workflow for setting up a knowledge base:

```bash
# 1. Setup the system
uv sync
docker-compose up -d
uv run scripts/setup.py

# 2. Prepare sample documents
mkdir -p data/knowledge-base
# Add your PDF, DOCX, TXT, MD files to data/knowledge-base/

# 3. Process documents
uv run rag-agent ingest --directory data/knowledge-base

# 4. Test retrieval
uv run rag-agent search --query "your question here"

# 5. Start MCP server for agent integration
python scripts/mcp_server.py
```

### Performance Tips

- **File sizes**: Keep individual files under 100MB for optimal processing
- **Batch processing**: Use `ingest` command for large document collections
- **Concurrent processing**: The system automatically handles multiple files concurrently
- **Embedding quality**: Ensure good text extraction by using supported file formats

### Troubleshooting

```bash
# Check if all services are healthy
uv run scripts/health_check.py

# View processing logs
tail -f logs/rag-agent.log

# Check database status
docker-compose logs postgres

# Verify Ollama models
curl http://localhost:11434/api/tags
```

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