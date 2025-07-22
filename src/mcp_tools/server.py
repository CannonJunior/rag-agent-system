import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime
import traceback

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    CallToolRequest, CallToolResult, ListResourcesRequest, ListResourcesResult,
    ListToolsRequest, ListToolsResult, ReadResourceRequest, ReadResourceResult
)
from loguru import logger

from config.settings import settings
from src.core.logging import setup_logging
from src.database.connection import db_manager
from src.mcp_tools.document_tools import DocumentTools
from src.mcp_tools.query_tools import QueryTools
from src.mcp_tools.memory_tools import MemoryTools
from src.mcp_tools.health_tools import HealthTools


class RAGMCPServer:
    """MCP Server for RAG Agent System."""
    
    def __init__(self):
        self.server = Server("rag-agent-system")
        self.document_tools = DocumentTools()
        self.query_tools = QueryTools()
        self.memory_tools = MemoryTools()
        self.health_tools = HealthTools()
        
        # Tool registry
        self.tools = {}
        self.resources = {}
        
        # Setup handlers
        self._setup_handlers()
        self._register_tools()
        self._register_resources()
    
    def _setup_handlers(self):
        """Setup MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return list(self.tools.values())
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> List[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls."""
            try:
                logger.info(f"Tool call: {name} with args: {arguments}")
                
                if name not in self.tools:
                    raise ValueError(f"Unknown tool: {name}")
                
                # Get tool handler
                tool_handler = getattr(self, f"_handle_{name}")
                result = await tool_handler(arguments or {})
                
                # Ensure result is properly formatted
                if isinstance(result, str):
                    return [TextContent(type="text", text=result)]
                elif isinstance(result, dict):
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                elif isinstance(result, list) and all(isinstance(item, (TextContent, ImageContent, EmbeddedResource)) for item in result):
                    return result
                else:
                    return [TextContent(type="text", text=str(result))]
                
            except Exception as e:
                logger.error(f"Error handling tool {name}: {e}")
                error_msg = f"Error in {name}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                return [TextContent(type="text", text=error_msg)]
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources."""
            return list(self.resources.values())
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read resource content."""
            try:
                if uri.startswith("document://"):
                    document_id = int(uri.split("//")[1])
                    return await self.document_tools.get_document_content(document_id)
                elif uri.startswith("stats://"):
                    stats_type = uri.split("//")[1]
                    if stats_type == "system":
                        stats = await self.health_tools.get_system_stats()
                        return json.dumps(stats, indent=2)
                    elif stats_type == "documents":
                        stats = await self.document_tools.get_document_stats()
                        return json.dumps(stats, indent=2)
                else:
                    raise ValueError(f"Unknown resource URI: {uri}")
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                return f"Error reading resource: {str(e)}"
    
    def _register_tools(self):
        """Register all available tools."""
        
        # Document management tools
        self.tools["list_documents"] = Tool(
            name="list_documents",
            description="List documents in the knowledge base with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of documents to return", "default": 50},
                    "offset": {"type": "integer", "description": "Number of documents to skip", "default": 0},
                    "filename_filter": {"type": "string", "description": "Filter by filename (partial match)"},
                    "content_type_filter": {"type": "string", "description": "Filter by content type"},
                    "status_filter": {"type": "string", "description": "Filter by processing status"},
                    "sort_by": {"type": "string", "enum": ["created_at", "updated_at", "filename", "file_size"], "default": "created_at"},
                    "sort_order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"}
                }
            }
        )
        
        self.tools["get_document"] = Tool(
            name="get_document",
            description="Get detailed information about a specific document",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "ID of the document"},
                    "include_content": {"type": "boolean", "description": "Whether to include full text content", "default": False},
                    "include_chunks": {"type": "boolean", "description": "Whether to include chunk information", "default": False}
                },
                "required": ["document_id"]
            }
        )
        
        self.tools["get_file_contents"] = Tool(
            name="get_file_contents",
            description="Get the original file contents of a document",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "ID of the document"},
                    "chunk_size": {"type": "integer", "description": "Max characters to return", "default": 10000}
                },
                "required": ["document_id"]
            }
        )
        
        # Query and search tools
        self.tools["query_documents"] = Tool(
            name="query_documents",
            description="Perform semantic search across documents using vector similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 10},
                    "similarity_threshold": {"type": "number", "description": "Minimum similarity score (0-1)", "default": 0.7},
                    "document_ids": {"type": "array", "items": {"type": "integer"}, "description": "Limit search to specific documents"},
                    "content_types": {"type": "array", "items": {"type": "string"}, "description": "Limit search to specific content types"},
                    "include_content": {"type": "boolean", "description": "Include chunk content in results", "default": True}
                },
                "required": ["query"]
            }
        )
        
        self.tools["get_document_chunks"] = Tool(
            name="get_document_chunks",
            description="Get text chunks from a specific document",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "ID of the document"},
                    "limit": {"type": "integer", "description": "Maximum number of chunks", "default": 20},
                    "offset": {"type": "integer", "description": "Number of chunks to skip", "default": 0},
                    "include_embeddings": {"type": "boolean", "description": "Include embedding vectors", "default": False}
                },
                "required": ["document_id"]
            }
        )
        
        self.tools["search_chunks"] = Tool(
            name="search_chunks",
            description="Search for specific text within document chunks",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_text": {"type": "string", "description": "Text to search for"},
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 10},
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False},
                    "whole_words": {"type": "boolean", "description": "Match whole words only", "default": False},
                    "document_ids": {"type": "array", "items": {"type": "integer"}, "description": "Limit search to specific documents"}
                },
                "required": ["search_text"]
            }
        )
        
        # Chat memory tools
        self.tools["store_chat_memory"] = Tool(
            name="store_chat_memory",
            description="Store a chat message in conversation memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Conversation session ID"},
                    "message_type": {"type": "string", "enum": ["user", "assistant", "system"], "description": "Type of message"},
                    "content": {"type": "string", "description": "Message content"},
                    "metadata": {"type": "object", "description": "Additional metadata for the message"}
                },
                "required": ["session_id", "message_type", "content"]
            }
        )
        
        self.tools["retrieve_chat_memory"] = Tool(
            name="retrieve_chat_memory",
            description="Retrieve chat conversation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Conversation session ID"},
                    "limit": {"type": "integer", "description": "Maximum number of messages", "default": 50},
                    "message_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by message types"},
                    "since": {"type": "string", "format": "date-time", "description": "Only messages after this time"}
                },
                "required": ["session_id"]
            }
        )
        
        self.tools["clear_chat_memory"] = Tool(
            name="clear_chat_memory",
            description="Clear chat memory for a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Conversation session ID"},
                    "confirm": {"type": "boolean", "description": "Confirmation to clear memory", "default": False}
                },
                "required": ["session_id", "confirm"]
            }
        )
        
        # Health and monitoring tools
        self.tools["health_check"] = Tool(
            name="health_check",
            description="Check system health and service status",
            inputSchema={
                "type": "object",
                "properties": {
                    "detailed": {"type": "boolean", "description": "Include detailed health information", "default": False}
                }
            }
        )
        
        self.tools["get_system_stats"] = Tool(
            name="get_system_stats",
            description="Get system statistics and metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_embedding_stats": {"type": "boolean", "description": "Include embedding performance stats", "default": True},
                    "include_processing_stats": {"type": "boolean", "description": "Include file processing stats", "default": True}
                }
            }
        )
    
    def _register_resources(self):
        """Register available resources."""
        
        self.resources["system_stats"] = Resource(
            uri="stats://system",
            name="System Statistics",
            description="Real-time system statistics and metrics",
            mimeType="application/json"
        )
        
        self.resources["document_stats"] = Resource(
            uri="stats://documents",
            name="Document Statistics",
            description="Document collection statistics",
            mimeType="application/json"
        )
    
    # Tool handlers
    async def _handle_list_documents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_documents tool call."""
        return await self.document_tools.list_documents(**args)
    
    async def _handle_get_document(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_document tool call."""
        return await self.document_tools.get_document(**args)
    
    async def _handle_get_file_contents(self, args: Dict[str, Any]) -> str:
        """Handle get_file_contents tool call."""
        return await self.document_tools.get_file_contents(**args)
    
    async def _handle_query_documents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query_documents tool call."""
        return await self.query_tools.query_documents(**args)
    
    async def _handle_get_document_chunks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_document_chunks tool call."""
        return await self.document_tools.get_document_chunks(**args)
    
    async def _handle_search_chunks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_chunks tool call."""
        return await self.query_tools.search_chunks(**args)
    
    async def _handle_store_chat_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle store_chat_memory tool call."""
        return await self.memory_tools.store_message(**args)
    
    async def _handle_retrieve_chat_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retrieve_chat_memory tool call."""
        return await self.memory_tools.retrieve_messages(**args)
    
    async def _handle_clear_chat_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clear_chat_memory tool call."""
        return await self.memory_tools.clear_session(**args)
    
    async def _handle_health_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health_check tool call."""
        return await self.health_tools.health_check(**args)
    
    async def _handle_get_system_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_system_stats tool call."""
        return await self.health_tools.get_system_stats(**args)
    
    async def run(self, transport_type: str = "stdio"):
        """Run the MCP server."""
        setup_logging()
        logger.info("Starting RAG MCP Server...")
        
        # Initialize database
        db_manager.initialize_all()
        
        # Check database connectivity
        if not await db_manager.check_connection():
            logger.error("Database connection failed")
            raise RuntimeError("Cannot connect to database")
        
        logger.info("Database connection established")
        
        if transport_type == "stdio":
            import mcp.server.stdio
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="rag-agent-system",
                        server_version="1.0.0",
                        capabilities={
                            "tools": {},
                            "resources": {}
                        }
                    )
                )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")


async def main():
    """Main entry point for MCP server."""
    server = RAGMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())