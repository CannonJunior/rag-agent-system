from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import asyncio

from loguru import logger

from src.core.health import health_checker
from src.database.connection import db_manager
from src.utils.ollama_manager import ollama_manager
from src.ingest.embedder import document_embedder
from src.ingest.processor import document_processor
from src.ingest.file_watcher import file_watcher


class HealthTools:
    """MCP tools for system health monitoring and diagnostics."""
    
    async def health_check(self, detailed: bool = False) -> Dict[str, Any]:
        """Check system health and service status."""
        try:
            # Get comprehensive health status
            health_status = await health_checker.check_all_services()
            
            if not detailed:
                # Return simplified health status
                return {
                    "overall_status": health_status["overall"]["status"],
                    "timestamp": health_status["overall"]["timestamp"],
                    "services": {
                        service: status["status"] 
                        for service, status in health_status.items() 
                        if service != "overall"
                    }
                }
            
            # Detailed health information
            detailed_health = health_status.copy()
            
            # Add database connection details
            if "postgres" in detailed_health:
                try:
                    db_connected = await db_manager.check_connection()
                    vector_available = await db_manager.check_vector_extension()
                    detailed_health["postgres"]["connection_test"] = db_connected
                    detailed_health["postgres"]["vector_extension"] = vector_available
                except Exception as e:
                    detailed_health["postgres"]["connection_error"] = str(e)
            
            # Add embedding service details
            if "ollama" in detailed_health:
                try:
                    embedding_stats = await document_embedder.get_embedding_stats()
                    detailed_health["ollama"]["embedding_stats"] = embedding_stats
                except Exception as e:
                    detailed_health["ollama"]["embedding_error"] = str(e)
            
            # Add file processing status
            try:
                processing_stats = await document_processor.get_processing_stats()
                detailed_health["file_processing"] = {
                    "status": "healthy" if processing_stats["is_running"] else "stopped",
                    "queue_size": processing_stats["queue_size"],
                    "active_workers": processing_stats["active_workers"],
                    "files_processed": processing_stats["files_processed"],
                    "files_failed": processing_stats["files_failed"]
                }
            except Exception as e:
                detailed_health["file_processing"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Add file watcher status
            try:
                watcher_stats = await file_watcher.get_stats()
                detailed_health["file_watcher"] = {
                    "status": "healthy" if watcher_stats["is_running"] else "stopped",
                    "events_received": watcher_stats["events_received"],
                    "events_processed": watcher_stats["events_processed"],
                    "queue_size": watcher_stats["queue_size"]
                }
            except Exception as e:
                detailed_health["file_watcher"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            return detailed_health
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_system_stats(
        self,
        include_embedding_stats: bool = True,
        include_processing_stats: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive system statistics and metrics."""
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_info": self._get_system_info(),
                "database_stats": await self._get_database_stats(),
                "service_stats": {}
            }
            
            # Embedding statistics
            if include_embedding_stats:
                try:
                    embedding_stats = await document_embedder.get_embedding_stats()
                    stats["service_stats"]["embedding"] = embedding_stats
                except Exception as e:
                    stats["service_stats"]["embedding"] = {"error": str(e)}
            
            # File processing statistics
            if include_processing_stats:
                try:
                    processing_stats = await document_processor.get_processing_stats()
                    stats["service_stats"]["file_processing"] = processing_stats
                    
                    watcher_stats = await file_watcher.get_stats()
                    stats["service_stats"]["file_watcher"] = watcher_stats
                except Exception as e:
                    stats["service_stats"]["processing"] = {"error": str(e)}
            
            # Ollama service statistics
            try:
                ollama_health = await ollama_manager.health_check()
                stats["service_stats"]["ollama"] = ollama_health
            except Exception as e:
                stats["service_stats"]["ollama"] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        try:
            # System performance
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            performance_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_performance": {
                    "cpu": {
                        "usage_percent": cpu_percent,
                        "count": psutil.cpu_count(),
                        "count_logical": psutil.cpu_count(logical=True)
                    },
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "used": memory.used,
                        "usage_percent": memory.percent,
                        "total_human": self._bytes_to_human(memory.total),
                        "available_human": self._bytes_to_human(memory.available),
                        "used_human": self._bytes_to_human(memory.used)
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "usage_percent": disk.used / disk.total * 100,
                        "total_human": self._bytes_to_human(disk.total),
                        "used_human": self._bytes_to_human(disk.used),
                        "free_human": self._bytes_to_human(disk.free)
                    }
                }
            }
            
            # Database performance
            try:
                db_metrics = await self._get_database_performance()
                performance_metrics["database_performance"] = db_metrics
            except Exception as e:
                performance_metrics["database_performance"] = {"error": str(e)}
            
            # Embedding performance
            try:
                embedding_metrics = document_embedder.client.get_metrics()
                performance_metrics["embedding_performance"] = embedding_metrics
            except Exception as e:
                performance_metrics["embedding_performance"] = {"error": str(e)}
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_service_diagnostics(self, service_name: str) -> Dict[str, Any]:
        """Get diagnostic information for a specific service."""
        try:
            diagnostics = {
                "service": service_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if service_name == "database":
                diagnostics.update(await self._diagnose_database())
            elif service_name == "ollama":
                diagnostics.update(await self._diagnose_ollama())
            elif service_name == "embedding":
                diagnostics.update(await self._diagnose_embedding_service())
            elif service_name == "file_processing":
                diagnostics.update(await self._diagnose_file_processing())
            elif service_name == "file_watcher":
                diagnostics.update(await self._diagnose_file_watcher())
            else:
                diagnostics["error"] = f"Unknown service: {service_name}"
                diagnostics["available_services"] = [
                    "database", "ollama", "embedding", "file_processing", "file_watcher"
                ]
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error getting diagnostics for {service_name}: {e}")
            return {
                "service": service_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            return {
                "platform": psutil.os.name,
                "boot_time": boot_time.isoformat(),
                "uptime_seconds": uptime.total_seconds(),
                "uptime_human": str(uptime),
                "python_version": f"{psutil.version_info}",
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            async with db_manager.get_async_session() as session:
                # Get table sizes and counts
                from sqlalchemy import text
                
                # Document counts
                doc_count_result = await session.execute(text("SELECT COUNT(*) FROM documents"))
                doc_count = doc_count_result.scalar()
                
                chunk_count_result = await session.execute(text("SELECT COUNT(*) FROM document_chunks"))
                chunk_count = chunk_count_result.scalar()
                
                memory_count_result = await session.execute(text("SELECT COUNT(*) FROM chat_memory"))
                memory_count = memory_count_result.scalar()
                
                # Database size
                try:
                    size_result = await session.execute(text("SELECT pg_database_size(current_database())"))
                    db_size = size_result.scalar()
                except Exception:
                    db_size = None
                
                return {
                    "documents": doc_count,
                    "document_chunks": chunk_count,
                    "chat_messages": memory_count,
                    "database_size_bytes": db_size,
                    "database_size_human": self._bytes_to_human(db_size) if db_size else "unknown"
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_database_performance(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        try:
            async with db_manager.get_async_session() as session:
                from sqlalchemy import text
                
                # Active connections
                connections_result = await session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                ))
                active_connections = connections_result.scalar()
                
                # Database statistics
                stats_result = await session.execute(text("""
                    SELECT 
                        tup_returned,
                        tup_fetched,
                        tup_inserted,
                        tup_updated,
                        tup_deleted
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """))
                stats = stats_result.first()
                
                return {
                    "active_connections": active_connections,
                    "tuples_returned": stats.tup_returned if stats else 0,
                    "tuples_fetched": stats.tup_fetched if stats else 0,
                    "tuples_inserted": stats.tup_inserted if stats else 0,
                    "tuples_updated": stats.tup_updated if stats else 0,
                    "tuples_deleted": stats.tup_deleted if stats else 0
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def _diagnose_database(self) -> Dict[str, Any]:
        """Diagnose database issues."""
        diagnostics = {}
        
        try:
            # Connection test
            connected = await db_manager.check_connection()
            diagnostics["connection"] = "healthy" if connected else "failed"
            
            # Vector extension test
            vector_available = await db_manager.check_vector_extension()
            diagnostics["pgvector_extension"] = "available" if vector_available else "missing"
            
            # Table existence
            async with db_manager.get_async_session() as session:
                from sqlalchemy import text
                
                tables_result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                tables = [row[0] for row in tables_result]
                
                expected_tables = ['documents', 'document_chunks', 'chat_memory']
                diagnostics["tables"] = {
                    "expected": expected_tables,
                    "present": tables,
                    "missing": [t for t in expected_tables if t not in tables]
                }
            
        except Exception as e:
            diagnostics["error"] = str(e)
        
        return diagnostics
    
    async def _diagnose_ollama(self) -> Dict[str, Any]:
        """Diagnose Ollama service issues."""
        try:
            health = await ollama_manager.health_check()
            return {
                "service_status": health["status"],
                "available_models": health.get("models", []),
                "missing_models": health.get("missing_models", []),
                "diagnostics": health
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _diagnose_embedding_service(self) -> Dict[str, Any]:
        """Diagnose embedding service issues."""
        try:
            health = await document_embedder.client.health_check()
            metrics = document_embedder.client.get_metrics()
            
            return {
                "service_health": health,
                "performance_metrics": metrics,
                "model": document_embedder.client.model
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _diagnose_file_processing(self) -> Dict[str, Any]:
        """Diagnose file processing issues."""
        try:
            stats = await document_processor.get_processing_stats()
            return {
                "processing_status": "running" if stats["is_running"] else "stopped",
                "queue_health": "healthy" if stats["queue_size"] < 100 else "congested",
                "worker_status": f"{stats['active_workers']}/{stats['total_workers']} active",
                "performance": {
                    "files_processed": stats["files_processed"],
                    "files_failed": stats["files_failed"],
                    "average_processing_time": stats["average_processing_time"]
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _diagnose_file_watcher(self) -> Dict[str, Any]:
        """Diagnose file watcher issues."""
        try:
            stats = await file_watcher.get_stats()
            return {
                "watcher_status": "running" if stats["is_running"] else "stopped",
                "watch_directory": stats["watch_directory"],
                "event_processing": {
                    "received": stats["events_received"],
                    "processed": stats["events_processed"],
                    "skipped": stats["events_skipped"],
                    "queue_size": stats["queue_size"]
                },
                "last_activity": stats["last_event_time"]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _bytes_to_human(self, bytes_count: int) -> str:
        """Convert bytes to human readable format."""
        if bytes_count is None:
            return "unknown"
        
        if bytes_count == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while bytes_count >= 1024.0 and i < len(size_names) - 1:
            bytes_count /= 1024.0
            i += 1
        
        return f"{bytes_count:.1f} {size_names[i]}"