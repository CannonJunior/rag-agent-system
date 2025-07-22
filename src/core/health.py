import asyncio
from typing import Dict, Any, List
import httpx
import psycopg2
from loguru import logger
from config.settings import settings


class HealthChecker:
    """Service health monitoring."""
    
    async def check_ollama(self) -> Dict[str, Any]:
        """Check Ollama service health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{settings.ollama.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    
                    # Check required models with proper name matching
                    def check_model_available(model_name: str, available_models: List[str]) -> bool:
                        # Check exact match first
                        if model_name in available_models:
                            return True
                        # Check with :latest suffix
                        if f"{model_name}:latest" in available_models:
                            return True
                        # Check without :latest suffix if model_name has it
                        if model_name.endswith(":latest"):
                            base_name = model_name[:-7]  # Remove :latest
                            if base_name in available_models:
                                return True
                        return False
                    
                    embedding_available = check_model_available(settings.ollama.embedding_model, model_names)
                    llm_available = check_model_available(settings.ollama.llm_model, model_names)
                    
                    return {
                        "status": "healthy" if embedding_available and llm_available else "degraded",
                        "available_models": model_names,
                        "embedding_model_available": embedding_available,
                        "llm_model_available": llm_available,
                        "missing_models": [
                            model for model, available in [
                                (settings.ollama.embedding_model, embedding_available),
                                (settings.ollama.llm_model, llm_available)
                            ] if not available
                        ]
                    }
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def check_postgres(self) -> Dict[str, Any]:
        """Check PostgreSQL health."""
        try:
            conn = psycopg2.connect(
                host=settings.database.host,
                port=settings.database.port,
                database=settings.database.database,
                user=settings.database.username,
                password=settings.database.password,
            )
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                vector_available = cursor.fetchone() is not None
            conn.close()
            
            return {
                "status": "healthy",
                "vector_extension": vector_available
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Check all service health."""
        results = {}
        
        # Check Ollama
        results["ollama"] = await self.check_ollama()
        
        # Check PostgreSQL
        results["postgres"] = self.check_postgres()
        
        # Overall status
        all_healthy = all(
            service["status"] == "healthy" 
            for service in results.values()
        )
        
        results["overall"] = {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return results
    
    async def wait_for_services(self, timeout: int = 60) -> bool:
        """Wait for all services to be healthy."""
        logger.info("Waiting for services to be ready...")
        
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            health = await self.check_all_services()
            
            if health["overall"]["status"] == "healthy":
                logger.info("All services are healthy")
                return True
            
            logger.info("Services not ready, retrying in 5 seconds...")
            await asyncio.sleep(5)
        
        logger.error(f"Services not ready after {timeout} seconds")
        return False


# Global health checker instance
health_checker = HealthChecker()