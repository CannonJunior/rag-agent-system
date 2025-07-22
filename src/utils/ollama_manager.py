import asyncio
from typing import Dict, List, Optional, Any
import httpx
from loguru import logger
from config.settings import settings


class OllamaManager:
    """Ollama service management and model operations."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=settings.ollama.base_url,
            timeout=settings.ollama.timeout
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        models = await self.list_models()
        model_names = [model["name"] for model in models]
        
        # Check exact match first
        if model_name in model_names:
            return True
        
        # Check with :latest suffix
        if f"{model_name}:latest" in model_names:
            return True
        
        # Check without :latest suffix if model_name has it
        if model_name.endswith(":latest"):
            base_name = model_name[:-7]  # Remove :latest
            if base_name in model_names:
                return True
        
        return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available."""
        if await self.check_model_availability(model_name):
            logger.info(f"Model {model_name} already available")
            return True
        
        logger.info(f"Pulling model {model_name}...")
        try:
            async with self.client.stream(
                "POST", 
                "/api/pull", 
                json={"name": model_name}
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        data = eval(line)  # JSON per line
                        if "status" in data:
                            logger.info(f"Pull status: {data['status']}")
                        if data.get("error"):
                            logger.error(f"Pull error: {data['error']}")
                            return False
            
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """Generate embedding for text."""
        model = model or settings.ollama.embedding_model
        
        try:
            response = await self.client.post(
                "/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None,
        context: Optional[List[str]] = None
    ) -> Optional[str]:
        """Generate text completion."""
        model = model or settings.ollama.llm_model
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.append({"role": "user", "content": f"Context: {' '.join(context)}"})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content")
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return None
    
    async def stream_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None
    ):
        """Stream text generation."""
        model = model or settings.ollama.llm_model
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            async with self.client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        data = eval(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to stream text: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health."""
        try:
            # Basic connectivity
            response = await self.client.get("/api/tags")
            if response.status_code != 200:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
            
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            # Check required models using the same logic as check_model_availability
            embedding_available = await self.check_model_availability(settings.ollama.embedding_model)
            llm_available = await self.check_model_availability(settings.ollama.llm_model)
            
            return {
                "status": "healthy" if embedding_available and llm_available else "degraded",
                "models": model_names,
                "embedding_model_available": embedding_available,
                "llm_model_available": llm_available,
                "missing_models": [
                    model for model, available in [
                        (settings.ollama.embedding_model, embedding_available),
                        (settings.ollama.llm_model, llm_available)
                    ] if not available
                ]
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def ensure_models(self) -> bool:
        """Ensure required models are available."""
        health = await self.health_check()
        
        if health["status"] == "healthy":
            return True
        
        missing_models = health.get("missing_models", [])
        if not missing_models:
            return True
        
        logger.info(f"Missing models: {missing_models}")
        
        # Try to pull missing models
        for model in missing_models:
            if not await self.pull_model(model):
                logger.error(f"Failed to pull required model: {model}")
                return False
        
        return True
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Global Ollama manager instance
ollama_manager = OllamaManager()