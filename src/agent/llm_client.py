import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

from loguru import logger
from config.settings import settings
from src.utils.ollama_manager import ollama_manager


class LLMClient:
    """Client for interacting with local LLM via Ollama."""
    
    def __init__(self):
        self.model = settings.ollama.llm_model
        self.timeout = settings.ollama.timeout
        self.max_retries = settings.ollama.max_retries
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_generated = 0
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        context_chunks: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate chat completion using the local LLM."""
        try:
            self.total_requests += 1
            start_time = asyncio.get_event_loop().time()
            
            # Prepare messages
            formatted_messages = self._format_messages(messages, system_prompt, context_chunks)
            
            if stream:
                return await self._stream_completion(formatted_messages)
            else:
                # Generate response
                response = await ollama_manager.generate_text(
                    prompt=formatted_messages[-1]["content"],
                    model=self.model,
                    system=formatted_messages[0]["content"] if formatted_messages[0]["role"] == "system" else None
                )
                
                if response:
                    self.successful_requests += 1
                    processing_time = asyncio.get_event_loop().time() - start_time
                    
                    # Estimate token count (rough approximation)
                    estimated_tokens = len(response.split())
                    self.total_tokens_generated += estimated_tokens
                    
                    return {
                        "success": True,
                        "response": response,
                        "model": self.model,
                        "processing_time": processing_time,
                        "estimated_tokens": estimated_tokens,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    self.failed_requests += 1
                    return {
                        "success": False,
                        "error": "Empty response from LLM",
                        "model": self.model
                    }
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Error in chat completion: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }
    
    async def _stream_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Stream chat completion."""
        try:
            response_chunks = []
            
            async for chunk in ollama_manager.stream_text(
                prompt=messages[-1]["content"],
                model=self.model,
                system=messages[0]["content"] if messages[0]["role"] == "system" else None
            ):
                response_chunks.append(chunk)
            
            full_response = "".join(response_chunks)
            
            if full_response:
                self.successful_requests += 1
                estimated_tokens = len(full_response.split())
                self.total_tokens_generated += estimated_tokens
                
                return {
                    "success": True,
                    "response": full_response,
                    "model": self.model,
                    "estimated_tokens": estimated_tokens,
                    "stream": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                self.failed_requests += 1
                return {
                    "success": False,
                    "error": "Empty streamed response",
                    "model": self.model
                }
                
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Error in streaming completion: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }
    
    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        context_chunks: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Format messages for the LLM with context injection."""
        formatted_messages = []
        
        # Add system prompt
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add context if provided
        if context_chunks:
            context_content = "Relevant information from documents:\n\n"
            for i, chunk in enumerate(context_chunks, 1):
                context_content += f"[Context {i}]:\n{chunk}\n\n"
            
            formatted_messages.append({
                "role": "system",
                "content": context_content
            })
        
        # Add conversation messages
        for message in messages:
            if "role" in message and "content" in message:
                formatted_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        return formatted_messages
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM service health."""
        try:
            test_messages = [
                {"role": "user", "content": "Say 'healthy' if you can respond."}
            ]
            
            result = await self.chat_completion(test_messages)
            
            if result["success"]:
                return {
                    "status": "healthy",
                    "model": self.model,
                    "response_time": result.get("processing_time", 0),
                    "test_successful": True
                }
            else:
                return {
                    "status": "unhealthy",
                    "model": self.model,
                    "error": result.get("error"),
                    "test_successful": False
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model,
                "error": str(e),
                "test_successful": False
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get LLM client metrics."""
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        avg_tokens_per_request = (
            self.total_tokens_generated / self.successful_requests 
            if self.successful_requests > 0 else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "total_tokens_generated": self.total_tokens_generated,
            "average_tokens_per_request": avg_tokens_per_request,
            "model": self.model
        }
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_generated = 0
        logger.info("LLM client metrics reset")


# Global LLM client instance
llm_client = LLMClient()