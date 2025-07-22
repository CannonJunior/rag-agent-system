import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from loguru import logger
from config.settings import settings
from src.utils.ollama_manager import ollama_manager


@dataclass
class EmbeddingRequest:
    """Container for embedding requests."""
    text: str
    request_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    request_id: str
    embedding: Optional[List[float]]
    error: Optional[str]
    processing_time: float
    model_used: str
    metadata: Dict[str, Any] = None


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests_per_second: float = 10.0):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit permission."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class EmbeddingClient:
    """Async embedding client with rate limiting and error handling."""
    
    def __init__(self):
        self.model = settings.ollama.embedding_model
        self.rate_limiter = RateLimiter(max_requests_per_second=1.0 / settings.ollama.request_delay)
        self.retry_attempts = settings.ollama.max_retries
        self.request_timeout = settings.ollama.timeout
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
    
    async def embed_text(self, text: str, request_id: str = None) -> EmbeddingResult:
        """Generate embedding for a single text."""
        if not request_id:
            request_id = f"req_{int(time.time() * 1000)}"
        
        start_time = time.time()
        self.total_requests += 1
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Retry logic
        last_error = None
        for attempt in range(self.retry_attempts + 1):
            try:
                logger.debug(f"Embedding request {request_id}, attempt {attempt + 1}")
                
                embedding = await ollama_manager.generate_embedding(text, self.model)
                
                if embedding:
                    processing_time = time.time() - start_time
                    self.successful_requests += 1
                    self.total_processing_time += processing_time
                    
                    logger.debug(f"Embedding successful for {request_id} in {processing_time:.2f}s")
                    
                    return EmbeddingResult(
                        request_id=request_id,
                        embedding=embedding,
                        error=None,
                        processing_time=processing_time,
                        model_used=self.model,
                        metadata={
                            'text_length': len(text),
                            'attempt': attempt + 1,
                            'total_attempts': self.retry_attempts + 1
                        }
                    )
                else:
                    last_error = "Empty embedding returned"
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Embedding attempt {attempt + 1} failed for {request_id}: {e}")
                
                if attempt < self.retry_attempts:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 1.0
                    await asyncio.sleep(wait_time)
        
        # All attempts failed
        processing_time = time.time() - start_time
        self.failed_requests += 1
        
        logger.error(f"Embedding failed for {request_id} after {self.retry_attempts + 1} attempts: {last_error}")
        
        return EmbeddingResult(
            request_id=request_id,
            embedding=None,
            error=last_error,
            processing_time=processing_time,
            model_used=self.model,
            metadata={
                'text_length': len(text),
                'total_attempts': self.retry_attempts + 1,
                'final_error': last_error
            }
        )
    
    async def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 5,
        request_prefix: str = "batch"
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts in batches."""
        if not texts:
            return []
        
        logger.info(f"Processing batch of {len(texts)} texts with batch size {batch_size}")
        
        results = []
        
        # Process in batches to control concurrency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_start = time.time()
            
            # Create concurrent tasks for this batch
            tasks = []
            for j, text in enumerate(batch):
                request_id = f"{request_prefix}_{i + j}"
                task = asyncio.create_task(self.embed_text(text, request_id))
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch task {i + j} failed with exception: {result}")
                    error_result = EmbeddingResult(
                        request_id=f"{request_prefix}_{i + j}",
                        embedding=None,
                        error=str(result),
                        processing_time=0.0,
                        model_used=self.model,
                        metadata={'batch_exception': True}
                    )
                    results.append(error_result)
                else:
                    results.append(result)
            
            batch_time = time.time() - batch_start
            logger.info(f"Batch {i//batch_size + 1} completed in {batch_time:.2f}s")
        
        successful = sum(1 for r in results if r.embedding is not None)
        logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check embedding service health."""
        try:
            test_text = "This is a test embedding."
            result = await self.embed_text(test_text, "health_check")
            
            if result.embedding:
                return {
                    'status': 'healthy',
                    'model': self.model,
                    'embedding_dimension': len(result.embedding),
                    'processing_time': result.processing_time,
                    'test_successful': True
                }
            else:
                return {
                    'status': 'unhealthy',
                    'model': self.model,
                    'error': result.error,
                    'test_successful': False
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.model,
                'error': str(e),
                'test_successful': False
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get embedding client metrics."""
        avg_processing_time = (
            self.total_processing_time / self.successful_requests 
            if self.successful_requests > 0 else 0.0
        )
        
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'total_processing_time': self.total_processing_time,
            'model': self.model,
            'rate_limit_rps': self.rate_limiter.max_requests_per_second
        }
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        logger.info("Embedding client metrics reset")


class DocumentEmbedder:
    """High-level document embedding processor."""
    
    def __init__(self):
        self.client = EmbeddingClient()
    
    async def embed_chunks(self, chunks: List[Any]) -> List[tuple]:
        """Embed document chunks and return (chunk, embedding_result) pairs."""
        if not chunks:
            return []
        
        logger.info(f"Embedding {len(chunks)} document chunks")
        
        # Extract text from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embedding_results = await self.client.embed_batch(
            texts, 
            batch_size=3,  # Conservative batch size for local models
            request_prefix="chunk"
        )
        
        # Pair chunks with their embedding results
        chunk_embedding_pairs = []
        for chunk, result in zip(chunks, embedding_results):
            chunk_embedding_pairs.append((chunk, result))
        
        # Log results
        successful_embeddings = sum(1 for _, result in chunk_embedding_pairs if result.embedding)
        logger.info(f"Generated {successful_embeddings}/{len(chunks)} chunk embeddings")
        
        return chunk_embedding_pairs
    
    async def embed_query(self, query: str) -> Optional[List[float]]:
        """Embed a search query."""
        result = await self.client.embed_text(query, "query")
        return result.embedding
    
    async def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Validate embedding quality and consistency."""
        if not embeddings:
            return False
        
        # Check dimensions consistency
        expected_dim = len(embeddings[0])
        if not all(len(emb) == expected_dim for emb in embeddings):
            logger.error("Inconsistent embedding dimensions")
            return False
        
        # Check for null/zero embeddings
        zero_embeddings = sum(1 for emb in embeddings if all(x == 0 for x in emb))
        if zero_embeddings > 0:
            logger.warning(f"Found {zero_embeddings} zero embeddings")
        
        # Check for NaN/inf values
        invalid_embeddings = 0
        for emb in embeddings:
            if any(not isinstance(x, (int, float)) or x != x or abs(x) == float('inf') for x in emb):
                invalid_embeddings += 1
        
        if invalid_embeddings > 0:
            logger.error(f"Found {invalid_embeddings} invalid embeddings (NaN/inf)")
            return False
        
        logger.info(f"Embedding validation passed: {len(embeddings)} embeddings, dimension {expected_dim}")
        return True
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        health = await self.client.health_check()
        metrics = self.client.get_metrics()
        
        return {
            'health': health,
            'metrics': metrics,
            'model': self.client.model,
            'rate_limit': self.client.rate_limiter.max_requests_per_second
        }


# Global embedder instance
document_embedder = DocumentEmbedder()