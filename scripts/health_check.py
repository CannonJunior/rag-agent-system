#!/usr/bin/env python3
"""Health check script for all services."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.core.logging import setup_logging
from src.core.health import health_checker


async def main():
    """Check health of all services."""
    setup_logging()
    
    logger.info("Checking service health...")
    
    try:
        health = await health_checker.check_all_services()
        
        print("\n=== Service Health Report ===")
        for service, status in health.items():
            if service == "overall":
                continue
                
            print(f"\n{service.upper()}:")
            print(f"  Status: {status['status']}")
            
            if status["status"] == "unhealthy":
                print(f"  Error: {status.get('error', 'Unknown')}")
            
            # Service-specific details
            if service == "ollama":
                if "available_models" in status:
                    print(f"  Available models: {len(status['available_models'])}")
                if "missing_models" in status and status["missing_models"]:
                    print(f"  Missing models: {status['missing_models']}")
            elif service == "postgres":
                if "vector_extension" in status:
                    print(f"  pgvector extension: {'✓' if status['vector_extension'] else '✗'}")
        
        overall = health["overall"]
        print(f"\nOVERALL STATUS: {overall['status'].upper()}")
        
        if overall["status"] != "healthy":
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())