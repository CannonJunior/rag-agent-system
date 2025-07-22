#!/usr/bin/env python3
"""Standalone MCP server script."""

import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mcp_tools.server import main

if __name__ == "__main__":
    asyncio.run(main())