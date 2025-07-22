from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_db"
    username: str = "rag_user"
    password: str = "rag_password"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "incept5/llama3.1-claude:latest"
    timeout: int = 300
    max_retries: int = 3
    request_delay: float = 0.1


class IngestSettings(BaseModel):
    watch_directory: Path = Path("data")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 100
    supported_extensions: List[str] = [".txt", ".pdf", ".docx", ".md"]
    processing_timeout: int = 600


class MCPSettings(BaseModel):
    server_host: str = "localhost"
    server_port: int = 8765
    max_connections: int = 10


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "{time} | {level} | {name}:{function}:{line} | {message}"
    rotation: str = "1 day"
    retention: str = "30 days"
    log_file: Optional[Path] = Path("logs/rag-agent.log")


class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    ollama: OllamaSettings = OllamaSettings()
    ingest: IngestSettings = IngestSettings()
    mcp: MCPSettings = MCPSettings()
    logging: LoggingSettings = LoggingSettings()
    
    # Environment settings
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    class Config:
        env_nested_delimiter = "__"
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()