"""
Configuration for DeepSeek-OCR API Service
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Model configuration
    MODEL_NAME: str = "deepseek-ai/DeepSeek-OCR"
    DEVICE: str = "cuda"  # cuda or cpu
    USE_BF16: bool = True  # Use bfloat16 precision (faster, less memory)
    MAX_NEW_TOKENS: int = 4096  # Maximum tokens to generate

    # API configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Security
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key-change-this")

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "*"  # Remove in production, specify exact origins
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
