import os
from typing import Optional, List
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "Mistral OCR Component"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Mistral API Configuration
    MISTRAL_API_KEY: str
    MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"
    MISTRAL_TEXT_MODEL: str = "mistral-small-latest"
    MISTRAL_BASE_URL: str = "https://api.mistral.ai"
    
    # Database Configuration
    DATABASE_URL: str
    DATABASE_ECHO: bool = False
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL: int = 3600  # 1 hour
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "jpg", "jpeg", "png", "tiff"]
    UPLOAD_DIR: str = "uploads"
    
    # Processing Configuration
    BATCH_SIZE: int = 100
    ENABLE_BATCH_API: bool = True
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    
    # Coordinate Validation
    MIN_LATITUDE: float = 6.0   # India bounds
    MAX_LATITUDE: float = 37.0
    MIN_LONGITUDE: float = 68.0
    MAX_LONGITUDE: float = 97.0
    
    # Task Queue Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    CORS_ORIGINS: List[str] = ["*"]
    
    @validator("MISTRAL_API_KEY")
    def validate_mistral_api_key(cls, v):
        if not v:
            raise ValueError("MISTRAL_API_KEY is required")
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("DATABASE_URL is required")
        if not v.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")
        return v
    
    @validator("ALLOWED_EXTENSIONS")
    def validate_extensions(cls, v):
        allowed = {"pdf", "jpg", "jpeg", "png", "tiff", "bmp", "gif"}
        for ext in v:
            if ext.lower() not in allowed:
                raise ValueError(f"Extension '{ext}' is not supported")
        return [ext.lower() for ext in v]
    
    @validator("MIN_LATITUDE", "MAX_LATITUDE")
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @validator("MIN_LONGITUDE", "MAX_LONGITUDE")
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v
    
    @property
    def upload_path(self) -> str:
        """Get absolute path to upload directory"""
        return os.path.abspath(self.UPLOAD_DIR)
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL for SQLAlchemy"""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.DEBUG
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.DEBUG
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class DevelopmentSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_ECHO: bool = True


class ProductionSettings(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_ECHO: bool = False
    SECRET_KEY: str  # Must be provided in production


class TestingSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_URL: str = "postgresql://test:test@localhost:5432/test_fra_ocr_db"
    REDIS_URL: str = "redis://localhost:6379/1"
    UPLOAD_DIR: str = "test_uploads"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings based on environment.
    Cached to avoid re-reading environment variables.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()


# Mistral OCR specific configurations
class MistralOCRConfig:
    """Mistral OCR specific configuration"""
    
    # OCR Processing
    INCLUDE_IMAGE_BASE64: bool = True
    MAX_PAGES_PER_DOCUMENT: int = 50
    
    # Batch API
    BATCH_UPLOAD_PURPOSE: str = "batch"
    BATCH_ENDPOINT: str = "/v1/ocr"
    BATCH_POLLING_INTERVAL: int = 5  # seconds
    BATCH_MAX_WAIT_TIME: int = 3600  # 1 hour
    
    # Tool Usage
    ENABLE_TOOL_USAGE: bool = True
    TOOL_TIMEOUT: int = 30  # seconds
    
    # Coordinate Extraction Patterns
    COORDINATE_PATTERNS = {
        "decimal_degrees": r"(\d+\.?\d*)[°\s]*([NS])[,\s]*(\d+\.?\d*)[°\s]*([EW])",
        "dms": r"(\d+)[°]\s*(\d+)['\s]*(\d+\.?\d*)[\"]\s*([NS])[,\s]*(\d+)[°]\s*(\d+)['\s]*(\d+\.?\d*)[\"]\s*([EW])",
        "utm": r"(\d+[NS])\s+(\d+)\s+(\d+)",
        "survey_numbers": r"(?:Sy\.?No\.?|Survey\s+No\.?|Plot\s+No\.?)\s*[:\-]?\s*(\d+[/\-]?\w*)"
    }
    
    # GeoJSON Configuration
    GEOJSON_CRS = {
        "type": "name",
        "properties": {
            "name": "EPSG:4326"
        }
    }


# Create Mistral OCR config instance
mistral_ocr_config = MistralOCRConfig()