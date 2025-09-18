import os
from typing import Optional, List
from functools import lru_cache

try:
    # For Pydantic v2
    from pydantic_settings import BaseSettings
    from pydantic import field_validator
except ImportError:
    try:
        # For Pydantic v1
        from pydantic import BaseSettings, validator as field_validator
    except ImportError:
        raise ImportError("Please install pydantic-settings: pip install pydantic-settings")


class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "Mistral OCR Component"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Mistral API Configuration
    MISTRAL_API_KEY: str = ""
    MISTRAL_OCR_MODEL: str = "mistral-large-vision"
    MISTRAL_TEXT_MODEL: str = "mistral-small-latest"
    MISTRAL_BASE_URL: str = "https://api.mistral.ai"
    
    # Database Configuration - SQLite by default
    DATABASE_URL: str = "sqlite:///./fra_ocr.db"
    DATABASE_ECHO: bool = False
    
    # Redis Configuration (optional for development)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL: int = 3600  # 1 hour
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "jpg", "jpeg", "png", "tiff"]  # Default value, no .env needed
    UPLOAD_DIR: str = "uploads"
    
    # Processing Configuration
    BATCH_SIZE: int = 100
    ENABLE_BATCH_API: bool = True
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    
    # Coordinate Validation - India bounds
    MIN_LATITUDE: float = 6.0
    MAX_LATITUDE: float = 37.0
    MIN_LONGITUDE: float = 68.0
    MAX_LONGITUDE: float = 97.0
    
    # Task Queue Configuration (for future use)
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    CORS_ORIGINS: List[str] = ["*"]
    
    @field_validator("MISTRAL_API_KEY")
    @classmethod
    def validate_mistral_api_key(cls, v):
        if not v or v == "your_mistral_api_key_here":
            # Allow empty for development setup
            return v
        return v
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("DATABASE_URL is required")
        return v
    
    @field_validator("ALLOWED_EXTENSIONS", mode='before')
    @classmethod
    def validate_extensions(cls, v):
        # If not provided, use default
        if not v:
            return ["pdf", "jpg", "jpeg", "png", "tiff"]
            
        if isinstance(v, str):
            # Handle comma-separated string from .env
            v = [ext.strip() for ext in v.split(",")]
        
        allowed = {"pdf", "jpg", "jpeg", "png", "tiff", "bmp", "gif"}
        validated_extensions = []
        
        for ext in v:
            ext_clean = ext.lower().strip()
            if ext_clean in allowed:
                validated_extensions.append(ext_clean)
            else:
                # Just skip invalid extensions instead of raising error
                continue
        
        return validated_extensions or ["pdf", "jpg", "jpeg", "png", "tiff"]  # Return default if all invalid
    
    @field_validator("MIN_LATITUDE", "MAX_LATITUDE")
    @classmethod
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @field_validator("MIN_LONGITUDE", "MAX_LONGITUDE")
    @classmethod
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v
    
    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v):
        if isinstance(v, str):
            # Handle comma-separated string or JSON array string from .env
            if v.startswith('[') and v.endswith(']'):
                # JSON array format
                import json
                return json.loads(v)
            else:
                # Comma-separated format
                return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def upload_path(self) -> str:
        """Get absolute path to upload directory"""
        return os.path.abspath(self.UPLOAD_DIR)
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL for SQLAlchemy"""
        if "sqlite:///" in self.DATABASE_URL:
            return self.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif "postgresql://" in self.DATABASE_URL:
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return self.DATABASE_URL
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database"""
        return "sqlite" in self.DATABASE_URL
    
    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL database"""
        return "postgresql" in self.DATABASE_URL
    
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
        extra = "ignore"  # Ignore extra fields from .env


class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_ECHO: bool = True
    DATABASE_URL: str = "sqlite:///./fra_ocr_dev.db"


class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_ECHO: bool = False
    SECRET_KEY: str  # Must be provided in production
    
    @field_validator("SECRET_KEY")
    @classmethod
    def validate_production_secret_key(cls, v):
        if not v or v == "your-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be set in production")
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v


class TestingSettings(Settings):
    """Testing environment settings"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_URL: str = "sqlite:///./test_fra_ocr.db"
    UPLOAD_DIR: str = "test_uploads"
    MISTRAL_API_KEY: str = "test_key"  # Allow test key


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
    
    # File Processing
    ALLOWED_MIME_TYPES = {
        'pdf': ['application/pdf'],
        'jpg': ['image/jpeg'],
        'jpeg': ['image/jpeg'], 
        'png': ['image/png'],
        'tiff': ['image/tiff'],
        'bmp': ['image/bmp'],
        'gif': ['image/gif']
    }
    
    # Validation Thresholds
    MIN_CONFIDENCE_SCORE: float = 0.0
    MAX_CONFIDENCE_SCORE: float = 1.0
    MIN_AREA_HECTARES: float = 0.001
    MAX_AREA_HECTARES: float = 10000
    MIN_COORDINATES_FOR_POLYGON: int = 3


# Create Mistral OCR config instance
mistral_ocr_config = MistralOCRConfig()


# Utility functions
def get_database_type() -> str:
    """Get the database type being used"""
    if settings.is_sqlite:
        return "sqlite"
    elif settings.is_postgresql:
        return "postgresql"
    else:
        return "unknown"


def is_mistral_api_configured() -> bool:
    """Check if Mistral API is properly configured"""
    return (
        bool(settings.MISTRAL_API_KEY) and 
        settings.MISTRAL_API_KEY != "your_mistral_api_key_here" and
        len(settings.MISTRAL_API_KEY) > 10
    )


def get_upload_path() -> str:
    """Get the upload directory path"""
    return settings.upload_path


def get_allowed_extensions() -> List[str]:
    """Get list of allowed file extensions"""
    return settings.ALLOWED_EXTENSIONS


# Export commonly used items
__all__ = [
    "settings",
    "mistral_ocr_config", 
    "get_database_type",
    "is_mistral_api_configured",
    "get_upload_path",
    "get_allowed_extensions",
    "Settings",
    "DevelopmentSettings",
    "ProductionSettings",
    "TestingSettings"
]