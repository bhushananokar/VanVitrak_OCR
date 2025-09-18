#!/usr/bin/env python3
"""
Local development server runner
"""
import os
import sys
from pathlib import Path
import uvicorn
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if environment is properly configured"""
    missing_vars = []
    
    # Check Mistral API key
    if not settings.MISTRAL_API_KEY or settings.MISTRAL_API_KEY == "your_mistral_api_key_here":
        missing_vars.append("MISTRAL_API_KEY (set to valid key)")
    
    # Check database URL
    if not settings.DATABASE_URL:
        missing_vars.append("DATABASE_URL")
    
    # Redis is optional for development
    # if not settings.REDIS_URL:
    #     missing_vars.append("REDIS_URL")
    
    if missing_vars:
        logger.error(f"Missing or invalid configuration: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        return False
    
    return True


def create_upload_directory():
    """Create upload directory if it doesn't exist"""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory: {upload_dir.absolute()}")


def main():
    """Main function to run the development server"""
    logger.info("Starting Mistral OCR Component - Local Development")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create necessary directories
    create_upload_directory()
    
    # Log configuration
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    logger.info(f"Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'Not configured'}")
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        reload_dirs=["app"],
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        reload_excludes=["*.pyc", "*.pyo", "*~"]
    )


if __name__ == "__main__":
    main()