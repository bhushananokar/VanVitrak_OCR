#!/usr/bin/env python3
"""
SQLite setup script - No PostgreSQL required!
"""
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sqlite_connection():
    """Test SQLite database creation"""
    try:
        import sqlite3
        
        # Test basic SQLite functionality
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        cursor.execute('SELECT sqlite_version();')
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        
        # Clean up test file
        os.remove('test.db')
        
        logger.info(f"✅ SQLite is working: version {version[0]}")
        return True
        
    except Exception as e:
        logger.error(f"❌ SQLite test failed: {e}")
        return False


def create_database_tables():
    """Create SQLite database tables"""
    try:
        from sqlalchemy import create_engine
        from app.models.database import Base
        
        # Create SQLite database
        engine = create_engine('sqlite:///./fra_ocr.db', echo=True)
        
        logger.info("Creating SQLite database tables...")
        Base.metadata.create_all(engine)
        
        # Test the database
        with engine.connect() as conn:
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = result.fetchall()
            logger.info(f"✅ Created {len(tables)} tables: {[t[0] for t in tables]}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database table creation failed: {e}")
        return False


def create_upload_directory():
    """Create upload directory"""
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        logger.info(f"✅ Upload directory: {upload_dir.absolute()}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create upload directory: {e}")
        return False


def install_requirements():
    """Check and install required packages"""
    try:
        import sqlite3  # Built into Python
        logger.info("✅ SQLite support available")
        
        # Check for aiosqlite
        try:
            import aiosqlite
            logger.info("✅ aiosqlite available")
        except ImportError:
            logger.info("📦 Installing aiosqlite...")
            os.system("pip install aiosqlite")
        
        return True
    except Exception as e:
        logger.error(f"❌ Requirements check failed: {e}")
        return False


def main():
    """Main setup function"""
    logger.info("🚀 Setting up Mistral OCR Component with SQLite")
    logger.info("(No PostgreSQL required! 🎉)")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test SQLite
    if not test_sqlite_connection():
        return False
    
    
    # Create upload directory
    if not create_upload_directory():
        return False
    
    # Create database tables
    if not create_database_tables():
        return False
    
    logger.info("")
    logger.info("🎉 SQLite setup completed successfully!")
    logger.info("")
    logger.info("📋 Next steps:")
    logger.info("1. Get your Mistral API key from: https://console.mistral.ai/")
    logger.info("2. Edit .env file and replace 'your_mistral_api_key_here'")
    logger.info("3. Run: python run_local.py")
    logger.info("4. Visit: http://localhost:8001/docs")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            input("\nPress Enter to exit...")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)