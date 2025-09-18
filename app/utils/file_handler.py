import os
import hashlib
import mimetypes
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, List
import aiofiles
import aiofiles.os
from fastapi import UploadFile
import logging
from datetime import datetime
import shutil

from ..core.config import settings
from .constants import (
    ALLOWED_FILE_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_FILE_SIZE,
    ERROR_MESSAGES, SUCCESS_MESSAGES
)

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Handles file operations for uploaded documents including saving, validation,
    encoding, and cleanup operations.
    """
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_path)
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(
        self, 
        upload_file: UploadFile
    ) -> Dict[str, Any]:
        """
        Save uploaded file to storage and return file information.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Dictionary with file information
        """
        try:
            # Validate file
            validation_result = await self.validate_file(upload_file)
            if not validation_result["is_valid"]:
                raise ValueError(f"File validation failed: {validation_result['errors']}")
            
            # Generate unique filename
            file_extension = self._get_file_extension(upload_file.filename)
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            # Save file
            await self._save_file_to_disk(upload_file, file_path)
            
            # Get file info
            file_info = await self._get_file_info(upload_file, file_path, unique_filename)
            
            logger.info(f"File saved: {unique_filename}, size: {file_info['file_size']} bytes")
            return file_info
            
        except Exception as e:
            logger.error(f"File save failed: {str(e)}")
            raise
    
    async def validate_file(self, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Validate uploaded file for size, type, and content.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Validation result with is_valid flag and errors list
        """
        errors = []
        
        try:
            # Check file size
            file_size = await self._get_file_size(upload_file)
            if file_size > self.max_file_size:
                errors.append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            if file_size == 0:
                errors.append("File is empty")
            
            # Check file extension
            if upload_file.filename:
                file_extension = self._get_file_extension(upload_file.filename)
                if file_extension not in self.allowed_extensions:
                    errors.append(f"File extension '{file_extension}' not allowed")
            else:
                errors.append("Filename is missing")
            
            # Check MIME type
            content_type = upload_file.content_type
            if content_type:
                if not self._is_mime_type_allowed(content_type, file_extension):
                    errors.append(f"MIME type '{content_type}' not allowed for extension '{file_extension}'")
            
            # Additional content validation
            content_errors = await self._validate_file_content(upload_file)
            errors.extend(content_errors)
            
            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "file_size": file_size,
                "file_extension": file_extension,
                "content_type": content_type
            }
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "file_size": 0,
                "file_extension": None,
                "content_type": None
            }
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                await aiofiles.os.remove(path)
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"File deletion failed: {str(e)}")
            return False
    
    async def get_file_hash(self, file_path: str, algorithm: str = "sha256") -> Optional[str]:
        """
        Calculate hash of file for integrity checking.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5, etc.)
            
        Returns:
            Hash string or None if failed
        """
        try:
            hash_obj = hashlib.new(algorithm)
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"File hash calculation failed: {str(e)}")
            return None
    
    async def create_backup(self, file_path: str) -> Optional[str]:
        """
        Create backup copy of file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Backup file path or None if failed
        """
        try:
            original_path = Path(file_path)
            if not original_path.exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{original_path.stem}_backup_{timestamp}{original_path.suffix}"
            backup_path = original_path.parent / "backups" / backup_filename
            
            # Create backups directory if it doesn't exist
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(original_path, backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return None
    
    async def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Clean up old files from upload directory.
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            current_time = datetime.now()
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_age.days > days_old:
                        await aiofiles.os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {file_path}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"File cleanup failed: {str(e)}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for upload directory.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            total_size = 0
            file_count = 0
            file_types = {}
            
            for file_path in self.upload_dir.rglob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    file_count += 1
                    
                    # Count by extension
                    extension = file_path.suffix.lower().lstrip('.')
                    file_types[extension] = file_types.get(extension, 0) + 1
            
            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "upload_directory": str(self.upload_dir)
            }
            
        except Exception as e:
            logger.error(f"Storage stats calculation failed: {str(e)}")
            return {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "file_types": {},
                "upload_directory": str(self.upload_dir),
                "error": str(e)
            }
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        return Path(filename).suffix.lower().lstrip('.')
    
    def _is_mime_type_allowed(self, mime_type: str, extension: str) -> bool:
        """Check if MIME type is allowed for given extension."""
        allowed_mimes = ALLOWED_MIME_TYPES.get(extension, [])
        return mime_type in allowed_mimes
    
    async def _get_file_size(self, upload_file: UploadFile) -> int:
        """Get file size by reading content."""
        # Seek to end to get size
        await upload_file.seek(0, 2)
        size = await upload_file.tell()
        # Reset to beginning
        await upload_file.seek(0)
        return size
    
    async def _save_file_to_disk(self, upload_file: UploadFile, file_path: Path):
        """Save uploaded file to disk."""
        async with aiofiles.open(file_path, 'wb') as f:
            # Read and write in chunks to handle large files
            while chunk := await upload_file.read(8192):
                await f.write(chunk)
        
        # Reset file pointer for potential future reads
        await upload_file.seek(0)
    
    async def _get_file_info(
        self, 
        upload_file: UploadFile, 
        file_path: Path, 
        unique_filename: str
    ) -> Dict[str, Any]:
        """Generate file information dictionary."""
        file_extension = self._get_file_extension(upload_file.filename)
        
        # Determine document type
        document_type = "pdf" if file_extension == "pdf" else "image"
        
        # Get file stats
        file_stats = file_path.stat()
        
        return {
            "filename": unique_filename,
            "original_filename": upload_file.filename,
            "file_path": str(file_path),
            "file_size": file_stats.st_size,
            "file_type": file_extension,
            "document_type": document_type,
            "mime_type": upload_file.content_type,
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        }
    
    async def _validate_file_content(self, upload_file: UploadFile) -> List[str]:
        """
        Validate file content by checking file headers.
        
        Args:
            upload_file: File to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            # Read first few bytes to check file signature
            header = await upload_file.read(512)
            await upload_file.seek(0)  # Reset position
            
            file_extension = self._get_file_extension(upload_file.filename)
            
            # Check file signatures
            if file_extension == "pdf":
                if not header.startswith(b'%PDF'):
                    errors.append("File does not appear to be a valid PDF")
            
            elif file_extension in ["jpg", "jpeg"]:
                if not (header.startswith(b'\xff\xd8\xff') or b'JFIF' in header[:20]):
                    errors.append("File does not appear to be a valid JPEG")
            
            elif file_extension == "png":
                if not header.startswith(b'\x89PNG\r\n\x1a\n'):
                    errors.append("File does not appear to be a valid PNG")
            
            elif file_extension == "tiff":
                if not (header.startswith(b'II*\x00') or header.startswith(b'MM\x00*')):
                    errors.append("File does not appear to be a valid TIFF")
            
            # Check for potentially malicious content
            if b'<script' in header.lower() or b'javascript:' in header.lower():
                errors.append("File contains potentially malicious content")
            
        except Exception as e:
            logger.warning(f"Content validation error: {str(e)}")
            # Don't add to errors as this is not critical
        
        return errors


class SecureFileHandler(FileHandler):
    """
    Extended file handler with additional security features.
    """
    
    def __init__(self):
        super().__init__()
        self.quarantine_dir = self.upload_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    async def scan_file_for_threats(self, file_path: str) -> Dict[str, Any]:
        """
        Basic file threat scanning (placeholder for future integration with antivirus).
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            Scan result dictionary
        """
        try:
            # Basic checks - in production, integrate with antivirus API
            threats_found = []
            
            # Check file size (extremely large files might be suspicious)
            file_size = Path(file_path).stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB
                threats_found.append("File size unusually large")
            
            # Check for suspicious patterns in filename
            filename = Path(file_path).name
            suspicious_patterns = ['.exe', '.bat', '.cmd', '.scr', '.com']
            
            for pattern in suspicious_patterns:
                if pattern in filename.lower():
                    threats_found.append(f"Suspicious file pattern: {pattern}")
            
            return {
                "clean": len(threats_found) == 0,
                "threats": threats_found,
                "scanned_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"File scan failed: {str(e)}")
            return {
                "clean": False,
                "threats": [f"Scan error: {str(e)}"],
                "scanned_at": datetime.now().isoformat()
            }
    
    async def quarantine_file(self, file_path: str, reason: str) -> bool:
        """
        Move suspicious file to quarantine directory.
        
        Args:
            file_path: Original file path
            reason: Reason for quarantine
            
        Returns:
            True if successful, False otherwise
        """
        try:
            original_path = Path(file_path)
            if not original_path.exists():
                return False
            
            # Create quarantine filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_filename = f"{timestamp}_{original_path.name}"
            quarantine_path = self.quarantine_dir / quarantine_filename
            
            # Move file to quarantine
            shutil.move(original_path, quarantine_path)
            
            # Create metadata file
            metadata = {
                "original_path": str(original_path),
                "quarantine_reason": reason,
                "quarantined_at": datetime.now().isoformat()
            }
            
            metadata_path = quarantine_path.with_suffix(quarantine_path.suffix + '.meta')
            async with aiofiles.open(metadata_path, 'w') as f:
                import json
                await f.write(json.dumps(metadata, indent=2))
            
            logger.warning(f"File quarantined: {file_path} -> {quarantine_path} (Reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"File quarantine failed: {str(e)}")
            return False