from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, status
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid
import logging
from pathlib import Path
import aiofiles
import os

from ..core.database import get_async_session
from ..core.config import settings
from ..models.schemas import (
    DocumentResponse, DocumentStatusResponse, OCRResponse, ProcessingResult,
    BatchJobRequest, BatchJobResponse, ErrorResponse, SingleResponse, ListResponse,
    ProcessingStatusEnum, GeoJSONFeatureCollection
)
from ..models.database import Document, OCRResult, ProcessingLog
from ..services.mistral_ocr_service import MistralOCRService
from ..services.document_processor import DocumentProcessor
from ..services.coordinate_parser import CoordinateParser
from ..services.geojson_converter import GeoJSONConverter
from ..utils.file_handler import FileHandler
from ..utils.validators import DocumentValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR Processing"])

# Dependencies
def get_mistral_service() -> MistralOCRService:
    return MistralOCRService()

def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()

def get_coordinate_parser() -> CoordinateParser:
    return CoordinateParser()

def get_geojson_converter() -> GeoJSONConverter:
    return GeoJSONConverter()

def get_file_handler() -> FileHandler:
    return FileHandler()

def get_document_validator() -> DocumentValidator:
    return DocumentValidator()


@router.post("/upload", response_model=SingleResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_batch: bool = False,
    db: AsyncSession = Depends(get_async_session),
    file_handler: FileHandler = Depends(get_file_handler),
    validator: DocumentValidator = Depends(get_document_validator)
):
    """
    Upload a document for OCR processing.
    
    Supports PDF, JPG, JPEG, PNG, TIFF formats.
    Can process immediately or queue for batch processing.
    """
    try:
        # Validate file
        validation_result = await validator.validate_upload_file(file)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File validation failed: {', '.join(validation_result.errors)}"
            )
        
        # Save file to storage
        file_info = await file_handler.save_uploaded_file(file)
        
        # Create document record
        document = Document(
            filename=file_info["filename"],
            original_filename=file.filename,
            file_path=file_info["file_path"],
            file_size=file_info["file_size"],
            file_type=file_info["file_type"],
            document_type=file_info["document_type"],
            mime_type=file.content_type or "application/octet-stream",
            status=ProcessingStatusEnum.PENDING
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # Log upload
        await _log_processing_event(
            db, document.id, "INFO", "Document uploaded successfully", "upload"
        )
        
        # Queue processing
        if use_batch:
            background_tasks.add_task(
                _queue_batch_processing, document.id, db
            )
        else:
            background_tasks.add_task(
                _process_document_immediately, document.id, db
            )
        
        response_data = DocumentResponse.from_orm(document)
        return SingleResponse(
            message="Document uploaded successfully. Processing started.",
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.get("/status/{document_id}", response_model=DocumentStatusResponse)
async def get_processing_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session)
):
    """Get processing status for a document."""
    try:
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Calculate progress if processing
        progress = await _calculate_progress(document, db)
        
        return DocumentStatusResponse(
            document_id=document.id,
            status=ProcessingStatusEnum(document.status),
            progress_percentage=progress,
            message=f"Document status: {document.status}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document status"
        )


@router.get("/result/{document_id}", response_model=ProcessingResult)
async def get_processing_result(
    document_id: uuid.UUID,
    include_geojson: bool = True,
    db: AsyncSession = Depends(get_async_session)
):
    """Get complete processing results for a document."""
    try:
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if document.status != ProcessingStatusEnum.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document processing not completed. Current status: {document.status}"
            )
        
        # Get all related data
        result = await _build_processing_result(document, db, include_geojson)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Result retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing result"
        )


@router.get("/result/{document_id}/geojson", response_model=GeoJSONFeatureCollection)
async def get_geojson_result(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    converter: GeoJSONConverter = Depends(get_geojson_converter)
):
    """Get GeoJSON representation of extracted claims."""
    try:
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Generate GeoJSON from claims and geometries
        geojson = await converter.create_geojson_from_document(document_id, db)
        
        return geojson
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GeoJSON generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate GeoJSON"
        )


@router.post("/batch", response_model=SingleResponse)
async def create_batch_job(
    batch_request: BatchJobRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    mistral_service: MistralOCRService = Depends(get_mistral_service)
):
    """Create a batch processing job for multiple documents."""
    try:
        # Validate all documents exist and are pending
        documents = []
        for doc_id in batch_request.document_ids:
            document = await db.get(Document, doc_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {doc_id} not found"
                )
            if document.status != ProcessingStatusEnum.PENDING:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Document {doc_id} is not in pending status"
                )
            documents.append(document)
        
        # Create batch job
        batch_job = await mistral_service.create_batch_job(documents, batch_request.metadata)
        
        # Update document statuses
        for document in documents:
            document.status = ProcessingStatusEnum.PROCESSING
            document.mistral_batch_id = batch_job.mistral_batch_id
        
        await db.commit()
        
        # Start monitoring batch job
        background_tasks.add_task(
            _monitor_batch_job, batch_job.id, db
        )
        
        return SingleResponse(
            message=f"Batch job created with {len(documents)} documents",
            data=BatchJobResponse.from_orm(batch_job)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch job creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch job"
        )


@router.get("/batch/{batch_id}", response_model=SingleResponse)
async def get_batch_status(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    mistral_service: MistralOCRService = Depends(get_mistral_service)
):
    """Get batch job status and progress."""
    try:
        batch_job = await mistral_service.get_batch_job_status(batch_id, db)
        if not batch_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch job not found"
            )
        
        return SingleResponse(
            message="Batch job status retrieved",
            data=BatchJobResponse.from_orm(batch_job)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get batch status"
        )


@router.post("/reprocess/{document_id}", response_model=SingleResponse)
async def reprocess_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    force: bool = False,
    db: AsyncSession = Depends(get_async_session)
):
    """Reprocess a document (e.g., after failed processing)."""
    try:
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if document.status == ProcessingStatusEnum.PROCESSING and not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document is currently being processed. Use force=true to override."
            )
        
        # Reset status and clear previous results if reprocessing
        document.status = ProcessingStatusEnum.PENDING
        document.processed_at = None
        document.mistral_task_id = None
        
        await db.commit()
        
        # Start reprocessing
        background_tasks.add_task(
            _process_document_immediately, document.id, db
        )
        
        return SingleResponse(
            message="Document queued for reprocessing",
            data=DocumentResponse.from_orm(document)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reprocessing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reprocess document"
        )


@router.delete("/document/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """Delete a document and all associated data."""
    try:
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete physical file
        await file_handler.delete_file(document.file_path)
        
        # Delete database record (cascade will handle related records)
        await db.delete(document)
        await db.commit()
        
        await _log_processing_event(
            db, document_id, "INFO", "Document deleted", "deletion"
        )
        
        return JSONResponse(
            content={"message": "Document deleted successfully"},
            status_code=status.HTTP_200_OK
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


# Background processing functions
async def _process_document_immediately(document_id: uuid.UUID, db: AsyncSession):
    """Process a single document immediately."""
    try:
        mistral_service = MistralOCRService()
        document_processor = DocumentProcessor()
        coordinate_parser = CoordinateParser()
        
        # Update status
        document = await db.get(Document, document_id)
        document.status = ProcessingStatusEnum.PROCESSING
        await db.commit()
        
        # Process with Mistral OCR
        ocr_results = await mistral_service.process_document(document, db)
        
        # Parse coordinates from OCR results
        for ocr_result in ocr_results:
            await coordinate_parser.extract_coordinates(ocr_result, db)
        
        # Extract claims
        await document_processor.extract_claims(document, db)
        
        # Update final status
        document.status = ProcessingStatusEnum.COMPLETED
        document.processed_at = func.now()
        await db.commit()
        
        await _log_processing_event(
            db, document_id, "INFO", "Document processing completed", "processing"
        )
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        
        # Update status to failed
        document = await db.get(Document, document_id)
        document.status = ProcessingStatusEnum.FAILED
        await db.commit()
        
        await _log_processing_event(
            db, document_id, "ERROR", f"Processing failed: {str(e)}", "processing"
        )


async def _queue_batch_processing(document_id: uuid.UUID, db: AsyncSession):
    """Queue document for batch processing."""
    # This would typically add to a batch queue
    # For now, we'll just mark as ready for batch
    await _log_processing_event(
        db, document_id, "INFO", "Document queued for batch processing", "batch_queue"
    )


async def _monitor_batch_job(batch_job_id: uuid.UUID, db: AsyncSession):
    """Monitor a batch job for completion."""
    try:
        mistral_service = MistralOCRService()
        await mistral_service.monitor_batch_job(batch_job_id, db)
        
    except Exception as e:
        logger.error(f"Batch monitoring failed: {str(e)}")


async def _calculate_progress(document: Document, db: AsyncSession) -> Optional[float]:
    """Calculate processing progress for a document."""
    if document.status == ProcessingStatusEnum.PENDING:
        return 0.0
    elif document.status == ProcessingStatusEnum.PROCESSING:
        # Could check Mistral API for actual progress
        return 50.0
    elif document.status == ProcessingStatusEnum.COMPLETED:
        return 100.0
    elif document.status == ProcessingStatusEnum.FAILED:
        return 0.0
    return None


async def _build_processing_result(
    document: Document, 
    db: AsyncSession, 
    include_geojson: bool = True
) -> ProcessingResult:
    """Build complete processing result for a document."""
    # This would fetch all related data and build the result
    # Implementation depends on your specific requirements
    return ProcessingResult(
        document_id=document.id,
        status=ProcessingStatusEnum(document.status),
        processing_time=0.0  # Calculate actual processing time
    )


async def _log_processing_event(
    db: AsyncSession,
    document_id: uuid.UUID,
    level: str,
    message: str,
    component: str,
    context_data: dict = None
):
    """Log processing events."""
    log_entry = ProcessingLog(
        document_id=document_id,
        level=level,
        message=message,
        component=component,
        context_data=context_data
    )
    
    db.add(log_entry)
    await db.commit()