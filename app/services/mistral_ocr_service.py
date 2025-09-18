import asyncio
import base64
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
import uuid
from datetime import datetime

from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage
import aiofiles
import httpx

from ..models.database import Document, OCRResult, BatchJob, ProcessingLog
from ..models.schemas import ProcessingStatusEnum
from ..core.config import settings
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)


class MistralOCRService:
    """
    Service for processing documents using Mistral's OCR API.
    Handles both individual document processing and batch operations.
    """
    
    def __init__(self):
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.ocr_model = settings.MISTRAL_OCR_MODEL
        self.text_model = settings.MISTRAL_TEXT_MODEL
        self.file_handler = FileHandler()
        
        # Batch processing configuration
        self.batch_config = {
            "endpoint": "/v1/ocr",
            "purpose": "batch",
            "polling_interval": 5,  # seconds
            "max_wait_time": 3600   # 1 hour
        }
    
    async def process_document(
        self, 
        document: Document, 
        db: AsyncSession
    ) -> List[OCRResult]:
        """
        Process a single document using Mistral OCR API.
        
        Args:
            document: Document record to process
            db: Database session
            
        Returns:
            List of OCR results (one per page)
        """
        try:
            logger.info(f"Starting OCR processing for document {document.id}")
            
            # Update document status
            document.status = ProcessingStatusEnum.PROCESSING.value
            await db.commit()
            
            # Encode document to base64
            base64_content = await self._encode_document_to_base64(document.file_path)
            
            # Determine document type and create request
            if document.document_type.lower() == "pdf":
                ocr_request = {
                    "model": self.ocr_model,
                    "document": {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{base64_content}"
                    },
                    "include_image_base64": True
                }
            else:
                # Image document
                mime_type = document.mime_type or "image/jpeg"
                ocr_request = {
                    "model": self.ocr_model,
                    "document": {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{base64_content}"
                    },
                    "include_image_base64": True
                }
            
            # Process with Mistral OCR
            start_time = time.time()
            response = await self._call_mistral_ocr_api(ocr_request)
            processing_time = time.time() - start_time
            
            # Create OCR results
            ocr_results = []
            for i, page in enumerate(response.pages):
                ocr_result = OCRResult(
                    document_id=document.id,
                    raw_text=page.text or "",
                    markdown_content=page.markdown or "",
                    page_number=i + 1,
                    mistral_response=response.dict() if hasattr(response, 'dict') else {},
                    confidence_score=self._extract_confidence_score(page),
                    processing_time=processing_time
                )
                
                db.add(ocr_result)
                ocr_results.append(ocr_result)
            
            # Update document status
            document.status = ProcessingStatusEnum.COMPLETED.value
            document.processed_at = datetime.now()
            
            await db.commit()
            
            await self._log_processing_event(
                db, document.id, "INFO", 
                f"OCR processing completed. {len(ocr_results)} pages processed.", 
                "mistral_ocr"
            )
            
            logger.info(f"OCR processing completed for document {document.id}")
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR processing failed for document {document.id}: {str(e)}")
            
            # Update document status to failed
            document.status = ProcessingStatusEnum.FAILED.value
            await db.commit()
            
            await self._log_processing_event(
                db, document.id, "ERROR", 
                f"OCR processing failed: {str(e)}", 
                "mistral_ocr"
            )
            
            raise
    
    async def create_batch_job(
        self, 
        documents: List[Document], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchJob:
        """
        Create a batch processing job for multiple documents.
        
        Args:
            documents: List of documents to process
            metadata: Optional job metadata
            
        Returns:
            BatchJob record
        """
        try:
            logger.info(f"Creating batch job for {len(documents)} documents")
            
            # Create batch file
            batch_file_path = await self._create_batch_file(documents)
            
            # Upload batch file to Mistral
            upload_response = await self._upload_batch_file(batch_file_path)
            
            # Create batch job with Mistral
            job_response = await self._create_mistral_batch_job(
                upload_response["id"], metadata
            )
            
            # Create BatchJob record
            batch_job = BatchJob(
                mistral_batch_id=job_response["id"],
                mistral_input_file_id=upload_response["id"],
                total_requests=len(documents),
                status="queued",
                metadata=metadata or {}
            )
            
            # Note: db.add and commit would be handled by the caller
            
            logger.info(f"Batch job created with ID {batch_job.mistral_batch_id}")
            return batch_job
            
        except Exception as e:
            logger.error(f"Batch job creation failed: {str(e)}")
            raise
    
    async def monitor_batch_job(
        self, 
        batch_job_id: uuid.UUID, 
        db: AsyncSession
    ) -> None:
        """
        Monitor a batch job until completion.
        
        Args:
            batch_job_id: BatchJob UUID
            db: Database session
        """
        try:
            batch_job = await db.get(BatchJob, batch_job_id)
            if not batch_job:
                raise ValueError(f"Batch job {batch_job_id} not found")
            
            logger.info(f"Starting monitoring for batch job {batch_job.mistral_batch_id}")
            
            while batch_job.status in ["queued", "running"]:
                # Check job status with Mistral
                job_status = await self._get_mistral_batch_status(batch_job.mistral_batch_id)
                
                # Update local status
                batch_job.status = job_status["status"]
                batch_job.completed_requests = job_status.get("succeeded_requests", 0)
                batch_job.failed_requests = job_status.get("failed_requests", 0)
                
                if job_status["status"] == "running" and not batch_job.started_at:
                    batch_job.started_at = datetime.now()
                
                await db.commit()
                
                # Break if completed
                if job_status["status"] in ["completed", "failed", "cancelled"]:
                    break
                
                # Wait before next check
                await asyncio.sleep(self.batch_config["polling_interval"])
            
            # Handle completion
            if batch_job.status == "completed":
                batch_job.completed_at = datetime.now()
                
                # Download and process results
                await self._process_batch_results(batch_job, db)
                
                logger.info(f"Batch job {batch_job.mistral_batch_id} completed successfully")
            else:
                logger.warning(f"Batch job {batch_job.mistral_batch_id} ended with status: {batch_job.status}")
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"Batch monitoring failed: {str(e)}")
            raise
    
    async def get_batch_job_status(
        self, 
        batch_job_id: uuid.UUID, 
        db: AsyncSession
    ) -> Optional[BatchJob]:
        """Get current status of a batch job."""
        try:
            batch_job = await db.get(BatchJob, batch_job_id)
            if not batch_job:
                return None
            
            # Update with latest Mistral status
            if batch_job.status in ["queued", "running"]:
                job_status = await self._get_mistral_batch_status(batch_job.mistral_batch_id)
                
                batch_job.status = job_status["status"]
                batch_job.completed_requests = job_status.get("succeeded_requests", 0)
                batch_job.failed_requests = job_status.get("failed_requests", 0)
                
                await db.commit()
            
            return batch_job
            
        except Exception as e:
            logger.error(f"Batch status check failed: {str(e)}")
            return None
    
    async def _encode_document_to_base64(self, file_path: str) -> str:
        """Encode document file to base64 string."""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
                return base64.b64encode(file_content).decode('utf-8')
        except Exception as e:
            logger.error(f"File encoding failed for {file_path}: {str(e)}")
            raise
    
    async def _call_mistral_ocr_api(self, request_data: Dict[str, Any]) -> Any:
        """Call Mistral OCR API with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Use the synchronous client in an async context
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.client.ocr.process(**request_data)
                )
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(f"OCR API call failed (attempt {attempt + 1}): {str(e)}")
                await asyncio.sleep(retry_delay * (2 ** attempt))
    
    def _extract_confidence_score(self, page_data: Any) -> Optional[float]:
        """Extract confidence score from OCR page data."""
        try:
            # This would depend on Mistral's actual response structure
            # For now, return a default confidence
            return 0.85
        except Exception:
            return None
    
    async def _create_batch_file(self, documents: List[Document]) -> str:
        """Create batch file for Mistral batch processing."""
        try:
            batch_data = []
            
            for i, document in enumerate(documents):
                # Encode document
                base64_content = await self._encode_document_to_base64(document.file_path)
                
                # Create batch entry
                if document.document_type.lower() == "pdf":
                    document_url = f"data:application/pdf;base64,{base64_content}"
                    entry = {
                        "custom_id": str(document.id),
                        "body": {
                            "document": {
                                "type": "document_url",
                                "document_url": document_url
                            },
                            "include_image_base64": True
                        }
                    }
                else:
                    mime_type = document.mime_type or "image/jpeg"
                    image_url = f"data:{mime_type};base64,{base64_content}"
                    entry = {
                        "custom_id": str(document.id),
                        "body": {
                            "document": {
                                "type": "image_url",
                                "image_url": image_url
                            },
                            "include_image_base64": True
                        }
                    }
                
                batch_data.append(entry)
            
            # Save batch file
            batch_file_path = f"/tmp/batch_{uuid.uuid4().hex}.jsonl"
            async with aiofiles.open(batch_file_path, 'w') as f:
                for entry in batch_data:
                    await f.write(json.dumps(entry) + '\n')
            
            logger.info(f"Created batch file with {len(batch_data)} entries: {batch_file_path}")
            return batch_file_path
            
        except Exception as e:
            logger.error(f"Batch file creation failed: {str(e)}")
            raise
    
    async def _upload_batch_file(self, file_path: str) -> Dict[str, Any]:
        """Upload batch file to Mistral."""
        try:
            # Use synchronous client in async context
            with open(file_path, 'rb') as f:
                upload_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.files.upload(
                        file={
                            "file_name": Path(file_path).name,
                            "content": f
                        },
                        purpose=self.batch_config["purpose"]
                    )
                )
            
            # Clean up temporary file
            Path(file_path).unlink(missing_ok=True)
            
            return {"id": upload_response.id}
            
        except Exception as e:
            logger.error(f"Batch file upload failed: {str(e)}")
            raise
    
    async def _create_mistral_batch_job(
        self, 
        input_file_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create batch job with Mistral API."""
        try:
            job_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.batch.jobs.create(
                    input_files=[input_file_id],
                    model=self.ocr_model,
                    endpoint=self.batch_config["endpoint"],
                    metadata=metadata or {}
                )
            )
            
            return {
                "id": job_response.id,
                "status": job_response.status,
                "input_files": job_response.input_files
            }
            
        except Exception as e:
            logger.error(f"Mistral batch job creation failed: {str(e)}")
            raise
    
    async def _get_mistral_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch job status from Mistral."""
        try:
            status_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.batch.jobs.get(job_id=batch_id)
            )
            
            return {
                "status": status_response.status,
                "total_requests": status_response.total_requests,
                "succeeded_requests": status_response.succeeded_requests,
                "failed_requests": status_response.failed_requests,
                "output_file": getattr(status_response, 'output_file', None)
            }
            
        except Exception as e:
            logger.error(f"Mistral batch status check failed: {str(e)}")
            raise
    
    async def _process_batch_results(self, batch_job: BatchJob, db: AsyncSession):
        """Process completed batch job results."""
        try:
            # Get batch status to get output file
            job_status = await self._get_mistral_batch_status(batch_job.mistral_batch_id)
            
            if not job_status.get("output_file"):
                logger.warning(f"No output file for batch job {batch_job.mistral_batch_id}")
                return
            
            # Download results file
            results_content = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.files.download(file_id=job_status["output_file"])
            )
            
            # Process each result
            lines = results_content.decode('utf-8').strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                
                try:
                    result_data = json.loads(line)
                    await self._process_single_batch_result(result_data, db)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in batch results: {line}")
                    continue
            
            batch_job.mistral_output_file_id = job_status["output_file"]
            await db.commit()
            
            logger.info(f"Processed batch results for job {batch_job.mistral_batch_id}")
            
        except Exception as e:
            logger.error(f"Batch results processing failed: {str(e)}")
            raise
    
    async def _process_single_batch_result(
        self, 
        result_data: Dict[str, Any], 
        db: AsyncSession
    ):
        """Process a single result from batch output."""
        try:
            document_id = result_data.get("custom_id")
            if not document_id:
                return
            
            # Get document
            document = await db.get(Document, uuid.UUID(document_id))
            if not document:
                logger.warning(f"Document {document_id} not found for batch result")
                return
            
            # Extract OCR response
            response_body = result_data.get("response", {}).get("body", {})
            pages = response_body.get("pages", [])
            
            # Create OCR results
            for i, page in enumerate(pages):
                ocr_result = OCRResult(
                    document_id=document.id,
                    raw_text=page.get("text", ""),
                    markdown_content=page.get("markdown", ""),
                    page_number=i + 1,
                    mistral_response=response_body,
                    confidence_score=0.85  # Default confidence for batch processing
                )
                
                db.add(ocr_result)
            
            # Update document status
            document.status = ProcessingStatusEnum.COMPLETED.value
            document.processed_at = datetime.now()
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"Single batch result processing failed: {str(e)}")
    
    async def _log_processing_event(
        self,
        db: AsyncSession,
        document_id: uuid.UUID,
        level: str,
        message: str,
        component: str,
        context_data: Optional[Dict[str, Any]] = None
    ):
        """Log processing events to database."""
        try:
            log_entry = ProcessingLog(
                document_id=document_id,
                level=level,
                message=message,
                component=component,
                context_data=context_data
            )
            
            db.add(log_entry)
            await db.commit()
            
        except Exception as e:
            logger.error(f"Logging failed: {str(e)}")


class EnhancedMistralOCRService(MistralOCRService):
    """
    Enhanced OCR service with additional features for FRA document processing.
    """
    
    def __init__(self):
        super().__init__()
        
        # FRA-specific processing tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_fra_details",
                    "description": "Extract FRA claim details from OCR text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "OCR extracted text"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
        ]
    
    async def process_with_ai_enhancement(
        self, 
        document: Document, 
        db: AsyncSession
    ) -> List[OCRResult]:
        """
        Process document with AI-enhanced analysis using Mistral's text model.
        """
        try:
            # First, perform standard OCR
            ocr_results = await self.process_document(document, db)
            
            # Enhance with AI analysis
            for ocr_result in ocr_results:
                enhanced_content = await self._enhance_with_ai(ocr_result.raw_text)
                if enhanced_content:
                    ocr_result.markdown_content = enhanced_content
                    await db.commit()
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"AI-enhanced processing failed: {str(e)}")
            raise
    
    async def _enhance_with_ai(self, ocr_text: str) -> Optional[str]:
        """Use Mistral's text model to enhance OCR results."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert at analyzing Forest Rights Act (FRA) documents. 
                    Extract and structure the key information from OCR text including:
                    - Claim type (IFR/CFR/CR)
                    - Holder details
                    - Survey numbers
                    - Coordinates
                    - Area information
                    - Location details
                    Format the output as structured markdown."""
                },
                {
                    "role": "user",
                    "content": f"Analyze this FRA document text:\n\n{ocr_text}"
                }
            ]
            
            response = await self.client.chat.complete_async(
                model=self.text_model,
                messages=messages,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {str(e)}")
            return None