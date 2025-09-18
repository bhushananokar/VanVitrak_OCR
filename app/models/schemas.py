from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from enum import Enum
import uuid
import re


# Enums
class ProcessingStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClaimTypeEnum(str, Enum):
    IFR = "IFR"  # Individual Forest Rights
    CFR = "CFR"  # Community Forest Rights
    CR = "CR"    # Community Rights


class DocumentTypeEnum(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    SCANNED_DOCUMENT = "scanned_document"


class CoordinateFormatEnum(str, Enum):
    DECIMAL_DEGREES = "decimal_degrees"
    DMS = "dms"
    UTM = "utm"
    SURVEY_GRID = "survey_grid"


# Base schemas
class BaseResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseResponse):
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Document schemas
class DocumentUpload(BaseModel):
    filename: str = Field(..., max_length=255)
    file_type: str = Field(..., max_length=50)
    document_type: DocumentTypeEnum = DocumentTypeEnum.PDF
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Filename cannot be empty')
        # Remove potentially dangerous characters
        safe_filename = re.sub(r'[^\w\s\-_\.]', '', v)
        return safe_filename[:255]
    
    @validator('file_type')
    def validate_file_type(cls, v):
        allowed_types = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp']
        if v.lower() not in allowed_types:
            raise ValueError(f'File type must be one of: {", ".join(allowed_types)}')
        return v.lower()


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    document_type: str
    status: ProcessingStatusEnum
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    mistral_task_id: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentStatusResponse(BaseResponse):
    document_id: uuid.UUID
    status: ProcessingStatusEnum
    progress_percentage: Optional[float] = None
    estimated_completion: Optional[datetime] = None


# OCR schemas
class OCRRequest(BaseModel):
    document_id: uuid.UUID
    use_batch_api: bool = False
    include_confidence_scores: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "use_batch_api": False,
                "include_confidence_scores": True
            }
        }


class OCRResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    raw_text: str
    markdown_content: Optional[str] = None
    page_number: int = 1
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# Coordinate schemas
class CoordinateInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    coordinate_format: CoordinateFormatEnum
    raw_text: str = Field(..., max_length=200)
    survey_number: Optional[str] = Field(None, max_length=50)
    plot_id: Optional[str] = Field(None, max_length=50)
    confidence_score: Optional[float] = Field(None, ge=0, le=1)


class CoordinateResponse(BaseModel):
    id: uuid.UUID
    latitude: float
    longitude: float
    coordinate_format: str
    raw_coordinate_text: str
    is_valid: bool
    validation_errors: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    survey_number: Optional[str] = None
    plot_id: Optional[str] = None
    extracted_at: datetime
    
    class Config:
        from_attributes = True


class CoordinateValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    corrected_coordinates: Optional[CoordinateInput] = None


# Claim schemas
class ClaimInput(BaseModel):
    claim_type: ClaimTypeEnum
    holder_name: Optional[str] = Field(None, max_length=200)
    area_hectares: Optional[float] = Field(None, gt=0)
    survey_numbers: Optional[List[str]] = None
    village: Optional[str] = Field(None, max_length=100)
    block: Optional[str] = Field(None, max_length=100)
    district: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=50)
    application_date: Optional[datetime] = None
    survey_date: Optional[datetime] = None
    rights_claimed: Optional[List[str]] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    
    @validator('survey_numbers')
    def validate_survey_numbers(cls, v):
        if v:
            # Clean and validate survey numbers
            cleaned = [s.strip() for s in v if s and s.strip()]
            return cleaned[:10]  # Limit to 10 survey numbers
        return v
    
    @validator('rights_claimed')
    def validate_rights_claimed(cls, v):
        if v:
            allowed_rights = [
                'cultivation', 'grazing', 'fishing', 'water_access',
                'forest_produce', 'hunting', 'traditional_use'
            ]
            return [r for r in v if r in allowed_rights]
        return v


class ClaimResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    claim_type: str
    holder_name: Optional[str] = None
    area_hectares: Optional[float] = None
    survey_numbers: Optional[List[str]] = None
    village: Optional[str] = None
    block: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    application_date: Optional[datetime] = None
    survey_date: Optional[datetime] = None
    rights_claimed: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# GeoJSON schemas
class GeoJSONGeometry(BaseModel):
    type: str = Field(..., regex="^(Point|LineString|Polygon|MultiPoint|MultiLineString|MultiPolygon)$")
    coordinates: List[Any]  # Coordinates structure varies by geometry type


class GeoJSONProperties(BaseModel):
    claim_id: Optional[uuid.UUID] = None
    claim_type: Optional[str] = None
    holder_name: Optional[str] = None
    area_hectares: Optional[float] = None
    survey_numbers: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    extraction_method: Optional[str] = None
    # Allow additional properties
    
    class Config:
        extra = "allow"


class GeoJSONFeature(BaseModel):
    type: str = Field(default="Feature", const=True)
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties
    id: Optional[Union[str, int, uuid.UUID]] = None


class GeoJSONFeatureCollection(BaseModel):
    type: str = Field(default="FeatureCollection", const=True)
    features: List[GeoJSONFeature]
    crs: Optional[Dict[str, Any]] = Field(
        default={
            "type": "name",
            "properties": {"name": "EPSG:4326"}
        }
    )
    
    class Config:
        schema_extra = {
            "example": {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "EPSG:4326"}
                },
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[77.5, 12.5], [77.6, 12.5], [77.6, 12.6], [77.5, 12.6], [77.5, 12.5]]]
                        },
                        "properties": {
                            "claim_id": "123e4567-e89b-12d3-a456-426614174000",
                            "claim_type": "IFR",
                            "holder_name": "Sample Holder",
                            "area_hectares": 2.5,
                            "confidence_score": 0.85
                        }
                    }
                ]
            }
        }


class GeometryInput(BaseModel):
    geometry: GeoJSONGeometry
    source_type: str = "coordinate_extraction"
    geometry_quality: str = Field(default="medium", regex="^(high|medium|low)$")


class GeometryResponse(BaseModel):
    id: uuid.UUID
    claim_id: uuid.UUID
    geojson: Dict[str, Any]
    area_calculated: Optional[float] = None
    perimeter_calculated: Optional[float] = None
    source_type: str
    geometry_quality: str
    validation_status: str
    validation_errors: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# Batch processing schemas
class BatchJobRequest(BaseModel):
    document_ids: List[uuid.UUID] = Field(..., min_items=1, max_items=1000)
    priority: str = Field(default="normal", regex="^(low|normal|high)$")
    metadata: Optional[Dict[str, Any]] = None


class BatchJobResponse(BaseModel):
    id: uuid.UUID
    mistral_batch_id: str
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Search and filtering schemas
class DocumentFilter(BaseModel):
    status: Optional[ProcessingStatusEnum] = None
    document_type: Optional[DocumentTypeEnum] = None
    uploaded_after: Optional[datetime] = None
    uploaded_before: Optional[datetime] = None
    filename_contains: Optional[str] = None


class ClaimFilter(BaseModel):
    claim_type: Optional[ClaimTypeEnum] = None
    village: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    min_area: Optional[float] = Field(None, gt=0)
    max_area: Optional[float] = Field(None, gt=0)
    holder_name_contains: Optional[str] = None
    
    @root_validator
    def validate_area_range(cls, values):
        min_area = values.get('min_area')
        max_area = values.get('max_area')
        if min_area and max_area and min_area > max_area:
            raise ValueError('min_area cannot be greater than max_area')
        return values


class SearchRequest(BaseModel):
    query: Optional[str] = None
    filters: Optional[Union[DocumentFilter, ClaimFilter]] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Optional[str] = "created_at"
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")


class SearchResponse(BaseModel):
    items: List[Union[DocumentResponse, ClaimResponse]]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


# API response wrappers
class SingleResponse(BaseResponse):
    data: Union[
        DocumentResponse, 
        OCRResponse, 
        CoordinateResponse, 
        ClaimResponse, 
        GeometryResponse,
        BatchJobResponse
    ]


class ListResponse(BaseResponse):
    data: List[Union[
        DocumentResponse, 
        OCRResponse, 
        CoordinateResponse, 
        ClaimResponse, 
        GeometryResponse
    ]]
    count: int


class ProcessingResult(BaseModel):
    document_id: uuid.UUID
    status: ProcessingStatusEnum
    ocr_results: List[OCRResponse] = []
    extracted_coordinates: List[CoordinateResponse] = []
    claims: List[ClaimResponse] = []
    geometries: List[GeometryResponse] = []
    errors: List[str] = []
    processing_time: Optional[float] = None


# Health check and system status
class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    database_status: str
    mistral_api_status: str
    redis_status: str
    uptime_seconds: float


class SystemStats(BaseModel):
    total_documents: int = 0
    pending_documents: int = 0
    processing_documents: int = 0
    completed_documents: int = 0
    failed_documents: int = 0
    total_claims: int = 0
    total_coordinates: int = 0
    total_geometries: int = 0
    active_batch_jobs: int = 0