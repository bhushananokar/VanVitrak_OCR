from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
import uuid
from datetime import datetime
from enum import Enum

from ..core.database import Base


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClaimType(str, Enum):
    IFR = "IFR"  # Individual Forest Rights
    CFR = "CFR"  # Community Forest Rights
    CR = "CR"    # Community Rights


class DocumentType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    SCANNED_DOCUMENT = "scanned_document"


class CoordinateFormat(str, Enum):
    DECIMAL_DEGREES = "decimal_degrees"
    DMS = "dms"  # Degrees, Minutes, Seconds
    UTM = "utm"
    SURVEY_GRID = "survey_grid"


class Document(Base):
    """Store uploaded documents for OCR processing"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    document_type = Column(String(20), nullable=False, default=DocumentType.PDF.value)
    mime_type = Column(String(100), nullable=False)
    
    # Processing information
    status = Column(String(20), nullable=False, default=ProcessingStatus.PENDING.value)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Mistral OCR information
    mistral_task_id = Column(String(100), nullable=True)
    mistral_batch_id = Column(String(100), nullable=True)
    
    # Relationships
    ocr_results = relationship("OCRResult", back_populates="document", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_status', 'status'),
        Index('idx_document_uploaded_at', 'uploaded_at'),
        Index('idx_mistral_task_id', 'mistral_task_id'),
        Index('idx_mistral_batch_id', 'mistral_batch_id'),
        CheckConstraint("file_size > 0", name="check_positive_file_size"),
    )


class OCRResult(Base):
    """Store OCR results from Mistral API"""
    __tablename__ = "ocr_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # OCR content
    raw_text = Column(Text, nullable=False)
    markdown_content = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=False, default=1)
    
    # Mistral specific data
    mistral_response = Column(JSONB, nullable=True)  # Full Mistral API response
    confidence_score = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="ocr_results")
    coordinates = relationship("ExtractedCoordinate", back_populates="ocr_result", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_ocr_document_id', 'document_id'),
        Index('idx_ocr_page_number', 'page_number'),
        Index('idx_ocr_created_at', 'created_at'),
        UniqueConstraint('document_id', 'page_number', name='uq_document_page'),
    )


class ExtractedCoordinate(Base):
    """Store coordinates extracted from OCR text"""
    __tablename__ = "extracted_coordinates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ocr_result_id = Column(UUID(as_uuid=True), ForeignKey("ocr_results.id"), nullable=False)
    
    # Coordinate data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    coordinate_format = Column(String(20), nullable=False)
    raw_coordinate_text = Column(String(200), nullable=False)
    
    # Validation
    is_valid = Column(Boolean, nullable=False, default=True)
    validation_errors = Column(ARRAY(String), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Survey information
    survey_number = Column(String(50), nullable=True)
    plot_id = Column(String(50), nullable=True)
    
    # Metadata
    extracted_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    ocr_result = relationship("OCRResult", back_populates="coordinates")
    
    # Constraints
    __table_args__ = (
        Index('idx_coordinates_lat_lon', 'latitude', 'longitude'),
        Index('idx_coordinates_survey', 'survey_number'),
        Index('idx_coordinates_plot', 'plot_id'),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="check_valid_latitude"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="check_valid_longitude"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_range"),
    )


class Claim(Base):
    """Store FRA claim information extracted from documents"""
    __tablename__ = "claims"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Claim information
    claim_type = Column(String(10), nullable=False)  # IFR, CFR, CR
    holder_name = Column(String(200), nullable=True)
    area_hectares = Column(Float, nullable=True)
    survey_numbers = Column(ARRAY(String), nullable=True)
    
    # Location information
    village = Column(String(100), nullable=True)
    block = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    
    # Dates
    application_date = Column(DateTime, nullable=True)
    survey_date = Column(DateTime, nullable=True)
    
    # Rights claimed
    rights_claimed = Column(ARRAY(String), nullable=True)
    
    # Processing metadata
    confidence_score = Column(Float, nullable=True)
    extraction_method = Column(String(50), nullable=False, default="mistral_ocr")
    status = Column(String(20), nullable=False, default=ProcessingStatus.PENDING.value)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="claims")
    geometries = relationship("ClaimGeometry", back_populates="claim", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_claim_type', 'claim_type'),
        Index('idx_claim_holder', 'holder_name'),
        Index('idx_claim_location', 'village', 'block', 'district'),
        Index('idx_claim_status', 'status'),
        Index('idx_claim_created_at', 'created_at'),
        CheckConstraint("area_hectares > 0", name="check_positive_area"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_claim_confidence_range"),
    )


class ClaimGeometry(Base):
    """Store spatial geometry for claims in GeoJSON format"""
    __tablename__ = "claim_geometries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=False)
    
    # Spatial data
    geometry = Column(Geometry('GEOMETRY', srid=4326), nullable=False)  # WGS84
    geojson = Column(JSONB, nullable=False)
    area_calculated = Column(Float, nullable=True)  # Area in hectares
    perimeter_calculated = Column(Float, nullable=True)  # Perimeter in meters
    
    # Source information
    source_type = Column(String(50), nullable=False, default="coordinate_extraction")  # coordinate_extraction, manual_digitization
    coordinate_source = Column(String(100), nullable=True)  # Reference to coordinate extraction
    
    # Quality metadata
    geometry_quality = Column(String(20), nullable=False, default="medium")  # high, medium, low
    validation_status = Column(String(20), nullable=False, default="pending")
    validation_errors = Column(ARRAY(String), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    claim = relationship("Claim", back_populates="geometries")
    
    # Indexes
    __table_args__ = (
        Index('idx_geometry_spatial', 'geometry', postgresql_using='gist'),
        Index('idx_geometry_claim_id', 'claim_id'),
        Index('idx_geometry_source', 'source_type'),
        Index('idx_geometry_quality', 'geometry_quality'),
    )


class ProcessingLog(Base):
    """Log processing events and errors"""
    __tablename__ = "processing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    
    # Log information
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    component = Column(String(50), nullable=False)  # mistral_ocr, coordinate_parser, etc.
    
    # Context data
    context_data = Column(JSONB, nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_log_document_id', 'document_id'),
        Index('idx_log_level', 'level'),
        Index('idx_log_component', 'component'),
        Index('idx_log_created_at', 'created_at'),
    )


class BatchJob(Base):
    """Track Mistral batch processing jobs"""
    __tablename__ = "batch_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Mistral batch information
    mistral_batch_id = Column(String(100), nullable=False, unique=True)
    mistral_input_file_id = Column(String(100), nullable=False)
    mistral_output_file_id = Column(String(100), nullable=True)
    
    # Job details
    total_requests = Column(Integer, nullable=False, default=0)
    completed_requests = Column(Integer, nullable=False, default=0)
    failed_requests = Column(Integer, nullable=False, default=0)
    
    # Status tracking
    status = Column(String(20), nullable=False, default="queued")  # queued, running, completed, failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_batch_mistral_id', 'mistral_batch_id'),
        Index('idx_batch_status', 'status'),
        Index('idx_batch_created_at', 'created_at'),
    )


# Create all indexes and constraints
def create_spatial_indexes():
    """Create additional spatial indexes for performance"""
    return [
        "CREATE INDEX IF NOT EXISTS idx_geometry_area ON claim_geometries USING btree(area_calculated);",
        "CREATE INDEX IF NOT EXISTS idx_coordinates_location ON extracted_coordinates USING btree(latitude, longitude);",
        "CREATE INDEX IF NOT EXISTS idx_claims_spatial_search ON claims USING gin(survey_numbers);",
    ]


# Helper functions for common queries
class DatabaseQueries:
    """Common database queries for the application"""
    
    @staticmethod
    def get_pending_documents():
        """Get documents pending OCR processing"""
        return "SELECT * FROM documents WHERE status = 'pending' ORDER BY uploaded_at ASC"
    
    @staticmethod
    def get_claims_by_location(village: str = None, district: str = None):
        """Get claims by location"""
        where_clause = []
        if village:
            where_clause.append(f"village ILIKE '%{village}%'")
        if district:
            where_clause.append(f"district ILIKE '%{district}%'")
        
        where_str = " AND ".join(where_clause) if where_clause else "1=1"
        return f"SELECT * FROM claims WHERE {where_str}"
    
    @staticmethod
    def get_overlapping_geometries(geometry_id: str):
        """Find overlapping claim geometries"""
        return f"""
        SELECT c1.id, c2.id, 
               ST_Area(ST_Intersection(c1.geometry, c2.geometry)) as overlap_area,
               ST_AsGeoJSON(ST_Intersection(c1.geometry, c2.geometry)) as overlap_geometry
        FROM claim_geometries c1, claim_geometries c2 
        WHERE c1.id = '{geometry_id}' 
        AND c2.id != c1.id 
        AND ST_Intersects(c1.geometry, c2.geometry)
        """