from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from enum import Enum

from ..core.database import Base
from ..core.config import settings


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


def generate_uuid():
    """Generate UUID string for primary keys"""
    return str(uuid.uuid4())


class Document(Base):
    """Store uploaded documents for OCR processing"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    document_type = Column(String(20), nullable=False, default=DocumentType.PDF.value)
    mime_type = Column(String(100), nullable=False)
    
    # Processing information
    status = Column(String(20), nullable=False, default=ProcessingStatus.PENDING.value)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Mistral OCR information
    mistral_task_id = Column(String(100), nullable=True)
    mistral_batch_id = Column(String(100), nullable=True)
    
    # Relationships
    ocr_results = relationship("OCRResult", back_populates="document", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="document", cascade="all, delete-orphan")
    processing_logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_document_status', 'status'),
        Index('idx_document_uploaded_at', 'uploaded_at'),
        Index('idx_mistral_task_id', 'mistral_task_id'),
        Index('idx_mistral_batch_id', 'mistral_batch_id'),
        CheckConstraint("file_size > 0", name="check_positive_file_size"),
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')", 
                       name="check_valid_status"),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class OCRResult(Base):
    """Store OCR results from Mistral API"""
    __tablename__ = "ocr_results"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    
    # OCR content
    raw_text = Column(Text, nullable=False)
    markdown_content = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=False, default=1)
    
    # Mistral specific data
    mistral_response = Column(JSON, nullable=True)  # Full Mistral API response
    confidence_score = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    # Relationships
    document = relationship("Document", back_populates="ocr_results")
    coordinates = relationship("ExtractedCoordinate", back_populates="ocr_result", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_ocr_document_id', 'document_id'),
        Index('idx_ocr_page_number', 'page_number'),
        Index('idx_ocr_created_at', 'created_at'),
        UniqueConstraint('document_id', 'page_number', name='uq_document_page'),
        CheckConstraint("page_number > 0", name="check_positive_page_number"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_range"),
    )
    
    def __repr__(self):
        return f"<OCRResult(id={self.id}, document_id={self.document_id}, page={self.page_number})>"


class ExtractedCoordinate(Base):
    """Store coordinates extracted from OCR text"""
    __tablename__ = "extracted_coordinates"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    ocr_result_id = Column(String(36), ForeignKey("ocr_results.id"), nullable=False)
    
    # Coordinate data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    coordinate_format = Column(String(20), nullable=False)
    raw_coordinate_text = Column(String(200), nullable=False)
    
    # Validation
    is_valid = Column(Boolean, nullable=False, default=True)
    validation_errors = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Survey information
    survey_number = Column(String(50), nullable=True)
    plot_id = Column(String(50), nullable=True)
    
    # Metadata
    extracted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    ocr_result = relationship("OCRResult", back_populates="coordinates")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_coordinates_lat_lon', 'latitude', 'longitude'),
        Index('idx_coordinates_survey', 'survey_number'),
        Index('idx_coordinates_plot', 'plot_id'),
        Index('idx_coordinates_extracted_at', 'extracted_at'),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="check_valid_latitude"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="check_valid_longitude"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_coordinate_confidence_range"),
        CheckConstraint("coordinate_format IN ('decimal_degrees', 'dms', 'utm', 'survey_grid')", 
                       name="check_valid_coordinate_format"),
    )
    
    def __repr__(self):
        return f"<ExtractedCoordinate(id={self.id}, lat={self.latitude}, lon={self.longitude})>"


class Claim(Base):
    """Store FRA claim information extracted from documents"""
    __tablename__ = "claims"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    
    # Claim information
    claim_type = Column(String(10), nullable=False)  # IFR, CFR, CR
    holder_name = Column(String(200), nullable=True)
    area_hectares = Column(Float, nullable=True)
    survey_numbers = Column(JSON, nullable=True)  # Store as JSON array
    
    # Location information
    village = Column(String(100), nullable=True)
    block = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    
    # Dates
    application_date = Column(DateTime, nullable=True)
    survey_date = Column(DateTime, nullable=True)
    
    # Rights claimed
    rights_claimed = Column(JSON, nullable=True)  # Store as JSON array
    
    # Processing metadata
    confidence_score = Column(Float, nullable=True)
    extraction_method = Column(String(50), nullable=False, default="mistral_ocr")
    status = Column(String(20), nullable=False, default=ProcessingStatus.PENDING.value)
    
    # Additional claim details
    claim_number = Column(String(100), nullable=True)  # Official claim number if available
    forest_type = Column(String(50), nullable=True)  # Type of forest (reserved, protected, etc.)
    land_classification = Column(String(50), nullable=True)  # Revenue/forest land classification
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="claims")
    geometries = relationship("ClaimGeometry", back_populates="claim", cascade="all, delete-orphan")
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_claim_type', 'claim_type'),
        Index('idx_claim_holder', 'holder_name'),
        Index('idx_claim_location', 'village', 'block', 'district'),
        Index('idx_claim_status', 'status'),
        Index('idx_claim_created_at', 'created_at'),
        Index('idx_claim_number', 'claim_number'),
        CheckConstraint("area_hectares > 0", name="check_positive_area"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_claim_confidence_range"),
        CheckConstraint("claim_type IN ('IFR', 'CFR', 'CR')", name="check_valid_claim_type"),
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')", 
                       name="check_claim_valid_status"),
    )
    
    def __repr__(self):
        return f"<Claim(id={self.id}, type={self.claim_type}, holder={self.holder_name})>"


class ClaimGeometry(Base):
    """Store spatial geometry for claims in GeoJSON format"""
    __tablename__ = "claim_geometries"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    claim_id = Column(String(36), ForeignKey("claims.id"), nullable=False)
    
    # Spatial data - stored as JSON and WKT for compatibility
    geojson = Column(JSON, nullable=False)
    wkt_geometry = Column(Text, nullable=True)  # Well-Known Text format
    area_calculated = Column(Float, nullable=True)  # Area in hectares
    perimeter_calculated = Column(Float, nullable=True)  # Perimeter in meters
    
    # Bounding box for quick spatial queries
    bbox_min_lat = Column(Float, nullable=True)
    bbox_max_lat = Column(Float, nullable=True)
    bbox_min_lon = Column(Float, nullable=True)
    bbox_max_lon = Column(Float, nullable=True)
    
    # Source information
    source_type = Column(String(50), nullable=False, default="coordinate_extraction")
    coordinate_source = Column(String(100), nullable=True)
    coordinate_count = Column(Integer, nullable=True)  # Number of coordinates used
    
    # Quality metadata
    geometry_quality = Column(String(20), nullable=False, default="medium")  # high, medium, low
    validation_status = Column(String(20), nullable=False, default="pending")
    validation_errors = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    
    # Relationships
    claim = relationship("Claim", back_populates="geometries")
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_geometry_claim_id', 'claim_id'),
        Index('idx_geometry_source', 'source_type'),
        Index('idx_geometry_quality', 'geometry_quality'),
        Index('idx_geometry_bbox', 'bbox_min_lat', 'bbox_max_lat', 'bbox_min_lon', 'bbox_max_lon'),
        CheckConstraint("area_calculated >= 0", name="check_positive_calculated_area"),
        CheckConstraint("perimeter_calculated >= 0", name="check_positive_perimeter"),
        CheckConstraint("geometry_quality IN ('high', 'medium', 'low')", name="check_valid_quality"),
        CheckConstraint("validation_status IN ('pending', 'validated', 'failed')", 
                       name="check_valid_validation_status"),
    )
    
    def __repr__(self):
        return f"<ClaimGeometry(id={self.id}, claim_id={self.claim_id}, quality={self.geometry_quality})>"


class ProcessingLog(Base):
    """Log processing events and errors"""
    __tablename__ = "processing_logs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=True)
    
    # Log information
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    component = Column(String(50), nullable=False)  # mistral_ocr, coordinate_parser, etc.
    
    # Context data
    context_data = Column(JSON, nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # User and session information
    user_id = Column(String(50), nullable=True)
    session_id = Column(String(100), nullable=True)
    request_id = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="processing_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_log_document_id', 'document_id'),
        Index('idx_log_level', 'level'),
        Index('idx_log_component', 'component'),
        Index('idx_log_created_at', 'created_at'),
        Index('idx_log_session', 'session_id'),
        CheckConstraint("level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", 
                       name="check_valid_log_level"),
    )
    
    def __repr__(self):
        return f"<ProcessingLog(id={self.id}, level={self.level}, component={self.component})>"


class BatchJob(Base):
    """Track Mistral batch processing jobs"""
    __tablename__ = "batch_jobs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
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
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Progress tracking
    progress_percentage = Column(Float, nullable=False, default=0.0)
    estimated_completion = Column(DateTime, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    
    # Metadata
    batch_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_batch_mistral_id', 'mistral_batch_id'),
        Index('idx_batch_status', 'status'),
        Index('idx_batch_created_at', 'created_at'),
        CheckConstraint("total_requests >= 0", name="check_positive_total_requests"),
        CheckConstraint("completed_requests >= 0", name="check_positive_completed_requests"),
        CheckConstraint("failed_requests >= 0", name="check_positive_failed_requests"),
        CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100", 
                       name="check_valid_progress_percentage"),
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", 
                       name="check_valid_batch_status"),
    )
    
    def __repr__(self):
        return f"<BatchJob(id={self.id}, status={self.status}, progress={self.progress_percentage}%)>"


class ConflictDetection(Base):
    """Store detected conflicts between overlapping claims"""
    __tablename__ = "conflict_detections"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    claim_id_1 = Column(String(36), ForeignKey("claims.id"), nullable=False)
    claim_id_2 = Column(String(36), ForeignKey("claims.id"), nullable=False)
    
    # Conflict details
    conflict_type = Column(String(50), nullable=False, default="spatial_overlap")
    overlap_area = Column(Float, nullable=True)  # Overlapping area in hectares
    overlap_percentage_1 = Column(Float, nullable=True)  # Overlap as % of claim 1
    overlap_percentage_2 = Column(Float, nullable=True)  # Overlap as % of claim 2
    
    # Conflict geometry
    conflict_geometry = Column(JSON, nullable=True)  # GeoJSON of overlapping area
    
    # Resolution tracking
    resolution_status = Column(String(20), nullable=False, default="detected")
    resolution_notes = Column(Text, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Metadata
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    severity = Column(String(20), nullable=False, default="medium")  # high, medium, low
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_conflict_claim_1', 'claim_id_1'),
        Index('idx_conflict_claim_2', 'claim_id_2'),
        Index('idx_conflict_status', 'resolution_status'),
        Index('idx_conflict_detected_at', 'detected_at'),
        UniqueConstraint('claim_id_1', 'claim_id_2', name='uq_claim_conflict'),
        CheckConstraint("claim_id_1 != claim_id_2", name="check_different_claims"),
        CheckConstraint("overlap_area >= 0", name="check_positive_overlap_area"),
        CheckConstraint("overlap_percentage_1 >= 0 AND overlap_percentage_1 <= 100", 
                       name="check_valid_overlap_percentage_1"),
        CheckConstraint("overlap_percentage_2 >= 0 AND overlap_percentage_2 <= 100", 
                       name="check_valid_overlap_percentage_2"),
        CheckConstraint("severity IN ('high', 'medium', 'low')", name="check_valid_severity"),
        CheckConstraint("resolution_status IN ('detected', 'reviewing', 'resolved', 'dismissed')", 
                       name="check_valid_resolution_status"),
    )
    
    def __repr__(self):
        return f"<ConflictDetection(id={self.id}, claims=({self.claim_id_1}, {self.claim_id_2}), severity={self.severity})>"


# Helper functions for common queries
class DatabaseQueries:
    """Common database queries for the application"""
    
    @staticmethod
    def get_pending_documents():
        """Get documents pending OCR processing"""
        return "SELECT * FROM documents WHERE status = 'pending' ORDER BY uploaded_at ASC"
    
    @staticmethod
    def get_claims_by_location(village: str = None, district: str = None, state: str = None):
        """Get claims by location"""
        conditions = []
        if village:
            conditions.append(f"village ILIKE '%{village}%'")
        if district:
            conditions.append(f"district ILIKE '%{district}%'")
        if state:
            conditions.append(f"state ILIKE '%{state}%'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return f"SELECT * FROM claims WHERE {where_clause}"
    
    @staticmethod
    def get_processing_statistics():
        """Get processing statistics"""
        return """
        SELECT 
            status,
            COUNT(*) as count,
            AVG(CASE WHEN processed_at IS NOT NULL THEN 
                (julianday(processed_at) - julianday(uploaded_at)) * 24 * 60 
                ELSE NULL END) as avg_processing_time_minutes
        FROM documents 
        GROUP BY status
        """
    
    @staticmethod
    def get_claims_with_conflicts():
        """Get claims that have conflicts"""
        return """
        SELECT DISTINCT c.* 
        FROM claims c 
        JOIN conflict_detections cd ON (c.id = cd.claim_id_1 OR c.id = cd.claim_id_2)
        WHERE cd.resolution_status = 'detected'
        ORDER BY c.created_at DESC
        """


# Create spatial indexes (for future PostGIS support)
def create_spatial_indexes():
    """Create additional spatial indexes for performance"""
    spatial_indexes = []
    
    if settings.is_postgresql:
        # PostGIS specific indexes
        spatial_indexes.extend([
            "CREATE INDEX IF NOT EXISTS idx_geometry_spatial ON claim_geometries USING gist(geometry);",
            "CREATE INDEX IF NOT EXISTS idx_geometry_area ON claim_geometries USING btree(area_calculated);",
        ])
    else:
        # SQLite spatial-like indexes using bounding box
        spatial_indexes.extend([
            "CREATE INDEX IF NOT EXISTS idx_geometry_bbox_spatial ON claim_geometries (bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon);",
            "CREATE INDEX IF NOT EXISTS idx_geometry_area ON claim_geometries (area_calculated);",
        ])
    
    spatial_indexes.extend([
        "CREATE INDEX IF NOT EXISTS idx_coordinates_location ON extracted_coordinates (latitude, longitude);",
        "CREATE INDEX IF NOT EXISTS idx_claims_spatial_search ON claims (village, district, state);",
    ])
    
    return spatial_indexes


# Model validation functions
def validate_claim_data(claim_dict: dict) -> tuple[bool, list[str]]:
    """Validate claim data before database insertion"""
    errors = []
    
    # Required fields
    if not claim_dict.get('claim_type') or claim_dict['claim_type'] not in ['IFR', 'CFR', 'CR']:
        errors.append("Valid claim_type (IFR, CFR, CR) is required")
    
    # Area validation
    if claim_dict.get('area_hectares') is not None:
        if claim_dict['area_hectares'] <= 0:
            errors.append("Area must be positive")
        if claim_dict['area_hectares'] > 10000:  # 10,000 hectares seems excessive for individual claims
            errors.append("Area seems unusually large")
    
    # Coordinate validation
    if claim_dict.get('coordinates'):
        for coord in claim_dict['coordinates']:
            lat, lon = coord.get('latitude'), coord.get('longitude')
            if lat is None or lon is None:
                errors.append("Coordinates must have latitude and longitude")
            elif not (-90 <= lat <= 90 and -180 <= lon <= 180):
                errors.append("Invalid coordinate values")
    
    return len(errors) == 0, errors


# Export all models and utilities
__all__ = [
    "Base",
    "Document",
    "OCRResult", 
    "ExtractedCoordinate",
    "Claim",
    "ClaimGeometry",
    "ProcessingLog",
    "BatchJob",
    "ConflictDetection",
    "ProcessingStatus",
    "ClaimType", 
    "DocumentType",
    "CoordinateFormat",
    "DatabaseQueries",
    "create_spatial_indexes",
    "validate_claim_data"
]