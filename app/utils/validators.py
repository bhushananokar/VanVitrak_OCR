import re
import math
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from fastapi import UploadFile
import logging

from ..core.config import settings
from .constants import (
    ALLOWED_FILE_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_FILE_SIZE,
    INDIA_BOUNDS, FRA_CLAIM_TYPES, VALIDATION_THRESHOLDS,
    INDIAN_STATES, ERROR_MESSAGES
)
from ..models.schemas import (
    ClaimTypeEnum, CoordinateFormatEnum, ProcessingStatusEnum
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Standard validation result structure"""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class CoordinateValidationResult(ValidationResult):
    """Validation result for coordinates with corrections"""
    corrected_latitude: Optional[float] = None
    corrected_longitude: Optional[float] = None
    coordinate_system: Optional[str] = None


class DocumentValidator:
    """Validates uploaded documents for OCR processing"""
    
    def __init__(self):
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
    
    async def validate_upload_file(self, upload_file: UploadFile) -> ValidationResult:
        """Validate uploaded file comprehensively"""
        errors = []
        warnings = []
        
        try:
            # Basic file checks
            if not upload_file.filename:
                errors.append("Filename is required")
                return ValidationResult(False, errors, warnings)
            
            # File size validation
            file_size = await self._get_file_size(upload_file)
            if file_size == 0:
                errors.append("File is empty")
            elif file_size > self.max_file_size:
                errors.append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            # File extension validation
            file_extension = self._get_file_extension(upload_file.filename)
            if file_extension not in self.allowed_extensions:
                errors.append(f"File extension '{file_extension}' not allowed. Allowed: {', '.join(self.allowed_extensions)}")
            
            # MIME type validation
            if upload_file.content_type:
                if not self._validate_mime_type(upload_file.content_type, file_extension):
                    warnings.append(f"MIME type '{upload_file.content_type}' may not match file extension '{file_extension}'")
            
            # Filename security check
            if self._has_suspicious_filename(upload_file.filename):
                warnings.append("Filename contains potentially unsafe characters")
            
            # Content validation
            content_errors = await self._validate_file_content(upload_file)
            errors.extend(content_errors)
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            errors.append(f"Validation failed: {str(e)}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    async def _get_file_size(self, upload_file: UploadFile) -> int:
        """Get file size"""
        current_pos = await upload_file.tell()
        await upload_file.seek(0, 2)  # Seek to end
        size = await upload_file.tell()
        await upload_file.seek(current_pos)  # Reset position
        return size
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension"""
        return Path(filename).suffix.lower().lstrip('.')
    
    def _validate_mime_type(self, mime_type: str, extension: str) -> bool:
        """Validate MIME type matches extension"""
        allowed_mimes = ALLOWED_MIME_TYPES.get(extension, [])
        return mime_type in allowed_mimes
    
    def _has_suspicious_filename(self, filename: str) -> bool:
        """Check for suspicious filename patterns"""
        suspicious_patterns = [
            r'\.exe$', r'\.bat$', r'\.cmd$', r'\.scr$', r'\.com$',
            r'[<>:"|?*]', r'\.\.',  # Path traversal
        ]
        
        filename_lower = filename.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, filename_lower):
                return True
        return False
    
    async def _validate_file_content(self, upload_file: UploadFile) -> List[str]:
        """Validate file content by checking headers"""
        errors = []
        
        try:
            current_pos = await upload_file.tell()
            header = await upload_file.read(512)
            await upload_file.seek(current_pos)
            
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
            
            # Check for malicious content
            if b'<script' in header.lower() or b'javascript:' in header.lower():
                errors.append("File contains potentially malicious content")
                
        except Exception as e:
            logger.warning(f"Content validation error: {str(e)}")
        
        return errors


class CoordinateValidator:
    """Validates coordinate data for FRA claims"""
    
    def __init__(self):
        self.bounds = INDIA_BOUNDS
    
    async def validate_coordinate(
        self, 
        latitude: float, 
        longitude: float, 
        coordinate_format: Optional[str] = None
    ) -> CoordinateValidationResult:
        """Validate a single coordinate pair"""
        errors = []
        warnings = []
        corrected_lat = latitude
        corrected_lon = longitude
        
        # Basic range validation
        if not (-90 <= latitude <= 90):
            errors.append(f"Latitude {latitude} is outside valid range (-90 to 90)")
        
        if not (-180 <= longitude <= 180):
            errors.append(f"Longitude {longitude} is outside valid range (-180 to 180)")
        
        # India bounds validation
        if not self._is_within_india_bounds(latitude, longitude):
            if self._is_near_india_bounds(latitude, longitude):
                warnings.append("Coordinates are near but outside Indian boundaries")
                # Try to correct minor errors
                corrected_lat, corrected_lon = self._correct_near_boundary_coordinates(
                    latitude, longitude
                )
            else:
                errors.append("Coordinates are outside Indian geographic boundaries")
        
        # Precision validation
        if self._has_excessive_precision(latitude, longitude):
            warnings.append("Coordinates have excessive precision, may be inaccurate")
        
        # Suspicious coordinate patterns
        if self._are_suspicious_coordinates(latitude, longitude):
            warnings.append("Coordinates appear to be placeholder or test values")
        
        return CoordinateValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_latitude=corrected_lat,
            corrected_longitude=corrected_lon,
            coordinate_system=coordinate_format
        )
    
    async def validate_coordinate_list(
        self, 
        coordinates: List[Tuple[float, float]]
    ) -> ValidationResult:
        """Validate a list of coordinates for polygon creation"""
        errors = []
        warnings = []
        
        if len(coordinates) < 3:
            errors.append(f"At least 3 coordinates required for polygon, got {len(coordinates)}")
        
        if len(coordinates) > 1000:
            warnings.append(f"Large number of coordinates ({len(coordinates)}) may impact performance")
        
        # Check for duplicate consecutive points
        duplicates = 0
        for i in range(1, len(coordinates)):
            if self._are_same_coordinates(coordinates[i-1], coordinates[i]):
                duplicates += 1
        
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate consecutive coordinates")
        
        # Check for self-intersecting polygon
        if len(coordinates) >= 4 and self._is_self_intersecting(coordinates):
            errors.append("Coordinates form a self-intersecting polygon")
        
        # Validate individual coordinates
        invalid_coords = 0
        for lat, lon in coordinates:
            result = await self.validate_coordinate(lat, lon)
            if not result.is_valid:
                invalid_coords += 1
        
        if invalid_coords > 0:
            errors.append(f"{invalid_coords} coordinates are invalid")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _is_within_india_bounds(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Indian boundaries"""
        return (
            self.bounds['MIN_LATITUDE'] <= lat <= self.bounds['MAX_LATITUDE'] and
            self.bounds['MIN_LONGITUDE'] <= lon <= self.bounds['MAX_LONGITUDE']
        )
    
    def _is_near_india_bounds(self, lat: float, lon: float, tolerance: float = 0.1) -> bool:
        """Check if coordinates are near Indian boundaries"""
        return (
            self.bounds['MIN_LATITUDE'] - tolerance <= lat <= self.bounds['MAX_LATITUDE'] + tolerance and
            self.bounds['MIN_LONGITUDE'] - tolerance <= lon <= self.bounds['MAX_LONGITUDE'] + tolerance
        )
    
    def _correct_near_boundary_coordinates(
        self, 
        lat: float, 
        lon: float
    ) -> Tuple[float, float]:
        """Attempt to correct coordinates that are slightly outside boundaries"""
        corrected_lat = max(
            self.bounds['MIN_LATITUDE'], 
            min(self.bounds['MAX_LATITUDE'], lat)
        )
        corrected_lon = max(
            self.bounds['MIN_LONGITUDE'], 
            min(self.bounds['MAX_LONGITUDE'], lon)
        )
        return corrected_lat, corrected_lon
    
    def _has_excessive_precision(self, lat: float, lon: float, max_decimals: int = 8) -> bool:
        """Check if coordinates have excessive decimal precision"""
        lat_str = str(lat)
        lon_str = str(lon)
        
        lat_decimals = len(lat_str.split('.')[-1]) if '.' in lat_str else 0
        lon_decimals = len(lon_str.split('.')[-1]) if '.' in lon_str else 0
        
        return lat_decimals > max_decimals or lon_decimals > max_decimals
    
    def _are_suspicious_coordinates(self, lat: float, lon: float) -> bool:
        """Check for suspicious coordinate patterns"""
        # Check for common placeholder values
        suspicious_patterns = [
            (0.0, 0.0),  # Null Island
            (lat == 0 and lon != 0),  # Zero latitude
            (lat != 0 and lon == 0),  # Zero longitude
            (lat == lon),  # Same lat/lon
            (lat % 10 == 0 and lon % 10 == 0),  # Round numbers
        ]
        
        for condition in suspicious_patterns:
            if isinstance(condition, tuple):
                if abs(lat - condition[0]) < 0.001 and abs(lon - condition[1]) < 0.001:
                    return True
            elif condition:
                return True
        
        return False
    
    def _are_same_coordinates(
        self, 
        coord1: Tuple[float, float], 
        coord2: Tuple[float, float], 
        tolerance: float = 0.0001
    ) -> bool:
        """Check if two coordinates are the same within tolerance"""
        return (
            abs(coord1[0] - coord2[0]) < tolerance and 
            abs(coord1[1] - coord2[1]) < tolerance
        )
    
    def _is_self_intersecting(self, coordinates: List[Tuple[float, float]]) -> bool:
        """Check if polygon is self-intersecting (simplified check)"""
        # This is a simplified implementation
        # For production, use a proper computational geometry library
        n = len(coordinates)
        if n < 4:
            return False
        
        # Check if any non-adjacent edges intersect
        for i in range(n):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # Skip adjacent edges
                
                if self._lines_intersect(
                    coordinates[i], coordinates[(i + 1) % n],
                    coordinates[j], coordinates[(j + 1) % n]
                ):
                    return True
        
        return False
    
    def _lines_intersect(
        self, 
        p1: Tuple[float, float], 
        p2: Tuple[float, float],
        p3: Tuple[float, float], 
        p4: Tuple[float, float]
    ) -> bool:
        """Check if two line segments intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


class ClaimValidator:
    """Validates FRA claim data"""
    
    def __init__(self):
        self.thresholds = VALIDATION_THRESHOLDS
        self.valid_states = INDIAN_STATES
        self.valid_claim_types = list(FRA_CLAIM_TYPES.keys())
    
    async def validate_claim(self, claim_data: Any) -> ValidationResult:
        """Validate FRA claim data"""
        errors = []
        warnings = []
        
        # Claim type validation
        if hasattr(claim_data, 'claim_type') and claim_data.claim_type:
            if claim_data.claim_type.value not in self.valid_claim_types:
                errors.append(f"Invalid claim type: {claim_data.claim_type}")
        else:
            errors.append("Claim type is required")
        
        # Holder name validation
        if hasattr(claim_data, 'holder_name') and claim_data.holder_name:
            name_errors = self._validate_holder_name(claim_data.holder_name)
            errors.extend(name_errors)
        else:
            warnings.append("Holder name is missing")
        
        # Area validation
        if hasattr(claim_data, 'area_hectares') and claim_data.area_hectares:
            area_errors = self._validate_area(claim_data.area_hectares)
            errors.extend(area_errors)
        
        # Survey numbers validation
        if hasattr(claim_data, 'survey_numbers') and claim_data.survey_numbers:
            survey_errors = self._validate_survey_numbers(claim_data.survey_numbers)
            errors.extend(survey_errors)
        
        # Location validation
        location_errors = self._validate_location_data(claim_data)
        errors.extend(location_errors)
        
        # Confidence score validation
        if hasattr(claim_data, 'confidence_score') and claim_data.confidence_score is not None:
            if not (0 <= claim_data.confidence_score <= 1):
                errors.append("Confidence score must be between 0 and 1")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_holder_name(self, name: str) -> List[str]:
        """Validate holder name"""
        errors = []
        
        if len(name.strip()) < self.thresholds['MIN_NAME_LENGTH']:
            errors.append(f"Holder name too short (minimum {self.thresholds['MIN_NAME_LENGTH']} characters)")
        
        if len(name) > self.thresholds['MAX_NAME_LENGTH']:
            errors.append(f"Holder name too long (maximum {self.thresholds['MAX_NAME_LENGTH']} characters)")
        
        # Check for suspicious patterns
        if re.match(r'^[0-9\s]+$', name):
            errors.append("Holder name appears to be only numbers")
        
        if re.match(r'^[^a-zA-Z\u0900-\u097F]+$', name):
            errors.append("Holder name should contain alphabetic characters")
        
        return errors
    
    def _validate_area(self, area: float) -> List[str]:
        """Validate area in hectares"""
        errors = []
        
        if area < self.thresholds['MIN_AREA_HECTARES']:
            errors.append(f"Area too small (minimum {self.thresholds['MIN_AREA_HECTARES']} hectares)")
        
        if area > self.thresholds['MAX_AREA_HECTARES']:
            errors.append(f"Area too large (maximum {self.thresholds['MAX_AREA_HECTARES']} hectares)")
        
        return errors
    
    def _validate_survey_numbers(self, survey_numbers: List[str]) -> List[str]:
        """Validate survey numbers"""
        errors = []
        
        for survey_num in survey_numbers:
            if len(survey_num.strip()) < self.thresholds['MIN_SURVEY_NUMBER_LENGTH']:
                errors.append(f"Survey number '{survey_num}' too short")
            
            if len(survey_num) > self.thresholds['MAX_SURVEY_NUMBER_LENGTH']:
                errors.append(f"Survey number '{survey_num}' too long")
            
            # Check format
            if not re.match(r'^[0-9A-Za-z/\-]+$', survey_num):
                errors.append(f"Survey number '{survey_num}' has invalid format")
        
        return errors
    
    def _validate_location_data(self, claim_data: Any) -> List[str]:
        """Validate location information"""
        errors = []
        
        # State validation
        if hasattr(claim_data, 'state') and claim_data.state:
            if claim_data.state not in self.valid_states:
                # Try fuzzy matching
                closest_state = self._find_closest_state(claim_data.state)
                if closest_state:
                    errors.append(f"Unknown state '{claim_data.state}'. Did you mean '{closest_state}'?")
                else:
                    errors.append(f"Unknown state: {claim_data.state}")
        
        return errors
    
    def _find_closest_state(self, state_name: str) -> Optional[str]:
        """Find closest matching state name"""
        state_lower = state_name.lower()
        
        for valid_state in self.valid_states:
            if state_lower in valid_state.lower() or valid_state.lower() in state_lower:
                return valid_state
        
        return None


class GeometryValidator:
    """Validates geometric data for claims"""
    
    def __init__(self):
        self.min_area = 0.0001  # 1 square meter
        self.max_area = 100000   # 100,000 hectares
    
    def validate_polygon(self, coordinates: List[List[float]]) -> ValidationResult:
        """Validate polygon coordinates"""
        errors = []
        warnings = []
        
        if len(coordinates) < 1:
            errors.append("Polygon must have at least one ring")
            return ValidationResult(False, errors, warnings)
        
        exterior_ring = coordinates[0]
        
        # Validate exterior ring
        ring_result = self._validate_ring(exterior_ring)
        errors.extend(ring_result.errors)
        warnings.extend(ring_result.warnings)
        
        # Validate interior rings (holes)
        for i, interior_ring in enumerate(coordinates[1:], 1):
            ring_result = self._validate_ring(interior_ring)
            if ring_result.errors:
                errors.extend([f"Interior ring {i}: {error}" for error in ring_result.errors])
            if ring_result.warnings:
                warnings.extend([f"Interior ring {i}: {warning}" for warning in ring_result.warnings])
        
        # Calculate and validate area
        area = self._calculate_polygon_area(exterior_ring)
        if area < self.min_area:
            errors.append(f"Polygon area too small: {area} hectares")
        elif area > self.max_area:
            errors.append(f"Polygon area too large: {area} hectares")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_ring(self, ring: List[List[float]]) -> ValidationResult:
        """Validate a polygon ring"""
        errors = []
        warnings = []
        
        if len(ring) < 4:
            errors.append("Ring must have at least 4 coordinates")
        
        # Check if ring is closed
        if len(ring) >= 2 and ring[0] != ring[-1]:
            errors.append("Ring must be closed (first and last coordinates must be the same)")
        
        # Check for minimum distinct points
        distinct_points = len(set(tuple(coord) for coord in ring))
        if distinct_points < 3:
            errors.append("Ring must have at least 3 distinct points")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _calculate_polygon_area(self, ring: List[List[float]]) -> float:
        """Calculate polygon area in hectares (simplified)"""
        if len(ring) < 3:
            return 0.0
        
        # Simple area calculation using shoelace formula
        # This is approximate and should be replaced with proper geodesic calculation
        area = 0.0
        n = len(ring)
        
        for i in range(n):
            j = (i + 1) % n
            area += ring[i][0] * ring[j][1]
            area -= ring[j][0] * ring[i][1]
        
        area = abs(area) / 2.0
        
        # Convert from degrees to approximate hectares
        # This is a rough conversion and should use proper geodesic calculations
        area_hectares = area * 12100  # Very rough approximation
        
        return area_hectares


# Utility functions
def validate_survey_number_format(survey_number: str) -> bool:
    """Check if survey number follows Indian patterns"""
    patterns = [
        r'^\d+$',           # Simple number: 123
        r'^\d+/\d+$',       # Fraction: 123/4
        r'^\d+[A-Za-z]$',   # Number with letter: 123A
        r'^\d+/\d+[A-Za-z]$',  # Complex: 123/4A
        r'^[A-Za-z]+\d+$',  # Letter then number: S123
    ]
    
    for pattern in patterns:
        if re.match(pattern, survey_number.strip()):
            return True
    
    return False


def normalize_location_name(name: str) -> str:
    """Normalize location names for comparison"""
    if not name:
        return ""
    
    # Remove extra spaces and convert to title case
    normalized = re.sub(r'\s+', ' ', name.strip()).title()
    
    # Handle common variations
    replacements = {
        'Distt.': 'District',
        'Teh.': 'Tehsil',
        'Vil.': 'Village',
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized


def is_valid_indian_name(name: str) -> bool:
    """Check if name follows Indian naming patterns"""
    if not name or len(name.strip()) < 2:
        return False
    
    # Allow English and major Indian scripts
    pattern = r'^[a-zA-Z\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]+$'
    
    return bool(re.match(pattern, name.strip()))