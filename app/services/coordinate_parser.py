import re
import math
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession
from dataclasses import dataclass
import logging

from ..models.database import OCRResult, ExtractedCoordinate
from ..models.schemas import CoordinateFormatEnum, CoordinateValidationResult
from ..core.config import settings, mistral_ocr_config
from ..utils.validators import CoordinateValidator

logger = logging.getLogger(__name__)


@dataclass
class ParsedCoordinate:
    """Data class for parsed coordinate information"""
    latitude: float
    longitude: float
    format_type: CoordinateFormatEnum
    raw_text: str
    confidence: float = 0.0
    survey_number: Optional[str] = None
    plot_id: Optional[str] = None
    validation_errors: List[str] = None


class CoordinateParser:
    """
    Parser for extracting coordinates from OCR text using various Indian coordinate formats.
    Supports decimal degrees, DMS, UTM, and survey grid references.
    """
    
    def __init__(self):
        self.patterns = mistral_ocr_config.COORDINATE_PATTERNS
        self.validator = CoordinateValidator()
        
        # Compile regex patterns for performance
        self.compiled_patterns = {
            format_type: re.compile(pattern, re.IGNORECASE)
            for format_type, pattern in self.patterns.items()
        }
        
        # Common Indian survey number patterns
        self.survey_patterns = [
            r"(?:Sy\.?\s*No\.?|Survey\s+No\.?|S\.?\s*No\.?)\s*[:\-]?\s*(\d+[/\-]?\w*)",
            r"(?:Plot\s+No\.?|P\.?\s*No\.?)\s*[:\-]?\s*(\d+[/\-]?\w*)",
            r"(?:Khasra\s+No\.?|K\.?\s*No\.?)\s*[:\-]?\s*(\d+[/\-]?\w*)",
            r"(?:Survey\s+Settlement)\s*[:\-]?\s*(\d+[/\-]?\w*)"
        ]
        
        # Location keywords that might precede coordinates
        self.location_keywords = [
            "coordinates?", "location", "lat", "long", "latitude", "longitude",
            "north", "south", "east", "west", "position", "point", "boundary",
            "corner", "vertex", "गुणांक", "अक्षांश", "देशांतर"  # Hindi terms
        ]
    
    async def extract_coordinates(
        self, 
        ocr_result: OCRResult, 
        db: AsyncSession
    ) -> List[ExtractedCoordinate]:
        """
        Extract all coordinates from OCR result text.
        
        Args:
            ocr_result: OCR result containing text to parse
            db: Database session
            
        Returns:
            List of extracted coordinate records
        """
        try:
            parsed_coordinates = []
            text = ocr_result.raw_text
            
            # Extract different coordinate formats
            parsed_coordinates.extend(self._extract_decimal_degrees(text))
            parsed_coordinates.extend(self._extract_dms_coordinates(text))
            parsed_coordinates.extend(self._extract_utm_coordinates(text))
            parsed_coordinates.extend(self._extract_survey_grid(text))
            
            # Save to database
            coordinate_records = []
            for coord in parsed_coordinates:
                # Validate coordinate
                validation = await self.validator.validate_coordinate(
                    coord.latitude, coord.longitude
                )
                
                record = ExtractedCoordinate(
                    ocr_result_id=ocr_result.id,
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    coordinate_format=coord.format_type.value,
                    raw_coordinate_text=coord.raw_text,
                    is_valid=validation.is_valid,
                    validation_errors=validation.errors if not validation.is_valid else None,
                    confidence_score=coord.confidence,
                    survey_number=coord.survey_number,
                    plot_id=coord.plot_id
                )
                
                db.add(record)
                coordinate_records.append(record)
            
            await db.commit()
            
            logger.info(f"Extracted {len(coordinate_records)} coordinates from OCR result {ocr_result.id}")
            return coordinate_records
            
        except Exception as e:
            logger.error(f"Coordinate extraction failed: {str(e)}")
            await db.rollback()
            raise
    
    def _extract_decimal_degrees(self, text: str) -> List[ParsedCoordinate]:
        """Extract decimal degree coordinates (e.g., 12.3456°N, 78.9012°E)"""
        coordinates = []
        
        # Pattern for decimal degrees with direction indicators
        pattern = self.compiled_patterns["decimal_degrees"]
        matches = pattern.finditer(text)
        
        for match in matches:
            try:
                lat_val = float(match.group(1))
                lat_dir = match.group(2).upper()
                lon_val = float(match.group(3))
                lon_dir = match.group(4).upper()
                
                # Convert to standard decimal degrees
                if lat_dir == 'S':
                    lat_val = -lat_val
                if lon_dir == 'W':
                    lon_val = -lon_val
                
                # Check if coordinates are within Indian bounds
                if self._is_within_indian_bounds(lat_val, lon_val):
                    coord = ParsedCoordinate(
                        latitude=lat_val,
                        longitude=lon_val,
                        format_type=CoordinateFormatEnum.DECIMAL_DEGREES,
                        raw_text=match.group(0),
                        confidence=0.9  # High confidence for clear decimal format
                    )
                    
                    # Look for nearby survey numbers
                    coord.survey_number = self._find_nearby_survey_number(text, match.start(), match.end())
                    
                    coordinates.append(coord)
                    
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse decimal coordinate: {e}")
                continue
        
        return coordinates
    
    def _extract_dms_coordinates(self, text: str) -> List[ParsedCoordinate]:
        """Extract DMS (Degrees, Minutes, Seconds) coordinates"""
        coordinates = []
        
        pattern = self.compiled_patterns["dms"]
        matches = pattern.finditer(text)
        
        for match in matches:
            try:
                # Parse latitude DMS
                lat_deg = int(match.group(1))
                lat_min = int(match.group(2))
                lat_sec = float(match.group(3))
                lat_dir = match.group(4).upper()
                
                # Parse longitude DMS
                lon_deg = int(match.group(5))
                lon_min = int(match.group(6))
                lon_sec = float(match.group(7))
                lon_dir = match.group(8).upper()
                
                # Convert to decimal degrees
                lat_decimal = self._dms_to_decimal(lat_deg, lat_min, lat_sec)
                lon_decimal = self._dms_to_decimal(lon_deg, lon_min, lon_sec)
                
                # Apply direction
                if lat_dir == 'S':
                    lat_decimal = -lat_decimal
                if lon_dir == 'W':
                    lon_decimal = -lon_decimal
                
                if self._is_within_indian_bounds(lat_decimal, lon_decimal):
                    coord = ParsedCoordinate(
                        latitude=lat_decimal,
                        longitude=lon_decimal,
                        format_type=CoordinateFormatEnum.DMS,
                        raw_text=match.group(0),
                        confidence=0.85  # Good confidence for DMS format
                    )
                    
                    coord.survey_number = self._find_nearby_survey_number(text, match.start(), match.end())
                    coordinates.append(coord)
                    
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse DMS coordinate: {e}")
                continue
        
        return coordinates
    
    def _extract_utm_coordinates(self, text: str) -> List[ParsedCoordinate]:
        """Extract UTM coordinates and convert to lat/lon"""
        coordinates = []
        
        pattern = self.compiled_patterns["utm"]
        matches = pattern.finditer(text)
        
        for match in matches:
            try:
                zone = match.group(1)
                easting = int(match.group(2))
                northing = int(match.group(3))
                
                # Convert UTM to lat/lon (simplified conversion)
                lat, lon = self._utm_to_latlon(zone, easting, northing)
                
                if lat and lon and self._is_within_indian_bounds(lat, lon):
                    coord = ParsedCoordinate(
                        latitude=lat,
                        longitude=lon,
                        format_type=CoordinateFormatEnum.UTM,
                        raw_text=match.group(0),
                        confidence=0.7  # Lower confidence for UTM conversion
                    )
                    
                    coord.survey_number = self._find_nearby_survey_number(text, match.start(), match.end())
                    coordinates.append(coord)
                    
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse UTM coordinate: {e}")
                continue
        
        return coordinates
    
    def _extract_survey_grid(self, text: str) -> List[ParsedCoordinate]:
        """Extract survey grid references and attempt coordinate resolution"""
        coordinates = []
        
        # Look for survey settlement patterns with potential coordinates
        survey_pattern = re.compile(
            r"(?:Survey\s+Settlement|S\.S\.)\s*[:;\-]?\s*(\d+[/\-]?\w*)"
            r".*?(?:(?:Lat|Latitude)\s*[:;\-]?\s*([\d.]+))"
            r".*?(?:(?:Lon|Long|Longitude)\s*[:;\-]?\s*([\d.]+))",
            re.IGNORECASE | re.DOTALL
        )
        
        matches = survey_pattern.finditer(text)
        
        for match in matches:
            try:
                survey_ref = match.group(1)
                lat = float(match.group(2))
                lon = float(match.group(3))
                
                if self._is_within_indian_bounds(lat, lon):
                    coord = ParsedCoordinate(
                        latitude=lat,
                        longitude=lon,
                        format_type=CoordinateFormatEnum.SURVEY_GRID,
                        raw_text=match.group(0),
                        confidence=0.6,  # Lower confidence for survey grid
                        survey_number=survey_ref
                    )
                    
                    coordinates.append(coord)
                    
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse survey grid coordinate: {e}")
                continue
        
        return coordinates
    
    def _find_nearby_survey_number(self, text: str, start_pos: int, end_pos: int) -> Optional[str]:
        """Find survey numbers near coordinate positions"""
        # Search in a window around the coordinate
        window_size = 200
        search_start = max(0, start_pos - window_size)
        search_end = min(len(text), end_pos + window_size)
        search_text = text[search_start:search_end]
        
        for pattern in self.survey_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _dms_to_decimal(self, degrees: int, minutes: int, seconds: float) -> float:
        """Convert DMS to decimal degrees"""
        return degrees + minutes/60.0 + seconds/3600.0
    
    def _utm_to_latlon(self, zone: str, easting: int, northing: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Simplified UTM to lat/lon conversion for Indian coordinates.
        This is a basic implementation - for production, use a proper geodetic library.
        """
        try:
            # Extract zone number and hemisphere
            zone_num = int(zone[:-1])
            hemisphere = zone[-1].upper()
            
            # Indian UTM zones are typically 43N to 45N
            if zone_num < 43 or zone_num > 45:
                return None, None
            
            # Basic conversion constants (WGS84)
            a = 6378137.0  # Semi-major axis
            e2 = 0.00669438  # First eccentricity squared
            k0 = 0.9996  # Scale factor
            
            # Zone central meridian
            lon_origin = (zone_num - 1) * 6 - 180 + 3
            
            # Simplified conversion (this is a basic approximation)
            x = easting - 500000
            y = northing if hemisphere == 'N' else northing - 10000000
            
            # Basic lat calculation
            lat = y / (a * k0) * 180 / math.pi
            
            # Basic lon calculation
            lon = lon_origin + (x / (a * k0 * math.cos(math.radians(lat)))) * 180 / math.pi
            
            return lat, lon
            
        except Exception as e:
            logger.debug(f"UTM conversion failed: {e}")
            return None, None
    
    def _is_within_indian_bounds(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Indian geographic bounds"""
        return (settings.MIN_LATITUDE <= lat <= settings.MAX_LATITUDE and 
                settings.MIN_LONGITUDE <= lon <= settings.MAX_LONGITUDE)
    
    async def parse_coordinate_text(self, text: str) -> List[ParsedCoordinate]:
        """
        Parse coordinate text without database operations.
        Useful for testing and validation.
        """
        coordinates = []
        
        coordinates.extend(self._extract_decimal_degrees(text))
        coordinates.extend(self._extract_dms_coordinates(text))
        coordinates.extend(self._extract_utm_coordinates(text))
        coordinates.extend(self._extract_survey_grid(text))
        
        return coordinates
    
    def get_coordinate_summary(self, coordinates: List[ParsedCoordinate]) -> Dict[str, Any]:
        """Generate summary statistics for extracted coordinates"""
        if not coordinates:
            return {"total": 0}
        
        format_counts = {}
        confidence_scores = []
        
        for coord in coordinates:
            format_type = coord.format_type.value
            format_counts[format_type] = format_counts.get(format_type, 0) + 1
            if coord.confidence:
                confidence_scores.append(coord.confidence)
        
        summary = {
            "total": len(coordinates),
            "formats": format_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "has_survey_numbers": sum(1 for c in coordinates if c.survey_number) > 0
        }
        
        if coordinates:
            lats = [c.latitude for c in coordinates]
            lons = [c.longitude for c in coordinates]
            summary["bounds"] = {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lon": min(lons),
                "max_lon": max(lons)
            }
        
        return summary


class EnhancedCoordinateParser(CoordinateParser):
    """
    Enhanced coordinate parser with additional Indian-specific patterns
    and improved accuracy for FRA documents.
    """
    
    def __init__(self):
        super().__init__()
        
        # Additional patterns for Indian land records
        self.indian_patterns = {
            # Village boundary descriptions
            "boundary_desc": r"(?:पूर्व|उत्तर|पश्चिम|दक्षिण|East|North|West|South)\s*[:;\-]?\s*([\d.]+)",
            
            # Tehsil and village coordinates
            "administrative": r"(?:Tehsil|तहसील|Village|गांव)\s*[:;\-]?\s*(\w+)\s*(?:Lat|अक्षांश)\s*[:;\-]?\s*([\d.]+)",
            
            # GPS readings from devices
            "gps_reading": r"GPS\s*[:;\-]?\s*(?:Reading|Position)\s*[:;\-]?\s*([\d.]+)[°,\s]+([\d.]+)",
            
            # Cadastral survey marks
            "survey_mark": r"(?:Survey\s+Mark|S\.M\.)\s*[:;\-]?\s*(\d+)\s*[:;\-]?\s*([\d.]+)[°,\s]+([\d.]+)"
        }
        
        # Compile additional patterns
        for pattern_name, pattern in self.indian_patterns.items():
            self.compiled_patterns[pattern_name] = re.compile(pattern, re.IGNORECASE)
    
    def _extract_administrative_coordinates(self, text: str) -> List[ParsedCoordinate]:
        """Extract coordinates from administrative boundary descriptions"""
        coordinates = []
        
        pattern = self.compiled_patterns["administrative"]
        matches = pattern.finditer(text)
        
        for match in matches:
            try:
                location_name = match.group(1)
                coordinate_value = float(match.group(2))
                
                # This would need enhancement to determine if it's lat or lon
                # and find the corresponding value
                # For now, this is a placeholder for the pattern
                
            except (ValueError, IndexError):
                continue
        
        return coordinates
    
    def extract_contextual_coordinates(self, text: str) -> List[ParsedCoordinate]:
        """
        Extract coordinates using contextual analysis.
        Looks for coordinate patterns in specific document sections.
        """
        coordinates = []
        
        # Split text into sections based on common FRA document structure
        sections = self._split_into_sections(text)
        
        for section_name, section_text in sections.items():
            section_coords = self.parse_coordinate_text(section_text)
            
            # Adjust confidence based on section context
            for coord in section_coords:
                if section_name in ["boundary", "survey", "location"]:
                    coord.confidence *= 1.1  # Boost confidence for relevant sections
                elif section_name in ["header", "footer"]:
                    coord.confidence *= 0.8  # Reduce confidence for less relevant sections
            
            coordinates.extend(section_coords)
        
        return coordinates
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split document text into logical sections"""
        sections = {"full": text}
        
        # Simple section detection based on keywords
        boundary_keywords = ["boundary", "सीमा", "हद", "बाउंड्री"]
        survey_keywords = ["survey", "सर्वे", "नक्शा", "measurement"]
        
        lines = text.split('\n')
        current_section = "general"
        section_text = {}
        
        for line in lines:
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in boundary_keywords):
                current_section = "boundary"
            elif any(keyword in line_lower for keyword in survey_keywords):
                current_section = "survey"
            
            if current_section not in section_text:
                section_text[current_section] = ""
            
            section_text[current_section] += line + "\n"
        
        return section_text