import re
from typing import List, Dict, Optional, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging
from dataclasses import dataclass

from ..models.database import Document, OCRResult, Claim, ExtractedCoordinate
from ..models.schemas import ClaimTypeEnum, ProcessingStatusEnum
from ..core.config import settings
from ..utils.validators import ClaimValidator

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    """Data class for extracted claim information"""
    claim_type: ClaimTypeEnum
    holder_name: Optional[str] = None
    area_hectares: Optional[float] = None
    survey_numbers: List[str] = None
    village: Optional[str] = None
    block: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    application_date: Optional[datetime] = None
    survey_date: Optional[datetime] = None
    rights_claimed: List[str] = None
    confidence_score: float = 0.0
    raw_text_source: str = ""


class DocumentProcessor:
    """
    Processes FRA documents to extract claim information from OCR results.
    Handles different document types and formats commonly found in Indian forest rights applications.
    """
    
    def __init__(self):
        self.validator = ClaimValidator()
        
        # FRA claim type patterns
        self.claim_type_patterns = {
            ClaimTypeEnum.IFR: [
                r"Individual\s+Forest\s+Rights?",
                r"IFR",
                r"व्यक्तिगत\s+वन\s+अधिकार",
                r"Individual\s+Rights?",
                r"Personal\s+Forest\s+Rights?"
            ],
            ClaimTypeEnum.CFR: [
                r"Community\s+Forest\s+Rights?",
                r"CFR",
                r"सामुदायिक\s+वन\s+अधिकार",
                r"Community\s+Rights?",
                r"Collective\s+Forest\s+Rights?"
            ],
            ClaimTypeEnum.CR: [
                r"Community\s+Rights?",
                r"CR",
                r"सामुदायिक\s+अधिकार",
                r"Traditional\s+Rights?",
                r"Customary\s+Rights?"
            ]
        }
        
        # Compile patterns for performance
        self.compiled_claim_patterns = {}
        for claim_type, patterns in self.claim_type_patterns.items():
            self.compiled_claim_patterns[claim_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Name extraction patterns
        self.name_patterns = [
            r"(?:Name|नाम)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)",
            r"(?:Applicant|आवेदक)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)",
            r"(?:Holder|धारक)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)",
            r"(?:Claimant|दावेदार)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)"
        ]
        
        # Area extraction patterns
        self.area_patterns = [
            r"(?:Area|क्षेत्रफल|एरिया)\s*[:;\-]?\s*([\d.]+)\s*(?:hectare|हेक्टेयर|acre|एकड़|ha)",
            r"(?:Land|भूमि)\s*[:;\-]?\s*([\d.]+)\s*(?:hectare|हेक्टेयर|acre|एकड़|ha)",
            r"(?:Total|कुल)\s*[:;\-]?\s*([\d.]+)\s*(?:hectare|हेक्टेयर|acre|एकड़|ha)"
        ]
        
        # Location patterns
        self.location_patterns = {
            "village": [
                r"(?:Village|गांव|ग्राम)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)",
                r"(?:Gram|ग्राम)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)"
            ],
            "block": [
                r"(?:Block|ब्लॉक|खंड)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)",
                r"(?:Tehsil|तहसील)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)"
            ],
            "district": [
                r"(?:District|जिला)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)",
                r"(?:Zilla|जिला)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)"
            ],
            "state": [
                r"(?:State|राज्य)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)",
                r"(?:Pradesh|प्रदेश)\s*[:;\-]?\s*([A-Za-z\s\u0900-\u097F]+)"
            ]
        }
        
        # Date patterns
        self.date_patterns = [
            r"(?:Date|दिनांक|तारीख)\s*[:;\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"(?:Application\s+Date|आवेदन\s+दिनांक)\s*[:;\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
            r"(?:Survey\s+Date|सर्वेक्षण\s+दिनांक)\s*[:;\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
        ]
        
        # Rights claimed patterns
        self.rights_patterns = {
            "cultivation": [
                r"(?:Cultivation|खेती|कृषि)",
                r"(?:Agricultural|कृषि)",
                r"(?:Farming|खेती)"
            ],
            "grazing": [
                r"(?:Grazing|चराई|पशु\s+चराई)",
                r"(?:Pasture|चारागाह)",
                r"(?:Cattle|गवाह)"
            ],
            "fishing": [
                r"(?:Fishing|मछली\s+पकड़ना)",
                r"(?:Fish|मछली)",
                r"(?:Aquaculture|मत्स्य\s+पालन)"
            ],
            "water_access": [
                r"(?:Water|जल|पानी)",
                r"(?:Well|कुआं)",
                r"(?:Pond|तालाब)"
            ],
            "forest_produce": [
                r"(?:Forest\s+Produce|वन\s+उत्पाद)",
                r"(?:NTFP|गैर\s+काष्ठ\s+वन\s+उत्पाद)",
                r"(?:Minor\s+Forest\s+Produce|लघु\s+वन\s+उत्पाद)"
            ]
        }
    
    async def extract_claims(
        self, 
        document: Document, 
        db: AsyncSession
    ) -> List[Claim]:
        """
        Extract FRA claims from a processed document.
        
        Args:
            document: Document record with OCR results
            db: Database session
            
        Returns:
            List of extracted claim records
        """
        try:
            # Get OCR results for the document
            ocr_results = await self._get_ocr_results(document.id, db)
            
            if not ocr_results:
                logger.warning(f"No OCR results found for document {document.id}")
                return []
            
            # Combine all OCR text
            combined_text = "\n".join([result.raw_text for result in ocr_results])
            
            # Extract claims from text
            extracted_claims = self._extract_claims_from_text(combined_text)
            
            # Enhance claims with coordinate information
            coordinates = await self._get_coordinates(document.id, db)
            enhanced_claims = self._enhance_claims_with_coordinates(extracted_claims, coordinates)
            
            # Save claims to database
            claim_records = []
            for claim_data in enhanced_claims:
                # Validate claim
                validation = await self.validator.validate_claim(claim_data)
                
                if validation.is_valid:
                    claim = Claim(
                        document_id=document.id,
                        claim_type=claim_data.claim_type.value,
                        holder_name=claim_data.holder_name,
                        area_hectares=claim_data.area_hectares,
                        survey_numbers=claim_data.survey_numbers,
                        village=claim_data.village,
                        block=claim_data.block,
                        district=claim_data.district,
                        state=claim_data.state,
                        application_date=claim_data.application_date,
                        survey_date=claim_data.survey_date,
                        rights_claimed=claim_data.rights_claimed,
                        confidence_score=claim_data.confidence_score,
                        status=ProcessingStatusEnum.PENDING.value
                    )
                    
                    db.add(claim)
                    claim_records.append(claim)
                else:
                    logger.warning(f"Claim validation failed: {validation.errors}")
            
            await db.commit()
            
            logger.info(f"Extracted {len(claim_records)} claims from document {document.id}")
            return claim_records
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {str(e)}")
            await db.rollback()
            raise
    
    def _extract_claims_from_text(self, text: str) -> List[ExtractedClaim]:
        """Extract claim information from OCR text"""
        claims = []
        
        # Split text into potential claim sections
        sections = self._split_text_into_sections(text)
        
        for section in sections:
            claim = self._extract_single_claim(section)
            if claim:
                claims.append(claim)
        
        # If no claims found in sections, try whole document
        if not claims:
            claim = self._extract_single_claim(text)
            if claim:
                claims.append(claim)
        
        return claims
    
    def _extract_single_claim(self, text: str) -> Optional[ExtractedClaim]:
        """Extract a single claim from text section"""
        # Determine claim type
        claim_type = self._identify_claim_type(text)
        if not claim_type:
            return None
        
        claim = ExtractedClaim(
            claim_type=claim_type,
            raw_text_source=text[:500]  # Store first 500 chars for reference
        )
        
        # Extract holder name
        claim.holder_name = self._extract_holder_name(text)
        
        # Extract area
        claim.area_hectares = self._extract_area(text)
        
        # Extract survey numbers
        claim.survey_numbers = self._extract_survey_numbers(text)
        
        # Extract location information
        location_info = self._extract_location_info(text)
        claim.village = location_info.get("village")
        claim.block = location_info.get("block")
        claim.district = location_info.get("district")
        claim.state = location_info.get("state")
        
        # Extract dates
        dates = self._extract_dates(text)
        claim.application_date = dates.get("application_date")
        claim.survey_date = dates.get("survey_date")
        
        # Extract rights claimed
        claim.rights_claimed = self._extract_rights_claimed(text)
        
        # Calculate confidence score
        claim.confidence_score = self._calculate_claim_confidence(claim, text)
        
        return claim
    
    def _identify_claim_type(self, text: str) -> Optional[ClaimTypeEnum]:
        """Identify the type of FRA claim from text"""
        scores = {claim_type: 0 for claim_type in ClaimTypeEnum}
        
        for claim_type, patterns in self.compiled_claim_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                scores[claim_type] += len(matches)
        
        # Return the claim type with highest score
        max_score = max(scores.values())
        if max_score > 0:
            for claim_type, score in scores.items():
                if score == max_score:
                    return claim_type
        
        return None
    
    def _extract_holder_name(self, text: str) -> Optional[str]:
        """Extract claim holder name from text"""
        for pattern in self.name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean and validate name
                name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
                if len(name) > 3 and len(name) < 100:  # Reasonable name length
                    return name
        
        return None
    
    def _extract_area(self, text: str) -> Optional[float]:
        """Extract area information from text"""
        for pattern in self.area_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    area = float(match.group(1))
                    # Convert acres to hectares if needed
                    if "acre" in match.group(0).lower():
                        area = area * 0.404686  # Acres to hectares conversion
                    
                    # Reasonable area bounds for FRA claims
                    if 0.01 <= area <= 1000:
                        return area
                except ValueError:
                    continue
        
        return None
    
    def _extract_survey_numbers(self, text: str) -> List[str]:
        """Extract survey numbers from text"""
        survey_numbers = []
        
        patterns = [
            r"(?:Sy\.?\s*No\.?|Survey\s+No\.?)\s*[:;\-]?\s*([\d/\-\w,\s]+)",
            r"(?:Plot\s+No\.?|Khasra\s+No\.?)\s*[:;\-]?\s*([\d/\-\w,\s]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                numbers_text = match.group(1)
                # Split by common delimiters
                numbers = re.split(r'[,\s]+', numbers_text.strip())
                for num in numbers:
                    num = num.strip()
                    if num and len(num) <= 20:  # Reasonable survey number length
                        survey_numbers.append(num)
        
        return list(set(survey_numbers))  # Remove duplicates
    
    def _extract_location_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract location information from text"""
        location_info = {}
        
        for location_type, patterns in self.location_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    location = match.group(1).strip()
                    location = re.sub(r'\s+', ' ', location)  # Normalize whitespace
                    if len(location) > 2 and len(location) < 50:
                        location_info[location_type] = location
                        break
        
        return location_info
    
    def _extract_dates(self, text: str) -> Dict[str, Optional[datetime]]:
        """Extract dates from text"""
        dates = {}
        
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = self._parse_date(date_str)
                
                if parsed_date:
                    # Determine date type based on context
                    context = match.group(0).lower()
                    if "application" in context:
                        dates["application_date"] = parsed_date
                    elif "survey" in context:
                        dates["survey_date"] = parsed_date
                    else:
                        # Default to application date if context unclear
                        if "application_date" not in dates:
                            dates["application_date"] = parsed_date
        
        return dates
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        date_formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%d/%m/%y", "%d-%m-%y", "%d.%m.%y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_rights_claimed(self, text: str) -> List[str]:
        """Extract types of rights claimed from text"""
        rights_claimed = []
        
        for right_type, patterns in self.rights_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    rights_claimed.append(right_type)
                    break
        
        return rights_claimed
    
    def _calculate_claim_confidence(self, claim: ExtractedClaim, text: str) -> float:
        """Calculate confidence score for extracted claim"""
        confidence = 0.0
        total_weight = 0.0
        
        # Claim type identified
        if claim.claim_type:
            confidence += 0.2
        total_weight += 0.2
        
        # Holder name found
        if claim.holder_name:
            confidence += 0.15
        total_weight += 0.15
        
        # Area information
        if claim.area_hectares:
            confidence += 0.15
        total_weight += 0.15
        
        # Survey numbers
        if claim.survey_numbers:
            confidence += 0.1
        total_weight += 0.1
        
        # Location information
        location_fields = [claim.village, claim.block, claim.district, claim.state]
        location_score = sum(1 for field in location_fields if field) / len(location_fields)
        confidence += location_score * 0.2
        total_weight += 0.2
        
        # Rights claimed
        if claim.rights_claimed:
            confidence += 0.1
        total_weight += 0.1
        
        # Dates
        if claim.application_date or claim.survey_date:
            confidence += 0.1
        total_weight += 0.1
        
        return confidence / total_weight if total_weight > 0 else 0.0
    
    def _split_text_into_sections(self, text: str) -> List[str]:
        """Split document text into logical sections for claim extraction"""
        # Simple section splitting based on common document patterns
        sections = []
        
        # Split by page breaks or large gaps
        page_sections = re.split(r'\n\s*\n\s*\n', text)
        
        for section in page_sections:
            # Further split by claim-related headers
            claim_sections = re.split(
                r'(?i)(?:claim\s+no\.?|application\s+no\.?|case\s+no\.?)\s*[:;\-]?\s*\d+',
                section
            )
            sections.extend([s.strip() for s in claim_sections if s.strip()])
        
        return sections if sections else [text]
    
    async def _get_ocr_results(self, document_id: str, db: AsyncSession) -> List[OCRResult]:
        """Get OCR results for a document"""
        from sqlalchemy import select
        
        stmt = select(OCRResult).where(OCRResult.document_id == document_id)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def _get_coordinates(self, document_id: str, db: AsyncSession) -> List[ExtractedCoordinate]:
        """Get extracted coordinates for a document"""
        from sqlalchemy import select
        
        stmt = select(ExtractedCoordinate).join(OCRResult).where(
            OCRResult.document_id == document_id
        )
        result = await db.execute(stmt)
        return result.scalars().all()
    
    def _enhance_claims_with_coordinates(
        self, 
        claims: List[ExtractedClaim], 
        coordinates: List[ExtractedCoordinate]
    ) -> List[ExtractedClaim]:
        """Enhance claims with coordinate information"""
        if not coordinates:
            return claims
        
        # Associate coordinates with claims based on survey numbers
        for claim in claims:
            if claim.survey_numbers:
                for coord in coordinates:
                    if coord.survey_number and coord.survey_number in claim.survey_numbers:
                        # Coordinate matches survey number - boost confidence
                        claim.confidence_score *= 1.1
                        break
        
        return claims