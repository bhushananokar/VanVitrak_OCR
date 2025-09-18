import json
from typing import List, Dict, Optional, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from geojson import Feature, FeatureCollection, Point, Polygon, LineString
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon
from shapely.ops import transform
import pyproj
import logging
import uuid
from datetime import datetime

from ..models.database import Document, Claim, ExtractedCoordinate, ClaimGeometry
from ..models.schemas import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry, GeoJSONProperties
from ..core.config import mistral_ocr_config
from ..utils.validators import GeometryValidator

logger = logging.getLogger(__name__)


class GeoJSONConverter:
    """
    Converts extracted claims and coordinates to GeoJSON format.
    Handles various coordinate systems and creates proper geometric representations.
    """
    
    def __init__(self):
        self.validator = GeometryValidator()
        self.crs = mistral_ocr_config.GEOJSON_CRS
        
        # Define coordinate transformations for Indian coordinate systems
        self.transformers = {
            'utm_43n': pyproj.Transformer.from_crs('EPSG:32643', 'EPSG:4326', always_xy=True),
            'utm_44n': pyproj.Transformer.from_crs('EPSG:32644', 'EPSG:4326', always_xy=True),
            'utm_45n': pyproj.Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True),
        }
    
    async def create_geojson_from_document(
        self, 
        document_id: uuid.UUID, 
        db: AsyncSession
    ) -> GeoJSONFeatureCollection:
        """
        Create GeoJSON FeatureCollection from all claims in a document.
        
        Args:
            document_id: Document UUID
            db: Database session
            
        Returns:
            GeoJSON FeatureCollection with all claim features
        """
        try:
            # Get document and validate
            document = await db.get(Document, document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Get all claims for the document
            claims = await self._get_document_claims(document_id, db)
            
            # Convert claims to GeoJSON features
            features = []
            for claim in claims:
                feature = await self._create_feature_from_claim(claim, db)
                if feature:
                    features.append(feature)
            
            # Create metadata
            metadata = {
                "document_id": str(document_id),
                "document_name": document.original_filename,
                "processed_at": datetime.now().isoformat(),
                "total_claims": len(claims),
                "total_features": len(features),
                "extraction_method": "mistral_ocr"
            }
            
            # Create FeatureCollection
            feature_collection = GeoJSONFeatureCollection(
                features=features,
                crs=self.crs
            )
            
            # Add metadata to the collection
            feature_collection_dict = feature_collection.dict()
            feature_collection_dict["metadata"] = metadata
            
            return GeoJSONFeatureCollection(**feature_collection_dict)
            
        except Exception as e:
            logger.error(f"GeoJSON creation failed for document {document_id}: {str(e)}")
            raise
    
    async def create_geometry_from_coordinates(
        self, 
        claim_id: uuid.UUID, 
        coordinates: List[ExtractedCoordinate],
        db: AsyncSession
    ) -> Optional[ClaimGeometry]:
        """
        Create ClaimGeometry from extracted coordinates.
        
        Args:
            claim_id: Claim UUID
            coordinates: List of extracted coordinates
            db: Database session
            
        Returns:
            ClaimGeometry record or None if cannot create geometry
        """
        try:
            if not coordinates or len(coordinates) < 3:
                logger.warning(f"Insufficient coordinates for claim {claim_id}")
                return None
            
            # Sort coordinates to create a logical boundary
            sorted_coords = self._sort_coordinates_for_polygon(coordinates)
            
            # Create Shapely polygon
            coord_pairs = [(coord.longitude, coord.latitude) for coord in sorted_coords]
            
            # Ensure polygon is closed
            if coord_pairs[0] != coord_pairs[-1]:
                coord_pairs.append(coord_pairs[0])
            
            shapely_polygon = ShapelyPolygon(coord_pairs)
            
            # Validate geometry
            if not shapely_polygon.is_valid:
                logger.warning(f"Invalid polygon geometry for claim {claim_id}")
                return None
            
            # Create GeoJSON geometry
            geojson_geometry = {
                "type": "Polygon",
                "coordinates": [coord_pairs]
            }
            
            # Calculate area and perimeter
            area_hectares = self._calculate_area_hectares(shapely_polygon)
            perimeter_meters = self._calculate_perimeter_meters(shapely_polygon)
            
            # Create ClaimGeometry record
            claim_geometry = ClaimGeometry(
                claim_id=claim_id,
                geometry=f"POLYGON(({','.join([f'{x} {y}' for x, y in coord_pairs])}))",
                geojson=geojson_geometry,
                area_calculated=area_hectares,
                perimeter_calculated=perimeter_meters,
                source_type="coordinate_extraction",
                geometry_quality=self._assess_geometry_quality(coordinates, shapely_polygon),
                validation_status="validated"
            )
            
            db.add(claim_geometry)
            await db.commit()
            await db.refresh(claim_geometry)
            
            logger.info(f"Created geometry for claim {claim_id} with area {area_hectares:.2f} hectares")
            return claim_geometry
            
        except Exception as e:
            logger.error(f"Geometry creation failed for claim {claim_id}: {str(e)}")
            await db.rollback()
            return None
    
    async def create_point_geometry(
        self, 
        claim_id: uuid.UUID, 
        coordinate: ExtractedCoordinate,
        db: AsyncSession
    ) -> Optional[ClaimGeometry]:
        """
        Create point geometry from a single coordinate.
        
        Args:
            claim_id: Claim UUID
            coordinate: Single coordinate point
            db: Database session
            
        Returns:
            ClaimGeometry record with point geometry
        """
        try:
            # Create GeoJSON point geometry
            geojson_geometry = {
                "type": "Point",
                "coordinates": [coordinate.longitude, coordinate.latitude]
            }
            
            # Create ClaimGeometry record
            claim_geometry = ClaimGeometry(
                claim_id=claim_id,
                geometry=f"POINT({coordinate.longitude} {coordinate.latitude})",
                geojson=geojson_geometry,
                source_type="coordinate_extraction",
                geometry_quality="medium",  # Point geometries have medium quality
                validation_status="validated"
            )
            
            db.add(claim_geometry)
            await db.commit()
            await db.refresh(claim_geometry)
            
            logger.info(f"Created point geometry for claim {claim_id}")
            return claim_geometry
            
        except Exception as e:
            logger.error(f"Point geometry creation failed for claim {claim_id}: {str(e)}")
            await db.rollback()
            return None
    
    def create_feature_from_data(
        self,
        geometry: Dict[str, Any],
        properties: Dict[str, Any],
        feature_id: Optional[str] = None
    ) -> GeoJSONFeature:
        """
        Create GeoJSON Feature from geometry and properties data.
        
        Args:
            geometry: GeoJSON geometry dict
            properties: Feature properties dict
            feature_id: Optional feature ID
            
        Returns:
            GeoJSON Feature
        """
        geojson_geometry = GeoJSONGeometry(**geometry)
        geojson_properties = GeoJSONProperties(**properties)
        
        return GeoJSONFeature(
            geometry=geojson_geometry,
            properties=geojson_properties,
            id=feature_id
        )
    
    async def _create_feature_from_claim(
        self, 
        claim: Claim, 
        db: AsyncSession
    ) -> Optional[GeoJSONFeature]:
        """Create GeoJSON feature from claim data"""
        try:
            # Get claim geometry
            geometry_data = await self._get_claim_geometry(claim.id, db)
            
            if not geometry_data:
                # Try to create geometry from coordinates
                coordinates = await self._get_claim_coordinates(claim.id, db)
                if coordinates:
                    if len(coordinates) >= 3:
                        geometry_record = await self.create_geometry_from_coordinates(
                            claim.id, coordinates, db
                        )
                        if geometry_record:
                            geometry_data = geometry_record.geojson
                    elif len(coordinates) == 1:
                        geometry_record = await self.create_point_geometry(
                            claim.id, coordinates[0], db
                        )
                        if geometry_record:
                            geometry_data = geometry_record.geojson
            
            if not geometry_data:
                logger.warning(f"No geometry available for claim {claim.id}")
                return None
            
            # Create properties
            properties = GeoJSONProperties(
                claim_id=claim.id,
                claim_type=claim.claim_type,
                holder_name=claim.holder_name,
                area_hectares=claim.area_hectares,
                survey_numbers=claim.survey_numbers,
                village=claim.village,
                block=claim.block,
                district=claim.district,
                state=claim.state,
                confidence_score=claim.confidence_score,
                extraction_method=claim.extraction_method,
                status=claim.status,
                rights_claimed=claim.rights_claimed,
                application_date=claim.application_date.isoformat() if claim.application_date else None,
                survey_date=claim.survey_date.isoformat() if claim.survey_date else None
            )
            
            # Create geometry
            geojson_geometry = GeoJSONGeometry(**geometry_data)
            
            # Create feature
            feature = GeoJSONFeature(
                geometry=geojson_geometry,
                properties=properties,
                id=str(claim.id)
            )
            
            return feature
            
        except Exception as e:
            logger.error(f"Feature creation failed for claim {claim.id}: {str(e)}")
            return None
    
    def _sort_coordinates_for_polygon(
        self, 
        coordinates: List[ExtractedCoordinate]
    ) -> List[ExtractedCoordinate]:
        """
        Sort coordinates to create a logical polygon boundary.
        Uses centroid-based angular sorting.
        """
        if len(coordinates) <= 3:
            return coordinates
        
        # Calculate centroid
        center_lat = sum(coord.latitude for coord in coordinates) / len(coordinates)
        center_lon = sum(coord.longitude for coord in coordinates) / len(coordinates)
        
        # Sort by angle from centroid
        def angle_from_center(coord):
            import math
            return math.atan2(coord.latitude - center_lat, coord.longitude - center_lon)
        
        return sorted(coordinates, key=angle_from_center)
    
    def _calculate_area_hectares(self, polygon: ShapelyPolygon) -> float:
        """Calculate polygon area in hectares using geodesic calculation"""
        try:
            # For rough calculation, use simple degree-to-meter conversion
            # For production, use proper geodesic area calculation
            
            # Get bounds
            minx, miny, maxx, maxy = polygon.bounds
            
            # Approximate meters per degree at the polygon centroid
            center_lat = (miny + maxy) / 2
            meters_per_deg_lat = 111320  # Approximately constant
            meters_per_deg_lon = 111320 * abs(math.cos(math.radians(center_lat)))
            
            # Transform polygon to projected coordinates (approximate)
            import math
            def transform_to_meters(x, y):
                return (x * meters_per_deg_lon, y * meters_per_deg_lat)
            
            # Transform polygon
            from shapely.ops import transform
            polygon_meters = transform(transform_to_meters, polygon)
            
            # Calculate area in square meters, then convert to hectares
            area_sqm = polygon_meters.area
            area_hectares = area_sqm / 10000  # 1 hectare = 10,000 sq meters
            
            return round(area_hectares, 4)
            
        except Exception as e:
            logger.warning(f"Area calculation failed: {e}")
            return 0.0
    
    def _calculate_perimeter_meters(self, polygon: ShapelyPolygon) -> float:
        """Calculate polygon perimeter in meters"""
        try:
            # Similar to area calculation, use approximate transformation
            center_lat = polygon.centroid.y
            meters_per_deg_lat = 111320
            meters_per_deg_lon = 111320 * abs(math.cos(math.radians(center_lat)))
            
            def transform_to_meters(x, y):
                return (x * meters_per_deg_lon, y * meters_per_deg_lat)
            
            from shapely.ops import transform
            polygon_meters = transform(transform_to_meters, polygon)
            
            return round(polygon_meters.length, 2)
            
        except Exception as e:
            logger.warning(f"Perimeter calculation failed: {e}")
            return 0.0
    
    def _assess_geometry_quality(
        self, 
        coordinates: List[ExtractedCoordinate], 
        polygon: ShapelyPolygon
    ) -> str:
        """Assess quality of generated geometry"""
        # Calculate average confidence
        avg_confidence = sum(c.confidence_score or 0 for c in coordinates) / len(coordinates)
        
        # Check polygon validity
        if not polygon.is_valid:
            return "low"
        
        # Check coordinate count
        if len(coordinates) < 4:
            return "low"
        elif len(coordinates) < 8:
            return "medium"
        
        # Check confidence score
        if avg_confidence > 0.8:
            return "high"
        elif avg_confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    async def _get_document_claims(self, document_id: uuid.UUID, db: AsyncSession) -> List[Claim]:
        """Get all claims for a document"""
        stmt = select(Claim).where(Claim.document_id == document_id)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def _get_claim_geometry(self, claim_id: uuid.UUID, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get geometry data for a claim"""
        stmt = select(ClaimGeometry).where(ClaimGeometry.claim_id == claim_id)
        result = await db.execute(stmt)
        geometry_record = result.scalar_one_or_none()
        
        return geometry_record.geojson if geometry_record else None
    
    async def _get_claim_coordinates(self, claim_id: uuid.UUID, db: AsyncSession) -> List[ExtractedCoordinate]:
        """Get coordinates associated with a claim"""
        # This would need to be implemented based on how coordinates are linked to claims
        # For now, return empty list
        return []
    
    def export_to_file(self, geojson_data: GeoJSONFeatureCollection, file_path: str) -> bool:
        """Export GeoJSON data to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_data.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"GeoJSON exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    def validate_geojson(self, geojson_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate GeoJSON structure and content"""
        errors = []
        
        try:
            # Check basic structure
            if geojson_data.get("type") != "FeatureCollection":
                errors.append("Invalid GeoJSON type, must be FeatureCollection")
            
            features = geojson_data.get("features", [])
            if not isinstance(features, list):
                errors.append("Features must be a list")
            
            # Validate each feature
            for i, feature in enumerate(features):
                if feature.get("type") != "Feature":
                    errors.append(f"Feature {i}: Invalid type")
                
                geometry = feature.get("geometry")
                if not geometry:
                    errors.append(f"Feature {i}: Missing geometry")
                
                properties = feature.get("properties")
                if not isinstance(properties, dict):
                    errors.append(f"Feature {i}: Properties must be an object")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]


# Import required math module
import math