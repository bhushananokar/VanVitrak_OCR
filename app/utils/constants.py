"""
Constants and configuration values for the Mistral OCR component.
Contains patterns, mappings, and static values used throughout the application.
"""

# File handling constants
ALLOWED_FILE_EXTENSIONS = [
    'pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif'
]

ALLOWED_MIME_TYPES = {
    'pdf': ['application/pdf'],
    'jpg': ['image/jpeg'],
    'jpeg': ['image/jpeg'], 
    'png': ['image/png'],
    'tiff': ['image/tiff'],
    'bmp': ['image/bmp'],
    'gif': ['image/gif']
}

# File size limits (in bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_SIZE = 1000  # Maximum documents per batch

# Indian geographic bounds
INDIA_BOUNDS = {
    'MIN_LATITUDE': 6.0,
    'MAX_LATITUDE': 37.0,
    'MIN_LONGITUDE': 68.0,
    'MAX_LONGITUDE': 97.0
}

# FRA claim types
FRA_CLAIM_TYPES = {
    'IFR': 'Individual Forest Rights',
    'CFR': 'Community Forest Resources Rights', 
    'CR': 'Community Rights'
}

# Processing status values
PROCESSING_STATUS = {
    'PENDING': 'pending',
    'PROCESSING': 'processing',
    'COMPLETED': 'completed',
    'FAILED': 'failed',
    'CANCELLED': 'cancelled'
}

# Coordinate format types
COORDINATE_FORMATS = {
    'DECIMAL_DEGREES': 'decimal_degrees',
    'DMS': 'dms',
    'UTM': 'utm',
    'SURVEY_GRID': 'survey_grid'
}

# Regular expressions for coordinate extraction
COORDINATE_PATTERNS = {
    'decimal_degrees': {
        'pattern': r'(\d+\.?\d*)[°\s]*([NS])[,\s]*(\d+\.?\d*)[°\s]*([EW])',
        'description': 'Decimal degrees with direction (12.34°N, 78.90°E)'
    },
    'dms': {
        'pattern': r'(\d+)[°]\s*(\d+)[\']\s*(\d+\.?\d*)["]\s*([NS])[,\s]*(\d+)[°]\s*(\d+)[\']\s*(\d+\.?\d*)["]\s*([EW])',
        'description': 'Degrees, Minutes, Seconds (12°34\'56"N, 78°90\'12"E)'
    },
    'utm': {
        'pattern': r'(\d+[NS])\s+(\d+)\s+(\d+)',
        'description': 'UTM coordinates with zone (43N 123456 7890123)'
    },
    'decimal_only': {
        'pattern': r'(\d{1,2}\.\d{4,8})[,\s]+(\d{2,3}\.\d{4,8})',
        'description': 'Plain decimal coordinates (12.3456, 78.9012)'
    }
}

# Survey number patterns for Indian land records
SURVEY_NUMBER_PATTERNS = [
    r'(?:Sy\.?\s*No\.?|Survey\s+No\.?|S\.?\s*No\.?)\s*[:\-]?\s*(\d+[/\-]?\w*)',
    r'(?:Plot\s+No\.?|P\.?\s*No\.?)\s*[:\-]?\s*(\d+[/\-]?\w*)',
    r'(?:Khasra\s+No\.?|K\.?\s*No\.?)\s*[:\-]?\s*(\d+[/\-]?\w*)',
    r'(?:Survey\s+Settlement|S\.S\.)\s*[:\-]?\s*(\d+[/\-]?\w*)',
    r'(?:Revenue\s+Survey|R\.S\.)\s*[:\-]?\s*(\d+[/\-]?\w*)'
]

# Name extraction patterns (English and Hindi)
NAME_PATTERNS = [
    r'(?:Name|नाम|नाव)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{3,50})',
    r'(?:Applicant|आवेदक|अर्जदार)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{3,50})',
    r'(?:Holder|धारक|धारणकर्ता)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{3,50})',
    r'(?:Claimant|दावेदार|हक्कदार)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{3,50})',
    r'(?:Beneficiary|लाभार्थी|फायदाप्राप्त)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{3,50})'
]

# Area extraction patterns
AREA_PATTERNS = [
    r'(?:Area|क्षेत्रफल|एरिया|क्षेत्र)\s*[:\-]?\s*([\d.]+)\s*(?:hectare|हेक्टेयर|acre|एकड़|ha|ac)',
    r'(?:Land|भूमि|जमीन)\s*[:\-]?\s*([\d.]+)\s*(?:hectare|हेक्टेयर|acre|एकड़|ha|ac)',
    r'(?:Total|कुल|एकूण)\s*[:\-]?\s*([\d.]+)\s*(?:hectare|हेक्टेयर|acre|एकड़|ha|ac)',
    r'([\d.]+)\s*(?:hectare|हेक्टेयर|ha)\s*(?:area|क्षेत्र)?',
    r'([\d.]+)\s*(?:acre|एकड़|ac)\s*(?:area|क्षेत्र)?'
]

# Location patterns for Indian administrative divisions
LOCATION_PATTERNS = {
    'village': [
        r'(?:Village|गांव|ग्राम|गाव)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})',
        r'(?:Gram|ग्राम)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})',
        r'(?:Gaon|गाओं)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})'
    ],
    'block': [
        r'(?:Block|ब्लॉक|खंड|ब्लाक)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})',
        r'(?:Tehsil|तहसील|तालुका)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})',
        r'(?:Taluka|तालुका)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})'
    ],
    'district': [
        r'(?:District|जिला|जिल्हा)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})',
        r'(?:Zilla|जिला)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})'
    ],
    'state': [
        r'(?:State|राज्य|राज्य)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})',
        r'(?:Pradesh|प्रदेश)\s*[:\-]?\s*([A-Za-z\s\u0900-\u097F]{2,30})'
    ]
}

# Date extraction patterns
DATE_PATTERNS = [
    r'(?:Date|दिनांक|तारीख|दिनांक)\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
    r'(?:Application\s+Date|आवेदन\s+दिनांक)\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
    r'(?:Survey\s+Date|सर्वेक्षण\s+दिनांक)\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
    r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',  # Generic date format
    r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2})'   # Short year format
]

# Rights claimed patterns
RIGHTS_PATTERNS = {
    'cultivation': [
        r'(?:Cultivation|खेती|कृषि|शेती)',
        r'(?:Agricultural|कृषि|शेतकी)',
        r'(?:Farming|खेती|शेती)',
        r'(?:Crop|फसल|पीक)'
    ],
    'grazing': [
        r'(?:Grazing|चराई|पशु\s+चराई|चरणे)',
        r'(?:Pasture|चारागाह|कुरणे)',
        r'(?:Cattle|गवाह|गुरे)',
        r'(?:Livestock|पशुधन|पशु)'
    ],
    'fishing': [
        r'(?:Fishing|मछली\s+पकड़ना|मासेमारी)',
        r'(?:Fish|मछली|मासा)',
        r'(?:Aquaculture|मत्स्य\s+पालन|मत्स्यपालन)'
    ],
    'water_access': [
        r'(?:Water|जल|पानी|पाणी)',
        r'(?:Well|कुआं|विहीर)',
        r'(?:Pond|तालाब|तलाव)',
        r'(?:Stream|नाला|ओढा)'
    ],
    'forest_produce': [
        r'(?:Forest\s+Produce|वन\s+उत्पाद|वन\s+उत्पादन)',
        r'(?:NTFP|गैर\s+काष्ठ\s+वन\s+उत्पाद)',
        r'(?:Minor\s+Forest\s+Produce|लघु\s+वन\s+उत्पाद)',
        r'(?:Timber|काष्ठ|लकड़ी)',
        r'(?:Bamboo|बांस|बांबू)'
    ],
    'traditional_use': [
        r'(?:Traditional|परंपरागत|पारंपारिक)',
        r'(?:Customary|प्रथागत|रूढ़िगत)',
        r'(?:Ancestral|पैतृक|वंशागत)'
    ]
}

# Common Indian state names for validation
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Delhi', 'Puducherry', 'Chandigarh', 'Dadra and Nagar Haveli', 
    'Daman and Diu', 'Lakshadweep', 'Ladakh', 'Jammu and Kashmir'
]

# Error messages
ERROR_MESSAGES = {
    'FILE_TOO_LARGE': 'File size exceeds maximum allowed size',
    'INVALID_FILE_TYPE': 'File type not supported',
    'INVALID_COORDINATES': 'Coordinates are outside valid geographic bounds',
    'OCR_FAILED': 'OCR processing failed',
    'BATCH_LIMIT_EXCEEDED': 'Batch size exceeds maximum allowed',
    'DOCUMENT_NOT_FOUND': 'Document not found',
    'INVALID_CLAIM_TYPE': 'Invalid FRA claim type',
    'INSUFFICIENT_COORDINATES': 'Insufficient coordinates to create geometry'
}

# Success messages
SUCCESS_MESSAGES = {
    'DOCUMENT_UPLOADED': 'Document uploaded successfully',
    'OCR_COMPLETED': 'OCR processing completed successfully',
    'BATCH_CREATED': 'Batch job created successfully',
    'COORDINATES_EXTRACTED': 'Coordinates extracted successfully',
    'GEOJSON_CREATED': 'GeoJSON created successfully'
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'MIN_CONFIDENCE_SCORE': 0.0,
    'MAX_CONFIDENCE_SCORE': 1.0,
    'MIN_AREA_HECTARES': 0.001,  # Minimum 0.001 hectares
    'MAX_AREA_HECTARES': 10000,  # Maximum 10,000 hectares
    'MIN_NAME_LENGTH': 3,
    'MAX_NAME_LENGTH': 100,
    'MIN_COORDINATES_FOR_POLYGON': 3,
    'MIN_SURVEY_NUMBER_LENGTH': 1,
    'MAX_SURVEY_NUMBER_LENGTH': 20
}

# Database configuration
DATABASE_DEFAULTS = {
    'POOL_SIZE': 20,
    'MAX_OVERFLOW': 0,
    'POOL_RECYCLE': 300,
    'POOL_PRE_PING': True
}

# OCR processing defaults
OCR_DEFAULTS = {
    'DEFAULT_CONFIDENCE': 0.85,
    'RETRY_ATTEMPTS': 3,
    'RETRY_DELAY': 1,  # seconds
    'TIMEOUT': 300,    # 5 minutes
    'BATCH_POLLING_INTERVAL': 5  # seconds
}

# GeoJSON configuration
GEOJSON_CONFIG = {
    'CRS': {
        'type': 'name',
        'properties': {
            'name': 'EPSG:4326'
        }
    },
    'PRECISION': 6,  # Decimal places for coordinates
    'AREA_PRECISION': 4,  # Decimal places for area calculations
    'PERIMETER_PRECISION': 2  # Decimal places for perimeter
}

# Logging configuration
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# API response codes
HTTP_STATUS_CODES = {
    'SUCCESS': 200,
    'CREATED': 201,
    'BAD_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'FORBIDDEN': 403,
    'NOT_FOUND': 404,
    'CONFLICT': 409,
    'UNPROCESSABLE_ENTITY': 422,
    'INTERNAL_SERVER_ERROR': 500
}

# Conversion factors
CONVERSION_FACTORS = {
    'ACRES_TO_HECTARES': 0.404686,
    'HECTARES_TO_ACRES': 2.47105,
    'METERS_PER_DEGREE_LAT': 111320,  # Approximate
    'SQUARE_METERS_TO_HECTARES': 0.0001
}

# Common abbreviations used in Indian land records
COMMON_ABBREVIATIONS = {
    'Sy.No.': 'Survey Number',
    'S.No.': 'Survey Number', 
    'P.No.': 'Plot Number',
    'K.No.': 'Khasra Number',
    'R.S.': 'Revenue Survey',
    'S.S.': 'Survey Settlement',
    'Vil.': 'Village',
    'Distt.': 'District',
    'Teh.': 'Tehsil',
    'ha': 'hectares',
    'ac': 'acres'
}

# Quality assessment criteria
QUALITY_CRITERIA = {
    'HIGH_QUALITY': {
        'min_confidence': 0.8,
        'min_coordinates': 8,
        'required_fields': ['holder_name', 'claim_type', 'area_hectares']
    },
    'MEDIUM_QUALITY': {
        'min_confidence': 0.6,
        'min_coordinates': 4,
        'required_fields': ['claim_type']
    },
    'LOW_QUALITY': {
        'min_confidence': 0.0,
        'min_coordinates': 1,
        'required_fields': []
    }
}

# Unicode ranges for Indian scripts
INDIAN_SCRIPT_RANGES = {
    'DEVANAGARI': (0x0900, 0x097F),  # Hindi, Marathi, Nepali
    'BENGALI': (0x0980, 0x09FF),     # Bengali, Assamese
    'GUJARATI': (0x0A80, 0x0AFF),    # Gujarati
    'GURMUKHI': (0x0A00, 0x0A7F),    # Punjabi
    'KANNADA': (0x0C80, 0x0CFF),     # Kannada
    'MALAYALAM': (0x0D00, 0x0D7F),   # Malayalam
    'ORIYA': (0x0B00, 0x0B7F),       # Odia
    'TAMIL': (0x0B80, 0x0BFF),       # Tamil
    'TELUGU': (0x0C00, 0x0C7F)       # Telugu
}