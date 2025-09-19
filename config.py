"""
Configuration file for the Document Information Extraction System
"""

# OCR Configuration
TESSERACT_CONFIG = '--psm 6'

# Image Preprocessing Settings
IMAGE_PREPROCESSING = {
    'denoise': True,
    'threshold': True,
    'morphology': True,
    'kernel_size': (1, 1)
}

# Field Extraction Patterns
PATTERNS = {
    'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
    'date_formats': [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
    ],
    'name_patterns': [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b'
    ],
    'address_pattern': r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)',
    'money_pattern': r'\$[\d,]+\.?\d*',
    'tax_year_pattern': r'(?:tax\s+year|year)\s*:?\s*(\d{4})'
}

# Document Type Keywords
DOCUMENT_KEYWORDS = {
    'driver_license': ['driver', 'license', 'dmv', 'department of motor vehicles'],
    'passport': ['passport', 'passport number', 'issuing country'],
    'birth_certificate': ['birth', 'certificate', 'born', 'birthplace'],
    'ssn_document': ['social security', 'ssn', 'ss#'],
    'utility_bill': ['utility', 'electric', 'gas', 'water', 'bill', 'statement'],
    'rental_agreement': ['rental', 'lease', 'tenant', 'landlord'],
    'tax_return': ['tax return', 'form 1040', 'irs', 'federal tax'],
    'w2_form': ['w-2', 'w2', 'wage and tax statement'],
    'bank_statement': ['bank', 'account', 'statement', 'checking', 'savings']
}

# Confidence Scoring Weights
CONFIDENCE_WEIGHTS = {
    'first_name': 1.0,
    'last_name': 1.0,
    'date_of_birth': 1.0,
    'ssn': 1.0,
    'current_address': 1.0,
    'document_type': 1.0,
    'marriage_date': 0.5,
    'birth_city': 0.5,
    'financial_data': 1.0
}

# Output Settings
OUTPUT_SETTINGS = {
    'save_json': True,
    'json_filename': 'extraction_results.json',
    'include_confidence': True,
    'include_metadata': True
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'extraction.log'
}
