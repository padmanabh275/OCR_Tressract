"""
Dynamic Field Extraction System
Automatically detects and extracts field names from scanned documents
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import pytesseract
from PIL import Image

@dataclass
class DynamicField:
    """Represents a dynamically extracted field"""
    field_name: str
    field_value: str
    confidence: float
    position: Tuple[int, int, int, int]  # x, y, width, height
    field_type: str  # 'text', 'number', 'date', 'email', 'phone', 'address'
    context: str  # surrounding text for validation

@dataclass
class DocumentFields:
    """Complete field extraction result"""
    document_type: str
    all_fields: List[DynamicField]
    common_fields: Dict[str, str]  # Standard fields like name, dob, etc.
    custom_fields: Dict[str, str]  # Document-specific fields
    confidence_score: float

class DynamicFieldExtractor:
    """Extracts fields dynamically from any document type"""
    
    def __init__(self):
        self.setup_field_patterns()
        self.setup_common_field_mappings()
        
    def setup_field_patterns(self):
        """Setup patterns for different field types"""
        self.field_patterns = {
            'label_value': [
                r'([A-Za-z\s]+):\s*([^\n\r]+)',  # Label: Value
                r'([A-Za-z\s]+)\s*:\s*([^\n\r]+)',  # Label : Value
                r'([A-Za-z\s]+)\s*-\s*([^\n\r]+)',  # Label - Value
                r'([A-Za-z\s]+)\s*=\s*([^\n\r]+)',  # Label = Value
            ],
            'form_field': [
                r'([A-Za-z\s]+)\s*\[([^\]]+)\]',  # Field [Value]
                r'([A-Za-z\s]+)\s*\(([^)]+)\)',  # Field (Value)
                r'([A-Za-z\s]+)\s*:\s*([^\n\r]+)',  # Field: Value
            ],
            'table_row': [
                r'([A-Za-z\s]+)\s+([^\n\r]+)',  # Field Value
            ],
            'numbered_field': [
                r'(\d+\.?\s*[A-Za-z\s]+):\s*([^\n\r]+)',  # 1. Field: Value
                r'(\d+\.?\s*[A-Za-z\s]+)\s*-\s*([^\n\r]+)',  # 1. Field - Value
            ]
        }
        
        # Field type detection patterns
        self.type_patterns = {
            'date': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
                r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
            ],
            'phone': [
                r'\d{3}[-.]?\d{3}[-.]?\d{4}',
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
                r'\+\d{1,3}\s*\d{3}[-.]?\d{3}[-.]?\d{4}'
            ],
            'email': [
                r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
            ],
            'ssn': [
                r'\d{3}-?\d{2}-?\d{4}',
                r'\d{9}'
            ],
            'address': [
                r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)',
                r'[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[A-Za-z0-9\s,.-]*'
            ],
            'money': [
                r'\$[\d,]+\.?\d*',
                r'[\d,]+\.?\d*\s*(?:USD|dollars?|cents?)'
            ],
            'number': [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
            ],
            'percentage': [
                r'\d+(?:\.\d+)?%'
            ]
        }
    
    def setup_common_field_mappings(self):
        """Setup mappings for common field names"""
        self.common_field_mappings = {
            # Name variations
            'name': ['name', 'full name', 'complete name', 'legal name', 'given name', 'first name', 'last name'],
            'first_name': ['first name', 'given name', 'fname', 'first'],
            'last_name': ['last name', 'surname', 'family name', 'lname', 'last'],
            'father_name': ['father name', 'father\'s name', 'fathers name', 'dad name', 'parent name'],
            'mother_name': ['mother name', 'mother\'s name', 'mothers name', 'mom name'],
            
            # Date variations
            'date_of_birth': ['date of birth', 'dob', 'birth date', 'born', 'birthday', 'birth'],
            'marriage_date': ['marriage date', 'wedding date', 'married on', 'marriage'],
            'issue_date': ['issue date', 'issued on', 'date issued', 'issued'],
            'expiry_date': ['expiry date', 'expires', 'expire', 'valid until', 'expiration'],
            
            # Address variations
            'address': ['address', 'residence', 'home address', 'mailing address', 'current address'],
            'permanent_address': ['permanent address', 'permanent residence', 'home address'],
            'temporary_address': ['temporary address', 'current address', 'present address'],
            
            # ID variations
            'id_number': ['id number', 'id no', 'identification number', 'id'],
            'passport_number': ['passport number', 'passport no', 'passport #', 'passport'],
            'license_number': ['license number', 'license no', 'dl number', 'driving license'],
            'ssn': ['ssn', 'social security', 'ss#', 'social security number'],
            
            # Contact variations
            'phone': ['phone', 'telephone', 'mobile', 'cell', 'contact number', 'tel'],
            'email': ['email', 'e-mail', 'email address', 'electronic mail'],
            
            # Other common fields
            'gender': ['gender', 'sex', 'male/female', 'm/f'],
            'age': ['age', 'years old', 'yrs'],
            'occupation': ['occupation', 'profession', 'job', 'work', 'employment'],
            'nationality': ['nationality', 'citizenship', 'citizen of'],
            'religion': ['religion', 'faith', 'belief'],
            'marital_status': ['marital status', 'married', 'single', 'divorced', 'widowed']
        }
    
    def extract_dynamic_fields(self, text: str, image: np.ndarray = None) -> DocumentFields:
        """Extract all fields dynamically from document text and image"""
        
        # Extract fields using different methods
        label_value_fields = self.extract_label_value_fields(text)
        form_fields = self.extract_form_fields(text)
        table_fields = self.extract_table_fields(text)
        numbered_fields = self.extract_numbered_fields(text)
        
        # Combine all fields
        all_fields = label_value_fields + form_fields + table_fields + numbered_fields
        
        # Remove duplicates and merge similar fields
        unique_fields = self.merge_duplicate_fields(all_fields)
        
        # Classify field types
        typed_fields = self.classify_field_types(unique_fields)
        
        # Separate common and custom fields
        common_fields, custom_fields = self.separate_common_custom_fields(typed_fields)
        
        # Calculate overall confidence
        confidence = self.calculate_overall_confidence(typed_fields)
        
        return DocumentFields(
            document_type='dynamic',
            all_fields=typed_fields,
            common_fields=common_fields,
            custom_fields=custom_fields,
            confidence_score=confidence
        )
    
    def extract_label_value_fields(self, text: str) -> List[DynamicField]:
        """Extract fields in Label: Value format"""
        fields = []
        
        for pattern in self.field_patterns['label_value']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                label = match.group(1).strip()
                value = match.group(2).strip()
                
                if len(label) > 2 and len(value) > 0:
                    field = DynamicField(
                        field_name=self.clean_field_name(label),
                        field_value=value,
                        confidence=0.8,
                        position=(0, 0, 0, 0),  # Will be filled by OCR position detection
                        field_type='text',
                        context=match.group(0)
                    )
                    fields.append(field)
        
        return fields
    
    def extract_form_fields(self, text: str) -> List[DynamicField]:
        """Extract fields in form format"""
        fields = []
        
        for pattern in self.field_patterns['form_field']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                label = match.group(1).strip()
                value = match.group(2).strip()
                
                if len(label) > 2 and len(value) > 0:
                    field = DynamicField(
                        field_name=self.clean_field_name(label),
                        field_value=value,
                        confidence=0.7,
                        position=(0, 0, 0, 0),
                        field_type='text',
                        context=match.group(0)
                    )
                    fields.append(field)
        
        return fields
    
    def extract_table_fields(self, text: str) -> List[DynamicField]:
        """Extract fields from table-like structures"""
        fields = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for two-column format
            parts = line.split(None, 1)  # Split on first whitespace
            if len(parts) == 2:
                label = parts[0].strip()
                value = parts[1].strip()
                
                if len(label) > 2 and len(value) > 0:
                    field = DynamicField(
                        field_name=self.clean_field_name(label),
                        field_value=value,
                        confidence=0.6,
                        position=(0, 0, 0, 0),
                        field_type='text',
                        context=line
                    )
                    fields.append(field)
        
        return fields
    
    def extract_numbered_fields(self, text: str) -> List[DynamicField]:
        """Extract numbered fields"""
        fields = []
        
        for pattern in self.field_patterns['numbered_field']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                label = match.group(1).strip()
                value = match.group(2).strip()
                
                if len(label) > 2 and len(value) > 0:
                    field = DynamicField(
                        field_name=self.clean_field_name(label),
                        field_value=value,
                        confidence=0.7,
                        position=(0, 0, 0, 0),
                        field_type='text',
                        context=match.group(0)
                    )
                    fields.append(field)
        
        return fields
    
    def clean_field_name(self, name: str) -> str:
        """Clean and normalize field names"""
        # Remove common prefixes/suffixes
        name = re.sub(r'^\d+\.?\s*', '', name)  # Remove numbers
        name = re.sub(r'[:\-=\[\]()]+$', '', name)  # Remove trailing punctuation
        name = re.sub(r'^[:\-=\[\]()]+', '', name)  # Remove leading punctuation
        
        # Normalize case
        name = name.strip().lower()
        
        # Replace spaces with underscores
        name = re.sub(r'\s+', '_', name)
        
        # Remove special characters
        name = re.sub(r'[^\w_]', '', name)
        
        return name
    
    def merge_duplicate_fields(self, fields: List[DynamicField]) -> List[DynamicField]:
        """Merge duplicate fields and keep the best one"""
        field_dict = {}
        
        for field in fields:
            key = field.field_name.lower()
            if key in field_dict:
                # Keep the field with higher confidence
                if field.confidence > field_dict[key].confidence:
                    field_dict[key] = field
            else:
                field_dict[key] = field
        
        return list(field_dict.values())
    
    def classify_field_types(self, fields: List[DynamicField]) -> List[DynamicField]:
        """Classify field types based on content"""
        for field in fields:
            field.field_type = self.detect_field_type(field.field_value)
            # Update confidence based on type detection
            if field.field_type != 'text':
                field.confidence = min(field.confidence + 0.1, 1.0)
        
        return fields
    
    def detect_field_type(self, value: str) -> str:
        """Detect the type of a field value"""
        for field_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return field_type
        return 'text'
    
    def separate_common_custom_fields(self, fields: List[DynamicField]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Separate common fields from custom fields"""
        common_fields = {}
        custom_fields = {}
        
        for field in fields:
            field_name_lower = field.field_name.lower()
            
            # Check if it matches any common field
            matched_common = False
            for common_key, variations in self.common_field_mappings.items():
                if field_name_lower in variations or any(var in field_name_lower for var in variations):
                    common_fields[common_key] = field.field_value
                    matched_common = True
                    break
            
            if not matched_common:
                custom_fields[field.field_name] = field.field_value
        
        return common_fields, custom_fields
    
    def calculate_overall_confidence(self, fields: List[DynamicField]) -> float:
        """Calculate overall confidence score"""
        if not fields:
            return 0.0
        
        total_confidence = sum(field.confidence for field in fields)
        return total_confidence / len(fields)
    
    def get_field_suggestions(self, document_type: str) -> List[str]:
        """Get suggested field names based on document type"""
        suggestions = {
            'driver_license': [
                'license_number', 'class', 'endorsements', 'restrictions',
                'height', 'weight', 'hair_color', 'eye_color', 'sex',
                'issue_date', 'expiry_date', 'state', 'country'
            ],
            'passport': [
                'passport_number', 'issuing_country', 'nationality',
                'place_of_birth', 'issue_date', 'expiry_date',
                'place_of_issue', 'authority'
            ],
            'birth_certificate': [
                'place_of_birth', 'time_of_birth', 'attending_physician',
                'mother_maiden_name', 'father_name', 'registrar',
                'file_number', 'certificate_number'
            ],
            'utility_bill': [
                'account_number', 'service_address', 'billing_address',
                'service_period', 'due_date', 'amount_due',
                'previous_reading', 'current_reading', 'usage'
            ],
            'bank_statement': [
                'account_number', 'routing_number', 'statement_period',
                'opening_balance', 'closing_balance', 'total_deposits',
                'total_withdrawals', 'service_charges'
            ]
        }
        
        return suggestions.get(document_type, [])
    
    def extract_with_ocr_positioning(self, image: np.ndarray) -> List[DynamicField]:
        """Extract fields with OCR positioning information"""
        # Get OCR data with bounding boxes
        try:
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            fields = []
            current_text = ""
            current_bbox = None
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text:
                    if current_text:
                        current_text += " " + text
                    else:
                        current_text = text
                        current_bbox = (
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['width'][i],
                            ocr_data['height'][i]
                        )
                else:
                    if current_text:
                        # Process the accumulated text
                        extracted_fields = self.extract_dynamic_fields(current_text)
                        for field in extracted_fields.all_fields:
                            field.position = current_bbox or (0, 0, 0, 0)
                        fields.extend(extracted_fields.all_fields)
                        current_text = ""
                        current_bbox = None
            
            return fields
            
        except Exception as e:
            print(f"OCR positioning error: {e}")
            return []

# Usage example
def test_dynamic_field_extraction():
    """Test the dynamic field extraction"""
    extractor = DynamicFieldExtractor()
    
    sample_text = """
    Name: John Doe
    Date of Birth: 01/15/1990
    Address: 123 Main Street, City, State
    Phone: (555) 123-4567
    Email: john.doe@email.com
    License Number: D123456789
    Class: C
    Expires: 12/25/2025
    Height: 6'0"
    Weight: 180 lbs
    Hair: Brown
    Eyes: Blue
    """
    
    result = extractor.extract_dynamic_fields(sample_text)
    
    print("Dynamic Field Extraction Results:")
    print(f"Common Fields: {result.common_fields}")
    print(f"Custom Fields: {result.custom_fields}")
    print(f"Total Fields: {len(result.all_fields)}")
    print(f"Confidence: {result.confidence_score:.2f}")

if __name__ == "__main__":
    test_dynamic_field_extraction()
