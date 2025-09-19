"""
Indian Document Enhancement System
Specialized accuracy improvements for Indian documents like PAN cards, Aadhaar, etc.
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

@dataclass
class IndianDocumentResult:
    """Result of Indian document processing"""
    document_type: str
    extracted_fields: Dict[str, str]
    confidence_score: float
    validation_results: Dict[str, bool]
    enhanced_features: Dict[str, any]

class IndianDocumentEnhancer:
    """Enhanced processing for Indian documents"""
    
    def __init__(self):
        self.setup_indian_patterns()
        self.setup_enhancement_techniques()
        self._initialize_enhancement_techniques()
        
    def setup_indian_patterns(self):
        """Setup patterns specific to Indian documents"""
        
        self.indian_document_patterns = {
            'pan_card': {
                'keywords': {
                    'primary': ['permanent account number', 'pan', 'income tax', 'govt of india'],
                    'secondary': ['name', 'father\'s name', 'date of birth', 'signature'],
                    'specific': ['pan no', 'pan number', 'card no', 'card number']
                },
                'patterns': [
                    r'PERMANENT\s+ACCOUNT\s+NUMBER',
                    r'P\.A\.N\.?\s*:?\s*([A-Z]{5}[0-9]{4}[A-Z]{1})',
                    r'PAN\s*:?\s*([A-Z]{5}[0-9]{4}[A-Z]{1})',
                    r'INCOME\s+TAX\s+DEPARTMENT',
                    r'GOVT\.?\s+OF\s+INDIA',
                    r'CARD\s+NO\.?\s*:?\s*([A-Z0-9]+)',
                    r'NAME\s*:?\s*([A-Z\s]+)',
                    r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'DOB\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'SIGNATURE\s*:?\s*([A-Z\s]+)'
                ],
                'pan_format': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$',
                'visual_indicators': ['pan_logo', 'income_tax_logo', 'government_seal'],
                'weight': 1.0
            },
            
            'aadhaar_card': {
                'keywords': {
                    'primary': ['aadhaar', 'uid', 'unique identification', 'government of india'],
                    'secondary': ['enrolment', 'enrollment', 'date of birth', 'gender'],
                    'specific': ['aadhaar no', 'aadhaar number', 'uid number']
                },
                'patterns': [
                    r'AADHAAR',
                    r'UNIQUE\s+IDENTIFICATION\s+AUTHORITY',
                    r'GOVERNMENT\s+OF\s+INDIA',
                    r'UID\s*:?\s*(\d{4}\s?\d{4}\s?\d{4})',
                    r'AADHAAR\s+NO\.?\s*:?\s*(\d{4}\s?\d{4}\s?\d{4})',
                    r'NAME\s*:?\s*([A-Z\s]+)',
                    r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'MOTHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'GENDER\s*:?\s*([MF])',
                    r'ADDRESS\s*:?\s*([A-Za-z0-9\s,.-]+)',
                    r'PIN\s*:?\s*(\d{6})'
                ],
                'aadhaar_format': r'^\d{4}\s?\d{4}\s?\d{4}$',
                'visual_indicators': ['aadhaar_logo', 'qr_code', 'barcode', 'photo'],
                'weight': 1.0
            },
            
            'driving_license': {
                'keywords': {
                    'primary': ['driving licence', 'driving license', 'transport authority', 'rto'],
                    'secondary': ['licence no', 'license no', 'valid from', 'valid upto'],
                    'state_specific': ['delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata']
                },
                'patterns': [
                    r'DRIVING\s+LICEN[CS]E',
                    r'LICEN[CS]E\s+NO\.?\s*:?\s*([A-Z0-9]+)',
                    r'NAME\s*:?\s*([A-Z\s]+)',
                    r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'VALID\s+FROM\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'VALID\s+UPTO\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'ADDRESS\s*:?\s*([A-Za-z0-9\s,.-]+)',
                    r'BLOOD\s+GROUP\s*:?\s*([A-Z]+[+-]?)',
                    r'VEHICLE\s+CLASS\s*:?\s*([A-Z0-9\s]+)'
                ],
                'visual_indicators': ['rto_logo', 'state_emblem', 'photo', 'signature'],
                'weight': 0.9
            },
            
            'voter_id': {
                'keywords': {
                    'primary': ['electoral photo identity card', 'epic', 'election commission'],
                    'secondary': ['voter id', 'elector\'s name', 'father\'s name'],
                    'specific': ['epic no', 'epic number', 'electoral roll']
                },
                'patterns': [
                    r'ELECTORAL\s+PHOTO\s+IDENTITY\s+CARD',
                    r'EPIC\s+NO\.?\s*:?\s*([A-Z]{3}[0-9]{7})',
                    r'ELECTOR\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'HUSBAND\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'GENDER\s*:?\s*([MF])',
                    r'AGE\s*:?\s*(\d{2})',
                    r'ADDRESS\s*:?\s*([A-Za-z0-9\s,.-]+)',
                    r'CONSTITUENCY\s*:?\s*([A-Z\s]+)',
                    r'PART\s+NO\.?\s*:?\s*(\d+)',
                    r'SERIAL\s+NO\.?\s*:?\s*(\d+)'
                ],
                'epic_format': r'^[A-Z]{3}[0-9]{7}$',
                'visual_indicators': ['election_commission_logo', 'national_symbol', 'photo'],
                'weight': 0.9
            },
            
            'passport': {
                'keywords': {
                    'primary': ['passport', 'passport no', 'passport number', 'ministry of external affairs'],
                    'secondary': ['place of birth', 'place of issue', 'nationality'],
                    'specific': ['passport no', 'passport number', 'file no']
                },
                'patterns': [
                    r'PASSPORT',
                    r'PASSPORT\s+NO\.?\s*:?\s*([A-Z]{1}[0-9]{7})',
                    r'FILE\s+NO\.?\s*:?\s*([A-Z0-9]+)',
                    r'NAME\s*:?\s*([A-Z\s]+)',
                    r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'MOTHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                    r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'PLACE\s+OF\s+BIRTH\s*:?\s*([A-Z\s]+)',
                    r'PLACE\s+OF\s+ISSUE\s*:?\s*([A-Z\s]+)',
                    r'NATIONALITY\s*:?\s*([A-Z\s]+)',
                    r'DATE\s+OF\s+ISSUE\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                    r'DATE\s+OF\s+EXPIRE\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})'
                ],
                'passport_format': r'^[A-Z]{1}[0-9]{7}$',
                'visual_indicators': ['passport_logo', 'national_emblem', 'photo', 'mrz'],
                'weight': 0.9
            }
        }
        
        # Indian name patterns
        self.indian_name_patterns = [
            r'^[A-Z\s]+$',  # All caps names
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last format
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$'  # First Middle Last format
        ]
        
        # Indian address patterns
        self.indian_address_patterns = [
            r'\d+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{6}',  # Full address with PIN
            r'[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{6}',  # Address with PIN
            r'PIN\s*:?\s*(\d{6})',  # PIN code
            r'POSTAL\s+CODE\s*:?\s*(\d{6})'  # Postal code
        ]
    
    def setup_enhancement_techniques(self):
        """Setup enhancement techniques for Indian documents"""
        # Define enhancement techniques after all methods are defined
        self.enhancement_techniques = {}
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text visibility"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                enhanced = clahe.apply(image)
            return enhanced
        except Exception as e:
            print(f"Contrast enhancement error: {e}")
            return image
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise in the image"""
        try:
            if len(image.shape) == 3:
                # Convert to grayscale for noise reduction
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                denoised = cv2.medianBlur(gray, 3)
                return denoised
            else:
                return cv2.medianBlur(image, 3)
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return image
    
    def sharpen_text(self, image: np.ndarray) -> np.ndarray:
        """Sharpen text for better OCR"""
        try:
            # Create sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened
        except Exception as e:
            print(f"Text sharpening error: {e}")
            return image
    
    def correct_colors(self, image: np.ndarray) -> np.ndarray:
        """Correct colors for better text visibility"""
        try:
            if len(image.shape) == 3:
                # Convert to HSV for color correction
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # Adjust saturation
                hsv[:,:,1] = hsv[:,:,1] * 1.2
                # Adjust value (brightness)
                hsv[:,:,2] = hsv[:,:,2] * 1.1
                corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                return corrected
            else:
                # For grayscale, just adjust brightness
                return cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        except Exception as e:
            print(f"Color correction error: {e}")
            return image
    
    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion"""
        try:
            # This is a simplified perspective correction
            # In a real implementation, you would detect document corners
            # and apply perspective transformation
            return image
        except Exception as e:
            print(f"Perspective correction error: {e}")
            return image
    
    def correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """Correct rotation of the document"""
        try:
            # This is a simplified rotation correction
            # In a real implementation, you would detect text orientation
            # and rotate accordingly
            return image
        except Exception as e:
            print(f"Rotation correction error: {e}")
            return image
    
    def detect_borders(self, image: np.ndarray) -> np.ndarray:
        """Detect document borders"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            return edges
        except Exception as e:
            print(f"Border detection error: {e}")
            return image
    
    def extract_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract text regions from the image"""
        try:
            # This is a simplified text region extraction
            # In a real implementation, you would use contour detection
            # and morphological operations to find text regions
            return [image]
        except Exception as e:
            print(f"Text region extraction error: {e}")
            return [image]
    
    def enhance_indian_document(self, image: np.ndarray = None, document_type: str = None) -> IndianDocumentResult:
        """Enhanced processing for Indian documents"""
        
        if image is not None:
            # Apply multiple enhancement techniques
            enhanced_image = self.apply_enhancement_pipeline(image)
            
            # Extract text with Indian-specific OCR settings
            text = self.extract_text_indian_optimized(enhanced_image)
        else:
            # For text-only processing (testing)
            text = ""
        
        # Classify document type if not provided
        if not document_type:
            document_type = self.classify_indian_document_type(text, image)
        
        # Extract fields using Indian-specific patterns
        extracted_fields = self.extract_indian_fields(text, document_type)
        
        # Validate extracted data
        validation_results = self.validate_indian_data(extracted_fields, document_type)
        
        # Calculate confidence score
        confidence = self.calculate_indian_confidence(extracted_fields, validation_results)
        
        return IndianDocumentResult(
            document_type=document_type,
            extracted_fields=extracted_fields,
            confidence_score=confidence,
            validation_results=validation_results,
            enhanced_features={
                'enhancement_applied': True,
                'text_length': len(text),
                'field_count': len(extracted_fields)
            }
        )
    
    def apply_enhancement_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive enhancement pipeline for Indian documents"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply enhancements
        enhanced = gray.copy()
        
        # 1. Noise reduction
        enhanced = cv2.medianBlur(enhanced, 3)
        
        # 2. Contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        
        # 3. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 4. Adaptive thresholding
        enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 5. Morphological operations
        kernel = np.ones((1,1), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def extract_text_indian_optimized(self, image: np.ndarray) -> str:
        """Extract text optimized for Indian documents"""
        
        # Multiple OCR configurations for Indian documents
        configs = [
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
            '--psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
            '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() '
        ]
        
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                confidence = self.calculate_text_confidence(text)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_text = text
            except Exception as e:
                continue
        
        return best_text
    
    def calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length factor
        score += min(len(text) / 100, 1.0) * 0.3
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        score += min(unique_chars / 26, 1.0) * 0.2
        
        # Indian document specific patterns
        indian_patterns = [
            r'[A-Z]{5}[0-9]{4}[A-Z]{1}',  # PAN format
            r'\d{4}\s?\d{4}\s?\d{4}',  # Aadhaar format
            r'[A-Z]{3}[0-9]{7}',  # EPIC format
            r'[A-Z]{1}[0-9]{7}',  # Passport format
            r'GOVT\.?\s+OF\s+INDIA',
            r'INCOME\s+TAX',
            r'ELECTION\s+COMMISSION'
        ]
        
        pattern_matches = sum(1 for pattern in indian_patterns if re.search(pattern, text))
        score += (pattern_matches / len(indian_patterns)) * 0.5
        
        return min(score, 1.0)
    
    def classify_indian_document_type(self, text: str, image: np.ndarray = None) -> str:
        """Classify Indian document type"""
        text_upper = text.upper()
        
        scores = {}
        for doc_type, config in self.indian_document_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in config['keywords']['primary']:
                if keyword.upper() in text_upper:
                    score += 3.0
            
            for keyword in config['keywords']['secondary']:
                if keyword.upper() in text_upper:
                    score += 2.0
            
            if 'specific' in config['keywords']:
                for keyword in config['keywords']['specific']:
                    if keyword.upper() in text_upper:
                        score += 1.5
            
            # Pattern matching
            for pattern in config['patterns']:
                if re.search(pattern, text_upper):
                    score += 2.0
            
            scores[doc_type] = score * config['weight']
        
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'unknown'
    
    def extract_indian_fields(self, text: str, document_type: str) -> Dict[str, str]:
        """Extract fields using Indian-specific patterns"""
        extracted = {}
        
        if document_type in self.indian_document_patterns:
            config = self.indian_document_patterns[document_type]
            
            for pattern in config['patterns']:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) == 2:
                            field_name = self.clean_indian_field_name(match[0])
                            field_value = match[1].strip()
                            if field_name and field_value:
                                extracted[field_name] = field_value
                        elif isinstance(match, str) and match.strip():
                            # Handle single group matches
                            if 'NAME' in pattern:
                                extracted['name'] = match.strip()
                            elif 'DATE' in pattern:
                                extracted['date_of_birth'] = match.strip()
                            elif 'ADDRESS' in pattern:
                                extracted['address'] = match.strip()
                            elif 'PAN' in pattern or r'P\.A\.N' in pattern:
                                extracted['pan'] = match.strip()
                            elif 'AADHAAR' in pattern or 'UID' in pattern:
                                extracted['aadhaar'] = match.strip()
                        elif hasattr(match, 'group'):  # Handle regex Match objects
                            # This shouldn't happen with findall, but just in case
                            match_str = str(match)
                            if match_str.strip():
                                if 'NAME' in pattern:
                                    extracted['name'] = match_str.strip()
                                elif 'DATE' in pattern:
                                    extracted['date_of_birth'] = match_str.strip()
                                elif 'ADDRESS' in pattern:
                                    extracted['address'] = match_str.strip()
                                elif 'PAN' in pattern or r'P\.A\.N' in pattern:
                                    extracted['pan'] = match_str.strip()
                                elif 'AADHAAR' in pattern or 'UID' in pattern:
                                    extracted['aadhaar'] = match_str.strip()
                except Exception as e:
                    # Skip problematic patterns
                    continue
        
        return extracted
    
    def clean_indian_field_name(self, name: str) -> str:
        """Clean Indian field names"""
        name = re.sub(r'[:\-=\[\]()]+$', '', name)  # Remove trailing punctuation
        name = re.sub(r'^[:\-=\[\]()]+', '', name)  # Remove leading punctuation
        name = name.strip().lower()
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^\w_]', '', name)
        return name
    
    def validate_indian_data(self, fields: Dict[str, str], document_type: str) -> Dict[str, bool]:
        """Validate Indian document data"""
        validation = {}
        
        if document_type == 'pan_card':
            validation['pan_format'] = self.validate_pan_format(fields.get('pan', ''))
            validation['name_format'] = self.validate_indian_name(fields.get('name', ''))
            validation['dob_format'] = self.validate_date_format(fields.get('date_of_birth', ''))
        
        elif document_type == 'aadhaar_card':
            validation['aadhaar_format'] = self.validate_aadhaar_format(fields.get('aadhaar', ''))
            validation['name_format'] = self.validate_indian_name(fields.get('name', ''))
            validation['dob_format'] = self.validate_date_format(fields.get('date_of_birth', ''))
        
        elif document_type == 'driving_license':
            validation['license_format'] = self.validate_license_format(fields.get('license_no', ''))
            validation['name_format'] = self.validate_indian_name(fields.get('name', ''))
            validation['dob_format'] = self.validate_date_format(fields.get('date_of_birth', ''))
        
        elif document_type == 'voter_id':
            validation['epic_format'] = self.validate_epic_format(fields.get('epic_no', ''))
            validation['name_format'] = self.validate_indian_name(fields.get('name', ''))
            validation['dob_format'] = self.validate_date_format(fields.get('date_of_birth', ''))
        
        elif document_type == 'passport':
            validation['passport_format'] = self.validate_passport_format(fields.get('passport_no', ''))
            validation['name_format'] = self.validate_indian_name(fields.get('name', ''))
            validation['dob_format'] = self.validate_date_format(fields.get('date_of_birth', ''))
        
        return validation
    
    def validate_pan_format(self, pan: str) -> bool:
        """Validate PAN card format"""
        return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan.upper()))
    
    def validate_aadhaar_format(self, aadhaar: str) -> bool:
        """Validate Aadhaar format"""
        cleaned = re.sub(r'\s', '', aadhaar)
        return bool(re.match(r'^\d{12}$', cleaned))
    
    def validate_license_format(self, license_no: str) -> bool:
        """Validate driving license format"""
        return len(license_no) >= 8 and license_no.isalnum()
    
    def validate_epic_format(self, epic: str) -> bool:
        """Validate EPIC format"""
        return bool(re.match(r'^[A-Z]{3}[0-9]{7}$', epic.upper()))
    
    def validate_passport_format(self, passport: str) -> bool:
        """Validate passport format"""
        return bool(re.match(r'^[A-Z]{1}[0-9]{7}$', passport.upper()))
    
    def validate_indian_name(self, name: str) -> bool:
        """Validate Indian name format"""
        if not name:
            return False
        return len(name) >= 2 and bool(re.match(r'^[A-Za-z\s]+$', name))
    
    def validate_date_format(self, date_str: str) -> bool:
        """Validate date format"""
        if not date_str:
            return False
        date_patterns = [
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{2}-\d{2}-\d{4}$',
            r'^\d{4}/\d{2}/\d{2}$',
            r'^\d{4}-\d{2}-\d{2}$'
        ]
        return any(re.match(pattern, date_str) for pattern in date_patterns)
    
    def calculate_indian_confidence(self, fields: Dict[str, str], validation: Dict[str, bool]) -> float:
        """Calculate confidence score for Indian documents"""
        if not fields:
            return 0.0
        
        # Base confidence from field count
        base_confidence = min(len(fields) / 10, 1.0) * 0.4
        
        # Validation bonus
        validation_score = sum(validation.values()) / max(len(validation), 1) * 0.4
        
        # Field completeness bonus
        completeness_score = 0.2
        
        return min(base_confidence + validation_score + completeness_score, 1.0)
    
    def _initialize_enhancement_techniques(self):
        """Initialize enhancement techniques after all methods are defined"""
        self.enhancement_techniques = {
            'contrast_enhancement': self.enhance_contrast,
            'noise_reduction': self.reduce_noise,
            'text_sharpening': self.sharpen_text,
            'color_correction': self.correct_colors,
            'perspective_correction': self.correct_perspective,
            'rotation_correction': self.correct_rotation,
            'border_detection': self.detect_borders,
            'text_region_extraction': self.extract_text_regions
        }

# Usage example
def test_indian_document_enhancement():
    """Test Indian document enhancement"""
    enhancer = IndianDocumentEnhancer()
    
    # Test with sample PAN card text
    sample_text = """
    PERMANENT ACCOUNT NUMBER
    P.A.N. : ABCDE1234F
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    NAME: RAJESH KUMAR SHARMA
    FATHER'S NAME: RAMESH KUMAR SHARMA
    DATE OF BIRTH: 15/01/1990
    SIGNATURE: RAJESH KUMAR SHARMA
    """
    
    result = enhancer.enhance_indian_document(None, 'pan_card')
    print(f"Document Type: {result.document_type}")
    print(f"Extracted Fields: {result.extracted_fields}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Validation: {result.validation_results}")

if __name__ == "__main__":
    test_indian_document_enhancement()
