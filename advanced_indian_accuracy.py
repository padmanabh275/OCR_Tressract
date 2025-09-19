"""
Advanced Accuracy Improvements for Indian Documents
Additional techniques beyond the basic Indian document enhancer
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import re
from typing import Dict, List, Tuple, Optional
import math

class AdvancedIndianAccuracy:
    """Advanced accuracy improvements for Indian documents"""
    
    def __init__(self):
        self.setup_advanced_patterns()
        self.setup_visual_detection()
        
    def setup_advanced_patterns(self):
        """Setup advanced patterns for Indian documents"""
        
        # PAN Card specific patterns
        self.pan_patterns = {
            'pan_number': [
                r'P\.A\.N\.?\s*:?\s*([A-Z]{5}[0-9]{4}[A-Z]{1})',
                r'PAN\s*:?\s*([A-Z]{5}[0-9]{4}[A-Z]{1})',
                r'Permanent\s+Account\s+Number\s*:?\s*([A-Z]{5}[0-9]{4}[A-Z]{1})',
                r'([A-Z]{5}[0-9]{4}[A-Z]{1})',  # Direct PAN format
            ],
            'name_patterns': [
                r'NAME\s*:?\s*([A-Z\s]+?)(?:\n|FATHER|MOTHER|DATE|SIGNATURE)',
                r'Name\s*:?\s*([A-Z\s]+?)(?:\n|Father|Mother|Date|Signature)',
                r'1\.\s*NAME\s*:?\s*([A-Z\s]+)',
                r'Name\s+of\s+the\s+Cardholder\s*:?\s*([A-Z\s]+)'
            ],
            'father_name_patterns': [
                r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|DATE|SIGNATURE)',
                r'Father\'?s?\s+Name\s*:?\s*([A-Z\s]+?)(?:\n|Date|Signature)',
                r'2\.\s*FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+)',
                r'Name\s+of\s+Father\s*:?\s*([A-Z\s]+)'
            ],
            'dob_patterns': [
                r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'Date\s+of\s+Birth\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'DOB\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'3\.\s*DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'Birth\s+Date\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})'
            ],
            'signature_patterns': [
                r'SIGNATURE\s*:?\s*([A-Z\s]+?)(?:\n|$)',
                r'Signature\s*:?\s*([A-Z\s]+?)(?:\n|$)',
                r'4\.\s*SIGNATURE\s*:?\s*([A-Z\s]+)',
                r'Signature\s+of\s+the\s+Cardholder\s*:?\s*([A-Z\s]+)'
            ]
        }
        
        # Aadhaar specific patterns
        self.aadhaar_patterns = {
            'aadhaar_number': [
                r'UID\s*:?\s*(\d{4}\s?\d{4}\s?\d{4})',
                r'AADHAAR\s+NO\.?\s*:?\s*(\d{4}\s?\d{4}\s?\d{4})',
                r'(\d{4}\s?\d{4}\s?\d{4})',  # Direct Aadhaar format
                r'Unique\s+Identification\s+Number\s*:?\s*(\d{4}\s?\d{4}\s?\d{4})'
            ],
            'gender_patterns': [
                r'GENDER\s*:?\s*([MF])',
                r'Gender\s*:?\s*([MF])',
                r'SEX\s*:?\s*([MF])',
                r'Male/Female\s*:?\s*([MF])'
            ],
            'pin_patterns': [
                r'PIN\s*:?\s*(\d{6})',
                r'Postal\s+Code\s*:?\s*(\d{6})',
                r'Pincode\s*:?\s*(\d{6})',
                r'(\d{6})'  # Direct PIN format
            ]
        }
        
        # Driving License patterns
        self.dl_patterns = {
            'license_number': [
                r'LICEN[CS]E\s+NO\.?\s*:?\s*([A-Z0-9]+)',
                r'License\s+No\.?\s*:?\s*([A-Z0-9]+)',
                r'DL\s+NO\.?\s*:?\s*([A-Z0-9]+)',
                r'([A-Z]{2}\d{2}\d{4}\d{7})'  # Common DL format
            ],
            'validity_patterns': [
                r'VALID\s+FROM\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'Valid\s+From\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'VALID\s+UPTO\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'Valid\s+Upto\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})'
            ],
            'blood_group_patterns': [
                r'BLOOD\s+GROUP\s*:?\s*([A-Z]+[+-]?)',
                r'Blood\s+Group\s*:?\s*([A-Z]+[+-]?)',
                r'B\.G\.\s*:?\s*([A-Z]+[+-]?)'
            ]
        }
    
    def setup_visual_detection(self):
        """Setup visual detection for Indian documents"""
        self.visual_indicators = {
            'pan_card': {
                'keywords': ['INCOME TAX', 'GOVT OF INDIA', 'PERMANENT ACCOUNT'],
                'logos': ['income_tax_logo', 'government_seal'],
                'layout': 'vertical_form'
            },
            'aadhaar_card': {
                'keywords': ['AADHAAR', 'UNIQUE IDENTIFICATION', 'GOVERNMENT OF INDIA'],
                'logos': ['aadhaar_logo', 'qr_code'],
                'layout': 'horizontal_form'
            },
            'driving_license': {
                'keywords': ['DRIVING LICENCE', 'TRANSPORT AUTHORITY', 'RTO'],
                'logos': ['rto_logo', 'state_emblem'],
                'layout': 'mixed_form'
            }
        }
    
    def enhance_image_for_indian_documents(self, image: np.ndarray) -> np.ndarray:
        """Advanced image enhancement for Indian documents"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Advanced noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Gamma correction
        gamma = 1.2
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
        enhanced = np.uint8(enhanced)
        
        # 4. Unsharp masking for text sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # 5. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 6. Adaptive thresholding with multiple methods
        thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
        
        # Combine both thresholding methods
        enhanced = cv2.bitwise_and(thresh1, thresh2)
        
        return enhanced
    
    def extract_text_with_multiple_psm(self, image: np.ndarray) -> Dict[str, str]:
        """Extract text using multiple PSM modes for better accuracy"""
        
        psm_modes = {
            'psm_3': '--psm 3',  # Fully automatic page segmentation
            'psm_4': '--psm 4',  # Assume a single column of text
            'psm_6': '--psm 6',  # Assume a single uniform block of text
            'psm_8': '--psm 8',  # Treat the image as a single word
            'psm_13': '--psm 13'  # Raw line. Treat the image as a single text line
        }
        
        results = {}
        
        for mode_name, psm_config in psm_modes.items():
            try:
                config = f"{psm_config} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() "
                text = pytesseract.image_to_string(image, config=config)
                results[mode_name] = text.strip()
            except Exception as e:
                results[mode_name] = ""
        
        return results
    
    def extract_pan_card_fields(self, text: str) -> Dict[str, str]:
        """Extract PAN card fields with high accuracy"""
        fields = {}
        
        # Extract PAN number
        for pattern in self.pan_patterns['pan_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['pan_number'] = match.group(1).strip()
                break
        
        # Extract name
        for pattern in self.pan_patterns['name_patterns']:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = re.sub(r'[^\w\s]', '', name)
                name = ' '.join(name.split())
                if len(name) > 3:  # Valid name length
                    fields['name'] = name
                    break
        
        # Extract father's name
        for pattern in self.pan_patterns['father_name_patterns']:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                father_name = match.group(1).strip()
                father_name = re.sub(r'[^\w\s]', '', father_name)
                father_name = ' '.join(father_name.split())
                if len(father_name) > 3:
                    fields['father_name'] = father_name
                    break
        
        # Extract date of birth
        for pattern in self.pan_patterns['dob_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dob = match.group(1).strip()
                if self.validate_date_format(dob):
                    fields['date_of_birth'] = dob
                    break
        
        # Extract signature
        for pattern in self.pan_patterns['signature_patterns']:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                signature = match.group(1).strip()
                signature = re.sub(r'[^\w\s]', '', signature)
                signature = ' '.join(signature.split())
                if len(signature) > 2:
                    fields['signature'] = signature
                    break
        
        return fields
    
    def extract_aadhaar_fields(self, text: str) -> Dict[str, str]:
        """Extract Aadhaar card fields with high accuracy"""
        fields = {}
        
        # Extract Aadhaar number
        for pattern in self.aadhaar_patterns['aadhaar_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                aadhaar = match.group(1).strip()
                # Clean and validate Aadhaar
                aadhaar = re.sub(r'\s', '', aadhaar)
                if len(aadhaar) == 12 and aadhaar.isdigit():
                    fields['aadhaar_number'] = aadhaar
                    break
        
        # Extract gender
        for pattern in self.aadhaar_patterns['gender_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['gender'] = match.group(1).strip().upper()
                break
        
        # Extract PIN code
        for pattern in self.aadhaar_patterns['pin_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pin = match.group(1).strip()
                if len(pin) == 6 and pin.isdigit():
                    fields['pin_code'] = pin
                    break
        
        return fields
    
    def extract_driving_license_fields(self, text: str) -> Dict[str, str]:
        """Extract driving license fields with high accuracy"""
        fields = {}
        
        # Extract license number
        for pattern in self.dl_patterns['license_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                license_no = match.group(1).strip()
                if len(license_no) >= 8:
                    fields['license_number'] = license_no
                    break
        
        # Extract validity dates
        for pattern in self.dl_patterns['validity_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group(1).strip()
                if self.validate_date_format(date):
                    if 'FROM' in pattern.upper():
                        fields['valid_from'] = date
                    else:
                        fields['valid_upto'] = date
        
        # Extract blood group
        for pattern in self.dl_patterns['blood_group_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                blood_group = match.group(1).strip().upper()
                if self.validate_blood_group(blood_group):
                    fields['blood_group'] = blood_group
                    break
        
        return fields
    
    def validate_date_format(self, date_str: str) -> bool:
        """Validate Indian date formats"""
        if not date_str:
            return False
        
        # Common Indian date formats
        patterns = [
            r'^\d{2}/\d{2}/\d{4}$',  # DD/MM/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{4}-\d{2}-\d{2}$'   # YYYY-MM-DD
        ]
        
        return any(re.match(pattern, date_str) for pattern in patterns)
    
    def validate_blood_group(self, blood_group: str) -> bool:
        """Validate blood group format"""
        valid_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        return blood_group in valid_groups
    
    def calculate_confidence_score(self, fields: Dict[str, str], document_type: str) -> float:
        """Calculate confidence score for extracted fields"""
        if not fields:
            return 0.0
        
        # Base score from field count
        base_score = min(len(fields) / 8, 1.0) * 0.4
        
        # Format validation bonus
        validation_score = 0.0
        if document_type == 'pan_card':
            if 'pan_number' in fields and self.validate_pan_format(fields['pan_number']):
                validation_score += 0.3
            if 'name' in fields and len(fields['name']) > 3:
                validation_score += 0.2
            if 'date_of_birth' in fields and self.validate_date_format(fields['date_of_birth']):
                validation_score += 0.1
        
        elif document_type == 'aadhaar_card':
            if 'aadhaar_number' in fields and len(fields['aadhaar_number']) == 12:
                validation_score += 0.3
            if 'name' in fields and len(fields['name']) > 3:
                validation_score += 0.2
            if 'pin_code' in fields and len(fields['pin_code']) == 6:
                validation_score += 0.1
        
        elif document_type == 'driving_license':
            if 'license_number' in fields and len(fields['license_number']) >= 8:
                validation_score += 0.3
            if 'name' in fields and len(fields['name']) > 3:
                validation_score += 0.2
            if 'blood_group' in fields and self.validate_blood_group(fields['blood_group']):
                validation_score += 0.1
        
        return min(base_score + validation_score, 1.0)
    
    def validate_pan_format(self, pan: str) -> bool:
        """Validate PAN card format"""
        return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan.upper()))
    
    def process_indian_document_advanced(self, image: np.ndarray, document_type: str) -> Dict:
        """Process Indian document with advanced accuracy"""
        
        # Enhance image
        enhanced_image = self.enhance_image_for_indian_documents(image)
        
        # Extract text with multiple PSM modes
        text_results = self.extract_text_with_multiple_psm(enhanced_image)
        
        # Combine all text results
        combined_text = ' '.join(text_results.values())
        
        # Extract fields based on document type
        fields = {}
        if document_type == 'pan_card':
            fields = self.extract_pan_card_fields(combined_text)
        elif document_type == 'aadhaar_card':
            fields = self.extract_aadhaar_fields(combined_text)
        elif document_type == 'driving_license':
            fields = self.extract_driving_license_fields(combined_text)
        
        # Calculate confidence
        confidence = self.calculate_confidence_score(fields, document_type)
        
        return {
            'success': True,
            'document_type': document_type,
            'extracted_fields': fields,
            'confidence_score': confidence,
            'text_results': text_results,
            'combined_text': combined_text,
            'enhancement_applied': True
        }

# Usage example
def test_advanced_indian_accuracy():
    """Test advanced Indian document accuracy"""
    enhancer = AdvancedIndianAccuracy()
    
    # Test PAN card text
    pan_text = """
    PERMANENT ACCOUNT NUMBER
    P.A.N. : ABCDE1234F
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    NAME: RAJESH KUMAR SHARMA
    FATHER'S NAME: RAMESH KUMAR SHARMA
    DATE OF BIRTH: 15/01/1990
    SIGNATURE: RAJESH KUMAR SHARMA
    """
    
    fields = enhancer.extract_pan_card_fields(pan_text)
    print(f"PAN Card Fields: {fields}")
    
    confidence = enhancer.calculate_confidence_score(fields, 'pan_card')
    print(f"Confidence Score: {confidence:.2f}")

if __name__ == "__main__":
    test_advanced_indian_accuracy()
